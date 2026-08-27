"""Classifier-free guidance over two prompts.

    logits = logits_neutral + gamma * (logits_asked - logits_neutral)

Both branches are the same weights on two prompts, so the perturbation is
bounded by what the prompt difference does to the distribution -- unlike
DExperts, which extrapolates a whole logit vector against a second model and
collapsed past beta=1 (fail.md section 1). ``gamma=1`` is ordinary conditional
decoding and reproduces it exactly.

This was written for control-token prefixes, where the adapter demonstrably
conditions on the control -- the true prefix scores the reference at loss 0.62
against an inverted prefix's 0.78, on 53 of 60 examples -- but the shift is too
small to move a greedy argmax, so plain generation ignores it entirely.
Guidance amplifies that exact difference. Nothing here is specific to controls,
though: any pair of prompts that differ in one intended way can be dialled this
way, which is why it lives in lib rather than in the ladder that needed it.

The knob does not come free. On the control ladder, raising gamma trades
calibration against faithfulness monotonically -- at an asked ratio of 0.40,
gamma=2 gives a control error of +0.320 at NLI 0.897, gamma=12 an error of
+0.081 at NLI 0.344. Read fail.md section 7 before reaching for a large gamma.
"""

from __future__ import annotations

import torch


@torch.inference_mode()
def generate(
    lm,
    asked: list[str],
    neutral: list[str],
    *,
    gamma: float,
    max_new_tokens: int,
    instruction: str | None = None,
    task: str | None = None,
) -> list[str]:
    """Greedy decode from the guided mixture of two prompt branches.

    ``asked`` and ``neutral`` are parallel source texts: same items, differing
    only in whatever is being dialled. They must be the same length.

    Two caches, not one. The branches have different prompts and therefore
    different keys and values; a shared cache would feed each branch the
    other's history and the mixture would be meaningless.
    """
    if len(asked) != len(neutral):
        raise ValueError(
            f"asked and neutral must be parallel, got {len(asked)} and "
            f"{len(neutral)}"
        )
    if not asked:
        return []

    branches = {}
    for key, srcs in (("asked", asked), ("neutral", neutral)):
        enc = lm.encode_batch([lm.build_prompt(t, instruction, task) for t in srcs])
        branches[key] = {
            "ids": enc["input_ids"],
            "mask": enc["attention_mask"],
            # Left padding, so positions come from the mask rather than arange.
            "pos": (enc["attention_mask"].cumsum(-1) - 1).clamp_min(0),
            "cache": None,
        }

    device = branches["asked"]["ids"].device
    n = branches["asked"]["ids"].shape[0]
    eos = lm.tokenizer.eos_token_id
    pad = lm.tokenizer.pad_token_id if lm.tokenizer.pad_token_id is not None else eos
    done = torch.zeros(n, dtype=torch.bool, device=device)
    out: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        logits = {}
        for key, b in branches.items():
            res = lm.model(
                input_ids=b["ids"],
                attention_mask=b["mask"],
                position_ids=b["pos"],
                past_key_values=b["cache"],
                use_cache=True,
            )
            b["cache"] = res.past_key_values
            logits[key] = res.logits[:, -1, :].float()

        mixed = logits["neutral"] + gamma * (logits["asked"] - logits["neutral"])
        nxt = torch.where(
            done,
            torch.full((n,), pad, device=device, dtype=torch.long),
            mixed.argmax(-1),
        )
        out.append(nxt)
        done = done | (nxt == eos)
        if bool(done.all()):
            break
        for b in branches.values():
            b["ids"] = nxt.unsqueeze(1)
            b["mask"] = torch.cat([b["mask"], (~done).long().unsqueeze(1)], dim=1)
            b["pos"] = b["pos"][:, -1:] + 1

    if not out:
        return ["" for _ in asked]
    gen = torch.stack(out, dim=1)
    return [
        lm.tokenizer.decode(
            [int(t) for t in row if int(t) not in (pad, eos)],
            skip_special_tokens=True,
        ).strip()
        for row in gen
    ]
