"""Classifier-free guidance over two prompts.

    logits = logits_neutral + gamma * (logits_asked - logits_neutral)

``gamma=1`` reduces to ordinary conditional decoding.
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

    ``asked`` and ``neutral`` are parallel source texts, same length, differing
    only in whatever is being dialled.
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
