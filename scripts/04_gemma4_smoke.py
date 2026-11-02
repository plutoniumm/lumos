"""Gemma-4 wiring gate. Run once the weights finish downloading.

    python scripts/04_gemma4_smoke.py                      # config.GEMMA4
    python scripts/04_gemma4_smoke.py google/gemma-4-E4B-it

Checks architecture-specific behaviour that ``scripts/02_hook_noop.py`` cannot,
since it runs on a 135M model.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import hub, models
from lib.hooks import InjectionSite, resolve_module, steered
from lib.metrics import readability

HARD = (
    "The proliferation of antimicrobial resistance determinants among "
    "commensal Enterobacteriaceae constitutes a substantial epidemiological "
    "concern, notwithstanding the attenuated virulence of these organisms."
)

MAX_NEW = 96
PROBE_MAX_NEW = 16
PROBE_ALPHA = 4.0


@dataclass
class Report:
    rows: list[tuple[str, bool]] = field(default_factory=list)

    def check(self, name: str, ok: bool, *lines: str) -> None:
        self.rows.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for line in lines:
            print(f"       {line}")

    @property
    def ok(self) -> bool:
        return all(ok for _, ok in self.rows)

    def summary(self) -> None:
        print("\n" + "-" * 78)
        for name, ok in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        passed = sum(ok for _, ok in self.rows)
        print(f"\n{passed}/{len(self.rows)} checks passed")


@torch.inference_mode()
def greedy(lm, batch: dict, n_new: int) -> torch.Tensor:
    """Deterministic generation, ids on CPU so two runs compare bit-for-bit.

    ``eos_token_id`` is deliberately not passed: Gemma-4 stops on a list.
    """
    torch.manual_seed(config.SEED)
    out = lm.model.generate(
        **batch,
        max_new_tokens=n_new,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=(
            lm.tokenizer.eos_token_id
            if lm.tokenizer.pad_token_id is None
            else lm.tokenizer.pad_token_id
        ),
    )
    return out.detach().to("cpu")


@torch.inference_mode()
def layer_output(lm, layer: int, batch: dict) -> torch.Tensor:
    """One decoder layer's raw output, float32 on CPU. Cast before move: a fused
    ``.to(device, dtype)`` drops the storage offset of an MPS half tensor."""
    mod = resolve_module(lm, layer, "resid_post")
    cap: dict[str, torch.Tensor] = {}

    def probe(_m, _a, out):
        hidden = out[0] if isinstance(out, tuple) else out
        cap.setdefault("h", hidden.detach().float().to("cpu"))
        return None

    handle = mod.register_forward_hook(probe)
    try:
        lm.model(**batch)
    finally:
        handle.remove()
    return cap["h"]


def preflight(repo_id: str) -> int:
    ok, why = hub.is_complete(repo_id)
    print(f"[gemma4] {repo_id}: {why}")
    if not ok and (hub.local_path(repo_id) / "config.json").exists():
        return 2
    template = hub.local_path(repo_id) / "chat_template.jinja"
    if ok and not template.exists():
        print(
            f"[gemma4] {template} missing -- the download predates '.jinja' being "
            f"added to config.DOWNLOAD_ALLOW_SUFFIXES. Re-run "
            f"scripts/00_download.py gemma4."
        )
        return 2
    return 0


def main(argv: list[str]) -> int:
    torch.manual_seed(config.SEED)
    repo_id = argv[0] if argv else config.GEMMA4

    rc = preflight(repo_id)
    if rc:
        return rc

    t0 = time.perf_counter()
    lm = models.load(repo_id)
    load_s = time.perf_counter() - t0

    text_cfg = getattr(lm.model.config, "text_config", lm.model.config)
    mid = lm.n_layers // 2
    rep = Report()

    print(
        f"\nmodel        {lm.repo_id}\n"
        f"class        {type(lm.model).__name__}\n"
        f"tokenizer    {type(lm.tokenizer).__name__}\n"
        f"n_layers     {lm.n_layers}\n"
        f"hidden_size  {lm.hidden_size}\n"
        f"layer class  {type(lm.layer(mid)).__name__}\n"
        f"mlp class    {type(lm.mlp(mid)).__name__}\n"
        f"dtype        {lm.dtype}\n"
        f"device       {lm.device}\n"
        f"stack        torch {torch.__version__}, "
        f"transformers {transformers.__version__}\n"
        f"load         {load_s:.1f}s\n"
    )

    want_layers = int(text_cfg.num_hidden_layers)
    want_hidden = int(text_cfg.hidden_size)
    rep.check(
        "1. the layers we hook are the text stack",
        lm.n_layers == want_layers and lm.hidden_size == want_hidden,
        f"found {lm.n_layers} layers of width {lm.hidden_size}; "
        f"text_config claims {want_layers} x {want_hidden}",
        "a mismatch means _find_layers walked into a vision or audio encoder",
    )

    prompt = lm.build_prompt(HARD, config.INSTRUCTIONS["simplify"])
    batch = lm.encode(prompt)
    prompt_len = int(batch["input_ids"].shape[1])
    rep.check(
        "2. the chat template renders and carries the source text",
        HARD in prompt and prompt != HARD and prompt_len > 0,
        f"{prompt_len} tokens, template head {prompt[:40]!r}",
        f"generation prompt tail {prompt[-40:]!r}",
    )

    base = layer_output(lm, mid, batch)
    rep.check(
        f"3. layer {mid} returns a 3-D [batch, seq, hidden] tensor",
        base.dim() == 3 and base.shape[-1] == lm.hidden_size,
        f"shape {tuple(base.shape)}; a 4-D shape here is the Gemma-3n altup "
        "stack, which the injector treats as [streams, batch, seq, hidden]",
    )

    t0 = time.perf_counter()
    out = greedy(lm, batch, MAX_NEW)
    gen_s = time.perf_counter() - t0
    rewrite = lm.decode_new(out, prompt_len)[0].strip()
    n_new = int(out.shape[1]) - prompt_len

    print(f"\nsource:\n  {HARD}\n")
    print(
        f"rewrite ({n_new} new tokens, {n_new / gen_s:.1f} tok/s greedy):\n"
        f"  {rewrite}\n"
    )

    src, dst = readability.measure(HARD), readability.measure(rewrite)
    print(
        f"fkgl         {src.fkgl:.1f} -> {dst.fkgl:.1f}\n"
        f"words/sent   {src.words_per_sentence:.1f} -> {dst.words_per_sentence:.1f}\n"
        f"hard words   {src.pct_hard_words:.2f} -> {dst.pct_hard_words:.2f}\n"
        f"degenerate   {dst.degenerate}\n"
    )
    rep.check(
        "4. the model rewrites rather than echoing or commenting",
        bool(rewrite) and not dst.degenerate and rewrite != HARD,
        "not a gate on quality -- an empty or degenerate rewrite here means the "
        "chat template or the eos set is wrong, not that steering failed",
    )

    ref = float(base.norm(dim=-1).median())
    g = torch.Generator().manual_seed(1)
    v = torch.randn(lm.hidden_size, generator=g)
    scaled = v / v.norm() * ref
    print(f"layer {mid}: ||h|| median {ref:.2f}, probe vector scaled to match\n")

    baseline = greedy(lm, batch, PROBE_MAX_NEW)
    site = InjectionSite(mid, scaled, 0.0, positions="all", norm_preserve=True)
    with steered(lm, [site]):
        noop = greedy(lm, batch, PROBE_MAX_NEW)
    rep.check(
        "5. alpha=0 is token-for-token identical to no hooks",
        torch.equal(baseline, noop),
        f"{PROBE_MAX_NEW} greedy tokens, positions=all, norm_preserve=True",
    )

    zeros = torch.zeros(lm.hidden_size)
    site = InjectionSite(mid, zeros, 3.0, positions="all", norm_preserve=True)
    with steered(lm, [site]):
        zero_vec = greedy(lm, batch, PROBE_MAX_NEW)
    h_zero = layer_output(lm, mid, batch)
    rep.check(
        "6. a zero vector at alpha=3 with norm_preserve is bit-exact",
        torch.equal(baseline, zero_vec) and torch.equal(h_zero, base),
        "the rescale divides a tensor's norm by its own norm, so the ratio is "
        "exactly 1.0 -- a zero delta alone would hide a renormalisation bug",
    )

    site = InjectionSite(mid, scaled, PROBE_ALPHA, positions="all", norm_preserve=True)
    with steered(lm, [site]):
        pushed = greedy(lm, batch, PROBE_MAX_NEW)
    rep.check(
        f"7. alpha={PROBE_ALPHA} at layer {mid} changes the tokens",
        not torch.equal(baseline, pushed),
        f"baseline: {lm.decode_new(baseline, prompt_len)[0][:60]!r}",
        f"steered : {lm.decode_new(pushed, prompt_len)[0][:60]!r}",
        "without this, check 5 passes on a hook the model never reads",
    )

    leaked = sum(
        len(resolve_module(lm, i, "resid_post")._forward_hooks)
        + len(resolve_module(lm, i, "mlp_out")._forward_hooks)
        for i in range(lm.n_layers)
    )
    rep.check(
        "8. no hook handles survive the with block",
        leaked == 0 and torch.equal(baseline, greedy(lm, batch, PROBE_MAX_NEW)),
        f"{leaked} forward hooks left across {lm.n_layers} layers, and a "
        "post-block generation still matches the baseline",
    )

    rep.summary()
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
