"""Correctness gate for the hook layer. Run this before trusting any sweep.

    python scripts/02_hook_noop.py

Seven checks, each printed PASS/FAIL. The invariants that matter:

1. ``alpha=0`` is bit-exact vs no hooks at all.
2. a zero *vector* at nonzero alpha is bit-exact even with ``norm_preserve``
   (a zero delta alone would hide a renormalisation bug).
3. a real vector at high alpha *does* change the tokens, i.e. the hook is on the
   module the model actually reads from.
4. ``norm_preserve`` leaves ``||h||`` alone.
5. ``positions`` means what it says: ``prompt_last`` fires on prefill only,
   ``gen_only`` never fires on prefill.
6. ``Recorder`` returns one ``[batch, hidden]`` entry per layer, and the values
   are the layer's real activations.
7. no hook handle leaks past the ``with`` block.

Everything runs on a small ungated model, so this is the gate you can run while
the real weights are still downloading. Diagnostics that are *not* gates -- real
defects found in ``lib/`` that the seven checks do not cover -- print at the end.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import hub, models
from lib.hooks import InjectionSite, Injector, record, resolve_module, steered

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"

TEXT = (
    "The proliferation of antimicrobial resistance determinants among "
    "commensal Enterobacteriaceae constitutes a substantial epidemiological "
    "concern, notwithstanding the attenuated virulence of these organisms."
)
SECOND_TEXT = "Photosynthesis converts radiant energy into chemical potential energy."

MAX_NEW = 24  # long enough for a divergence to show, short enough to iterate

# bfloat16 carries 8 mantissa bits, so a single rounding is <= 2^-9 = 2.0e-3
# relative. norm_preserve rounds the sum, both norms and the rescaled product,
# so ~4 ULP of drift is expected; 2e-2 is 10 ULP. The control below (the same
# injection with norm_preserve=False) moves the norm by 18-390%, two orders of
# magnitude above the gate, so the tolerance is not hiding a failure.
NORM_RTOL = 2e-2


@dataclass
class Report:
    rows: list[tuple[str, bool, bool]] = field(default_factory=list)

    def check(self, name: str, ok: bool, *lines: str, gate: bool = True) -> None:
        self.rows.append((name, ok, gate))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for line in lines:
            print(f"       {line}")

    def note(self, *lines: str) -> None:
        for line in lines:
            print(f"       {line}")

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.rows)

    def summary(self) -> None:
        print("\n" + "-" * 78)
        for name, ok, _ in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        gates = [ok for _, ok, gate in self.rows if gate]
        pre = [ok for _, ok, gate in self.rows if not gate]
        print(
            f"\n{sum(gates)}/{len(gates)} gates passed"
            f"   (preconditions: {sum(pre)}/{len(pre)})"
        )


def load_small() -> models.SteerableLM:
    for repo in (SMALL, FALLBACK):
        try:
            return models.load(repo)
        except Exception as exc:  # network, gating, missing arch
            print(f"[load] {repo} unavailable: {type(exc).__name__}: {exc}")
    raise SystemExit("no throwaway model could be loaded")


@torch.inference_mode()
def greedy(lm, batch: dict, n_new: int = MAX_NEW, **kw) -> torch.Tensor:
    """Deterministic generation. Returns token ids on CPU so they compare cleanly."""
    torch.manual_seed(config.SEED)
    out = lm.model.generate(
        **batch,
        max_new_tokens=n_new,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=lm.tokenizer.eos_token_id,
        **kw,
    )
    return out.detach().to("cpu")


@torch.inference_mode()
def hidden_at(lm, layer: int, batch: dict, sites: list | None = None) -> torch.Tensor:
    """Layer output for one forward pass, ``[seq, hidden]`` float32 on CPU.

    The probe registers *after* the injector, so it sees the post-injection
    tensor. Casts on-device before moving, which the Recorder does not (see the
    diagnostics at the end of this script).
    """
    mod = resolve_module(lm, layer, "resid_post")
    cap: dict[str, torch.Tensor] = {}

    def probe(_m, _a, out):
        cap.setdefault("h", out.detach().float().to("cpu"))
        return None

    if sites:
        with Injector(lm, sites):
            handle = mod.register_forward_hook(probe)
            try:
                lm.model(**batch)
            finally:
                handle.remove()
    else:
        handle = mod.register_forward_hook(probe)
        try:
            lm.model(**batch)
        finally:
            handle.remove()
    return cap["h"][0]


@dataclass
class Call:
    """One invocation of the hooked module during a generate."""

    seq: int
    changed: list[int]  # positions the injector actually modified

    @property
    def fired(self) -> bool:
        return bool(self.changed)

    def show(self, index: int) -> str:
        """One line per call, labelled with how the *injector* classified it."""
        tag = "prefill" if index == 0 else "decode"
        if not self.changed:
            where = "-"
        elif len(self.changed) == self.seq:
            where = f"all {self.seq}"
        else:
            where = str(self.changed if len(self.changed) <= 3 else self.changed[:3])
        return (
            f"call {index:<3} seq_len={self.seq:<4} injector says {tag:<8} "
            f"modified={where}"
        )


def trace(lm, site: InjectionSite, batch: dict, n_new: int = 3, **kw):
    """Run a generate and record, per module call, which positions changed.

    A hook registered before the injector snapshots the module's own output; one
    registered after it sees whatever the injector returned. The difference is
    the branch the injector took, without touching ``lib/``.
    """
    mod = resolve_module(lm, site.layer, site.kind)
    calls: list[Call] = []
    snap: dict[str, torch.Tensor] = {}

    def before(_m, _a, out):
        snap["h"] = out.detach().clone()
        return None

    def after(_m, _a, out):
        diff = (out != snap["h"]).any(dim=-1)[0]
        calls.append(
            Call(int(out.shape[1]), [i for i, c in enumerate(diff.tolist()) if c])
        )
        return None

    h_before = mod.register_forward_hook(before)
    inj = Injector(lm, [site])
    with inj:
        h_after = mod.register_forward_hook(after)
        try:
            greedy(lm, batch, n_new, **kw)
        finally:
            h_after.remove()
    h_before.remove()
    return calls, inj.call_counts()


def cite(rel: str, needle: str) -> str:
    """``file:line`` for the first line containing *needle*.

    Looked up at runtime rather than hard-coded, so the citations below stay
    correct while ``lib/`` is still being written.
    """
    path = Path(__file__).resolve().parents[1] / rel
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if needle in line:
            return f"{rel}:{i}"
    return f"{rel}:?? ({needle!r} not found -- the code moved)"


def unit_vector(hidden_size: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_size, generator=g)
    return v / v.norm()


@torch.inference_mode()
def layer_output_kind(lm, layer: int, batch: dict) -> str:
    """What the hooked module hands to ``_split``: bare tensor or tuple?"""
    mod = resolve_module(lm, layer, "resid_post")
    seen: dict[str, str] = {}

    def probe(_m, _a, out):
        if isinstance(out, tuple):
            shapes = [
                tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in out
            ]
            seen["kind"] = (
                f"tuple of {len(out)} {shapes} -> _split keeps rest={out[1:]!r}"
            )
        else:
            seen["kind"] = (
                f"bare {type(out).__name__} {tuple(out.shape)} -> _split returns "
                "rest=None, _rejoin returns the tensor unchanged"
            )
        return None

    handle = mod.register_forward_hook(probe)
    try:
        lm.model(**batch)
    finally:
        handle.remove()
    return seen["kind"]


def main(argv: list[str]) -> int:
    torch.manual_seed(config.SEED)
    rep = Report()

    lm = load_small()
    prompt = lm.build_prompt(TEXT, config.INSTRUCTIONS["simplify"])
    batch = lm.encode(prompt)
    prompt_len = int(batch["input_ids"].shape[1])
    mid = lm.n_layers // 2

    print(
        f"\nmodel   {lm.repo_id} ({type(lm.model).__name__})\n"
        f"layers  {lm.n_layers}   hidden {lm.hidden_size}   "
        f"dtype {lm.dtype}   device {lm.device}\n"
        f"stack   torch {torch.__version__}, transformers {transformers.__version__}\n"
        f"prompt  {prompt_len} tokens, {MAX_NEW} new tokens per generation\n"
        f"resid_post output: {layer_output_kind(lm, mid, batch)}\n"
    )

    # Scale the probe vector to this layer's own activation magnitude, so alpha
    # means "this fraction of a typical hidden state" as lib/vectors.py intends.
    base = hidden_at(lm, mid, batch)
    ref = float(base.norm(dim=-1).median())
    v = unit_vector(lm.hidden_size)
    scaled = v * ref
    print(
        f"layer {mid}: ||h|| median {ref:.2f}  min {base.norm(dim=-1).min():.2f}  "
        f"max {base.norm(dim=-1).max():.2f}\n"
    )

    baseline = greedy(lm, batch)
    again = greedy(lm, batch)
    rep.check(
        "S. precondition: greedy generation is reproducible run to run",
        torch.equal(baseline, again),
        f"{baseline.shape[1] - prompt_len} new tokens, two unhooked runs bit-identical; "
        "without this nothing below is meaningful",
        gate=False,
    )

    site = InjectionSite(mid, scaled, alpha=0.0, positions="all", norm_preserve=True)
    with steered(lm, [site]):
        zero_alpha = greedy(lm, batch)
    rep.check(
        "1. alpha=0.0 is token-for-token identical to no hooks",
        torch.equal(baseline, zero_alpha),
        f"vector ||v||={float(scaled.norm()):.2f}, positions=all, norm_preserve=True",
        f"note: {cite('lib/hooks.py', 'if site.alpha == 0.0')} short-circuits before "
        "any arithmetic, so this exercises the plumbing, not the maths",
    )

    zeros = torch.zeros(lm.hidden_size)
    site = InjectionSite(mid, zeros, alpha=3.0, positions="all", norm_preserve=True)
    with steered(lm, [site]):
        zero_vec = greedy(lm, batch)
    h_zero = hidden_at(lm, mid, batch, [site])
    rep.check(
        "2. zero vector at alpha=3.0 with norm_preserve=True is bit-exact",
        torch.equal(baseline, zero_vec) and torch.equal(h_zero, base),
        "tokens identical and the layer output is bit-identical: the rescale "
        "computes before/after from the same tensor, so the ratio is exactly 1.0",
    )

    site = InjectionSite(mid, scaled, alpha=4.0, positions="all", norm_preserve=True)
    with steered(lm, [site]):
        steered_ids = greedy(lm, batch)
    changed = not torch.equal(baseline, steered_ids)
    rep.check(
        f"3. alpha=4.0 at layer {mid} changes the generated tokens",
        changed,
        f"baseline: {lm.decode_new(baseline, prompt_len)[0][:60]!r}",
        f"steered : {lm.decode_new(steered_ids, prompt_len)[0][:60]!r}",
    )

    keep = InjectionSite(mid, scaled, 2.0, positions="prompt_all", norm_preserve=True)
    free = InjectionSite(mid, scaled, 2.0, positions="prompt_all", norm_preserve=False)
    bn = base.norm(dim=-1)

    def max_rel_dnorm(site: InjectionSite) -> float:
        after = hidden_at(lm, mid, batch, [site]).norm(dim=-1)
        return float(((after - bn).abs() / bn).max())

    rel_keep, rel_free = max_rel_dnorm(keep), max_rel_dnorm(free)
    rep.check(
        f"4. norm_preserve=True holds ||h|| to within rtol={NORM_RTOL:.0e}",
        rel_keep <= NORM_RTOL < rel_free,
        f"max relative |d||h|||  over all {prompt_len} positions: "
        f"{rel_keep:.3e} with norm_preserve, {rel_free:.3e} without",
        f"tolerance {NORM_RTOL:.0e} = 10 bfloat16 ULP (eps=2^-9={2**-9:.1e}); the "
        "rescale rounds the sum, both norms and the product, so ~4 ULP is the "
        "floor. The no-preserve control is ~100x the gate, so the gate is real.",
    )

    print(
        "[ .. ] 5. tracing positions semantics (call 0 is the prefill, "
        f"seq_len == {prompt_len})"
    )
    traces: dict[str, list[Call]] = {}
    for positions in ("gen_only", "prompt_last", "prompt_all", "all"):
        probe = InjectionSite(
            mid, scaled, 2.0, positions=positions, norm_preserve=False
        )
        calls, counter = trace(lm, probe, batch, n_new=3)
        traces[positions] = calls
        rep.note(f"positions={positions!r}  call_counts={counter}")
        for i, call in enumerate(calls):
            rep.note("  " + call.show(i))

    gen_only, p_last = traces["gen_only"], traces["prompt_last"]
    p_all, every = traces["prompt_all"], traces["all"]
    ok5 = (
        not gen_only[0].fired
        and all(c.fired for c in gen_only[1:])
        and p_last[0].changed == [prompt_len - 1]
        and not any(c.fired for c in p_last[1:])
        # not required by the spec, but the other two names should also hold
        and p_all[0].changed == list(range(prompt_len))
        and not any(c.fired for c in p_all[1:])
        and all(c.fired for c in every)
    )
    rep.check(
        "5. prompt_last fires on prefill only; gen_only never fires on prefill",
        ok5,
        "prompt_last touched exactly the final prompt position and nothing after; "
        "gen_only skipped prefill and fired on every decode step",
        "prompt_all and all also match their names (with use_cache=True -- see "
        "diagnostic D2)",
    )

    want = [0, mid, lm.n_layers - 1]
    last = record(lm, prompt, layers=want, pooling="last")
    mean = record(lm, prompt, layers=want, pooling="mean_prompt")
    shapes_ok = (
        sorted(last) == want
        and sorted(mean) == want
        and all(tuple(t.shape) == (1, lm.hidden_size) for t in last.values())
        and all(tuple(t.shape) == (1, lm.hidden_size) for t in mean.values())
    )
    pooling_differs = all(not torch.allclose(last[l], mean[l]) for l in want)
    # The values must be the layer's actual activations, not merely the right shape.
    truth = {l: hidden_at(lm, l, batch)[-1] for l in want}
    values_ok = all(torch.equal(last[l][0], truth[l]) for l in want)
    rep.check(
        "6. Recorder returns one [batch, hidden] entry per layer, with real values",
        shapes_ok and pooling_differs and values_ok,
        f"layers {want} -> shapes {[tuple(last[l].shape) for l in want]}, "
        f"hidden == lm.hidden_size == {lm.hidden_size}: {shapes_ok}",
        f"mean_prompt differs from last on a {prompt_len}-token prompt: {pooling_differs}",
        f"recorded 'last' equals an independent capture of the same layer: {values_ok}",
    )
    if not values_ok:
        where = cite("lib/hooks.py", 'to("cpu", torch.float32)')
        print(f"\n       >>> BUG in {where} -- Recorder returns garbage on MPS.")
        for l in want:
            a, b = last[l][0], truth[l]
            denom = float(a.norm() * b.norm())
            cos = (
                f"{float(torch.dot(a, b)) / denom:+.4f}"
                if denom
                else "n/a (recorded 0)"
            )
            print(
                f"           layer {l:<3} recorded ||v||={float(a.norm()):10.4f}   "
                f"true ||v||={float(b.norm()):10.4f}   cos={cos}"
            )
        norms = {float(last[l][0].norm()) for l in want}
        print(
            f"           the recorded norm is identical at every layer ({norms}), "
            "which no real residual stream is"
        )

    site = InjectionSite(mid, scaled, alpha=4.0, positions="all", norm_preserve=True)
    with steered(lm, [site]):
        greedy(lm, batch)
    after_exit = greedy(lm, batch)
    leaked = sum(
        len(resolve_module(lm, l, "resid_post")._forward_hooks)
        for l in range(lm.n_layers)
    )
    record(lm, prompt, layers=[mid])  # Recorder must clean up too
    leaked += sum(
        len(resolve_module(lm, l, "resid_post")._forward_hooks)
        for l in range(lm.n_layers)
    )
    rep.check(
        "7. no handles leak past the with block",
        torch.equal(baseline, after_exit) and leaked == 0,
        f"post-block generation matches baseline; {leaked} forward hooks left on "
        f"{lm.n_layers} layer modules after Injector and Recorder exits",
    )

    rep.summary()
    diagnostics(lm, batch, prompt_len, mid, scaled)
    return 0 if rep.ok else 1


def diagnostics(lm, batch, prompt_len: int, mid: int, scaled: torch.Tensor) -> None:
    """Defects the seven gates do not cover. Informational; do not change exit code."""
    print("\n" + "=" * 78)
    print("DIAGNOSTICS -- real defects, not gates. Report these; do not silently rely")
    print("on the current behaviour.")
    print("=" * 78)

    # D1 -- the MPS cast that breaks Recorder, reduced to pure torch.
    print(
        f"\nD1  {cite('lib/hooks.py', 'to(\"cpu\", torch.float32)')}"
        '  `pooled.detach().to("cpu", torch.float32)`'
    )
    x = torch.arange(12 * 5, dtype=torch.float32).reshape(1, 12, 5)
    x = x.to(config.DEVICE).to(getattr(torch, config.DTYPE))
    view = x[:, -1, :]
    bad = view.to("cpu", torch.float32).flatten().tolist()
    good = view.float().to("cpu").flatten().tolist()
    print(f"    x = arange(60).reshape(1,12,5) as {config.DTYPE} on {config.DEVICE}")
    print(f"    x[:, -1, :].to('cpu', torch.float32) -> {[f'{v:.3e}' for v in bad]}")
    print(f"    x[:, -1, :].float().to('cpu')        -> {good}   <- correct")
    print(
        f"    storage_offset of the view = {view.storage_offset()} (0 for x[:, 0, :], "
        "which is why the bug is invisible on 1-token prompts)"
    )
    print("    A device+dtype change in one .to() call ignores the source storage")
    print("    offset for half-precision MPS tensors, so pooling='last' records")
    print("    denormal garbage. pooling='mean_prompt' and 'mean_span' survive")
    print("    because .mean()/.stack() allocate fresh at offset 0. Fix: cast on")
    print('    the device first, move second -- `pooled.detach().float().to("cpu")`.')
    for batch_size in (1, 2, 4):
        t = torch.arange(batch_size * 12 * 5, dtype=torch.float32)
        t = t.reshape(batch_size, 12, 5).to(config.DEVICE)
        t = t.to(getattr(torch, config.DTYPE))[:, -1, :]
        ok = torch.equal(t.to("cpu", torch.float32), t.float().to("cpu"))
        print(
            f"    batch={batch_size}: hidden[:, -1, :] is_contiguous="
            f"{str(t.is_contiguous()):5} offset={t.storage_offset():<4} "
            f"cast correct={ok}"
        )
    print("    The bug fires exactly when the view *reports* contiguous while")
    print("    sitting at a nonzero offset, i.e. batch=1 -- the only shape record()")
    print("    produces, and what vectors.py `_pooled` assumes. A batched view is")
    print("    strided, takes the gather path instead of the memcpy, and survives.")

    # D2 -- prefill/decode is inferred from a call counter, not from seq_len.
    print(
        f"\nD2  {cite('lib/hooks.py', 'is_prefill = n == 0')}"
        "  prefill is `_calls == 0`, and decode assumes seq_len == 1"
    )
    site = InjectionSite(mid, scaled, 2.0, positions="gen_only", norm_preserve=False)
    calls, _ = trace(lm, site, batch, n_new=2, use_cache=False)
    for i, call in enumerate(calls):
        print("    " + call.show(i))
    print("    With use_cache=False every call carries the whole sequence, and the")
    print(
        f"    else-branch at {cite('lib/hooks.py', 'delta += site.alpha')} injects into"
    )
    print("    every prompt position -- exactly what gen_only exists to prevent.")

    # D3 -- the counter is only reset on __enter__.
    print(
        f"\nD3  {cite('lib/hooks.py', 'def reset')} / "
        f"{cite('lib/hooks.py', 'self.reset()')}  `_calls` resets on __enter__, "
        "not per generate"
    )
    second = lm.encode(lm.build_prompt(SECOND_TEXT, config.INSTRUCTIONS["simplify"]))
    for positions in ("prompt_last", "gen_only"):
        site = InjectionSite(mid, scaled, 4.0, positions=positions, norm_preserve=True)
        mod = resolve_module(lm, mid, "resid_post")
        log: list[Call] = []
        snap: dict[str, torch.Tensor] = {}
        h_before = mod.register_forward_hook(
            lambda _m, _a, out: snap.__setitem__("h", out.detach().clone())
        )
        shared: list[torch.Tensor] = []
        with Injector(lm, [site]):
            h_after = mod.register_forward_hook(
                lambda _m, _a, out: log.append(
                    Call(
                        int(out.shape[1]),
                        [
                            i
                            for i, c in enumerate(
                                (out != snap["h"]).any(dim=-1)[0].tolist()
                            )
                            if c
                        ],
                    )
                )
            )
            for b in (batch, second):
                shared.append(greedy(lm, b, 8))
            h_after.remove()
        h_before.remove()
        with steered(lm, [site]):
            fresh = greedy(lm, second, 8)
        prefills = [(i, c) for i, c in enumerate(log) if c.seq > 1]
        print(
            f"    positions={positions!r}: prefill calls in one shared `with` block ->"
        )
        for i, call in prefills:
            print("      " + call.show(i))
        print(
            f"      2nd generate identical to a freshly-entered block? "
            f"{torch.equal(shared[1], fresh)}"
        )
    print("    Second and later generates inside one `with steered(...)` block see")
    print("    _calls != 0, so their prefill is misread as a decode step:")
    print("    prompt_last/prompt_all silently stop steering, and gen_only steers")
    print("    the entire prompt. Only the first generate in a block is correct.")

    # D4 -- Positions is a Literal with no runtime check.
    print(
        f"\nD4  {cite('lib/hooks.py', 'Positions = Literal')}  `positions` is never "
        "validated at runtime"
    )
    site = InjectionSite(mid, scaled, 2.0, positions="prompt_alll", norm_preserve=False)
    calls, _ = trace(lm, site, batch, n_new=2)
    for i, call in enumerate(calls):
        print("    " + call.show(i))
    print("    positions='prompt_alll' (one typo) falls through to the else branch")
    print("    and behaves as 'all'. resolve_module raises on a bad `kind` and")
    print("    Recorder raises on a bad `pooling`; this path silently mis-steers.")

    # D5 -- static findings, not reachable from this model.
    print("\nD5  static findings not reachable with this model")
    gemma_at = cite("lib/models.py", "Gemma3nForConditionalGeneration")
    print(f"    {gemma_at}: in transformers {transformers.__version__},")
    print("      Gemma3nTextDecoderLayer.forward returns `corrected_predictions`,")
    print("      i.e. [altup_num_inputs, batch, seq, hidden] -- 4-D. For config.GEMMA")
    print("      every hidden[:, -1, :] and delta[:, :, :] in hooks.py is off by one")
    print("      dimension: dim 1 is batch, not seq. kind='mlp_out' stays 3-D there")
    print("      and is unaffected; kind='resid_post' (the default) is not.")
    mean_at = cite("lib/hooks.py", "pooled = hidden.mean(dim=1)")
    pad_at = cite("lib/models.py", "padding_side")
    print(f"    {mean_at}: mean_prompt averages over padding -- attention_mask never")
    print(f"      reaches the hook, and {pad_at} left-pads every batch. Reachable now")
    print("      that record_inputs takes batches; of the three poolings only")
    print("      'mean_span' is padding-safe.")
    local = hub.local_path(config.ACTIVE_MODEL)
    aria = [p.name for p in local.glob("*.aria2") if p.is_file()]
    print(f"    {cite('lib/hub.py', 'def resolve')}: resolve() returns a local dir as")
    print("      soon as config.json is there, ignoring the .aria2 control files aria2")
    print("      leaves beside unfinished shards and the .download_meta.json that")
    print("      hub.download writes only on success. Mid-download, from_pretrained")
    print("      is handed a truncated model. ACTIVE_MODEL right now:")
    print(f"      resolve -> {hub.resolve(config.ACTIVE_MODEL)!r}")
    print(
        f"      unfinished control files: {len(aria)}   "
        f".download_meta.json: {(local / '.download_meta.json').exists()}"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
