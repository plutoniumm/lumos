#!/usr/bin/env python
"""Generate several candidates per item, ship the simplest one that passes the gate.

The knob
--------
Every other arm in this project picks one setting and applies it to every input.
That is the wrong shape for the problem: ASSET sources range from 12 to 40 words
and from grade 8 to grade 20, and a single alpha that is safe on the hard end is
inert on the easy end while a single alpha that moves the easy end collapses the
hard one. The alpha ladder in the paper shows this directly -- the best FKGL in
the whole sweep belongs to a setting that breaks two thirds of its outputs.

Best-of-n sidesteps the choice. Generate the whole ladder, score each candidate,
and let the gate decide per item. The cost is n forward passes instead of one.
The gain is that a collapsed output can no longer be shipped, because the same
degeneracy rules that flagged it in the results table now reject it before it
leaves.

The selection rule uses only the source and the candidate
---------------------------------------------------------
No references. SARI needs 10 human rewrites per sentence and would make this a
measurement of an oracle rather than of a deployable system. The rule is
``Scored.passes`` in ``lib/bench.py``:

    keep candidates that are not degenerate
        and entail the source in both directions at >= NLI_ENTAILMENT_MIN;
    among those, take the lowest FKGL;
    if none survive, fall back to the unsteered output.

Conditional perplexity is still measured and written to every row, but it is no
longer part of the rule -- ``Scored.passes`` in ``lib/bench.py`` has the
evidence for why a ratio against the model's own baseline cannot be gated at an
absolute constant.

The fallback is not a formality. It is what makes the arm honest: an item where
every intervention fails is reported as "no simplification", not dropped from
the average. Dropping it would let the arm buy its score by refusing the hard
half of the eval.

Usage
-----
    python src/best_of_n.py --n 32 --out results/eval_main.jsonl \
        --layers-from results/best_layers_Llama-3.2-3B-Instruct_instruction_diff.json
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import config  # noqa: E402
from lib import jsonio, tables  # noqa: E402
from lib.bench import Bench, Scored  # noqa: E402
from lib.cli import AxisSettings  # noqa: E402
from lib.steering import SteeringSpec  # noqa: E402

# Selection costs seven NLI passes and seven perplexity passes per item, so the
# scoring loop is the slow half of the run and gets a bar. It goes to stderr and
# switches itself off there when that is not a terminal, and the console emits
# no escape codes off a tty -- this arm is usually run into a log file.


@dataclass(frozen=True)
class Candidate:
    """One way of producing a rewrite. The pool, not a configuration to tune."""

    label: str
    instruction: str | None
    axis: str | None  # None means no steering
    alpha_scale: float  # multiplies the swept best alpha for that axis

    def spec(self, settings: AxisSettings) -> SteeringSpec:
        """The candidate's rung, at whatever width the axis is configured to.

        ``alpha_scale`` multiplies every layer in the axis's schedule rather
        than a single alpha, so a depth-scheduled axis ladders as one dial and
        keeps its shape at each rung.
        """
        if self.axis is None:
            return SteeringSpec.off()
        return settings.spec(self.axis).scaled(self.alpha_scale)


# Chosen to span the tradeoff rather than to be exhaustive. The unsteered and
# prompted ends are already measured as the floor and the over-shooting bar; the
# structural ladder covers the range between them; the two extras are there so a
# per-item winner is not forced to come from one mechanism.
POOL: tuple[Candidate, ...] = (
    Candidate("unsteered", None, None, 0.0),
    Candidate("prompted", config.INSTRUCTIONS["simplify"], None, 0.0),
    Candidate("stru@0.5x", None, "structural", 0.5),
    Candidate("stru@1.0x", None, "structural", 1.0),
    Candidate("stru@1.5x", None, "structural", 1.5),
    Candidate("stru@2.0x", None, "structural", 2.0),
    Candidate("lexi@1.0x", None, "lexical", 1.0),
)


def select(scored: list[Scored]) -> tuple[Scored, list[Scored]]:
    """The winner and the survivors it was chosen from.

    Falls back to the item's own unsteered output rather than to the least-bad
    candidate: shipping a rewrite that failed the gate is exactly what the gate
    exists to prevent. ``scored`` only holds candidates that produced text, so
    the unsteered entry can be absent if the baseline itself came back empty --
    the least-bad candidate is the only option left there, and the row records
    that it was a fallback.
    """
    survivors = [s for s in scored if s.passes()]
    if survivors:
        return min(survivors, key=lambda s: s.read.fkgl), survivors
    unsteered = [s for s in scored if s.label == "unsteered"]
    return (unsteered[0] if unsteered else min(scored, key=lambda s: s.read.fkgl)), []


def tally(picked: dict[str, int], n_items: int) -> Table:
    """Which mechanisms actually won. A pool member that never wins is dead
    weight in the n of best-of-n, and this is where that shows up."""
    table = tables.table(header_style="bold")
    table.add_column("candidate")
    table.add_column("chosen", justify="right")
    table.add_column("share", justify="right")
    for label, n in sorted(picked.items(), key=lambda kv: -kv[1]):
        table.add_row(label, f"{n}", f"{100 * n / max(n_items, 1):.1f}%")
    return table


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    p.add_argument("--estimator", default="instruction_diff")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--asset-split", default="test")
    p.add_argument("--layers-from", type=Path)
    p.add_argument("--structural", help="LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... (shapes: lib/schedules.py)")
    p.add_argument("--lexical", help="LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... (shapes: lib/schedules.py)")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "eval_main.jsonl")
    p.add_argument("--variant", default="best_of_n", help="row label")
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--trace", type=Path, help="write per-candidate scores here")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    torch.manual_seed(config.SEED)

    settings = AxisSettings.from_args(args)
    settings.require(
        sorted({c.axis for c in POOL if c.axis}), needed_by="the best-of-n pool"
    )

    bench = Bench.open(args)
    baselines = bench.baselines(
        args.out.with_suffix(args.out.suffix + ".baselines.json"),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    generated: dict[str, list[str]] = {}
    t0 = time.perf_counter()
    for cand in POOL:
        spec = cand.spec(settings)
        t = time.perf_counter()
        generated[cand.label] = bench.generate(
            spec,
            instruction=cand.instruction,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        print(f"  [{cand.label:<12}] {spec.label():<22} {time.perf_counter() - t:6.1f}s")
    gen_sec = (time.perf_counter() - t0) / max(len(bench.items), 1)
    print(f"[gen]  {len(POOL)} candidate(s) x {len(bench.items)} item(s)")

    # The "unsteered" *candidate* is replaced by the baseline string, so that a
    # fallback ships byte-identical text to what A/B 1 scored and its perplexity
    # ratio is exactly 1.0. Greedy decoding is deterministic in principle, but
    # MPS batch padding is not bit-stable across runs with different batch
    # composition, and a fallback that differs from the baseline by a token
    # would make the ratio look like a real effect.
    generated["unsteered"] = list(baselines)

    trace_rows: list[dict] = []
    n_fallback = 0
    picked: dict[str, int] = {}

    with bench.writer(args.out) as out:
        for k, item in enumerate(
            tables.progress(bench.items, desc="select", unit="item")
        ):
            base = baselines[k]
            base_ppl = bench.conditional_ppl(item, base)
            scored = [
                bench.score(
                    item,
                    generated[c.label][k],
                    baseline_ppl=base_ppl,
                    label=c.label,
                    skip_sari=True,
                )
                for c in POOL
                if generated[c.label][k]
            ]
            pick, survivors = select(scored)
            if not survivors:
                n_fallback += 1
            picked[pick.label] = picked.get(pick.label, 0) + 1

            # SARI is deliberately absent from selection but present in the
            # report, so it is filled in once, on the winner only.
            pick = bench.with_sari(item, pick)
            out.write(
                bench.row(
                    item,
                    pick,
                    base,
                    variant=args.variant,
                    spec={"pool": [c.label for c in POOL], "picked": pick.label},
                    spec_label=f"bo{len(POOL)}/{pick.label}",
                    gen_sec=gen_sec,
                    n_survivors=len(survivors),
                    fell_back=not survivors,
                )
            )
            trace_rows.append(
                {
                    "item_index": item.index,
                    "picked": pick.label,
                    "n_survivors": len(survivors),
                    "candidates": [
                        {
                            "label": s.label,
                            "fkgl": s.read.fkgl,
                            "nli": s.nli,
                            "ppl_ratio": s.ppl_ratio,
                            "degenerate": s.degenerate,
                            "passes": s.passes(),
                        }
                        for s in scored
                    ],
                }
            )
            # tables.write rather than print: the per-item line is the record of
            # which candidate won and why, and it still has to be readable when
            # a bar is redrawing over the same terminal.
            tables.write(
                f"  item {item.index:3d}  {len(survivors)}/{len(scored)} pass  -> "
                f"{pick.label:<12} fkgl {pick.read.fkgl:5.2f} nli {pick.nli:.3f}"
                + ("  [fallback]" if not survivors else "")
            )

    print(f"\n[pick] {n_fallback}/{len(bench.items)} item(s) fell back to unsteered")
    tables.show(tally(picked, len(bench.items)))
    print(f"[out]  {args.out}")

    if args.trace:
        jsonio.dump(args.trace, trace_rows)
        print(f"[out]  {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
