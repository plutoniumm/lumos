#!/usr/bin/env python
"""Generate a candidate ladder per item and ship the simplest one that passes the
gate, falling back to the item's own unsteered output when none do."""

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


@dataclass(frozen=True)
class Candidate:
    """One way of producing a rewrite. The pool, not a configuration to tune."""

    label: str
    instruction: str | None
    axis: str | None
    alpha_scale: float

    def spec(self, settings: AxisSettings) -> SteeringSpec:
        """The candidate's rung, at whatever width the axis is configured to."""
        if self.axis is None:
            return SteeringSpec.off()
        return settings.spec(self.axis).scaled(self.alpha_scale)


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
    """Return the picked candidate and its passing survivors, falling back to
    the item's own unsteered output if none survive."""
    survivors = [s for s in scored if s.passes()]
    if survivors:
        return min(survivors, key=lambda s: s.read.fkgl), survivors
    unsteered = [s for s in scored if s.label == "unsteered"]
    return (unsteered[0] if unsteered else min(scored, key=lambda s: s.read.fkgl)), []


def tally(picked: dict[str, int], n_items: int) -> Table:
    """Tally how often each pool candidate was picked."""
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
    p.add_argument(
        "--structural",
        help="LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... (shapes: lib/schedules.py)",
    )
    p.add_argument(
        "--lexical",
        help="LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... (shapes: lib/schedules.py)",
    )
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
        print(
            f"  [{cand.label:<12}] {spec.label():<22} {time.perf_counter() - t:6.1f}s"
        )
    gen_sec = (time.perf_counter() - t0) / max(len(bench.items), 1)
    print(f"[gen]  {len(POOL)} candidate(s) x {len(bench.items)} item(s)")

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
