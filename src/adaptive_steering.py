#!/usr/bin/env python
"""Per-item alpha, set from how hard the source already is.

The waste a flat alpha creates
------------------------------
Every steered result in this project applies one alpha to every input. ASSET
test sources run from roughly grade 6 to grade 20. At a single alpha the easy
end is pushed for no reason -- it was already readable, and the push only costs
it fluency and faithfulness -- while the hard end is barely moved. The averaged
result reports the mean of an intervention that was wrong in both directions.

This is the cheap half of "conditional steering". The literature version trains
a probe on activations to decide when to fire. A probe is not needed to answer
*this* question: the source text is available before generation, its reading
grade is a CPU-side function of it, and that is exactly the quantity the dial is
supposed to respond to. Alpha ramps linearly from zero at ADAPTIVE_FKGL_FLOOR to
the full swept alpha at ADAPTIVE_FKGL_CEIL.

Alphas are snapped to a grid so items sharing one alpha generate in a batch.
Without that, per-item alpha means per-item generation, which is roughly eight
times slower for no change in what is measured.

Sources shorter than the readability module's minimum are handled by the module
itself, which returns a degenerate reading rather than raising; those items land
at zero alpha, i.e. unsteered, which is the correct thing to do with a sentence
too short to judge.

Usage
-----
    python src/adaptive_steering.py --n 32 --structural 15,0.75
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from lib import tables  # noqa: E402
from lib.bench import Bench, Item  # noqa: E402
from lib.cli import AxisSettings  # noqa: E402
from lib.metrics import readability  # noqa: E402
from lib.steering import SteeringSpec  # noqa: E402

# Both of these have to behave when stdout is a log file, which is how a full
# run is normally started: the bar goes to stderr and switches itself off there
# when that is not a terminal, and the console emits no escape codes off a tty.


def strength(source_fkgl: float) -> float:
    """0 below the floor, 1 above the ceiling, linear between."""
    if not math.isfinite(source_fkgl):
        return 0.0
    lo, hi = config.ADAPTIVE_FKGL_FLOOR, config.ADAPTIVE_FKGL_CEIL
    return min(1.0, max(0.0, (source_fkgl - lo) / (hi - lo)))


def snap(alpha: float, step: float) -> float:
    return round(round(alpha / step) * step, 6)


class Schedule:
    """Which alpha each item gets, grouped so a shared alpha generates in one batch.

    Built before any weights are touched: it is entirely a CPU-side function of
    the source text, and it decides the whole run, so it is printed first. A
    schedule that is obviously wrong is worth finding out about in two seconds
    rather than after a six-gigabyte load.
    """

    def __init__(self, items: list[Item], axis: str, layer: int, ceiling: float):
        self.axis, self.layer, self.ceiling = axis, layer, ceiling
        self.fkgl = [readability.measure(it.source).fkgl for it in items]
        self.alpha = [snap(ceiling * strength(f), config.ADAPTIVE_ALPHA_STEP) for f in self.fkgl]
        self.groups: dict[float, list[int]] = defaultdict(list)
        for k, a in enumerate(self.alpha):
            self.groups[a].append(k)

    def spec(self, alpha: float) -> SteeringSpec:
        if alpha == 0.0:
            return SteeringSpec.off()
        return SteeringSpec.single(self.axis, self.layer, alpha)

    def describe(self, n_items: int) -> None:
        print(
            f"[plan] axis={self.axis} layer={self.layer} alpha ceiling={self.ceiling} "
            f"ramp {config.ADAPTIVE_FKGL_FLOOR}->{config.ADAPTIVE_FKGL_CEIL} FKGL"
        )
        table = tables.table(header_style="bold")
        table.add_column("alpha", justify="right")
        table.add_column("items", justify="right")
        table.add_column("source FKGL", justify="right")
        for a in sorted(self.groups):
            ks = self.groups[a]
            lo = min(self.fkgl[k] for k in ks)
            hi = max(self.fkgl[k] for k in ks)
            table.add_row(f"{a:.3f}", f"{len(ks)}", f"{lo:.2f}-{hi:.2f}")
        tables.show(table)
        print(f"[plan] {len(self.groups.get(0.0, []))}/{n_items} item(s) get no steering at all")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    p.add_argument("--estimator", default="instruction_diff")
    p.add_argument("--axis", default="structural", choices=list(config.AXES))
    p.add_argument("--structural", help="LAYER,ALPHA -- alpha is the *ceiling*")
    p.add_argument("--lexical", help="LAYER,ALPHA -- alpha is the *ceiling*")
    p.add_argument("--layers-from", type=Path)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--asset-split", default="test")
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "eval_main.jsonl")
    p.add_argument("--variant", default="adaptive_structural")
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config.SEED)

    settings = AxisSettings.from_args(args)
    settings.require([args.axis], needed_by=f"adaptive steering on {args.axis!r}")
    layer, ceiling = settings[args.axis]

    bench = Bench.open(args)
    plan = Schedule(bench.items, args.axis, layer, ceiling)
    plan.describe(len(bench.items))

    outputs: list[str] = [""] * len(bench.items)
    t0 = time.perf_counter()
    for a in sorted(plan.groups):
        ks = plan.groups[a]
        t = time.perf_counter()
        got = bench.generate(
            plan.spec(a),
            [bench.items[k].source for k in ks],
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        for k, text in zip(ks, got):
            outputs[k] = text
        print(f"  alpha {a:5.3f}  {len(ks):3d} item(s)  {time.perf_counter() - t:6.1f}s")
    gen_sec = (time.perf_counter() - t0) / max(len(bench.items), 1)

    baselines = bench.baselines(
        args.out.with_suffix(args.out.suffix + ".baselines.json"),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    with bench.writer(args.out) as out:
        for k, item in enumerate(
            tables.progress(bench.items, desc="score", unit="item")
        ):
            a = plan.alpha[k]
            scored = bench.score(item, outputs[k], baseline=baselines[k])
            out.write(
                bench.row(
                    item,
                    scored,
                    baselines[k],
                    variant=args.variant,
                    spec={
                        "axis": args.axis,
                        "layer": layer,
                        "alpha": a,
                        "alpha_ceiling": ceiling,
                        "source_fkgl": plan.fkgl[k],
                    },
                    spec_label=f"adapt@L{layer}a{a:g}",
                    gen_sec=gen_sec,
                )
            )
    print(f"\n[out] {args.out}  variant={args.variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
