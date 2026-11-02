#!/usr/bin/env python
"""Measure whether spreading the steering budget across layers (a per-depth alpha
schedule) beats one-layer injection, paired against an unsteered baseline.
Writes results/alpha_schedule.json."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from lib import jsonio, tables  # noqa: E402
from lib.bench import Bench, Scored  # noqa: E402
from lib.cli import AxisSettings, die  # noqa: E402
from lib import schedules  # noqa: E402
from lib.steering import AxisSetting, SteeringSpec  # noqa: E402
from src.report import boot_ci  # noqa: E402


DEFAULT_LADDER = (
    "single,flat3,flat3_l2,flat5,flat5_l2,tent5,ramp5_deep,ramp5_shallow"
)
DOSE_LADDER = "single,single_x0.7,single_x0.6,single_x0.5,single_x0.35,flat3,tent5"


COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("FKGL", "fkgl", 2),
    ("w/sent", "words_per_sentence", 2),
    ("hard", "pct_hard_words", 3),
    ("NLI", "nli", 3),
    ("gate%", "pct_gate_passed", 1),
    ("pplR", "ppl_ratio", 2),
    ("SARI", "sari", 2),
    ("degen%", "pct_degenerate", 1),
    ("sec", "gen_seconds", 1),
)


def aggregate(scored: list[Scored], seconds: float, items) -> dict:
    return {
        "fkgl": jsonio.mean(s.read.fkgl for s in scored),
        "words_per_sentence": jsonio.mean(s.read.words_per_sentence for s in scored),
        "pct_hard_words": jsonio.mean(s.read.pct_hard_words for s in scored),
        "nli": jsonio.mean(s.nli for s in scored),
        "pct_gate_passed": jsonio.pct(s.passes() for s in scored),
        "ppl_ratio": jsonio.mean(s.ppl_ratio for s in scored),
        "sari": jsonio.mean(s.sari for s in scored if s.sari is not None),
        "pct_degenerate": jsonio.pct(s.degenerate for s in scored),
        "gen_seconds": seconds,
        "items": [
            {
                "item_index": it.index,
                "output": s.text,
                "fkgl": s.read.fkgl,
                "words_per_sentence": s.read.words_per_sentence,
                "pct_hard_words": s.read.pct_hard_words,
                "nli": s.nli,
                "ppl_ratio": s.ppl_ratio,
                "sari": s.sari,
                "degenerate": s.degenerate,
            }
            for it, s in zip(items, scored)
        ],
    }


def cell(value, places: int) -> str:
    """"--" for a metric with no finite value."""
    return f"{value:.{places}f}" if isinstance(value, (int, float)) else "--"


def levels_table(arms: list[dict]):
    table = tables.table(header_style="bold")
    table.add_column("arm")
    table.add_column("layers x alpha")
    for head, _, _ in COLUMNS:
        table.add_column(head, justify="right")
    for arm in arms:
        table.add_row(
            arm["name"], arm["dial"], *(cell(arm[k], p) for _, k, p in COLUMNS)
        )
    return table


def paired(arms: list[dict], keys, baseline: str, n_boot: int, seed: int) -> list[dict]:
    """Within-item deltas of every arm against one reference arm."""
    ref = {a["name"]: a for a in arms}.get(baseline)
    if ref is None:
        die(f"no arm named {baseline!r} to pair against")
    ref_items = {i["item_index"]: i for i in ref["items"]}

    rows = []
    for arm in arms:
        if arm["name"] == baseline:
            continue
        row = {"name": arm["name"], "dial": arm["dial"]}
        for key in keys:
            deltas = [
                i[key] - ref_items[i["item_index"]][key]
                for i in arm["items"]
                if i["item_index"] in ref_items
                and i[key] is not None
                and ref_items[i["item_index"]][key] is not None
            ]
            mean, lo, hi = boot_ci(deltas, n_boot, seed)
            row[key] = {"mean": mean, "lo": lo, "hi": hi, "n": len(deltas)}
        rows.append(row)
    return rows


def delta_table(rows: list[dict], keys, title: str):
    table = tables.table(header_style="bold", title=title)
    table.add_column("arm")
    for key in keys:
        table.add_column(key, justify="right")
    for row in rows:
        cells = []
        for key in keys:
            d = row[key]
            star = "*" if d["lo"] > 0 or d["hi"] < 0 else ""
            cells.append(f"{d['mean']:+.2f}{star} [{d['lo']:+.2f}, {d['hi']:+.2f}]")
        table.add_row(row["name"], *cells)
    return table


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    p.add_argument("--estimator", default="instruction_diff")
    p.add_argument("--axis", default="structural", choices=list(config.AXES))
    p.add_argument("--layers-from", help="sweep report to read (layer, alpha) from")
    p.add_argument(
        "--structural",
        help="ladder centre; LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... "
        "(shapes: lib/schedules.py)",
    )
    p.add_argument(
        "--lexical",
        help="ladder centre; LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... "
        "(shapes: lib/schedules.py)",
    )
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--asset-split", default="test")
    p.add_argument(
        "--schedules",
        default=DEFAULT_LADDER,
        help="comma-separated shape names from lib/schedules.py, optionally "
        "with an _xFRAC dose suffix; --schedules dose runs the dose ladder",
    )
    p.add_argument(
        "--baselines",
        type=Path,
        help="an evaluate.py .baselines.json, so the unsteered text this is "
        "paired against is byte-identical to the published arms'",
    )
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument(
        "--out", type=Path, default=config.RESULTS_DIR / "alpha_schedule.json"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the layer/alpha table each schedule resolves to and stop, "
        "without loading the model",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config.SEED)

    if args.schedules.strip() == "dose":
        args.schedules = DOSE_LADDER
    names = [n.strip() for n in args.schedules.split(",") if n.strip()]
    try:
        chosen = [schedules.resolve(n) for n in names]
    except KeyError as exc:
        die(str(exc))

    settings = AxisSettings.from_args(args)
    settings.require([args.axis], needed_by="the schedule ladder")
    center, budget = settings.peak(args.axis)

    print(f"\n[cfg]  axis={args.axis} center=L{center} budget={budget:g} "
          f"({args.model})")
    for sch in chosen:
        print(f"       {sch.name:<14} {schedules.label(sch.alphas(center, budget))}")

    if args.dry_run:
        return 0

    bench = Bench.open(args)
    n_layers = bench.lm.n_layers
    bad = [
        sch.name
        for sch in chosen
        for layer, _ in sch.alphas(center, budget)
        if not 0 <= layer < n_layers
    ]
    if bad:
        die(f"schedule(s) {sorted(set(bad))} reach outside 0..{n_layers - 1} "
            f"from center L{center}")

    if args.baselines:
        base_texts = bench.baselines(
            args.baselines,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    else:
        print("\n[base] no --baselines given; generating unsteered fresh")
        base_texts = bench.generate(
            SteeringSpec.off(),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    base_ppl = [
        bench.conditional_ppl(it, b)
        for it, b in tables.progress(
            list(zip(bench.items, base_texts)), desc="base ppl", unit="item"
        )
    ]
    base_scored = [
        bench.score(it, text, baseline_ppl=bp)
        for it, text, bp in tables.progress(
            list(zip(bench.items, base_texts, base_ppl)),
            desc="score unsteered",
            unit="item",
        )
    ]

    arms = [
        {
            "name": "unsteered",
            "dial": "--",
            **aggregate(base_scored, 0.0, bench.items),
        }
    ]

    for sch in chosen:
        pairs = sch.alphas(center, budget)
        spec = SteeringSpec(
            settings=tuple(
                AxisSetting(args.axis, layer, alpha) for layer, alpha in pairs
            )
        )
        print(f"\n[arm]  {sch.name}  {schedules.label(pairs)}")
        t0 = time.perf_counter()
        outs = bench.generate(
            spec,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        seconds = time.perf_counter() - t0
        scored = [
            bench.score(it, text, baseline_ppl=bp)
            for it, text, bp in tables.progress(
                list(zip(bench.items, outs, base_ppl)),
                desc=f"score {sch.name}",
                unit="item",
            )
        ]
        arms.append(
            {
                "name": sch.name,
                "dial": schedules.label(pairs),
                "schedule": {
                    "offsets": list(sch.offsets),
                    "weights": list(sch.weights),
                    "norm": sch.norm,
                    "scale": sch.scale,
                    "layers": [layer for layer, _ in pairs],
                    "alphas": [alpha for _, alpha in pairs],
                },
                "spec": spec.as_dict(),
                **aggregate(scored, seconds, bench.items),
            }
        )

    keys = ("fkgl", "words_per_sentence", "nli", "sari")
    vs_unsteered = paired(arms, keys, "unsteered", args.n_boot, args.seed)
    vs_single = (
        paired([a for a in arms if a["name"] != "unsteered"], keys, "single",
               args.n_boot, args.seed)
        if any(a["name"] == "single" for a in arms)
        else []
    )

    print()
    tables.show(levels_table(arms))
    print()
    tables.show(delta_table(vs_unsteered, keys, "paired delta vs unsteered"))
    if vs_single:
        print()
        tables.show(
            delta_table(vs_single, keys, "paired delta vs single-layer incumbent")
        )

    jsonio.dump(
        args.out,
        {
            "model": args.model,
            "estimator": args.estimator,
            "axis": args.axis,
            "center": center,
            "budget": budget,
            "n": len(bench.items),
            "asset_split": args.asset_split,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "baselines_from": str(args.baselines) if args.baselines else None,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "arms": arms,
            "paired_vs_unsteered": vs_unsteered,
            "paired_vs_single": vs_single,
        },
    )
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
