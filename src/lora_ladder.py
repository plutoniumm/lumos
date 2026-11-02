#!/usr/bin/env python
"""Sweep the LoRA adapter's strength scale continuously, scored with the
adapter off via ``Adapter.scoring``; writes results/lora_ladder.json."""

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
from lib.steering import SteeringSpec  # noqa: E402

COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("scale", "scale", 2),
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


def rung(scale: float, scored: list[Scored], seconds: float, items) -> dict:
    """One step of the ladder, aggregated."""
    return {
        "scale": scale,
        "fkgl": jsonio.mean(s.read.fkgl for s in scored),
        "words_per_sentence": jsonio.mean(s.read.words_per_sentence for s in scored),
        "pct_hard_words": jsonio.mean(s.read.pct_hard_words for s in scored),
        "nli": jsonio.mean(s.nli for s in scored),
        "pct_gate_passed": jsonio.pct(s.nli_passed for s in scored),
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
    """``--`` for a metric with no finite value."""
    return f"{value:.{places}f}" if isinstance(value, (int, float)) else "--"


def ladder(rungs: list[dict]) -> Table:
    table = tables.table(header_style="bold")
    for head, _, _ in COLUMNS:
        table.add_column(head, justify="right")
    for r in rungs:
        table.add_row(*(cell(r[key], places) for _, key, places in COLUMNS))
    return table


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--asset-split", default="test")
    p.add_argument(
        "--scales",
        default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0",
        help="adapter strength multipliers; 1.0 is the adapter as trained",
    )
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "lora_ladder.json")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config.SEED)

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    bench = Bench.open(args, needs_banks=False)

    print("\n[base] generating s=0.0 reference outputs")
    with bench.adapter.scaled(0.0):
        baselines = bench.generate(
            SteeringSpec.off(),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    base_ppl = [
        bench.conditional_ppl(it, b)
        for it, b in tables.progress(
            list(zip(bench.items, baselines)),
            desc="base ppl",
            unit="item",
        )
    ]

    report = {
        "model": args.model,
        "lora": str(args.lora),
        "n": len(bench.items),
        "asset_split": args.asset_split,
        "scales": scales,
        "rungs": [],
    }

    for scale in scales:
        t0 = time.perf_counter()
        if scale == 0.0:
            outs = baselines
        else:
            with bench.adapter.scaled(scale):
                outs = bench.generate(
                    SteeringSpec.off(),
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                )
        seconds = time.perf_counter() - t0
        scored = [
            bench.score(it, text, baseline_ppl=bp)
            for it, text, bp in tables.progress(
                list(zip(bench.items, outs, base_ppl)),
                desc=f"score s={scale:g}",
                unit="item",
            )
        ]
        report["rungs"].append(rung(scale, scored, seconds, bench.items))

    print()
    tables.show(ladder(report["rungs"]))

    jsonio.dump(args.out, report)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
