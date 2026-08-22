#!/usr/bin/env python
"""Paired A/B tables over an evaluate.py JSONL. Prints markdown for report.md.

Every number here is *paired within item*. Each variant is compared against the
``unsteered`` row for the same ASSET index, and the confidence interval comes
from resampling items, not from resampling the two variants independently. That
matters more than usual on this eval: ASSET sources differ enormously in
difficulty, so an unpaired interval on 32 items is dominated by which sentences
were drawn rather than by what the intervention did.

The interval is a 10k-resample percentile bootstrap, drawn with numpy. The
pure-Python version built each resample element by element and took 3.0s to
print ``eval_main_v2``, against 0.15s now -- 78 calls x 10k x 32 draws is a lot
of interpreter for a table that is expected to be reprinted after every metric
change.

    python src/report.py results/eval_main.jsonl
    python src/report.py results/eval_main.jsonl --baseline unsteered --n-boot 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from lib import jsonio  # noqa: E402

# (column header, how to pull it off a row, "lower is better")
METRICS = (
    ("SARI", lambda r: _dig(r, "sari", "sari"), False),
    ("FKGL", lambda r: _dig(r, "readability", "fkgl"), True),
    ("words/sent", lambda r: _dig(r, "readability", "words_per_sentence"), True),
    ("hard words", lambda r: _dig(r, "readability", "pct_hard_words"), True),
    ("NLI", lambda r: _dig(r, "faithfulness", "score"), False),
    ("ppl ratio", lambda r: r.get("ppl_ratio"), True),
)


def _dig(row: dict, *path):
    cur = row
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load(path: Path) -> dict[tuple[str, str, int], dict]:
    """Last row wins per key -- evaluate.py appends, it does not rewrite.

    Deliberately not ``pd.read_json(lines=True)`` + ``drop_duplicates``, which
    reads as the tidier version of this and is wrong twice. Its parser does not
    round-trip doubles: 5623 of the floats inside the nested ``readability`` /
    ``faithfulness`` / ``sari`` dicts of ``eval_main_v2.jsonl`` come back altered
    (worst relative error 1e-12), so every level and delta printed below would
    stop matching the numbers already in the paper. And a missing ``error`` field
    arrives as ``nan`` rather than ``None`` -- ``bool(nan)`` is ``True``, so the
    obvious ``if r.get("error"): continue`` would silently drop every good row.
    It is also ~4x slower here (30ms vs 8ms), because the rows are mostly text
    and nested dicts that pandas has to box back into objects anyway.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        out[(r["variant"], r["eval_set"], r["item_index"])] = r
    return out


def boot_ci(deltas: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean and 95% percentile interval over item resamples."""
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    n = len(deltas)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.asarray(deltas, dtype=float)[idx].mean(axis=1)
    # np.percentile, not the old means[int(0.025 * n_boot)] on a sorted list.
    # That indexed one order statistic high at both ends, so the whole interval
    # sat slightly high -- ~1e-5 on these deltas, far under the spread the seed
    # itself causes, but this estimator is what decides the `*` markers and
    # several of them turn on a bound within 1e-4 of zero, so it should be the
    # interpolating rule rather than a rounded index.
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Python's left-to-right sum, not means.mean(): the point estimate is not a
    # resampling quantity and must not move when the bootstrap does.
    return sum(deltas) / n, float(lo), float(hi)


def fmt(x, places=2) -> str:
    if x is None or x != x:
        return "--"
    return f"{x:.{places}f}"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path)
    p.add_argument("--baseline", default="unsteered")
    p.add_argument("--eval-set", default="asset")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", type=Path, help="also write the raw numbers as JSON")
    args = p.parse_args(argv)

    rows = load(args.jsonl)
    variants: list[str] = []
    for (variant, eval_set, _), _row in rows.items():
        if eval_set == args.eval_set and variant not in variants:
            variants.append(variant)

    indices = sorted(
        i for (v, e, i) in rows if v == args.baseline and e == args.eval_set
    )
    if not indices:
        print(f"no {args.baseline!r} rows for eval set {args.eval_set!r}")
        return 2

    print(f"\n`{args.jsonl}` -- {args.eval_set}, {len(indices)} items, "
          f"paired against `{args.baseline}`, {args.n_boot} bootstrap resamples\n")

    # Absolute levels.
    head = "| variant | " + " | ".join(m[0] for m in METRICS) + " | degen % | n |"
    print(head)
    print("|" + "---|" * (len(METRICS) + 3))
    payload: dict = {"baseline": args.baseline, "eval_set": args.eval_set, "levels": {}, "deltas": {}}
    for variant in variants:
        cells, record = [], {}
        got = [rows[(variant, args.eval_set, i)] for i in indices
               if (variant, args.eval_set, i) in rows]
        for name, get, _ in METRICS:
            vals = [get(r) for r in got]
            vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
            m = sum(vals) / len(vals) if vals else None
            record[name] = m
            cells.append(fmt(m))
        degen = [bool(_dig(r, "readability", "degenerate")) for r in got]
        pct = 100 * sum(degen) / len(degen) if degen else None
        record["degenerate_pct"] = pct
        payload["levels"][variant] = record
        print(f"| {variant} | " + " | ".join(cells) + f" | {fmt(pct, 1)} | {len(got)} |")

    # Paired deltas with intervals.
    print(f"\nPaired deltas vs `{args.baseline}`, 95% bootstrap CI:\n")
    print("| variant | " + " | ".join(m[0] for m in METRICS) + " |")
    print("|" + "---|" * (len(METRICS) + 1))
    for variant in variants:
        if variant == args.baseline:
            continue
        cells, record = [], {}
        for j, (name, get, lower_better) in enumerate(METRICS):
            deltas = []
            for i in indices:
                a = rows.get((variant, args.eval_set, i))
                b = rows.get((args.baseline, args.eval_set, i))
                if a is None or b is None:
                    continue
                va, vb = get(a), get(b)
                if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                    continue
                if va != va or vb != vb:
                    continue
                deltas.append(va - vb)
            mean, lo, hi = boot_ci(deltas, args.n_boot, args.seed + j)
            record[name] = {"mean": mean, "lo": lo, "hi": hi, "n": len(deltas)}
            # A star marks an interval that excludes zero, i.e. a real move.
            star = "*" if (lo > 0 or hi < 0) else ""
            cells.append(f"{fmt(mean)}{star}<br>[{fmt(lo)}, {fmt(hi)}]")
        payload["deltas"][variant] = record
        print(f"| {variant} | " + " | ".join(cells) + " |")

    print("\n`*` = 95% CI excludes zero.")
    if args.out:
        jsonio.dump(args.out, payload)
        print(f"\nraw -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
