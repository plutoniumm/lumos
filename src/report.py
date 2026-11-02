#!/usr/bin/env python
"""Paired A/B tables over an evaluate.py JSONL. Prints markdown for report.md.

    python src/report.py results/eval_main.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from lib import jsonio  # noqa: E402

# header, value getter, lower-is-better
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
    """Last row wins per key -- evaluate.py appends, it does not rewrite."""
    out: dict[tuple[str, str, int], dict] = {}
    for n, r in jsonio.read_jsonl(path):
        if r is None:
            raise ValueError(f"{path}:{n}: invalid JSON")
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
    lo, hi = np.percentile(means, [2.5, 97.5])
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
