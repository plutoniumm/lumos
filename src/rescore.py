#!/usr/bin/env python
"""Re-run metrics over generations already on disk, writing a new JSONL.

    python src/rescore.py results/eval_main.jsonl --out results/eval_main_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from lib import cleanup, jsonio, models, tables  # noqa: E402
from lib.cli import die  # noqa: E402
from src.evaluate import EvalItem, score_item  # noqa: E402


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--model",
        default=None,
        help=(
            "model to score with; must match the model that generated the "
            "rows (each row records it under 'model'). Inferred from the "
            f"file when omitted -- never {config.ACTIVE_MODEL!r} by default, "
            "because that is whichever model happens to be active on this "
            "machine right now, not necessarily the one this file's text "
            "came from"
        ),
    )
    args = p.parse_args(argv)

    rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    print(f"[in]   {len(rows)} row(s) from {args.jsonl}")
    n_pre = sum(1 for r in rows if r.get("output") and cleanup.had_preamble(r["output"]))
    print(f"[in]   {n_pre} row(s) carry a preamble ({100 * n_pre / max(len(rows), 1):.1f}%)")

    row_models = sorted({r["model"] for r in rows if r.get("model")})
    if args.model is None:
        if len(row_models) != 1:
            die(
                f"--model not given, and {args.jsonl} does not name exactly "
                f"one model across its rows (found {row_models or 'none'}); "
                f"cannot infer which model to score with. Pass --model "
                f"explicitly."
            )
        args.model = row_models[0]
        print(f"[model] inferred {args.model!r} from {args.jsonl}")
    elif row_models and args.model not in row_models:
        die(
            f"--model {args.model!r} does not match the model recorded in "
            f"{args.jsonl}'s rows ({row_models}); rescoring under a "
            f"different model than generated the text would silently "
            f"recompute NLI and the perplexity ratio against the wrong "
            f"reference. Pass --model matching the file, or omit --model to "
            f"infer it."
        )

    lm = models.load(args.model)
    fh = args.out.open("w", encoding="utf-8")
    cache: dict[str, float] = {}
    changed = 0
    for row in tables.progress(rows, desc="rescore", unit="row"):
        if row.get("error") or not row.get("output"):
            fh.write(json.dumps(row) + "\n")
            continue
        before = row.get("faithfulness", {}).get("score")
        patch = score_item(
            lm,
            EvalItem(row["item_index"], row["source"], []),
            row["output"],
            row.get("baseline_output"),
            cache,
            f"{row['eval_set']}|{row['mode']}|{row['item_index']}",
            skip_nli=False,
            skip_sari=True,
        )
        if "sari" in row:
            # SARI needs the item's references, which the row doesn't store;
            # reuse the value already computed instead of recomputing it.
            patch["sari"] = row["sari"]
            patch["metrics"] = patch["metrics"] + ["sari"]
        row.update(patch)
        row["output_scored"] = cleanup.strip_preamble(row["output"])
        row["had_preamble"] = cleanup.had_preamble(row["output"])
        after = row.get("faithfulness", {}).get("score")
        if before is not None and after is not None and abs(after - before) > 1e-6:
            changed += 1
        fh.write(json.dumps(jsonio.clean(row), allow_nan=False) + "\n")
    fh.close()
    print(f"[out]  {args.out}  ({changed} row(s) changed score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
