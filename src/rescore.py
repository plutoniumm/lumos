#!/usr/bin/env python
"""Re-run metrics over generations already on disk. No model output changes.

Exists because a scoring bug found late is not a reason to spend two hours
regenerating text that was never wrong. The preamble strip in ``lib.cleanup``
changed what the metrics see, not what the model produced, so every row in an
existing JSONL can be brought onto the new convention by re-scoring it.

Writes a new file rather than editing in place: the old numbers are what was
reported, and overwriting them would leave nothing to check the correction
against.

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
from src.evaluate import EvalItem, score_item  # noqa: E402

# A rescore is an NLI pass and two perplexity passes per row over a file that
# can hold thousands, so the loop needs a bar -- but it is also the thing you
# start and redirect, and a redrawing bar in a log file is megabytes of \r.


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    args = p.parse_args(argv)

    rows = [json.loads(l) for l in args.jsonl.read_text().splitlines() if l.strip()]
    print(f"[in]   {len(rows)} row(s) from {args.jsonl}")
    n_pre = sum(1 for r in rows if r.get("output") and cleanup.had_preamble(r["output"]))
    print(f"[in]   {n_pre} row(s) carry a preamble ({100 * n_pre / max(len(rows), 1):.1f}%)")

    lm = models.load(args.model)
    fh = args.out.open("w", encoding="utf-8")
    cache: dict[str, float] = {}
    changed = 0
    for row in tables.progress(rows, desc="rescore", unit="row"):
        if row.get("error") or not row.get("output"):
            fh.write(json.dumps(row) + "\n")
            continue
        # score_item needs the real references to compute SARI; rows carry only
        # a count, so SARI is preserved from the original row rather than faked.
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
