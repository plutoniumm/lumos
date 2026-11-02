"""Measure the logit gap between plausible continuations, per model.

Scales ``config.EASY_WORD_PENALTY`` to each model's own logit scale.

    python scripts/05_logit_gaps.py                       # both models
    python scripts/05_logit_gaps.py google/gemma-4-12B-it  # one
"""

import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import data, models

RANK = 10
N_TEXTS = 24


def gaps_for(repo_id: str) -> dict:
    lm = models.load(repo_id)
    texts = data.load_asset("valid")[:N_TEXTS]
    gaps: list[float] = []
    with torch.inference_mode():
        for item in texts:
            prompt = lm.build_prompt(item.source.strip())
            batch = lm.encode(prompt)
            logits = lm.model(**batch).logits[0, -1].float()
            top = torch.topk(logits, RANK).values
            gaps.append(float(top[0] - top[RANK - 1]))
    del lm
    if config.DEVICE == "mps":
        torch.mps.empty_cache()
    return {
        "repo_id": repo_id,
        "median_gap": st.median(gaps),
        "mean_gap": st.mean(gaps),
        "min_gap": min(gaps),
        "max_gap": max(gaps),
        "n": len(gaps),
    }


def write_override(repo_id: str, penalty: float) -> None:
    """Put the measured value in config.py, where every other tunable lives."""
    path = Path(config.ROOT) / "config.py"
    text = path.read_text()
    marker = "EASY_WORD_PENALTY_OVERRIDES: dict[str, float] = "
    start = text.index(marker) + len(marker)
    end = text.index("\n", start)
    entries = dict(config.EASY_WORD_PENALTY_OVERRIDES)
    entries[repo_id] = round(penalty, 2)
    rendered = "{" + ", ".join(f'"{k}": {v}' for k, v in sorted(entries.items())) + "}"
    path.write_text(text[:start] + rendered + text[end:])
    print(f"  wrote {repo_id} -> {penalty:.2f} into config.py")


def main(argv: list[str]) -> int:
    write = "--write" in argv
    argv = [a for a in argv if a != "--write"]
    repos = argv or [config.LLAMA, config.GEMMA4]
    out = []
    for repo in repos:
        print(f"\n[measure] {repo}")
        g = gaps_for(repo)
        out.append(g)
        print(
            f"  rank1 - rank{RANK} gap: median {g['median_gap']:.2f} nats "
            f"(mean {g['mean_gap']:.2f}, range {g['min_gap']:.2f}-{g['max_gap']:.2f})"
        )

    ref = next((g for g in out if g["repo_id"] == config.LLAMA), None)
    if ref is None or len(out) < 2:
        return 0
    print(
        f"\nreference: {config.LLAMA} at penalty {config.EASY_WORD_PENALTY} "
        f"= {config.EASY_WORD_PENALTY / ref['median_gap']:.2f}x its median gap"
    )
    for g in out:
        if g["repo_id"] == ref["repo_id"]:
            continue
        scaled = config.EASY_WORD_PENALTY * g["median_gap"] / ref["median_gap"]
        print(f"\n{g['repo_id']}: equivalent penalty = {scaled:.2f}")
        if write:
            write_override(g["repo_id"], scaled)
        else:
            print(f'  EASY_WORD_PENALTY_OVERRIDES = {{"{g["repo_id"]}": {scaled:.2f}}}')

    import json

    out_path = Path(config.RESULTS_DIR) / "logit_gaps.json"
    out_path.write_text(
        json.dumps({"rank": RANK, "n_texts": N_TEXTS, "models": out}, indent=2)
    )
    print(f"\n[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
