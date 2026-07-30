"""Fetch every model the project needs. Safe to re-run; skips complete files.

python scripts/00_download.py            # all three
python scripts/00_download.py llama nli  # subset
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import hub

TARGETS = {"llama": config.LLAMA, "gemma": config.GEMMA, "nli": config.NLI_MODEL}


def main(argv: list[str]) -> int:
    keys = argv or list(TARGETS)
    unknown = [k for k in keys if k not in TARGETS]
    if unknown:
        print(f"unknown target(s): {unknown}; choose from {list(TARGETS)}")
        return 2

    failures = []
    for key in keys:
        repo = TARGETS[key]
        try:
            hub.download(repo)
        except Exception as exc:  # keep going; report everything at the end
            print(f"[hub] FAILED {repo}: {type(exc).__name__}: {exc}")
            failures.append((repo, exc))

    if failures:
        print(f"\n{len(failures)} download(s) failed:")
        for repo, exc in failures:
            print(f"  {repo}: {exc}")
        return 1
    print("\nall downloads complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
