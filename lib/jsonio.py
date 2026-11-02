"""Writing result files that a strict reader can open.

No NaN/Infinity (``allow_nan=False``), flushed per row, append-only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator


def finite(x: Any) -> Any:
    """``None`` for a non-finite float, *x* unchanged otherwise."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def clean(obj: Any) -> Any:
    """*obj* with every non-finite float replaced by ``None``, recursively."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return finite(obj)


def mean(values) -> float | None:
    """Mean over the finite entries; ``None`` when there are none."""
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def pct(flags) -> float | None:
    """Percentage of true flags; ``None`` for an empty sequence."""
    flags = list(flags)
    return 100.0 * sum(bool(f) for f in flags) / len(flags) if flags else None


def dump(path: Path, payload: Any, *, indent: int | None = 2) -> Path:
    """Write *payload* as JSON, cleaned. Returns *path* so callers can log it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=indent, allow_nan=False))
    return path


def read_jsonl(path: Path) -> Iterator[tuple[int, dict | None]]:
    """``(line_number, row)`` for each line; ``None`` where the line was junk."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError:
                yield n, None


class JsonlWriter:
    """Append-only JSONL sink, one flushed line per row."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self.n_written = 0

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(clean(row), allow_nan=False) + "\n")
        self._fh.flush()
        self.n_written += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
