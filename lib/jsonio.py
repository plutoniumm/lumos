"""Writing result files that a strict reader can open.

Why this is a module and not four copies
----------------------------------------
Every arm in ``src/`` ends the same way: build a dict of floats, drop the ones
JSON cannot represent, write a line, flush. Four files grew their own ``_finite``
and ``_clean`` to do it and the four disagreed -- ``src/evaluate.py`` passed
non-numbers through untouched while the other three quietly turned every string
they were handed into ``null``. That difference never fired only because each
copy happened to be called on floats. It is the kind of thing that fires the
first time someone reuses the helper one line further down.

The rules, in one place:

* ``allow_nan=False`` everywhere. ``NaN`` and ``Infinity`` are not JSON, and a
  file that emits them loads in Python and fails in anything else -- which is
  the worst possible time to find out, since these files are the paper's data.
  A metric that could not be computed is ``null``, and ``null`` is a value a
  reader has to handle rather than one it can mistake for a number.
* Flush per row. A sweep is hours long and is expected to be killed; a block
  buffer would lose the tail of the run that was actually completed.
* Never edit a result file in place. ``JsonlWriter`` appends, so a resumed run
  adds to what is on disk instead of replacing it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator


def finite(x: Any) -> Any:
    """``None`` for a non-finite float, *x* unchanged otherwise.

    Deliberately a passthrough for everything that is not a float: this runs
    over whole rows, and those hold the source text, the variant name and the
    spec label alongside the numbers.
    """
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
    """Mean over the finite entries; ``None`` when there are none.

    ``None`` rather than ``nan`` so the result survives ``allow_nan=False``
    without a second pass, and so "no items had this metric" is distinguishable
    downstream from "the mean happened to be zero".
    """
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
    """``(line_number, row)`` for each line; ``None`` where the line was junk.

    A partial line is what a killed run leaves behind, so it is a normal thing
    to find rather than an error -- but it is reported rather than skipped in
    silence, because the alternative is a resume that quietly drops work.
    """
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
    """Append-only JSONL sink, one flushed line per row.

    Used as a context manager so an interrupted run closes its file on the way
    out rather than relying on interpreter teardown.
    """

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
