"""Terminal tables and progress bars that survive being redirected to a file.

No ANSI colour, a fixed width (rich sizes to the terminal it finds, which
breaks a redirected log), and progress bars that disable themselves unless
stderr is a tty.
"""

from __future__ import annotations

import sys

from rich import box  # noqa: F401
from rich.console import Console
from rich.table import Table
from tqdm import tqdm as _tqdm

TABLE_STYLE = {
    "box": box.SIMPLE,
    "pad_edge": False,
    "show_edge": False,
    "caption_justify": "left",
}


def console(width: int = 140) -> Console:
    """Colour when a human is watching, plain when redirected."""
    tty = sys.stdout.isatty()
    return Console(
        force_terminal=None if tty else False,
        no_color=not tty,
        width=width,
        soft_wrap=False,
        highlight=False,
        markup=False,
    )


def table(title: str | None = None, caption: str | None = None, **kw) -> Table:
    """A ``rich`` table styled for a log file."""
    return Table(title=title, caption=caption, **(TABLE_STYLE | kw))


def show(t: Table, width: int = 140) -> None:
    console(width).print(t)


def write(message: str) -> None:
    """Print a line without a live progress bar eating it."""
    _tqdm.write(message)


def progress(iterable=None, *, enabled: bool = True, **kw):
    """``tqdm`` that stays silent unless stderr is a terminal."""
    kw["disable"] = not enabled or not sys.stderr.isatty()
    kw.setdefault("leave", False)
    return _tqdm(iterable, **kw)
