"""Terminal tables and progress bars that survive being redirected to a file.

Why this is centralised
-----------------------
Every long-running entry point in this project is normally started with its
output redirected -- a sweep is hours, a full eval is most of an afternoon, and
both are expected to be killed and resumed. So the terminal is not the audience
for these tables; a log file read days later is. Two things follow, and neither
is rich's or tqdm's default:

* **No ANSI.** Colour codes written into a log file turn every subsequent read
  of it into an exercise in ignoring escape sequences, and they break `grep`
  on the values. ``console()`` forces them off rather than relying on rich's
  tty detection, which guesses wrong under `nohup`, `tee` and CI.
* **A fixed width.** rich sizes to the terminal it finds. A sweep started in a
  narrow window would wrap every row of the depth profile into two, and the
  file would be unreadable for reasons that have nothing to do with the data.
  140 columns fits the widest table here (the depth profile at two axes).

``progress()`` is the same argument for tqdm: a progress bar redraws its line
thousands of times, and in a redirected run each redraw is a separate line in
the file. It is disabled unless stderr is a terminal, so an interactive run
gets the bar and a logged run gets nothing rather than megabytes of noise.

Nothing here touches a result file. These are presentation helpers, and no
number should be computed inside one.
"""

from __future__ import annotations

import sys

from rich import box  # noqa: F401  (re-exported for callers picking a style)
from rich.console import Console
from rich.table import Table
from tqdm import tqdm as _tqdm

# SIMPLE gives a single rule under the header and no outer frame: the columns
# stay alignable by eye in a plain-text file, which the heavier box styles do
# not manage once a terminal is not interpreting them.
TABLE_STYLE = {
    "box": box.SIMPLE,
    "pad_edge": False,
    "show_edge": False,
    "caption_justify": "left",
}


def console(width: int = 140) -> Console:
    """Colour when a human is watching, provably plain when redirected.

    ``force_terminal=False`` rather than leaving it to rich's own detection:
    under `nohup`, `tee` and CI, ``FORCE_COLOR``/``TERM`` can convince rich it
    has a terminal while stdout is a file, and then the log gets escape codes
    after all. ``no_color`` alone would not be enough -- it suppresses colour
    but not bold.
    """
    tty = sys.stdout.isatty()
    return Console(
        force_terminal=None if tty else False,
        no_color=not tty,
        width=width,
        soft_wrap=False,
        highlight=False,
        # Cell text is data -- variant names and spec labels carry brackets, and
        # rich would read "[lexical]" as a style tag and swallow it.
        markup=False,
    )


def table(title: str | None = None, caption: str | None = None, **kw) -> Table:
    """A ``rich`` table styled for a log file."""
    return Table(title=title, caption=caption, **(TABLE_STYLE | kw))


def show(t: Table, width: int = 140) -> None:
    console(width).print(t)


def write(message: str) -> None:
    """Print a line without a live bar eating it.

    ``tqdm`` repaints its bar on the current line, so an ordinary ``print``
    from inside a loop gets overwritten. This is the escape hatch, and it is
    a plain ``print`` when no bar is active.
    """
    _tqdm.write(message)


def progress(iterable=None, *, enabled: bool = True, **kw):
    """``tqdm`` that stays silent unless stderr is a terminal.

    ``enabled`` is the caller's own switch (several estimators take a
    ``progress`` flag) and is ANDed with the tty check rather than replacing
    it -- passing ``disable`` straight through would let a caller that wants a
    bar interactively also spray one into a redirected log.
    """
    kw["disable"] = not enabled or not sys.stderr.isatty()
    kw.setdefault("leave", False)
    return _tqdm(iterable, **kw)
