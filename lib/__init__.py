"""Package-wide setup that has to happen before any model is loaded.

Nothing here is project logic. It exists because two of transformers' defaults
are wrong for how this project is actually run -- long jobs, started with output
redirected to a log that gets read days later.
"""

from __future__ import annotations

import sys


def _plain_load_report() -> None:
    """Stop transformers colouring its load report into our log files.

    transformers 5 writes the "LOAD REPORT" banner with a hardcoded
    ``PALETTE['bold']`` / ``PALETTE['reset']`` -- see
    ``transformers/utils/loading_report.py``, which interpolates the palette
    directly instead of going through its own isatty-aware ``_style()``. The
    result is two escape bytes in every redirected log, which is enough to
    break ``grep`` on the first line of the report and to make the file render
    as garbage in anything that does not interpret ANSI.

    Blanking the palette rather than silencing the report: the report is worth
    having -- it is what tells you a checkpoint had unexpected keys -- and only
    the styling is wrong, with no env var separating the two. No-op when stdout
    is a terminal, where the colour is doing its job.

    This lives in the package __init__ rather than in ``lib.models`` because
    ``lib.metrics.faithfulness`` loads the NLI model through transformers
    directly and never imports ``lib.models``; the patch has to be in force for
    any ``lib.*`` import, not just the ones that go through the model loader.
    """
    if sys.stdout.isatty():
        return
    try:
        from transformers.utils import loading_report
    except ImportError:  # a transformers version without the report at all
        return
    for key in loading_report.PALETTE:
        loading_report.PALETTE[key] = ""


_plain_load_report()
