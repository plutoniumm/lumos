"""Package-wide setup: disables ANSI colour in transformers' load report."""

from __future__ import annotations

import sys


def _plain_load_report() -> None:
    """Blank transformers' load-report palette so redirected logs stay plain text."""
    if sys.stdout.isatty():
        return
    try:
        from transformers.utils import loading_report
    except ImportError:
        return
    for key in loading_report.PALETTE:
        loading_report.PALETTE[key] = ""


_plain_load_report()
