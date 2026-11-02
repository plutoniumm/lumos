"""Strip the wrapper an instruct model puts around a rewrite it was asked for."""

from __future__ import annotations

import re

_PREAMBLE = re.compile(
    r"""^\s*
    (?:here(?:'s|\s+is)\b[^\n]*
      |[^\n]*\b(?:rewritten|simplified|revised)\s+version\b[^\n]*
      |[^\n]*\bhere\s+are\s+a\s+few\s+ways\b[^\n]*
    )
    :\s*\n+""",
    re.IGNORECASE | re.VERBOSE,
)

_WRAPPING_QUOTES = re.compile(r'^"(.*)"$', re.DOTALL)


def strip_preamble(text: str) -> str:
    """Return the rewrite alone; *text* unchanged if there is no wrapper."""
    if not text:
        return text
    out = _PREAMBLE.sub("", text, count=1).strip()
    match = _WRAPPING_QUOTES.match(out)
    if match and '"' not in match.group(1):
        out = match.group(1).strip()
    return out or text


def had_preamble(text: str) -> bool:
    return strip_preamble(text) != (text or "").strip()
