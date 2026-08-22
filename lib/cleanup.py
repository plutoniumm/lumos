"""Strip the wrapper an instruct model puts around a rewrite it was asked for.

Why this is a scoring concern and not a cosmetic one
----------------------------------------------------
Asked to rewrite a sentence, Llama-3.2-3B usually just rewrites it. Asked to
rewrite a 130-word abstract, it opens with "Here's a rewritten version of the
given text:" 30% of the time -- against 3% on ASSET. That line is not part of
the rewrite, but every metric treats it as if it were.

Faithfulness pays the most. The score averages entailment over output
*sentences*, and a preamble entails nothing, so it enters the mean as a near
zero. Measured on one abstract: a rewrite that scores 0.990 once the wrapper is
removed scores 0.401 with it, and 0.084 once a shattered citation trailer is
added on top. That is a wrong answer about meaning preservation, produced by a
formatting habit.

Readability pays too, in the other direction -- "Here's a rewritten version of
the given text" is easy prose, and it drags FKGL down for free.

Applied uniformly to every arm, so it cannot favour one. The raw generation is
kept alongside the cleaned one in every row, because a stripper that silently
edits model output is exactly the sort of thing that should be auditable.
"""

from __future__ import annotations

import re

# A single leading line that announces the rewrite and ends in a colon. Anchored
# and count-limited: a rewrite whose real first sentence happens to end in a
# colon must survive, so the line also has to *say* it is a preamble.
_PREAMBLE = re.compile(
    r"""^\s*
    (?:here(?:'s|\s+is)\b[^\n]*
      |[^\n]*\b(?:rewritten|simplified|revised)\s+version\b[^\n]*
      |[^\n]*\bhere\s+are\s+a\s+few\s+ways\b[^\n]*
    )
    :\s*\n+""",
    re.IGNORECASE | re.VERBOSE,
)

# Models often wrap the rewrite in quotes once the preamble is gone.
_WRAPPING_QUOTES = re.compile(r'^"(.*)"$', re.DOTALL)


def strip_preamble(text: str) -> str:
    """The rewrite alone. Returns *text* unchanged when there is no wrapper."""
    if not text:
        return text
    out = _PREAMBLE.sub("", text, count=1).strip()
    match = _WRAPPING_QUOTES.match(out)
    if match and '"' not in match.group(1):
        out = match.group(1).strip()
    return out or text


def had_preamble(text: str) -> bool:
    return strip_preamble(text) != (text or "").strip()
