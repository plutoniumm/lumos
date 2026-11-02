"""Sentence splitting and rejoining. No external dependencies."""

from __future__ import annotations

import re

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
    "inc", "ltd", "co", "corp", "dept", "univ", "est", "e.g", "i.e", "etc",
    "cf", "al", "vs", "viz", "ibid", "op", "fig", "figs", "eq", "eqs", "ref",
    "refs", "no", "nos", "vol", "pp", "approx", "min", "max", "avg", "cal",
    "sec", "yr", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
}

_SENT_END = re.compile(r"([.!?][\"')\]]*)(\s+)(?=[\"'(\[]*[A-Z0-9])")
_WS = re.compile(r"\s+")


def _is_abbrev(chunk: str) -> bool:
    """Does ``chunk`` end in an abbreviation rather than a real sentence stop?"""
    tail = chunk.rstrip()
    if not tail.endswith("."):
        return False
    word = re.split(r"[\s(\[\"']", tail)[-1].rstrip(".").lower()
    if word in _ABBREV:
        return True
    if re.fullmatch(r"(?:[a-z]\.)+[a-z]?", word + "."):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split into sentences, preserving each sentence's own text verbatim."""
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    start = 0
    for match in _SENT_END.finditer(text):
        end = match.end(1)
        candidate = text[start:end]
        if _is_abbrev(candidate):
            continue
        out.append(candidate.strip())
        start = match.end(2)

    remainder = text[start:].strip()
    if remainder:
        out.append(remainder)
    return [s for s in out if s]


def join_sentences(sentences: list[str]) -> str:
    """Rejoin simplified sentences into a passage."""
    cleaned = [_WS.sub(" ", s.strip()) for s in sentences if s and s.strip()]
    return " ".join(cleaned)


def rolling_context(simplified: list[str], n: int) -> str:
    """The last *n* already-simplified sentences, as context for the next one."""
    if n <= 0 or not simplified:
        return ""
    return join_sentences(simplified[-n:])
