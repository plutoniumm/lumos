"""Readability metrics, split by the axis each one measures.

Effects are compared against the unsteered rewrite, not a fixed constant.

FKGL is invalid on single sentences. It is length-dominated, so it rewards
padding: a faithful 17-word paraphrase scored 14.3 while a 60-word invented
rewrite scored 7.0. Judge sentences by SARI and the per-axis signals.

Degeneracy rules stay narrow because genuine simplification is statistically
similar to degeneration. Perplexity is not a backstop: repetition is low-entropy,
so a loop scores 18.8 where valid repetitive prose scores 37.0.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import textstat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.chunking import split_sentences  # noqa: E402

# Hyphens separate words rather than joining them. With hyphens treated as word
# characters, degenerate output like "Photos-emp-am-emp-am-emp" tokenises to a
# single 55-character "word" and every diversity statistic reads 1.00.
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# Markers that introduce an extra clause. Counting these per sentence is a cheap
# stand-in for parse depth and needs no parser.
_CLAUSE = re.compile(
    r"[;:,]|\b(?:because|although|though|whereas|while|since|unless|whether|"
    r"which|who|whom|whose|that|when|where|if|however|therefore|moreover|"
    r"furthermore|nevertheless|thereby|wherein|whereby|thus)\b",
    re.IGNORECASE,
)

_TERMINATOR = re.compile(r"[.!?]")

LONG_SENTENCE_WORDS = 25
POLYSYLLABIC = 3

# Degeneracy thresholds, set so that valid simplified prose passes. Verified
# against both hand-written simple text and real collapsed generations.
MIN_WORDS = 3
MAX_PUNCT_RATIO = 0.30
MIN_DISTINCT_1 = 0.35
MAX_TOKEN_DOMINANCE = 0.45
MIN_REPEAT_RUN = 3
MAX_REPEATED_BIGRAM_FRAC = 0.60
NO_TERMINATOR_WORDS = 30


def _distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def _repeated_bigram_frac(tokens: list[str]) -> float:
    """Fraction of bigram positions whose bigram occurs more than once."""
    if len(tokens) < 3:
        return 0.0
    grams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    counts = Counter(grams)
    return sum(1 for g in grams if counts[g] > 1) / len(grams)


def _max_repeat_run(tokens: list[str], n: int = 4) -> int:
    """Longest run of an immediately repeating n-gram, e.g. 'a b a b a b' -> 3."""
    if len(tokens) < 2 * n:
        return 0
    best = 0
    for i in range(len(tokens) - n):
        gram = tokens[i : i + n]
        run, j = 1, i + n
        while j + n <= len(tokens) and tokens[j : j + n] == gram:
            run += 1
            j += n
        best = max(best, run)
    return best if best > 1 else 0


# Bigram loops built from function words are ordinary English -- "on and on and
# on", "row after row after row", "what makes a cat a cat". Loops built only from
# content words are not.
_FUNCTION_WORDS = frozenset(
    """a an the and or but nor so yet of to in on at by for with from into onto
    over under above below after before during until while as than that this
    these those it its is are was were be been being am do does did have has had
    not no if then when where who whom whose which what how very much many more
    most some any all both each other another such own same too also just only
    up down out off again once here there i you he she we they me him her us them
    my your his their our one two three""".split()
)


def _loop_signature(tokens_by_sentence: list[list[str]]) -> tuple[int, int]:
    """Widest immediate n-gram loop within a single sentence. Per sentence, and
    function-word loops excluded, or ordinary prose flags.
    """
    worst = (0, 0)
    for tokens in tokens_by_sentence:
        for n in (1, 2, 3, 4):
            run = _max_repeat_run(tokens, n)
            if n == 1:
                ok = run >= 4
            else:
                # A single adjacent repeat can be ordinary English when function
                # words are involved -- "the federal territory: the Federal
                # Territory", restating a term after a colon. Three or more
                # repeats of the same phrase is pathological either way.
                ok = run >= 3 or (run >= 2 and not _function_loop(tokens, n))
            if ok and run > worst[1]:
                worst = (n, run)
    return worst


def _function_loop(tokens: list[str], n: int) -> bool:
    """Whether any immediately repeating n-gram here contains a function word."""
    for i in range(len(tokens) - 2 * n + 1):
        gram = tokens[i : i + n]
        if tokens[i + n : i + 2 * n] == gram and any(
            t in _FUNCTION_WORDS for t in gram
        ):
            return True
    return False


def _safe(fn, text: str, default: float = 0.0) -> float:
    try:
        val = fn(text)
        return float(val) if val is not None else default
    except Exception:
        return default


@dataclass
class Readability:
    n_words: int
    n_sentences: int

    # --- structural axis ---
    words_per_sentence: float
    clauses_per_sentence: float
    pct_long_sentences: float

    # --- lexical axis ---
    syllables_per_word: float
    chars_per_word: float
    pct_hard_words: float
    pct_polysyllabic: float

    # --- standard composites ---
    fkgl: float
    fre: float
    dale_chall: float

    # --- degeneracy signals ---
    distinct_1: float
    distinct_2: float
    max_repeat_run: int
    punct_ratio: float
    token_dominance: float
    repeated_bigram_frac: float
    degeneracy: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structural_complexity(self) -> float:
        """Primary structural signal; lower is simpler."""
        return self.words_per_sentence

    @property
    def lexical_complexity(self) -> float:
        """Primary lexical signal; lower is simpler."""
        return self.pct_hard_words

    @property
    def degenerate(self) -> bool:
        """Unambiguously not language. Subtler breakage is perplexity's job."""
        return bool(self.degeneracy)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["degeneracy"] = list(self.degeneracy)
        d["degenerate"] = self.degenerate
        return d


def _hard_word_count(words: list[str]) -> int:
    """Per-token Dale-Chall hard-word count. textstat.difficult_words deduplicates,
    which inverts the measure.
    """
    try:
        return sum(1 for w in words if textstat.is_difficult_word(w))
    except Exception:
        return int(_safe(textstat.difficult_words, " ".join(words)))


def _degeneracy_reasons(
    text: str, words: list[str], lowered: list[str], sentences: list[str]
) -> tuple:
    reasons = []
    n = len(words)

    if n < MIN_WORDS:
        reasons.append("too_short")

    non_space = [c for c in text if not c.isspace()]
    punct = sum(1 for c in non_space if not c.isalnum())
    punct_ratio = punct / len(non_space) if non_space else 1.0
    if punct_ratio > MAX_PUNCT_RATIO:
        reasons.append("punct_heavy")

    per_sentence = [[w.lower() for w in _WORD.findall(s)] for s in sentences] or [
        lowered
    ]
    loop_n, loop_run = _loop_signature(per_sentence)
    if loop_run:
        reasons.append(f"loop_{loop_n}gram_x{loop_run}")

    if n >= 8:
        if _distinct_n(lowered, 1) < MIN_DISTINCT_1:
            reasons.append("low_diversity")
        top = Counter(lowered).most_common(1)[0][1] / n
        if top > MAX_TOKEN_DOMINANCE:
            reasons.append("token_dominance")
        if _repeated_bigram_frac(lowered) > MAX_REPEATED_BIGRAM_FRAC:
            reasons.append("repeated_bigrams")

    if n >= NO_TERMINATOR_WORDS and not _TERMINATOR.search(text):
        reasons.append("no_terminator")

    return tuple(reasons)


def measure(text: str) -> Readability:
    text = (text or "").strip()
    words = _WORD.findall(text)
    lowered = [w.lower() for w in words]
    sentences = split_sentences(text)

    n_words = len(words)
    n_sents = len(sentences)
    reasons = _degeneracy_reasons(text, words, lowered, sentences)

    non_space = [c for c in text if not c.isspace()]
    punct_ratio = (
        sum(1 for c in non_space if not c.isalnum()) / len(non_space)
        if non_space
        else 1.0
    )

    if n_words == 0:
        return Readability(
            n_words=0,
            n_sentences=n_sents,
            words_per_sentence=0.0,
            clauses_per_sentence=0.0,
            pct_long_sentences=0.0,
            syllables_per_word=0.0,
            chars_per_word=0.0,
            pct_hard_words=0.0,
            pct_polysyllabic=0.0,
            fkgl=0.0,
            fre=0.0,
            dale_chall=0.0,
            distinct_1=0.0,
            distinct_2=0.0,
            max_repeat_run=0,
            punct_ratio=punct_ratio,
            token_dominance=1.0,
            repeated_bigram_frac=0.0,
            degeneracy=reasons or ("too_short",),
        )

    clauses = sum(len(_CLAUSE.findall(s)) for s in sentences)
    long_sents = sum(
        1 for s in sentences if len(_WORD.findall(s)) > LONG_SENTENCE_WORDS
    )

    syllables = _safe(textstat.syllable_count, text)
    hard = _hard_word_count(words)
    poly = sum(1 for w in words if _safe(textstat.syllable_count, w) >= POLYSYLLABIC)

    return Readability(
        n_words=n_words,
        n_sentences=n_sents,
        words_per_sentence=n_words / max(n_sents, 1),
        clauses_per_sentence=clauses / max(n_sents, 1),
        pct_long_sentences=long_sents / max(n_sents, 1),
        syllables_per_word=syllables / n_words,
        chars_per_word=sum(len(w) for w in words) / n_words,
        pct_hard_words=hard / n_words,
        pct_polysyllabic=poly / n_words,
        fkgl=_safe(textstat.flesch_kincaid_grade, text),
        fre=_safe(textstat.flesch_reading_ease, text),
        dale_chall=_safe(textstat.dale_chall_readability_score, text),
        distinct_1=_distinct_n(lowered, 1),
        distinct_2=_distinct_n(lowered, 2),
        max_repeat_run=_max_repeat_run(lowered),
        punct_ratio=punct_ratio,
        token_dominance=Counter(lowered).most_common(1)[0][1] / n_words,
        repeated_bigram_frac=_repeated_bigram_frac(lowered),
        degeneracy=reasons,
    )


def relative_deltas(baseline: Readability, steered: Readability) -> dict[str, float]:
    """Fractional change per metric, negative meaning simpler."""
    out: dict[str, float] = {}
    for key in (
        "words_per_sentence",
        "clauses_per_sentence",
        "pct_long_sentences",
        "syllables_per_word",
        "chars_per_word",
        "pct_hard_words",
        "pct_polysyllabic",
        "fkgl",
        "dale_chall",
    ):
        base = getattr(baseline, key)
        new = getattr(steered, key)
        out[f"d_{key}"] = (new - base) / base if base else 0.0
    out["d_fre"] = steered.fre - baseline.fre  # already an absolute 0-100 scale
    return out


def axis_selectivity(baseline: Readability, steered: Readability, axis: str) -> float:
    """How cleanly a vector hits its own axis rather than the other one."""
    d = relative_deltas(baseline, steered)
    structural = -d["d_words_per_sentence"]
    lexical = -d["d_pct_hard_words"]
    if axis == "structural":
        return structural - abs(lexical)
    if axis == "lexical":
        return lexical - abs(structural)
    raise ValueError(f"unknown axis {axis!r}")
