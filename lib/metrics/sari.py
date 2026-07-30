"""SARI (Xu et al. 2016), the metric simplification systems are actually judged on.

Readability scores can be gamed by truncation: delete half a sentence and FKGL
falls. SARI cannot, because it scores *edits* against what R human references
did rather than scoring the output in isolation. Dropping content the references
kept costs keep-recall; inventing content no reference used costs add-precision.
That is why it sits alongside readability here instead of replacing it: FKGL says
how simple the output is, SARI says whether the simplification was the right one.

Three implementation details separate a correct SARI from the many subtly wrong
reimplementations:

*Clipped counts, not set membership.* The reference side is one Counter over the
concatenation of all R references, so an n-gram appearing in 3 of 4 references
has count 3; the source and prediction sides are multiplied by R before being
intersected with it. Every keep/delete precision and recall term is therefore
weighted by the fraction of references that agree with the edit. Implementations
that ask "is this n-gram in any reference" throw that weighting away and inflate
scores on n-grams a single reference happened to use.

*Deletion is precision-only.* The paper omits deletion recall on purpose. An
n-gram the references dropped but the prediction kept is already penalised by
keep-precision; scoring it a second time as missed deletion would reward
indiscriminate deletion, which is the exact failure mode SARI exists to catch.

*Addition is set-based.* Asymmetric with keep/delete, but that asymmetry is in
the original and in HuggingFace ``evaluate``, so it is preserved: an added
n-gram counts once regardless of how many references used it. Deviating here
makes numbers incomparable with everything published.

SARI is tokenizer-sensitive, and published numbers use mteval's 13a tokenizer
via sacrebleu, which is not a dependency of this project. We instead lowercase
and apply one regex: word characters with intra-word apostrophes kept together
("don't" is one token), and every other punctuation character emitted as its own
token. It is applied identically to source, prediction and references, so
comparisons *within* this project are sound; absolute values may differ slightly
from 13a-based papers and should not be quoted against them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

MAX_N = 4  # SARI averages orders 1..4, then the three operations

_TOKEN = re.compile(r"\w+(?:'\w+)*|[^\w\s]")

_Gram = tuple[str, ...]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> list[_Gram]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _f1(precision: float, recall: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _order_scores(
    src_grams: list[_Gram],
    pred_grams: list[_Gram],
    ref_grams: list[list[_Gram]],
    n_refs: int,
) -> tuple[float, float, float]:
    """(keep_f1, del_precision, add_f1) for a single n-gram order, each in [0, 1]."""
    ref_c: Counter[_Gram] = Counter(g for grams in ref_grams for g in grams)
    # Scaling by R puts source/prediction counts on the same scale as the pooled
    # reference counts, so the clipped intersections below are reference-weighted.
    src_c: Counter[_Gram] = Counter(
        {g: c * n_refs for g, c in Counter(src_grams).items()}
    )
    pred_c: Counter[_Gram] = Counter(
        {g: c * n_refs for g, c in Counter(pred_grams).items()}
    )

    kept = src_c & pred_c  # source n-grams the prediction retained
    kept_good = kept & ref_c  # ... that the references retained too
    kept_all = src_c & ref_c  # every source n-gram the references retained

    keep_precision = (
        sum(kept_good[g] / kept[g] for g in kept_good) / len(kept) if kept else 1.0
    )
    keep_recall = sum(kept_good.values()) / sum(kept_all.values()) if kept_all else 1.0
    keep_f1 = _f1(keep_precision, keep_recall)

    # Counter subtraction drops non-positive counts, so every gram in
    # ``deleted_good`` has a positive count in ``deleted``.
    deleted = src_c - pred_c
    deleted_good = deleted - ref_c
    del_precision = (
        sum(deleted_good[g] / deleted[g] for g in deleted_good) / len(deleted)
        if deleted
        else 1.0
    )

    added = set(pred_c) - set(src_c)
    added_good = added & set(ref_c)
    added_all = set(ref_c) - set(src_c)
    add_precision = len(added_good) / len(added) if added else 1.0
    add_recall = len(added_good) / len(added_all) if added_all else 1.0
    add_f1 = _f1(add_precision, add_recall)

    return keep_f1, del_precision, add_f1


@dataclass
class SariScore:
    sari: float
    add_f1: float
    keep_f1: float
    del_precision: float  # SARI scores deletion by precision only, by design

    def as_dict(self) -> dict:
        return asdict(self)


def sari(source: str, prediction: str, references: list[str]) -> SariScore:
    """SARI for one (source, prediction) pair against one or more references."""
    if not references:
        raise ValueError("SARI requires at least one reference")

    src_tok = tokenize(source)
    pred_tok = tokenize(prediction)
    ref_tok = [tokenize(r) for r in references]
    n_refs = len(references)

    keep = delete = add = 0.0
    for n in range(1, MAX_N + 1):
        k, d, a = _order_scores(
            _ngrams(src_tok, n),
            _ngrams(pred_tok, n),
            [_ngrams(r, n) for r in ref_tok],
            n_refs,
        )
        keep += k
        delete += d
        add += a

    keep /= MAX_N
    delete /= MAX_N
    add /= MAX_N
    return SariScore(
        sari=100.0 * (add + keep + delete) / 3.0,
        add_f1=100.0 * add,
        keep_f1=100.0 * keep,
        del_precision=100.0 * delete,
    )


def corpus_sari(
    sources: list[str], predictions: list[str], references: list[list[str]]
) -> SariScore:
    """Macro-average of sentence-level SARI, matching the standard scorers."""
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError(
            f"length mismatch: {len(sources)} sources, {len(predictions)} "
            f"predictions, {len(references)} reference sets"
        )
    if not sources:
        raise ValueError("corpus_sari requires at least one item")

    scores = [sari(s, p, r) for s, p, r in zip(sources, predictions, references)]
    n = len(scores)
    return SariScore(
        sari=sum(x.sari for x in scores) / n,
        add_f1=sum(x.add_f1 for x in scores) / n,
        keep_f1=sum(x.keep_f1 for x in scores) / n,
        del_precision=sum(x.del_precision for x in scores) / n,
    )


def _show(label: str, score: SariScore) -> None:
    print(
        f"{label:<34} sari={score.sari:6.2f}  add={score.add_f1:6.2f}  "
        f"keep={score.keep_f1:6.2f}  del={score.del_precision:6.2f}"
    )


def _main() -> int:
    src = (
        "The Hubble Space Telescope, which was deployed in 1990, has "
        "substantially augmented humanity's comprehension of cosmology."
    )
    refs = [
        "The Hubble Space Telescope was launched in 1990. It has greatly "
        "improved what we know about the universe.",
        "The Hubble Space Telescope went up in 1990. It has taught us much "
        "more about the universe.",
    ]
    # Neither reference verbatim: some wording from each, some of its own.
    good = (
        "The Hubble Space Telescope was launched in 1990. It has taught us a "
        "lot more about the universe."
    )

    identical = sari(src, src, refs)
    copy_ref = sari(src, refs[0], refs)
    empty = sari(src, "", refs)
    realistic = sari(src, good, refs)

    _show("prediction == source", identical)
    _show("prediction == reference[0]", copy_ref)
    _show("prediction == '' (empty)", empty)
    _show("realistic simplification", realistic)

    # The canonical example from HuggingFace `evaluate`'s SARI card; agreement to
    # 1e-9 is the real correctness check, since the inputs are pre-tokenized and
    # our tokenizer therefore cannot disagree with theirs on them.
    hf_src = ["About 95 species are currently accepted ."]
    hf_pred = ["About 95 you now get in ."]
    hf_refs = [
        [
            "About 95 species are currently known .",
            "About 95 species are now accepted .",
            "95 species are now accepted .",
        ]
    ]
    hf = corpus_sari(hf_src, hf_pred, hf_refs)
    _show("huggingface reference example", hf)
    assert abs(hf.sari - 26.953601953601954) < 1e-9, hf.sari

    # Copying a reference must beat copying the source.
    assert copy_ref.sari > identical.sari, (copy_ref.sari, identical.sari)
    assert realistic.sari > identical.sari, (realistic.sari, identical.sari)

    for label, score in [
        ("identical", identical),
        ("copy_ref", copy_ref),
        ("empty", empty),
        ("realistic", realistic),
        ("hf", hf),
    ]:
        for key, val in score.as_dict().items():
            assert 0.0 <= val <= 100.0, (label, key, val)

    # corpus_sari over one item is that item.
    single = corpus_sari([src], [good], [refs])
    assert single.as_dict() == realistic.as_dict(), (single, realistic)

    # Degenerate inputs must not raise.
    _show("empty source and prediction", sari("", "", [""]))
    _show("single token everywhere", sari("a", "a", ["a"]))

    print("\nall assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
