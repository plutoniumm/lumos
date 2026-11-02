"""SARI (Xu et al. 2016): scores edits against multiple human references,
rather than the output in isolation, so it can't be gamed by truncation the
way FKGL can.

Uses a lowercase regex tokenizer, not mteval's 13a, so absolute values are
comparable within this project but not to 13a-based papers.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

MAX_N = 4

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
    # Source/prediction counts are scaled by n_refs so their intersection with
    # the pooled reference counts is reference-weighted, not just set membership.
    src_c: Counter[_Gram] = Counter(
        {g: c * n_refs for g, c in Counter(src_grams).items()}
    )
    pred_c: Counter[_Gram] = Counter(
        {g: c * n_refs for g, c in Counter(pred_grams).items()}
    )

    kept = src_c & pred_c
    kept_good = kept & ref_c
    kept_all = src_c & ref_c

    keep_precision = (
        sum(kept_good[g] / kept[g] for g in kept_good) / len(kept) if kept else 1.0
    )
    keep_recall = (
        sum(kept_good.values()) / sum(kept_all.values()) if kept_all else 1.0
    )
    keep_f1 = _f1(keep_precision, keep_recall)

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
    del_precision: float  # deletion is scored by precision only, by design

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

    single = corpus_sari([src], [good], [refs])
    assert single.as_dict() == realistic.as_dict(), (single, realistic)

    _show("empty source and prediction", sari("", "", [""]))
    _show("single token everywhere", sari("a", "a", ["a"]))

    print("\nall assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
