"""The corpora this project needs, with leakage between them made impossible.

Three roles that must never mix: ASSET test is the sentence-level eval set,
SQuAD paragraphs are the passage-level eval set, and the complex corpus is what
the steering vectors are *estimated from*. If an estimation text also appears in
an eval set, every reported SARI/FKGL number is measuring memorisation instead
of steering, so the disjointness is enforced in code rather than assumed from
how the upstream splits happen to be built.

Two cache layers under ``config.DATA_DIR`` (gitignored). Raw Hub downloads go to
``data/hf`` so nothing lands in the user's global HF cache; the derived records
go to ``data/derived`` as JSON. The second layer is what makes a re-run instant
and fully offline - deriving the passage pool means deduplicating 87k SQuAD rows
and sentence-splitting 19k paragraphs, which is seconds of work to repeat for no
reason, and it also means a re-run never touches the network at all.

On dataset ids: ASSET's canonical repo still resolves and now serves parquet
directly, so ``facebook/asset`` is used as-is. OneStopEnglish is the awkward one
- see ``load_onestop``.
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.chunking import split_sentences  # noqa: E402
from lib.hub import hf_token  # noqa: E402

ASSET_REPO = "facebook/asset"
ASSET_CONFIG = "simplification"  # the other config, "ratings", is human judgements

# OneStopEnglish reaches the Hub twice, and only one of the two is usable:
#   iastate/onestop_english  - 567 rows as a flat 3-class classification set.
#     189 per level and the levels *look* index-parallel, but they are not: at
#     least five ele/int pairs are off by one or two (a duplicated elementary
#     text at index 164/165 shifts everything after it). Aligning by position
#     silently pairs the vaping article with the BB King article.
#   malmaud/onestop_qa      - the same Guardian articles carrying an explicit
#     (title, paragraph_index, level) key, so the triples need no guessing.
# The second one is therefore the source of truth here. See load_onestop.
ONESTOP_ALIGNED_REPO = "malmaud/onestop_qa"
ONESTOP_UNALIGNED_REPO = "iastate/onestop_english"

# Wikipedia paragraphs. ASSET is single sentences drawn from Wikipedia, so
# SQuAD's contexts put the passage-level path in the same register as the
# sentence-level path instead of introducing a second domain shift.
PASSAGE_REPO = "rajpurkar/squad"

_HF_CACHE = config.DATA_DIR / "hf"
_DERIVED = config.DATA_DIR / "derived"


# What counts as "genuinely hard prose" for vector estimation. Long words are a
# crude but parser-free proxy for the Latinate/technical register we want the
# lexical vector estimated on; lib.metrics.readability is the real measurement
# and is deliberately not imported here, so selecting the estimation set can
# never be tuned against the metric that scores it.
MIN_SOURCE_WORDS = 20
MIN_LONG_WORD_RATE = 0.10
LONG_WORD_CHARS = 8

# Passages need enough sentences that the rolling-context path in lib.chunking
# actually does something, and a length a 3B model can hold in one prompt.
MIN_PASSAGE_SENTENCES = 4
PASSAGE_WORDS = (80, 220)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Whitespace/case-insensitive key for cross-corpus identity checks."""
    return _WS.sub(" ", text.strip().lower())


def _is_hard(text: str) -> bool:
    words = _WORD.findall(text)
    if len(words) < MIN_SOURCE_WORDS:
        return False
    long_words = sum(1 for w in words if len(w) >= LONG_WORD_CHARS)
    return long_words / len(words) >= MIN_LONG_WORD_RATE


def _hf(repo: str, name: str | None = None, split: str = "train"):
    return load_dataset(
        repo, name, split=split, cache_dir=str(_HF_CACHE), token=hf_token()
    )


def _cached(key: str, build):
    """Return derived records for *key*, building and persisting them once."""
    path = _DERIVED / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))
    return obj


def _sample(pool: list[str], n: int, seed: int) -> list[str]:
    """Deterministic subset. Same (pool, n, seed) always gives the same texts in
    the same order, which is what makes a layer sweep comparable across runs."""
    if n > len(pool):
        raise ValueError(f"asked for {n} texts but the pool holds {len(pool)}")
    rng = random.Random(seed)
    return rng.sample(pool, n)


@dataclass
class SimplificationItem:
    source: str
    references: list[str]  # may be empty when the corpus has no refs


_ASSET_SPLITS = {"test": "test", "valid": "validation", "validation": "validation"}


def load_asset(split: str = "test") -> list[SimplificationItem]:
    """ASSET: 359 test / 2000 valid Wikipedia sentences, 10 human references each."""
    if split not in _ASSET_SPLITS:
        raise ValueError(
            f"unknown ASSET split {split!r}; expected {sorted(_ASSET_SPLITS)}"
        )
    hf_split = _ASSET_SPLITS[split]

    def build() -> list[dict]:
        ds = _hf(ASSET_REPO, ASSET_CONFIG, hf_split)
        return [
            {"source": row["original"], "references": list(row["simplifications"])}
            for row in ds
        ]

    return [SimplificationItem(**rec) for rec in _cached(f"asset_{hf_split}", build)]


def _asset_sources(split: str) -> list[str]:
    return [item.source for item in load_asset(split)]


@dataclass
class GradedItem:
    advanced: str
    intermediate: str
    elementary: str


def load_onestop() -> list[GradedItem]:
    """OneStopEnglish paragraphs at three reading levels, genuinely aligned."""

    def build() -> list[dict]:
        try:
            ds = _hf(ONESTOP_ALIGNED_REPO, split="train")
        except (
            Exception
        ) as exc:  # noqa: BLE001 - the id, not the traceback, is the news
            raise RuntimeError(
                "no aligned OneStopEnglish source loaded. Tried "
                f"{ONESTOP_ALIGNED_REPO!r} (explicit title/paragraph_index/level key) "
                f"and rejected {ONESTOP_UNALIGNED_REPO!r} plus its mirrors "
                "('onestop_english', 'SetFit/onestop_english'), which ship a flat "
                "3-class layout whose levels are not reliably index-parallel. "
                "Closest workable substitutes for graded supervision: "
                "'GEM/wiki_auto_asset_turk' (2-level, sentence-aligned) or Newsela "
                "(5-level, aligned, but gated behind a signed data agreement). "
                f"Underlying error: {exc}"
            ) from exc

        levels = ds.features["level"].int2str  # names are ['Adv', 'Int', 'Ele']
        groups: dict[tuple[str, int], dict[str, str]] = {}
        for row in ds:
            key = (row["title"], int(row["paragraph_index"]))
            level = levels(row["level"])
            slot = groups.setdefault(key, {})
            previous = slot.get(level)
            # Three questions share each paragraph, so repeats are expected -
            # disagreeing text under the same key would mean the key is not one.
            if previous is not None and previous != row["paragraph"]:
                raise RuntimeError(
                    f"{ONESTOP_ALIGNED_REPO}: two texts for {key} at level {level}"
                )
            slot[level] = row["paragraph"]

        out: list[dict] = []
        for key in sorted(groups):
            slot = groups[key]
            missing = {"Adv", "Int", "Ele"} - set(slot)
            if missing:
                raise RuntimeError(
                    f"{ONESTOP_ALIGNED_REPO}: {key} missing {sorted(missing)}"
                )
            out.append(
                {
                    "advanced": slot["Adv"].strip(),
                    "intermediate": slot["Int"].strip(),
                    "elementary": slot["Ele"].strip(),
                }
            )
        return out

    return [GradedItem(**rec) for rec in _cached("onestop_triples", build)]


def complex_pool() -> list[str]:
    """Every hard, non-leaking sentence available for vector estimation."""

    def build() -> list[str]:
        forbidden = {_norm(s) for s in _asset_sources("test")}

        pool: list[str] = []
        seen: set[str] = set()
        for source in _asset_sources("validation"):
            key = _norm(source)
            if key in forbidden or key in seen or not _is_hard(source):
                continue
            seen.add(key)
            pool.append(source)

        # Only reached for n far above config.VECTOR_N_TEXTS; kept so raising the
        # estimation budget does not silently fall back to easy text.
        if len(pool) < config.VECTOR_N_TEXTS:
            for paragraph in _squad_contexts():
                for sentence in split_sentences(paragraph):
                    key = _norm(sentence)
                    if key in forbidden or key in seen or not _is_hard(sentence):
                        continue
                    seen.add(key)
                    pool.append(sentence)
        return pool

    return _cached("complex_pool", build)


def load_complex_corpus(n: int = 256, seed: int = 0) -> list[str]:
    """Complex sentences used to ESTIMATE steering vectors."""
    pool = complex_pool()

    # Re-checked against the live eval set on every call rather than trusted from
    # build time, so a stale data/derived cannot reintroduce leakage unnoticed.
    forbidden = {_norm(s) for s in _asset_sources("test")}
    overlap = [t for t in pool if _norm(t) in forbidden]
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} estimation texts also appear in ASSET test, "
            f"e.g. {overlap[0][:80]!r}. Delete {_DERIVED} and re-run."
        )
    return _sample(pool, n, seed)


def _squad_contexts() -> list[str]:
    """Deduplicated SQuAD contexts, unfiltered, in first-seen order."""
    ds = _hf(PASSAGE_REPO, split="train")
    return list(dict.fromkeys(ds["context"]))


def passage_pool() -> list[str]:
    """Every SQuAD paragraph that clears the length, sentence and leakage bars."""

    def build() -> list[str]:
        asset_sentences = {
            _norm(s) for split in ("test", "validation") for s in _asset_sources(split)
        }
        lo, hi = PASSAGE_WORDS

        out: list[str] = []
        for paragraph in _squad_contexts():
            n_words = len(_WORD.findall(paragraph))
            if not lo <= n_words <= hi:
                continue
            sentences = split_sentences(paragraph)
            if len(sentences) < MIN_PASSAGE_SENTENCES:
                continue
            if any(_norm(s) in asset_sentences for s in sentences):
                continue
            if not _is_hard(paragraph):
                continue
            out.append(_WS.sub(" ", paragraph.strip()))
        return out

    return _cached("passage_pool", build)


def load_eval_passages(n: int = 32, seed: int = 0) -> list[str]:
    """Multi-sentence passages for the passage-level path."""
    return _sample(passage_pool(), n, seed)


if __name__ == "__main__":
    import time

    def show(label: str, text: str) -> None:
        flat = _WS.sub(" ", text.strip())
        print(f"  {label}: {flat[:150]}{'...' if len(flat) > 150 else ''}")

    started = time.perf_counter()

    print(f"[asset] {ASSET_REPO} ({ASSET_CONFIG})")
    test = load_asset("test")
    valid = load_asset("validation")
    print(f"  test={len(test)} items  validation={len(valid)} items")
    print(f"  references per test item: {sorted({len(i.references) for i in test})}")
    show("source", test[0].source)
    show("ref[0]", test[0].references[0])
    assert len(test) == 359, f"ASSET test should hold 359 items, got {len(test)}"
    assert (
        len(valid) == 2000
    ), f"ASSET validation should hold 2000 items, got {len(valid)}"
    assert all(
        len(i.references) == 10 for i in test
    ), "every ASSET test item needs 10 references"

    print(f"[onestop] {ONESTOP_ALIGNED_REPO}")
    graded = load_onestop()
    print(f"  {len(graded)} aligned triples")
    show("advanced", graded[0].advanced)
    show("intermediate", graded[0].intermediate)
    show("elementary", graded[0].elementary)
    assert graded, "no graded triples loaded"
    for item in graded:
        assert item.advanced.strip(), "empty advanced level"
        assert item.intermediate.strip(), "empty intermediate level"
        assert item.elementary.strip(), "empty elementary level"
    # 4 of the 162 paragraphs are byte-identical at advanced and intermediate:
    # the editors left them alone at that level. Real corpus behaviour, so it is
    # reported rather than asserted away. A fully collapsed triple would be a bug.
    unchanged = sum(1 for i in graded if i.advanced == i.intermediate)
    print(f"  triples where advanced == intermediate: {unchanged}")
    assert all(
        len({i.advanced, i.intermediate, i.elementary}) > 1 for i in graded
    ), "a triple is identical at all three levels"

    print(f"[complex] {ASSET_REPO} validation, hard-filtered")
    complex_texts = load_complex_corpus(config.VECTOR_N_TEXTS)
    print(f"  {len(complex_texts)} texts drawn from a pool of {len(complex_pool())}")
    show("text", complex_texts[0])
    test_sources = {_norm(i.source) for i in test}
    leaked = [t for t in complex_texts if _norm(t) in test_sources]
    assert (
        not leaked
    ), f"{len(leaked)} estimation texts leak into ASSET test: {leaked[:1]}"
    assert len(set(map(_norm, complex_texts))) == len(
        complex_texts
    ), "duplicate estimation texts"

    print(f"[passages] {PASSAGE_REPO}")
    passages = load_eval_passages()
    print(f"  {len(passages)} passages drawn from a pool of {len(passage_pool())}")
    print(
        f"  sentences per passage: min={min(len(split_sentences(p)) for p in passages)}"
    )
    show("passage", passages[0])
    assert len(passages) == 32, f"expected 32 passages, got {len(passages)}"
    assert all(len(split_sentences(p)) >= MIN_PASSAGE_SENTENCES for p in passages)
    complex_norm = {_norm(t) for t in complex_texts}
    crossover = [
        p for p in passages if any(_norm(s) in complex_norm for s in split_sentences(p))
    ]
    assert not crossover, f"{len(crossover)} eval passages contain an estimation text"

    print(f"[ok] all loaders passed in {time.perf_counter() - started:.2f}s")
    print(f"     cache: {config.DATA_DIR}")
