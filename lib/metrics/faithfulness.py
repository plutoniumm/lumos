"""Meaning preservation via bidirectional NLI.

One direction is gameable. source-entails-output rewards dropping facts; the
reverse rewards inventing them. Scored per sentence so one dropped fact shows up
in the minimum instead of averaging away.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from lib import hub  # noqa: E402
from lib.chunking import split_sentences  # noqa: E402

MAX_PAIRS_PER_BATCH = 16


@lru_cache(maxsize=1)
def _nli():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = hub.resolve(config.NLI_MODEL)
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to(config.DEVICE).eval()
    model.requires_grad_(False)

    entail_idx = next(
        (i for i, lbl in model.config.id2label.items() if "ENTAIL" in lbl.upper()),
        None,
    )
    if entail_idx is None:
        raise RuntimeError(f"no entailment label in {model.config.id2label}")
    return tok, model, int(entail_idx)


@torch.inference_mode()
def entailment_probs(pairs: list[tuple[str, str]]) -> list[float]:
    """P(entailment) for each (premise, hypothesis) pair."""
    if not pairs:
        return []
    tok, model, idx = _nli()
    out: list[float] = []
    for start in range(0, len(pairs), MAX_PAIRS_PER_BATCH):
        chunk = pairs[start : start + MAX_PAIRS_PER_BATCH]
        batch = tok(
            [p for p, _ in chunk],
            [h for _, h in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(config.DEVICE)
        probs = model(**batch).logits.softmax(-1)[:, idx]
        out.extend(probs.float().cpu().tolist())
    return out


@dataclass
class Faithfulness:
    # source |= each output sentence: low means the output invented something
    no_hallucination_mean: float
    no_hallucination_min: float
    # output |= each source sentence: low means the output dropped something
    no_omission_mean: float
    no_omission_min: float

    @property
    def score(self) -> float:
        """Single gate value: the weaker of the two directions."""
        return min(self.no_hallucination_mean, self.no_omission_mean)

    @property
    def worst(self) -> float:
        return min(self.no_hallucination_min, self.no_omission_min)

    def passed(self, threshold: float | None = None) -> bool:
        thr = config.NLI_ENTAILMENT_MIN if threshold is None else threshold
        return self.score >= thr

    def as_dict(self) -> dict:
        return asdict(self) | {"score": self.score, "worst": self.worst}


def faithfulness(source: str, output: str) -> Faithfulness:
    src_sents = split_sentences(source) or [source]
    out_sents = split_sentences(output) or [output]

    if not (source.strip() and output.strip()):
        return Faithfulness(0.0, 0.0, 0.0, 0.0)

    fwd = entailment_probs([(source, s) for s in out_sents])
    bwd = entailment_probs([(output, s) for s in src_sents])

    return Faithfulness(
        no_hallucination_mean=sum(fwd) / len(fwd) if fwd else 0.0,
        no_hallucination_min=min(fwd) if fwd else 0.0,
        no_omission_mean=sum(bwd) / len(bwd) if bwd else 0.0,
        no_omission_min=min(bwd) if bwd else 0.0,
    )
