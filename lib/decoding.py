"""Decode-time constraints: a logits bias against starting a hard word.

Computed once as a static per-token bias, added to logits every step, rather
than tracking partial words during generation. Applies only to word-initial
tokens; the word list is the same Dale-Chall list
``lib.metrics.readability`` uses for ``pct_hard_words``. Capitalised
word-initial tokens are exempt, since Dale-Chall has no proper nouns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import LogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

_SPACE_MARKERS = ("Ġ", "▁", " ")


def easy_words() -> set[str]:
    """The Dale-Chall easy list, straight from the package the metric reads."""
    from textstat.backend.utils._get_lang_easy_words import get_lang_easy_words

    return get_lang_easy_words("en")


def _prefixes(words: set[str]) -> set[str]:
    """Every prefix of every easy word, so a lookup is one set membership test."""
    out: set[str] = set()
    for w in words:
        for i in range(1, len(w) + 1):
            out.add(w[:i])
    return out


def _syllables(word: str) -> int:
    from textstat.backend.counts._count_syllables import count_syllables

    try:
        return count_syllables(word, "en")
    except Exception:
        return 2


def word_start_penalty_mask(tokenizer) -> torch.Tensor:
    """Boolean mask, True where a token starts a word no easy word starts with."""
    easy = easy_words()
    prefixes = _prefixes(easy)
    vocab = tokenizer.get_vocab()
    mask = torch.zeros(max(vocab.values()) + 1, dtype=torch.bool)

    for token, idx in vocab.items():
        if not token.startswith(_SPACE_MARKERS):
            continue
        body = token[1:]
        if not body or not body.isalpha():
            continue
        if body[0].isupper():
            continue
        low = body.lower()
        if low in prefixes:
            continue
        if _syllables(low) < 2:
            continue
        mask[idx] = True
    return mask


class EasyWordBias(LogitsProcessor):
    """Subtract ``penalty`` from tokens that can only start a hard word."""

    def __init__(self, mask: torch.Tensor, penalty: float):
        self.mask = mask
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor):
        mask = self.mask.to(scores.device)
        if mask.shape[0] != scores.shape[-1]:
            pad = scores.shape[-1] - mask.shape[0]
            if pad < 0:
                raise ValueError(
                    f"bias mask has {mask.shape[0]} entries but the model scores "
                    f"{scores.shape[-1]}; the mask is not for this tokenizer"
                )
            mask = torch.cat(
                [mask, torch.zeros(pad, dtype=torch.bool, device=mask.device)]
            )
        if self.penalty == float("inf"):
            return scores.masked_fill(mask, float("-inf"))
        return scores - mask.to(scores.dtype) * self.penalty


def build(
    tokenizer, penalty: float | None = None, repo_id: str | None = None
) -> EasyWordBias:
    """The bias for *repo_id*'s logit scale."""
    penalty = config.easy_word_penalty(repo_id) if penalty is None else penalty
    return EasyWordBias(word_start_penalty_mask(tokenizer), penalty)
