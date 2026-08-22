"""Decode-time constraints. Currently: a bias against starting a hard word.

Why a static bias over the vocabulary
-------------------------------------
The obvious implementation of "only use easy words" tracks the partially
generated word at each step and masks every vocabulary entry that could not
continue it. That is exact and unusably slow: 128k trie lookups per token per
sequence, in Python, inside the generation loop.

The bias here is computed once and added to the logits every step. It applies to
*word-initial* tokens only -- the ones whose text begins a new word -- and asks a
single question of each: could any easy word start this way? A token that can
only begin a hard word gets ``-penalty``; everything else gets zero. Nothing is
tracked between steps, so the cost is one vector add.

What that buys and what it misses
---------------------------------
It buys the thing that matters: the model rarely *starts* a hard word, so hard
words rarely appear. It misses hard words whose first token is also the start of
an easy word -- "photosynthesis" begins with "photo", and "photo" is easy -- so
the constraint is a strong preference, not a guarantee. Reporting it as a
guarantee would be wrong; ``pct_hard_words`` in the results is the measurement
that decides whether it worked.

Agreement with the metric
-------------------------
The word list is the one ``textstat.is_difficult_word`` reads, which is the same
call ``lib.metrics.readability`` uses to compute ``pct_hard_words``. Constraint
and metric therefore disagree only where the prefix approximation above lets
something through, not because they consult different dictionaries.

Proper nouns
------------
Capitalised word-initial tokens are exempt. Dale-Chall contains no names, so
constraining them would delete every entity in the source -- which is precisely
the deletion failure that costs prompting its faithfulness score. The cost of the
exemption is that ``pct_hard_words`` cannot reach zero, because the metric does
count proper nouns as difficult.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import LogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Llama/GPT byte-level BPE marks a leading space with U+0120; SentencePiece
# models use U+2581. Both are checked so the mask is not silently empty on a
# model whose tokenizer uses the other convention.
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
        return 2  # treat an uncountable string as polysyllabic, i.e. constrained


def word_start_penalty_mask(tokenizer) -> torch.Tensor:
    """Boolean mask, True where a token starts a word no easy word starts with.

    A token is exempt when it is not word-initial, is not alphabetic, is
    capitalised, or could still turn into a monosyllable -- ``is_difficult_word``
    never flags a word of fewer than two syllables, so neither does this.
    """
    easy = easy_words()
    prefixes = _prefixes(easy)
    vocab = tokenizer.get_vocab()
    mask = torch.zeros(max(vocab.values()) + 1, dtype=torch.bool)

    for token, idx in vocab.items():
        if not token.startswith(_SPACE_MARKERS):
            continue  # continuation token: it cannot start a word
        body = token[1:]
        if not body or not body.isalpha():
            continue
        if body[0].isupper():
            continue  # proper nouns; see the module docstring
        low = body.lower()
        if low in prefixes:
            continue  # some easy word starts this way
        if _syllables(low) < 2:
            continue  # the metric would not call the finished word difficult
        mask[idx] = True
    return mask


class EasyWordBias(LogitsProcessor):
    """Subtract ``penalty`` from tokens that can only start a hard word.

    ``penalty=inf`` is a hard ban. A finite penalty is the useful setting: the
    model can still reach for a hard word when nothing else fits, which is what
    keeps proper technical terms with no easy synonym from being mangled into
    something wrong.
    """

    def __init__(self, mask: torch.Tensor, penalty: float):
        self.mask = mask
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor):
        mask = self.mask.to(scores.device)
        if mask.shape[0] != scores.shape[-1]:
            # A tokenizer whose vocab is smaller than the model's embedding
            # matrix (Llama pads to a multiple of 256). Pad rather than crash;
            # the extra rows are unused ids and must not be biased.
            pad = scores.shape[-1] - mask.shape[0]
            if pad < 0:
                raise ValueError(
                    f"bias mask has {mask.shape[0]} entries but the model scores "
                    f"{scores.shape[-1]}; the mask is not for this tokenizer"
                )
            mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool, device=mask.device)])
        if self.penalty == float("inf"):
            return scores.masked_fill(mask, float("-inf"))
        return scores - mask.to(scores.dtype) * self.penalty


def build(tokenizer, penalty: float | None = None, repo_id: str | None = None) -> EasyWordBias:
    """The bias for *repo_id*'s logit scale. See config.easy_word_penalty."""
    penalty = config.easy_word_penalty(repo_id) if penalty is None else penalty
    return EasyWordBias(word_start_penalty_mask(tokenizer), penalty)
