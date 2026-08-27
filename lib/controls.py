"""ACCESS-style control tokens: ask for a target, instead of turning a dial.

Why this exists
---------------
Every dial in this project is opaque. Steering's alpha and the LoRA's scale both
answer "how hard should I push", and neither answers "produce text at grade 8".
The alpha ladder is non-monotone and the LoRA scale peaks below its trained
strength, so even reading a dial off a table is unreliable.

A control token turns the request around. During training the adapter sees the
ratio the reference *actually achieved* on each axis, prepended to the source:

    [len=0.80 wps=0.65 hard=0.90] <source>

so it learns to associate the request with the outcome. At inference the ratio is
a request rather than a measurement, and the dial has units.

The three controls are the project's own axes plus length, so a result here is
comparable to everything else measured:

    len   words(target) / words(source)          how much is kept
    wps   words_per_sentence ratio               the structural axis
    hard  pct_hard_words ratio                   the lexical axis

Deliberately not ACCESS's full set. Dependency-tree depth needs a parser that is
not a dependency of this repo, and word-rank needs a frequency table; the
Dale-Chall list is already what ``pct_hard_words`` consults, so ``hard`` is
measured with the same instrument the evaluation scores with.

Rendered as plain text, not as added vocabulary. New special tokens would mean
resizing the embedding matrix, which a LoRA does not train -- the ratios would
land on untrained rows.
"""

from __future__ import annotations

from lib.metrics import readability

CONTROLS = ("len", "wps", "hard")

# Ratios are quantised so the model sees a few dozen distinct requests rather
# than a continuum it can only memorise. 0.05 is ACCESS's step.
STEP = 0.05
LO, HI = 0.20, 1.50


def quantise(value: float, step: float = STEP, lo: float = LO, hi: float = HI) -> float:
    return round(min(max(round(value / step) * step, lo), hi), 2)


def _ratio(target: float, source: float) -> float:
    """Target over source, with the degenerate cases pinned rather than raised.

    A source with no hard words at all is not an error and not a request for
    infinite reduction; it is a source the lexical control has nothing to say
    about, so it gets 1.0 -- change nothing.
    """
    if source is None or target is None or source <= 0:
        return 1.0
    return target / source


def ratios(source: str, target: str) -> dict[str, float]:
    """What the target actually did to the source, on each axis."""
    s, t = readability.measure(source), readability.measure(target)
    return {
        "len": quantise(_ratio(t.n_words, s.n_words)),
        "wps": quantise(_ratio(t.words_per_sentence, s.words_per_sentence)),
        "hard": quantise(_ratio(t.pct_hard_words, s.pct_hard_words)),
    }


def render(values: dict[str, float]) -> str:
    """The prefix as the model sees it. Fixed key order, fixed precision."""
    inner = " ".join(f"{k}={values[k]:.2f}" for k in CONTROLS)
    return f"[{inner}] "


def prefix(source: str, target: str) -> str:
    """Training-time prefix: the ratios this reference achieved."""
    return render(ratios(source, target))


def request(**values: float) -> str:
    """Inference-time prefix: the ratios being asked for. Unset axes mean 1.0,
    i.e. leave that axis alone, which is what an absent request should mean."""
    return render({k: quantise(values.get(k, 1.0)) for k in CONTROLS})
