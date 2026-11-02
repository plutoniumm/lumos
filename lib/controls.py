"""ACCESS-style control tokens: ask for a target ratio, instead of turning a dial.

Three axes -- len, wps, hard -- rendered as a plain-text prefix like
``[len=0.80 wps=0.65 hard=0.90]`` that the LoRA is trained to condition on.
"""

from __future__ import annotations

from lib.metrics import readability

CONTROLS = ("len", "wps", "hard")

STEP = 0.05
LO, HI = 0.20, 1.50


def quantise(value: float, step: float = STEP, lo: float = LO, hi: float = HI) -> float:
    return round(min(max(round(value / step) * step, lo), hi), 2)


def _ratio(target: float, source: float) -> float:
    """Target over source; degenerate cases (no source signal) pin to 1.0."""
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
    """The prefix as the model sees it."""
    inner = " ".join(f"{k}={values[k]:.2f}" for k in CONTROLS)
    return f"[{inner}] "


def prefix(source: str, target: str) -> str:
    """Training-time prefix: the ratios this reference achieved."""
    return render(ratios(source, target))


def request(**values: float) -> str:
    """Inference-time prefix: the ratios being asked for; unset axes mean 1.0."""
    return render({k: quantise(values.get(k, 1.0)) for k in CONTROLS})
