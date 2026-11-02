"""Depth schedules: how one axis's steering budget is spread across layers.

``Shape`` is a set of relative weights over layer offsets plus a normalisation
rule (``sum`` or ``l2``) for scaling them to a target budget. ``sum`` is the
default: adjacent-layer steering vectors are near-parallel, so ``l2`` (matching
in quadrature) over-doses.
"""

from __future__ import annotations

from dataclasses import dataclass

NORMS = ("sum", "l2")


@dataclass(frozen=True)
class Shape:
    """Relative weights over layer offsets, plus how they are scaled."""

    name: str
    offsets: tuple[int, ...]
    weights: tuple[float, ...]
    norm: str = "sum"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.norm not in NORMS:
            raise ValueError(f"unknown normalisation {self.norm!r}; expected {NORMS}")
        if len(self.offsets) != len(self.weights):
            raise ValueError(
                f"shape {self.name!r} has {len(self.offsets)} offsets and "
                f"{len(self.weights)} weights"
            )
        if not self.weights or min(self.weights) < 0:
            raise ValueError(f"shape {self.name!r} needs non-negative weights")

    @property
    def width(self) -> int:
        return len(self.offsets)

    def alphas(self, center: int, budget: float) -> list[tuple[int, float]]:
        """Resolve to concrete ``(layer, alpha)`` pairs around *center*."""
        w = [float(x) for x in self.weights]
        if self.norm == "sum":
            unit = budget / sum(w)
        else:
            unit = budget / (sum(x * x for x in w) ** 0.5)
        return [(center + o, wi * unit * self.scale) for o, wi in zip(self.offsets, w)]


def _band(width: int) -> tuple[int, ...]:
    """Offsets centred on zero: width 3 -> (-1, 0, 1)."""
    half = width // 2
    return tuple(range(-half, half + 1))


SHAPES: dict[str, Shape] = {
    s.name: s
    for s in (
        Shape("single", (0,), (1,)),
        Shape("flat3", _band(3), (1, 1, 1)),
        Shape("flat5", _band(5), (1, 1, 1, 1, 1)),
        Shape("tent5", _band(5), (1, 2, 3, 2, 1)),
        Shape("ramp5_deep", _band(5), (1, 2, 3, 4, 5)),
        Shape("ramp5_shallow", _band(5), (5, 4, 3, 2, 1)),
        Shape("flat3_l2", _band(3), (1, 1, 1), "l2"),
        Shape("flat5_l2", _band(5), (1, 1, 1, 1, 1), "l2"),
    )
}

ALIASES = {
    "flat3_sum": "flat3",
    "flat5_sum": "flat5",
    "tent5_sum": "tent5",
}


def get(name: str) -> Shape:
    """Look up a shape by name, honouring the legacy ``_sum`` spellings."""
    key = ALIASES.get(name, name)
    if key not in SHAPES:
        raise KeyError(
            f"unknown shape {name!r}; choose from {', '.join(sorted(SHAPES))}"
        )
    return SHAPES[key]


def dosed(name: str, scale: float) -> Shape:
    """A shape at a fraction of its budget, for iso-dose comparisons."""
    base = get(name)
    return Shape(
        f"{base.name}_x{scale:g}", base.offsets, base.weights, base.norm, scale
    )


def resolve(name: str) -> Shape:
    """``get``, plus the ``NAME_xFRAC`` spelling for a dosed shape."""
    if "_x" in name:
        head, _, tail = name.rpartition("_x")
        try:
            return dosed(head, float(tail))
        except ValueError:
            pass
    return get(name)


def label(pairs: list[tuple[int, float]]) -> str:
    """``L30:0.167 L31:0.167`` -- the form printed in logs and tables."""
    return " ".join(f"L{layer}:{alpha:.3f}" for layer, alpha in pairs)
