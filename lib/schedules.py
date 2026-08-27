"""Depth schedules: how one axis's steering budget is spread across layers.

Why this is a library and not a flag in one script
--------------------------------------------------
Every steered arm in this project used to inject at exactly one layer, because
the sweep is a two-dimensional grid and its best cell is a single (layer,
alpha). That was never an argument that depth-one injection is right -- the
sweep's top band is contiguous, which is what you would expect if the
neighbouring layers carry near-parallel vectors.

Spreading turned out to matter, so it has to be reachable from every entry
point that steers, not just from the ladder that discovered it. The shapes live
here and ``lib.cli.AxisSettings`` parses them, so ``--structural 31,0.5,flat3``
works in evaluate, best_of_n, adaptive_steering and simplify alike.

Budget normalisation
--------------------
A schedule is a set of relative weights plus a rule for scaling them to a
budget:

    sum   sum(alpha_i) = A     the pushes add linearly
    l2    ||alpha||_2  = A     the pushes are near-orthogonal, adding in
                               quadrature

``sum`` is the default because it is the one the model agrees with. Matching in
quadrature spends about sqrt(k) times the real budget across k layers, and it
shows: on Gemma at L31, ``flat3_l2`` degenerates 37.5% of outputs and
``flat5_l2`` 78.1%, against 0.0% for every sum-matched schedule at the same
nominal budget. Adjacent-layer steering vectors are near-parallel -- the same
direction carried forward -- so quadrature is the wrong model of how they add.

Norm preservation still eats some of the budget: each injection is followed by
a rescale back to the original ||h||, so a later layer partly undoes an earlier
one and a sum-matched spread lands slightly under the single layer it is
matched to. ``scale`` exists to walk that difference out when the comparison
has to be iso-dose rather than iso-budget.
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
    # Fraction of the budget actually spent. 1.0 for every shape that is
    # testing depth; the dose controls use it to walk a single layer down to
    # the realised push a spread delivers, which is what separates "spreading
    # helps" from "spreading is quietly less alpha".
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
        else:  # l2
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
        # Flat: does spreading help at all.
        Shape("flat3", _band(3), (1, 1, 1)),
        Shape("flat5", _band(5), (1, 1, 1, 1, 1)),
        # Tent: should the swept peak stay dominant.
        Shape("tent5", _band(5), (1, 2, 3, 2, 1)),
        # Ramps: is the useful direction of the spread deeper or shallower.
        # The sweep's own depth ordering (structural before lexical) says depth
        # is not neutral, so the two directions are not interchangeable.
        Shape("ramp5_deep", _band(5), (1, 2, 3, 4, 5)),
        Shape("ramp5_shallow", _band(5), (5, 4, 3, 2, 1)),
        # Quadrature-matched twins, kept so the normalisation claim above stays
        # reproducible rather than being a remark in a docstring.
        Shape("flat3_l2", _band(3), (1, 1, 1), "l2"),
        Shape("flat5_l2", _band(5), (1, 1, 1, 1, 1), "l2"),
    )
}

# The ladder that discovered this wrote `flat3_sum` for what is now `flat3`;
# sum is the default, so the suffix is noise in a flag. Old names keep working
# so a command in a log or a results file still runs.
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
    """``get``, plus the ``NAME_xFRAC`` spelling for a dosed shape.

    ``single_x0.7`` is the single layer at 70% of its budget. Written as one
    name so a ladder, a flag and a results file can all refer to a dose control
    the same way.
    """
    if "_x" in name:
        head, _, tail = name.rpartition("_x")
        try:
            return dosed(head, float(tail))
        except ValueError:
            pass  # not a dose suffix; fall through to a plain lookup
    return get(name)


def label(pairs: list[tuple[int, float]]) -> str:
    """``L30:0.167 L31:0.167`` -- the form printed in logs and tables."""
    return " ".join(f"L{layer}:{alpha:.3f}" for layer, alpha in pairs)
