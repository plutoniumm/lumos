"""The argument plumbing every entry point in ``src/`` repeats.

Four scripts steer, so four scripts had to answer the same question: which
``(layer, alpha)`` does each axis run at, and where did that pair come from --
a sweep report, or a flag typed by hand. Each grew its own parser, and the four
were not equivalent. ``src/evaluate.py`` validated the sweep file and said what
to do about a bad one; the other three indexed straight into
``payload["axes"][axis]["best"]["layer"]`` and raised ``KeyError`` on a file
that was merely the wrong kind of JSON.

That asymmetry matters more than it looks. The sweep report is the artifact
that connects "where does the axis live" to every number downstream of it, and
it is passed by path on the command line. Getting a ``KeyError`` two hours into
a run because ``--layers-from`` pointed at an eval summary is a real failure
mode, and the fix -- say which file, what was expected, and what to run to
produce it -- only has to be written once.

``AxisSettings`` is that answer, and it also owns the step the callers used to
each do by hand: turning a pair into a ``SteeringSpec``.

Depth schedules
---------------
An axis is no longer one layer. Spreading a fixed budget across a band beats
concentrating it at the swept peak, so every entry point that steers has to be
able to say so, and the flag grammar carries it:

    --structural 31,0.5                      one layer, as before
    --structural 31,0.5,flat3                a named shape from lib.schedules,
                                             centred on L31, budget 0.5
    --structural 30:0.167+31:0.167+32:0.167  explicit, when no name fits

The first form is what every existing command and every recorded run uses, and
it still means exactly what it did. The others expand to more injection sites
on the same axis, which ``SteeringSpec`` already supported -- ``settings`` is a
tuple of ``AxisSetting`` and the ``Injector`` takes a list of sites -- so
nothing below this layer had to change to make depth tunable.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import config
from lib import schedules
from lib.steering import AxisSetting, SteeringSpec

_EXPECTED_SWEEP_SHAPE = (
    '{"axes": {"structural": {"top_layers": [...], '
    '"best": {"layer": 14, "alpha": 1.0, ...}}, "lexical": {...}}}'
)


def die(message: str) -> None:
    """Stop with a message on stderr. Exit 2, as for a usage error."""
    print(f"\nFATAL: {message}\n", file=sys.stderr)
    raise SystemExit(2)


_GRAMMAR = (
    "LAYER,ALPHA            (e.g. {flag} 14,1.5)\n"
    "  LAYER,ALPHA,SHAPE      (e.g. {flag} 14,1.5,flat3)\n"
    "  L:A+L:A+...            (e.g. {flag} 13:0.5+14:0.5+15:0.5)"
)


def _grammar(flag: str) -> str:
    return _GRAMMAR.format(flag=flag)


def parse_pair(text: str, flag: str) -> tuple[int, float]:
    """``"14,1.5"`` -> ``(14, 1.5)``. The single-layer form, kept separate
    because the sweep report and the calibration search both deal in one pair
    and should not have to unwrap a one-element schedule."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        die(f"{flag} wants LAYER,ALPHA (e.g. {flag} 14,1.5), got {text!r}")
    try:
        return int(parts[0]), float(parts[1])
    except ValueError:
        die(f"{flag} wants LAYER,ALPHA with an int layer and float alpha, got {text!r}")


def _parse_explicit(text: str, flag: str) -> list[tuple[int, float]]:
    """``"30:0.1+31:0.2"`` -> ``[(30, 0.1), (31, 0.2)]``."""
    out: list[tuple[int, float]] = []
    for term in text.split("+"):
        term = term.strip()
        if not term:
            continue
        bits = term.split(":")
        if len(bits) != 2:
            die(f"{flag}: {term!r} is not LAYER:ALPHA\nExpected {_grammar(flag)}")
        try:
            out.append((int(bits[0]), float(bits[1])))
        except ValueError:
            die(f"{flag}: {term!r} needs an int layer and a float alpha")
    if not out:
        die(f"{flag} named no layers\nExpected {_grammar(flag)}")
    seen = [layer for layer, _ in out]
    if len(set(seen)) != len(seen):
        # Two sites on one layer would both fire, and the second would steer a
        # stream the first had already rescaled -- the alphas would not add to
        # what was asked for.
        die(f"{flag}: layer(s) repeated in {text!r}; one alpha per layer per axis")
    return out


def parse_axis(text: str, flag: str) -> list[tuple[int, float]]:
    """One axis flag -> the ``(layer, alpha)`` schedule it names.

    Three forms, in the order they are tried: explicit (contains ``:``), named
    shape (three comma-separated fields), single layer (two).
    """
    text = text.strip()
    if ":" in text:
        return _parse_explicit(text, flag)

    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 3:
        layer, alpha = parse_pair(",".join(parts[:2]), flag)
        try:
            shape = schedules.get(parts[2])
        except KeyError as exc:
            die(f"{flag}: {exc}")
        return shape.alphas(layer, alpha)
    if len(parts) == 2:
        return [parse_pair(text, flag)]
    die(f"{flag} got {text!r}\nExpected {_grammar(flag)}")


@dataclass
class AxisSettings:
    """Which ``(layer, alpha)`` each axis is steered at.

    Built from a sweep report, from explicit flags, or from both -- flags win,
    so a single axis can be overridden without abandoning the sweep for the
    other one.

    Each axis holds a *schedule*: a list of ``(layer, alpha)``. A sweep report
    and the plain ``LAYER,ALPHA`` flag both produce a one-element list, which is
    the single-layer arm every published number was measured under.
    """

    per_axis: dict[str, list[tuple[int, float]]]

    @classmethod
    def from_args(cls, args) -> "AxisSettings":
        """The standard ``--layers-from`` / ``--structural`` / ``--lexical`` trio.

        Every steering entry point in ``src/`` accepts exactly these three, so
        they are read off the namespace rather than passed one at a time.
        """
        settings: dict[str, list[tuple[int, float]]] = {}
        path = getattr(args, "layers_from", None)
        if path:
            if not Path(path).exists():
                die(f"--layers-from {path} does not exist")
            settings.update(cls.from_sweep(Path(path)).per_axis)
        for axis in config.AXES:
            raw = getattr(args, axis, None)
            if raw:
                settings[axis] = parse_axis(raw, f"--{axis}")
        return cls(settings)

    @classmethod
    def from_sweep(cls, path: Path) -> "AxisSettings":
        """Best cell per axis, as ``src/sweep_layers.py --report`` writes it."""
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            die(f"--layers-from {path} is not valid JSON: {exc}")
        axes = payload.get("axes") if isinstance(payload, dict) else None
        if not isinstance(axes, dict):
            top = list(payload) if isinstance(payload, dict) else type(payload).__name__
            die(
                f"--layers-from {path} is not a sweep report: expected a top-level "
                f'"axes" object, found {top}.\n'
                f"Write one with:  {sys.executable} src/sweep_layers.py --report\n"
                f"Expected shape: {_EXPECTED_SWEEP_SHAPE}\n"
                f"Or pass --structural L,A --lexical L,A instead."
            )

        out: dict[str, list[tuple[int, float]]] = {}
        for axis in config.AXES:
            if axis not in axes:
                continue
            entry = axes[axis]
            if not isinstance(entry, dict):
                die(f"--layers-from {path}: axes.{axis} must be an object, got {entry!r}")
            # ``best`` is null for an axis where no cell passed the sweep's
            # degeneration/NLI gates. That is a sweep outcome, not a malformed
            # file, so it is a warning and the axis is simply left unset.
            best = entry.get("best")
            if best is None:
                print(f"  [warn] {path}: axes.{axis} has no best cell; no setting taken")
                continue
            try:
                out[axis] = [(int(best["layer"]), float(best["alpha"]))]
            except (KeyError, TypeError, ValueError):
                die(
                    f"--layers-from {path}: axes.{axis}.best must hold an int 'layer' "
                    f"and a float 'alpha', got {best!r}"
                )
        if not out:
            die(
                f"--layers-from {path} named no (layer, alpha) for any of "
                f"{list(config.AXES)}.\n"
                f"Expected shape: {_EXPECTED_SWEEP_SHAPE}\n"
                f"Or pass --structural L,A --lexical L,A instead."
            )
        return cls(out)

    def require(self, axes, *, needed_by: str = "this run") -> None:
        """Stop unless every axis in *axes* has a setting."""
        missing = [a for a in axes if a not in self.per_axis]
        if missing:
            die(
                f"{needed_by} needs a (layer, alpha) for {missing} and none was "
                f"given.\nPass --layers-from <sweep json>, or "
                + " ".join(f"--{a} LAYER,ALPHA" for a in missing)
                + "."
            )

    def _settings(self, axis: str) -> tuple[AxisSetting, ...]:
        return tuple(
            AxisSetting(axis, layer, alpha) for layer, alpha in self.per_axis[axis]
        )

    def spec(self, axis: str | None, **kw) -> SteeringSpec:
        """A one-axis spec, or the unsteered spec when *axis* is ``None``.

        One injection site per layer in the axis's schedule, which for a
        single-layer axis is exactly what ``SteeringSpec.single`` built before.
        """
        if axis is None:
            return SteeringSpec.off(**kw)
        return SteeringSpec(settings=self._settings(axis), **kw)

    def dual_spec(self, **kw) -> SteeringSpec:
        """Both dials at once -- the two-axis product configuration."""
        return SteeringSpec(
            settings=self._settings("structural") + self._settings("lexical"), **kw
        )

    def peak(self, axis: str) -> tuple[int, float]:
        """The dominant ``(layer, alpha)`` of an axis's schedule.

        For the callers that genuinely deal in one layer -- the bank coverage
        check, the adaptive ramp's ceiling, best-of-n's alpha rungs -- rather
        than in the whole shape.

        Ties break toward the middle of the band, not toward the shallowest
        layer. A flat schedule has no single largest alpha, and its centre is
        the layer the sweep actually chose; answering ``L30`` for a band
        centred on ``L31`` would misreport the arm in every log line that
        prints one layer.
        """
        sched = self.per_axis[axis]
        layers = sorted(layer for layer, _ in sched)
        middle = layers[len(layers) // 2]
        return max(sched, key=lambda pair: (pair[1], -abs(pair[0] - middle)))

    def is_scheduled(self, axis: str) -> bool:
        """True when the axis spreads over more than one layer."""
        return len(self.per_axis.get(axis, ())) > 1

    def layers(self, axis: str) -> list[int]:
        return [layer for layer, _ in self.per_axis[axis]]

    def __contains__(self, axis: str) -> bool:
        return axis in self.per_axis

    def __getitem__(self, axis: str) -> list[tuple[int, float]]:
        """The whole schedule. Callers wanting one layer want ``peak``.

        This used to return a bare ``(layer, alpha)``, so ``layer, alpha =
        settings[axis]`` was the idiom. That now raises rather than silently
        binding a tuple to ``layer``, which is why the return type changed
        instead of a parallel accessor being added.
        """
        return self.per_axis[axis]

    def as_dict(self) -> dict:
        """Provenance for the result row.

        A single-layer axis still writes ``[layer, alpha]``, the shape every
        recorded run so far uses, so an old reader keeps working and old files
        stay comparable. Only a real schedule writes the list of pairs.
        """
        out = {}
        for axis, sched in self.per_axis.items():
            out[axis] = (
                list(sched[0]) if len(sched) == 1 else [list(p) for p in sched]
            )
        return out

    def __repr__(self) -> str:
        if not self.per_axis:
            return "(none needed)"
        parts = []
        for axis, sched in sorted(self.per_axis.items()):
            if len(sched) == 1:
                parts.append(f"{axis}@L{sched[0][0]}a{sched[0][1]:g}")
            else:
                inner = "+".join(f"L{l}a{a:g}" for l, a in sched)
                parts.append(f"{axis}@[{inner}]")
        return " ".join(parts)
