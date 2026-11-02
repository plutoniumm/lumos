"""The argument plumbing every entry point in ``src/`` repeats.

``AxisSettings`` resolves each axis's ``(layer, alpha)`` schedule from a sweep
report and/or explicit flags, and turns it into a ``SteeringSpec``.

Depth schedules: an axis can spread its budget across a band of layers.

    --structural 31,0.5                      one layer
    --structural 31,0.5,flat3                a named shape from lib.schedules
    --structural 30:0.167+31:0.167+32:0.167  explicit pairs
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
    """``"14,1.5"`` -> ``(14, 1.5)``."""
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
        die(f"{flag}: layer(s) repeated in {text!r}; one alpha per layer per axis")
    return out


def parse_axis(text: str, flag: str) -> list[tuple[int, float]]:
    """One axis flag -> the ``(layer, alpha)`` schedule it names.

    Three forms, tried in order: explicit (contains ``:``), named shape
    (three comma-separated fields), single layer (two).
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
    """Which ``(layer, alpha)`` schedule each axis is steered at.

    Built from a sweep report, from explicit flags, or from both -- flags win.
    """

    per_axis: dict[str, list[tuple[int, float]]]

    @classmethod
    def from_args(cls, args) -> "AxisSettings":
        """Read the standard ``--layers-from``/``--structural``/``--lexical`` trio."""
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
                die(
                    f"--layers-from {path}: axes.{axis} must be an object, "
                    f"got {entry!r}"
                )
            best = entry.get("best")
            if best is None:
                print(
                    f"  [warn] {path}: axes.{axis} has no best cell; no setting taken"
                )
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
        """A one-axis spec, or the unsteered spec when *axis* is ``None``."""
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

        Ties break toward the middle of the band rather than the shallowest
        layer.
        """
        sched = self.per_axis[axis]
        layers = sorted(layer for layer, _ in sched)
        middle = layers[len(layers) // 2]
        return max(sched, key=lambda pair: (pair[1], -abs(pair[0] - middle)))

    def layers(self, axis: str) -> list[int]:
        return [layer for layer, _ in self.per_axis[axis]]

    def __getitem__(self, axis: str) -> list[tuple[int, float]]:
        """The whole schedule. Callers wanting one layer want ``peak``."""
        return self.per_axis[axis]

    def as_dict(self) -> dict:
        """Provenance for the result row."""
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
