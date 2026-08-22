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
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import config
from lib.steering import SteeringSpec

_EXPECTED_SWEEP_SHAPE = (
    '{"axes": {"structural": {"top_layers": [...], '
    '"best": {"layer": 14, "alpha": 1.0, ...}}, "lexical": {...}}}'
)


def die(message: str) -> None:
    """Stop with a message on stderr. Exit 2, as for a usage error."""
    print(f"\nFATAL: {message}\n", file=sys.stderr)
    raise SystemExit(2)


def parse_pair(text: str, flag: str) -> tuple[int, float]:
    """``"14,1.5"`` -> ``(14, 1.5)``."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        die(f"{flag} wants LAYER,ALPHA (e.g. {flag} 14,1.5), got {text!r}")
    try:
        return int(parts[0]), float(parts[1])
    except ValueError:
        die(f"{flag} wants LAYER,ALPHA with an int layer and float alpha, got {text!r}")


@dataclass
class AxisSettings:
    """Which ``(layer, alpha)`` each axis is steered at.

    Built from a sweep report, from explicit flags, or from both -- flags win,
    so a single axis can be overridden without abandoning the sweep for the
    other one.
    """

    per_axis: dict[str, tuple[int, float]]

    @classmethod
    def from_args(cls, args) -> "AxisSettings":
        """The standard ``--layers-from`` / ``--structural`` / ``--lexical`` trio.

        Every steering entry point in ``src/`` accepts exactly these three, so
        they are read off the namespace rather than passed one at a time.
        """
        settings: dict[str, tuple[int, float]] = {}
        path = getattr(args, "layers_from", None)
        if path:
            if not Path(path).exists():
                die(f"--layers-from {path} does not exist")
            settings.update(cls.from_sweep(Path(path)).per_axis)
        for axis in config.AXES:
            raw = getattr(args, axis, None)
            if raw:
                settings[axis] = parse_pair(raw, f"--{axis}")
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

        out: dict[str, tuple[int, float]] = {}
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
                out[axis] = int(best["layer"]), float(best["alpha"])
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

    def spec(self, axis: str | None, **kw) -> SteeringSpec:
        """A one-axis spec, or the unsteered spec when *axis* is ``None``."""
        if axis is None:
            return SteeringSpec.off(**kw)
        layer, alpha = self.per_axis[axis]
        return SteeringSpec.single(axis, layer, alpha, **kw)

    def dual_spec(self, **kw) -> SteeringSpec:
        """Both dials at once -- the two-axis product configuration."""
        return SteeringSpec.dual(
            self.per_axis["structural"], self.per_axis["lexical"], **kw
        )

    def __contains__(self, axis: str) -> bool:
        return axis in self.per_axis

    def __getitem__(self, axis: str) -> tuple[int, float]:
        return self.per_axis[axis]

    def as_dict(self) -> dict:
        return {axis: list(pair) for axis, pair in self.per_axis.items()}

    def __repr__(self) -> str:
        if not self.per_axis:
            return "(none needed)"
        return " ".join(
            f"{a}@L{l}a{alpha:g}" for a, (l, alpha) in sorted(self.per_axis.items())
        )
