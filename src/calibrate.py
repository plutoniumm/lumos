"""Turn alpha from an arbitrary float into a named reading level.

Alpha is the project's selling point over plain prompting: a continuous dial that
renders the same passage at whatever reading level a student needs. A dial only
helps if its numbers mean something, so this script maps each entry in
``config.TARGET_FKGL`` onto the alpha that actually lands there -- subject to the
gates that make the output usable at all (meaning preserved, fluency intact, no
degeneration).

Why a ladder before a bisection. Bisection assumes FKGL falls monotonically as
alpha rises. It mostly does, and then it stops: past some alpha the model starts
repeating itself, sentences fragment, and FKGL *drops* for reasons that have
nothing to do with readability -- a degenerate output can score grade 2. A naive
bisection walks straight into that region and reports it as a success. So we
first sweep a coarse ladder, record what FKGL and the gates actually do at each
rung, and only bisect inside a bracket where two neighbouring rungs both pass the
gates and FKGL genuinely decreases between them. If no such bracket exists, that
is the finding, and it is reported as such rather than papered over with the
closest violating alpha.

``max_safe_alpha`` is the other half of the output: the highest alpha that still
passes every gate. That is the ceiling the product UI must not let a dial exceed,
independent of any reading level.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import jsonio, tables
from lib.metrics import fluency, readability
from lib.metrics.faithfulness import faithfulness
from lib.models import SteerableLM, load
from lib.steering import SteeredLM, SteeringSpec, load_banks
from lib.vectors import VectorBank

# An empty or near-empty generation is the failure mode readability.measure()
# scores most generously: zero words gives fkgl 0.0 with degenerate False, which
# reads as a flawless grade-0 result. Anything this short is treated as
# degenerate here rather than trusted.
MIN_OUTPUT_WORDS = 6

# A single degenerate output in the calibration set rejects the alpha. This alpha
# ships as a named reading level to every student who picks that level; a
# degeneration rate that is "only" a few percent is a few percent of students
# handed garbage they cannot tell is garbage.
MAX_DEGENERATE_RATE = 0.0


@dataclass(frozen=True)
class Trial:
    """One alpha evaluated over the whole calibration set."""

    alpha: float
    mean_fkgl: float
    mean_nli: float
    mean_ppl_ratio: float
    degenerate_rate: float
    fails: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.fails

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "mean_fkgl": jsonio.finite(self.mean_fkgl),
            "mean_nli": jsonio.finite(self.mean_nli),
            "mean_ppl_ratio": jsonio.finite(self.mean_ppl_ratio),
            "degenerate_rate": self.degenerate_rate,
            "passed": self.passed,
            "fails": list(self.fails),
        }

    def line(self) -> str:
        verdict = "pass" if self.passed else "FAIL " + ",".join(self.fails)
        return (
            f"a={self.alpha:5.2f}  fkgl={self.mean_fkgl:6.2f}  "
            f"nli={self.mean_nli:.3f}  ppl={self.mean_ppl_ratio:6.2f}  "
            f"degen={self.degenerate_rate:4.0%}  {verdict}"
        )


@dataclass
class LevelResult:
    """The outcome of searching one axis for one named reading level."""

    level: str
    target_fkgl: float
    status: str  # ok | off_target | no_gate_passing_alpha
    alpha: float | None
    trial: Trial | None
    iterations: int
    note: str
    # Kept only as a diagnostic: a gate-passing alpha that missed the target. It
    # is deliberately not returned as the level's alpha.
    closest: Trial | None = None

    def as_dict(self) -> dict:
        out: dict = {
            "target_fkgl": self.target_fkgl,
            "status": self.status,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "note": self.note,
        }
        if self.trial is not None:
            out["achieved"] = self.trial.as_dict()
        if self.closest is not None:
            out["closest_passing_rejected"] = self.closest.as_dict()
        return out


@dataclass
class Calibrator:
    """Evaluates specs against one fixed calibration set, with a trial cache."""

    lm: SteerableLM
    steered: SteeredLM
    texts: list[str]
    max_new_tokens: int
    fkgl_tol: float
    batch_size: int
    baseline_fkgl: float = 0.0
    _prompts: list[str] = field(default_factory=list)
    _ref_ppl: list[float] = field(default_factory=list)
    _cache: dict[str, Trial] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The unsteered rewrite of each item is both the fluency reference and
        # the honest baseline for the printed summary. It never changes with
        # alpha, so it is computed once here.
        self._prompts = [self.lm.build_prompt(t) for t in self.texts]
        refs = self.steered.generate_batch(
            self.texts,
            SteeringSpec.off(),
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
        )
        self._ref_ppl = [
            fluency.conditional_ppl(self.lm, p, r) for p, r in zip(self._prompts, refs)
        ]
        self.baseline_fkgl = sum(readability.measure(r).fkgl for r in refs) / max(
            len(refs), 1
        )

    def _ratio(self, prompt: str, output: str, ref_ppl: float) -> float:
        """``fluency.ppl_ratio`` with the reference hoisted out of the alpha loop."""
        if not math.isfinite(ref_ppl) or ref_ppl <= 0:
            return float("inf")
        return fluency.conditional_ppl(self.lm, prompt, output) / ref_ppl

    def evaluate(self, spec: SteeringSpec, axis: str) -> Trial:
        alpha = next((s.alpha for s in spec.settings if s.axis == axis), 0.0)

        # Zero-alpha axes are dropped from the cache key: the injector treats
        # alpha 0 as an exact no-op, so "structural at 0 plus lexical at 0.4" is
        # the same generation as "lexical at 0.4" and must not be paid for twice.
        # That matters when the joint search holds an axis at 0. The searched
        # axis's alpha is re-attached on a hit, because the same generation can be
        # reached while searching either axis.
        live = tuple(s for s in spec.settings if s.alpha != 0.0)
        key = replace(spec, settings=live).label()
        cached = self._cache.get(key)
        if cached is not None:
            return cached if cached.alpha == alpha else replace(cached, alpha=alpha)

        outs = self.steered.generate_batch(
            self.texts,
            spec,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
        )

        fkgls, nlis, ratios, n_degen = [], [], [], 0
        for src, prompt, ref, out in zip(
            self.texts, self._prompts, self._ref_ppl, outs
        ):
            read = readability.measure(out)
            fkgls.append(read.fkgl)
            if read.degenerate or read.n_words < MIN_OUTPUT_WORDS:
                n_degen += 1
            nlis.append(faithfulness(src, out).score)
            ratios.append(self._ratio(prompt, out, ref))

        n = max(len(outs), 1)
        mean_fkgl = sum(fkgls) / n
        mean_nli = sum(nlis) / n
        mean_ppl = (
            sum(ratios) / n if all(math.isfinite(r) for r in ratios) else math.inf
        )
        degen_rate = n_degen / n

        fails: list[str] = []
        if mean_nli < config.NLI_ENTAILMENT_MIN:
            fails.append("nli")
        if not mean_ppl <= config.PPL_RATIO_MAX:
            fails.append("ppl")
        if degen_rate > MAX_DEGENERATE_RATE:
            fails.append("degenerate")

        trial = Trial(alpha, mean_fkgl, mean_nli, mean_ppl, degen_rate, tuple(fails))
        self._cache[key] = trial
        return trial

    def ladder(self, base: SteeringSpec, axis: str, alphas: list[float]) -> list[Trial]:
        trials = []
        bar = tables.progress(alphas, unit="rung", desc=f"ladder {axis}")
        for alpha in bar:
            trial = self.evaluate(base.with_alpha(axis, alpha), axis)
            # Written rather than printed: the per-rung line is the record a
            # redirected run keeps, and the bar is only for a watching human.
            bar.write(f"  {trial.line()}")
            trials.append(trial)
        return trials

    def search_level(
        self,
        base: SteeringSpec,
        axis: str,
        level: str,
        target: float,
        rungs: list[Trial],
    ) -> tuple[LevelResult, list[Trial]]:
        """Bisect for ``target`` inside a bracket the ladder shows to be sound."""
        refined: list[Trial] = []

        hit = _on_target(rungs, target, self.fkgl_tol)
        if hit is not None:
            return (
                LevelResult(
                    level, target, "ok", hit.alpha, hit, 0, "hit on the coarse ladder"
                ),
                refined,
            )

        bracket = _bracket(rungs, target)
        if bracket is None:
            passing = [t for t in rungs if t.passed]
            if not passing:
                return (
                    LevelResult(
                        level,
                        target,
                        "no_gate_passing_alpha",
                        None,
                        None,
                        0,
                        "no rung on the ladder passed the gates; nothing to search",
                    ),
                    refined,
                )
            best = min(passing, key=lambda t: abs(t.mean_fkgl - target))
            return (
                LevelResult(
                    level,
                    target,
                    "off_target",
                    None,
                    None,
                    0,
                    f"no monotone gate-passing bracket straddles fkgl {target}; "
                    f"reachable gate-passing fkgl spans "
                    f"{min(t.mean_fkgl for t in passing):.2f}"
                    f"..{max(t.mean_fkgl for t in passing):.2f}",
                    closest=best,
                ),
                refined,
            )

        lo, hi = bracket
        alpha_lo, alpha_hi = lo.alpha, hi.alpha
        print(f"  bracket [{alpha_lo:g}, {alpha_hi:g}] for {level} (fkgl {target})")

        iters = 0
        for _ in range(config.CALIBRATE_MAX_ITERS):
            mid = (alpha_lo + alpha_hi) / 2.0
            trial = self.evaluate(base.with_alpha(axis, mid), axis)
            refined.append(trial)
            iters += 1
            print(f"  {trial.line()}")

            if not trial.passed:
                # Gates broke strictly inside a bracket whose endpoints both
                # passed, so the bracket is not as well behaved as the ladder
                # suggested. Retreat toward the lower, safer alpha.
                alpha_hi = mid
                continue
            if abs(trial.mean_fkgl - target) <= self.fkgl_tol:
                break
            if trial.mean_fkgl > target:
                alpha_lo = mid
            else:
                alpha_hi = mid

        candidates = rungs + refined
        hit = _on_target(candidates, target, self.fkgl_tol)
        if hit is not None:
            return (
                LevelResult(level, target, "ok", hit.alpha, hit, iters, "bisected"),
                refined,
            )

        passing = [t for t in candidates if t.passed]
        best = (
            min(passing, key=lambda t: abs(t.mean_fkgl - target)) if passing else None
        )
        return (
            LevelResult(
                level,
                target,
                "off_target",
                None,
                None,
                iters,
                f"bisection did not reach fkgl {target} +/- {self.fkgl_tol} "
                f"within {config.CALIBRATE_MAX_ITERS} iterations while passing gates",
                closest=best,
            ),
            refined,
        )


def _on_target(trials: list[Trial], target: float, tol: float) -> Trial | None:
    """Gentlest gate-passing trial within ``tol`` of the target FKGL."""
    ok = [t for t in trials if t.passed and abs(t.mean_fkgl - target) <= tol]
    return min(ok, key=lambda t: t.alpha) if ok else None


def _bracket(trials: list[Trial], target: float) -> tuple[Trial, Trial] | None:
    """Lowest adjacent rung pair that brackets ``target`` and behaves monotonically."""
    ordered = sorted(trials, key=lambda t: t.alpha)
    for lo, hi in zip(ordered, ordered[1:]):
        if not (lo.passed and hi.passed):
            continue
        if hi.mean_fkgl < lo.mean_fkgl and hi.mean_fkgl <= target <= lo.mean_fkgl:
            return lo, hi
    return None


def _max_safe(trials: list[Trial]) -> float | None:
    passing = [t for t in trials if t.passed]
    return max(t.alpha for t in passing) if passing else None


def _read_texts(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        return [r["text"] if isinstance(r, dict) else str(r) for r in rows]
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("texts", [])
        return [p if isinstance(p, str) else p["text"] for p in payload]
    # Plain text: one item per blank-line-separated block, so multi-sentence
    # passages survive intact.
    return [b.strip() for b in path.read_text().split("\n\n") if b.strip()]


def load_calibration_texts(n: int, path: Path | None) -> tuple[list[str], str]:
    if path is not None:
        if not path.exists():
            raise SystemExit(f"[calibrate] no calibration texts at {path}")
        return _read_texts(path)[:n], str(path)

    try:
        from lib import data
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"[calibrate] cannot load a calibration corpus ({exc}). Pass "
            "--data PATH (a .txt with one passage per blank-line block, or a "
            ".json/.jsonl list) or provide lib/data.py."
        ) from exc

    # Two constraints on the calibration set, in this order:
    #
    #   1. it must not be the ASSET test set the project reports SARI on --
    #      fitting alpha on the eval set and then reporting eval numbers at that
    #      alpha is exactly the leak lib/data.py asserts against elsewhere;
    #   2. it should not be the texts the vectors were estimated from either.
    #
    # So: same pool as estimation, with the estimation sample subtracted
    # outright. Subtracting rather than slicing a bigger draw, because
    # load_complex_corpus samples with random.sample and being prefix-stable in
    # n is an implementation detail of CPython, not a promise.
    try:
        pool = data.complex_pool()
        used = set(data.load_complex_corpus(min(config.VECTOR_N_TEXTS, len(pool))))
        held = [t for t in pool if t not in used]
        if len(held) < n:
            print(
                "[calibrate] WARNING: only "
                f"{len(held)} texts left after excluding the estimation set; "
                "these alphas are fitted partly on text the vectors saw"
            )
            held = pool
        texts = random.Random(config.SEED).sample(held, min(n, len(held)))
        return texts, "lib.data.complex_pool minus the estimation sample"
    except (AttributeError, ValueError) as exc:
        print(f"[calibrate] holdout unavailable ({exc}); falling back")
        return data.load_complex_corpus(n), "lib.data.load_complex_corpus"


def _ladder_alphas(override: str | None, steps: int) -> list[float]:
    """Coarse ladder over ``CALIBRATE_ALPHA_BOUNDS``, denser at the bottom."""
    if override:
        return sorted({float(x) for x in override.split(",") if x.strip()})
    lo, hi = config.CALIBRATE_ALPHA_BOUNDS
    if steps < 2:
        raise SystemExit("[calibrate] --ladder-steps must be at least 2")
    return [lo + (hi - lo) * (i / (steps - 1)) ** 2 for i in range(steps)]


def _levels(raw: str | None) -> list[str]:
    if not raw:
        return list(config.TARGET_FKGL)
    names = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [n for n in names if n not in config.TARGET_FKGL]
    if unknown:
        raise SystemExit(
            f"[calibrate] unknown level(s) {unknown}; "
            f"choose from {list(config.TARGET_FKGL)}"
        )
    return names


def _resolve_layer(
    axis: str, given: int | None, lm: SteerableLM, bank: VectorBank
) -> int:
    layer = given
    if layer is None:
        layer = lm.n_layers // 2
        print(
            f"[calibrate] WARNING: no --{axis}-layer given, defaulting to the "
            f"middle layer {layer}. The layer sweep, not this default, is what "
            f"should choose it."
        )
    if layer < 0:
        layer += lm.n_layers
    if layer not in bank.vectors:
        raise SystemExit(
            f"[calibrate] bank {axis!r} has no vector for layer {layer}; "
            f"available layers {bank.layers[:3]}...{bank.layers[-3:]}"
        )
    return layer


def run(args: argparse.Namespace) -> int:
    levels = _levels(args.levels)
    repo = args.model or config.ACTIVE_MODEL

    # Banks first: a missing bank is a one-second failure, and loading a 3B model
    # only to discover it is a six-gigabyte one.
    try:
        banks = load_banks(repo, args.estimator)
    except FileNotFoundError as exc:
        print(f"[calibrate] {exc}")
        return 2

    lm = load(repo)
    print(f"[model] {lm.repo_id}  layers={lm.n_layers}  hidden={lm.hidden_size}")

    layers = {
        "structural": _resolve_layer(
            "structural", args.structural_layer, lm, banks["structural"]
        ),
        "lexical": _resolve_layer("lexical", args.lexical_layer, lm, banks["lexical"]),
    }
    print(f"[layers] {layers}")

    texts, source = load_calibration_texts(args.n, args.data)
    if not texts:
        raise SystemExit("[calibrate] calibration set is empty")
    print(f"[data] {len(texts)} calibration items from {source}")

    alphas = _ladder_alphas(args.ladder, args.ladder_steps)
    cal = Calibrator(
        lm=lm,
        steered=SteeredLM(lm, banks),
        texts=texts,
        max_new_tokens=args.max_new_tokens,
        fkgl_tol=args.fkgl_tol,
        batch_size=args.batch_size,
    )
    print(f"[baseline] unsteered mean fkgl = {cal.baseline_fkgl:.2f}")

    ladders: dict[str, list[Trial]] = {}
    per_axis: dict[str, dict[str, LevelResult]] = {}
    max_safe: dict[str, float | None] = {}

    for axis in config.AXES:
        base = SteeringSpec.single(axis, layers[axis], 0.0)
        print(f"\n[ladder] {axis} @ L{layers[axis]}")
        rungs = cal.ladder(base, axis, alphas)
        ladders[axis] = list(rungs)
        seen = list(rungs)

        results: dict[str, LevelResult] = {}
        for level in levels:
            target = config.TARGET_FKGL[level]
            print(f"[search] {axis} -> {level} (fkgl {target})")
            res, refined = cal.search_level(base, axis, level, target, rungs)
            results[level] = res
            seen += refined
            print(f"[search] {axis} {level}: {res.status} alpha={res.alpha}")
        per_axis[axis] = results
        max_safe[axis] = _max_safe(seen)

    joint: dict[str, dict] = {}
    if args.joint:
        held_axis = args.hold
        search_axis = next(a for a in config.AXES if a != held_axis)
        for level in levels:
            target = config.TARGET_FKGL[level]
            anchor_res = per_axis[held_axis][level]
            anchor = anchor_res.alpha
            anchor_src = "level"
            if anchor is None:
                # No calibrated value for the held axis at this level, so the
                # joint search runs with that dial off rather than guessing.
                anchor, anchor_src = 0.0, "fallback_zero"

            # Coordinate search, not a joint optimum: the held axis is pinned at
            # its own single-axis value and only the other dial moves. A true
            # joint fit would search the (structural, lexical) plane, which costs
            # a full grid per level and is not what this produces.
            base = SteeringSpec.dual(
                (layers["structural"], anchor if held_axis == "structural" else 0.0),
                (layers["lexical"], anchor if held_axis == "lexical" else 0.0),
            )
            print(
                f"\n[ladder] joint {level}: {held_axis} held at {anchor:g} "
                f"({anchor_src}), searching {search_axis}"
            )
            rungs = cal.ladder(base, search_axis, alphas)
            ladders[f"joint:{level}"] = list(rungs)
            res, refined = cal.search_level(base, search_axis, level, target, rungs)
            print(f"[search] joint {level}: {res.status} alpha={res.alpha}")
            joint[level] = {
                "held_axis": held_axis,
                "held_alpha": anchor,
                "held_alpha_source": anchor_src,
                "searched_axis": search_axis,
                "max_safe_alpha_searched_axis": _max_safe(rungs + refined),
                **res.as_dict(),
            }

    payload = _payload(
        lm,
        args,
        layers,
        texts,
        source,
        levels,
        alphas,
        cal,
        ladders,
        per_axis,
        max_safe,
        joint,
    )
    out_path = args.out or (
        config.RESULTS_DIR
        / f"calibration_{lm.repo_id.split('/')[-1]}_{args.estimator}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonio.dump(out_path, payload)
    print(f"\n[save] {out_path}")

    _summarise(payload)
    return 0


def _payload(
    lm: SteerableLM,
    args: argparse.Namespace,
    layers: dict[str, int],
    texts: list[str],
    source: str,
    levels: list[str],
    alphas: list[float],
    cal: Calibrator,
    ladders: dict[str, list[Trial]],
    per_axis: dict[str, dict[str, LevelResult]],
    max_safe: dict[str, float | None],
    joint: dict[str, dict],
) -> dict:
    """Assemble the JSON. ``levels[*].alphas`` is the part simplify.py reads."""
    out_levels: dict[str, dict] = {}
    for level in levels:
        target = config.TARGET_FKGL[level]
        entry: dict = {"target_fkgl": target}
        if level in joint:
            j = joint[level]
            held, searched = j["held_axis"], j["searched_axis"]
            entry["status"] = j["status"]
            entry["alphas"] = {
                held: j["held_alpha"] if j["status"] == "ok" else None,
                searched: j["alpha"],
            }
            entry["achieved"] = j.get("achieved")
            entry["source"] = "joint"
        else:
            # Never "ok": each alpha was measured with the other axis switched
            # off, so the pair has no measurement behind it and applying both
            # will overshoot the target. Only the joint search can call a pair ok.
            ok_alone = all(per_axis[a][level].status == "ok" for a in per_axis)
            entry["status"] = "single_axis_only" if ok_alone else "partial"
            entry["alphas"] = {a: per_axis[a][level].alpha for a in per_axis}
            entry["achieved"] = None
            entry["source"] = "single_axis"
            entry["note"] = (
                "each axis was calibrated with the other axis at zero; using "
                "both alphas together was never measured and will overshoot the "
                "target. Re-run without --no-joint for a searched pair."
            )
        entry["per_axis"] = {a: per_axis[a][level].as_dict() for a in per_axis}
        if level in joint:
            entry["joint"] = joint[level]
        out_levels[level] = entry

    return {
        "model": lm.repo_id,
        "estimator": args.estimator,
        "layers": layers,
        "n_items": len(texts),
        "calibration_source": source,
        "max_new_tokens": args.max_new_tokens,
        "fkgl_tol": args.fkgl_tol,
        "baseline_unsteered_fkgl": jsonio.finite(cal.baseline_fkgl),
        "gates": {
            "nli_entailment_min": config.NLI_ENTAILMENT_MIN,
            "ppl_ratio_max": config.PPL_RATIO_MAX,
            "max_degenerate_rate": MAX_DEGENERATE_RATE,
            "min_output_words": MIN_OUTPUT_WORDS,
        },
        "ladder_alphas": alphas,
        "max_safe_alpha": max_safe,
        "levels": out_levels,
        "ladders": {k: [t.as_dict() for t in v] for k, v in ladders.items()},
    }


def _fmt(x: float | None, places: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def _summarise(payload: dict) -> None:
    print("\n" + "=" * 72)
    print(f"calibration  {payload['model']}  {payload['estimator']}")
    print(
        f"  layers {payload['layers']}   items {payload['n_items']}   "
        f"unsteered fkgl {_fmt(payload['baseline_unsteered_fkgl'])}"
    )
    print(
        "  max_safe_alpha "
        + ", ".join(f"{k}={_fmt(v, 3)}" for k, v in payload["max_safe_alpha"].items())
        + "   (product UI ceiling)"
    )

    for name, rungs in payload["ladders"].items():
        table = tables.table(title=f"ladder {name}")
        for col in ("alpha", "fkgl", "nli", "ppl", "degen", "verdict"):
            table.add_column(col, justify="left" if col == "verdict" else "right")
        for r in rungs:
            table.add_row(
                f"{r['alpha']:.3f}",
                _fmt(r["mean_fkgl"]),
                _fmt(r["mean_nli"], 3),
                _fmt(r["mean_ppl_ratio"]),
                f"{r['degenerate_rate']:.0%}",
                "pass" if r["passed"] else "FAIL " + ",".join(r["fails"]),
            )
        tables.show(table, width=72)

    print("\n  levels")
    for level, entry in payload["levels"].items():
        alphas = ", ".join(f"{k}={v}" for k, v in entry["alphas"].items())
        print(f"    {level} (target fkgl {entry['target_fkgl']}): {entry['status']}")
        print(f"      alphas   {alphas}   [{entry['source']}]")
        got = entry.get("achieved")
        if got:
            print(
                f"      achieved fkgl={_fmt(got['mean_fkgl'])} "
                f"nli={_fmt(got['mean_nli'], 3)} ppl={_fmt(got['mean_ppl_ratio'])}"
            )
        joint = entry.get("joint")
        if joint:
            print(
                f"      joint    {joint['held_axis']} held at "
                f"{_fmt(joint['held_alpha'], 3)} "
                f"({joint['held_alpha_source']}), searched "
                f"{joint['searched_axis']} -> {joint['status']}; ceiling for "
                f"{joint['searched_axis']} under that anchor "
                f"{_fmt(joint['max_safe_alpha_searched_axis'], 3)}"
            )
        for axis, res in entry["per_axis"].items():
            line = f"      {axis:10s} {res['status']:22s} alpha={res['alpha']}"
            if res["status"] != "ok":
                line += f"  -- {res['note']}"
            print(line)
        if entry["status"] != "ok":
            if all(v is not None for v in entry["alphas"].values()):
                print(f"      NOTE: {entry.get('note', 'status ' + entry['status'])}")
            else:
                print(
                    "      NOTE: no alpha reached this level inside the gates. "
                    "Do not ship this level; the closest passing alpha is "
                    "reported as a diagnostic only."
                )
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--estimator", default="instruction_diff")
    p.add_argument("--structural-layer", type=int, default=None)
    p.add_argument("--lexical-layer", type=int, default=None)
    p.add_argument("--n", type=int, default=24, help="calibration items")
    p.add_argument("--levels", default=None, help="e.g. grade4,grade6")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--data", type=Path, default=None, help="calibration texts")
    p.add_argument(
        "--ladder", default=None, help="explicit alpha ladder, comma separated"
    )
    p.add_argument("--ladder-steps", type=int, default=7)
    p.add_argument("--fkgl-tol", type=float, default=0.5, help="grade levels")
    p.add_argument("--max-new-tokens", type=int, default=config.SWEEP_MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--hold", default="structural", choices=config.AXES)
    p.add_argument("--no-joint", dest="joint", action="store_false")
    p.set_defaults(joint=True)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
