"""Simplify text with two dials, or with a named reading level.

    python src/simplify.py --text "..." --structural 0.8 --lexical 1.2
    python src/simplify.py --file in.txt --level grade4 --out out.txt
    python src/simplify.py --text "..." --level grade6 --explain
    echo "..." | python src/simplify.py --level grade4

This is the end-user entry point, so it is the one place in the project where
being wrong has a cost outside a results table. The reader is a student who
cannot check the simplification against the source -- that is why they asked for
it. Handing them text that quietly dropped a clause or invented a fact is worse
than handing them the original, so every rendering is checked for meaning
preservation before it is printed, alpha is walked back down and retried if the
check fails, and the tool falls back to the plain prompted rewrite rather than
shipping an unverified steered output. Failures are announced on stderr; they are
never silent.

stdout carries the simplified text and nothing else unless ``--explain``, so the
tool composes in a pipeline. Warnings, progress and diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib.chunking import split_sentences
from lib.metrics import readability
from lib.metrics.faithfulness import faithfulness
from lib.models import SteerableLM, load
from lib.steering import SteeredLM, SteeringSpec, load_banks

# Each retry multiplies both dials by this. 0.7 twice lands near half strength,
# which is far enough to matter and close enough to still be a simplification.
RETRY_SCALE = 0.7
MAX_RETRIES = 2

# readability.measure() scores an empty generation as fkgl 0.0 with
# degenerate=False, i.e. a flawless grade-0 result. Anything this short is
# rejected here instead.
MIN_OUTPUT_WORDS = 6


def warn(msg: str) -> None:
    print(f"[simplify] {msg}", file=sys.stderr)


@dataclass(frozen=True)
class Dials:
    """The two-axis control surface, plus the layer each axis is injected at."""

    structural: float
    lexical: float
    structural_layer: int
    lexical_layer: int

    def spec(self) -> SteeringSpec:
        return SteeringSpec.dual(
            (self.structural_layer, self.structural),
            (self.lexical_layer, self.lexical),
        )

    def scaled(self, factor: float) -> "Dials":
        return replace(
            self,
            structural=self.structural * factor,
            lexical=self.lexical * factor,
        )

    def label(self) -> str:
        return (
            f"structural={self.structural:.3g}@L{self.structural_layer} "
            f"lexical={self.lexical:.3g}@L{self.lexical_layer}"
        )


@dataclass
class Attempt:
    """One rendering and its verdict."""

    label: str
    text: str
    nli: float
    degenerate: bool

    @property
    def ok(self) -> bool:
        return not self.degenerate and self.nli >= config.NLI_ENTAILMENT_MIN

    def why(self) -> str:
        if self.degenerate:
            return "output is degenerate (repetition, or too short to be a rewrite)"
        return (
            f"meaning check failed: NLI {self.nli:.2f} < "
            f"{config.NLI_ENTAILMENT_MIN:.2f}"
        )


@dataclass
class Result:
    text: str
    outcome: str  # steered | prompt_only | source | unverified
    attempts: list[Attempt]
    final: Attempt | None


def _check(source: str, output: str, label: str) -> Attempt:
    read = readability.measure(output)
    degenerate = read.degenerate or read.n_words < MIN_OUTPUT_WORDS
    nli = faithfulness(source, output).score
    return Attempt(label, output, nli, degenerate)


def _render(
    steered: SteeredLM,
    text: str,
    spec: SteeringSpec,
    chunked: bool,
    max_new_tokens: int,
    instruction: str | None = None,
) -> str:
    if chunked:
        return steered.generate_chunked(
            text, spec, instruction=instruction, max_new_tokens=max_new_tokens
        )
    return steered.generate(
        text, spec, instruction=instruction, max_new_tokens=max_new_tokens
    )


def simplify(
    steered: SteeredLM,
    text: str,
    dials: Dials,
    chunked: bool,
    max_new_tokens: int,
    allow_unverified: bool = False,
) -> Result:
    """Render ``text``, backing off alpha until the meaning check passes."""
    attempts: list[Attempt] = []
    current = dials

    for i in range(1 + MAX_RETRIES):
        out = _render(steered, text, current.spec(), chunked, max_new_tokens)
        attempt = _check(text, out, current.label())
        attempts.append(attempt)
        if attempt.ok:
            return Result(out, "steered", attempts, attempt)
        if i < MAX_RETRIES:
            warn(
                f"attempt {i + 1} ({current.label()}): {attempt.why()}; "
                f"retrying at {RETRY_SCALE:g}x alpha"
            )
            current = current.scaled(RETRY_SCALE)
        else:
            warn(f"attempt {i + 1} ({current.label()}): {attempt.why()}")

    # Steering could not preserve meaning at any strength tried. The prompted
    # rewrite is the baseline the whole project is measured against, so it is the
    # right thing to fall back to -- but it gets checked too.
    fallback = _render(
        steered,
        text,
        SteeringSpec.off(),
        chunked,
        max_new_tokens,
        instruction=config.INSTRUCTIONS["simplify"],
    )
    checked = _check(text, fallback, "prompt-only")
    attempts.append(checked)
    if checked.ok:
        warn(
            "steering could not preserve meaning at these dials; returned the "
            "prompt-only simplification instead (no steering applied)"
        )
        return Result(fallback, "prompt_only", attempts, checked)

    best = max(attempts, key=lambda a: (not a.degenerate, a.nli))
    if allow_unverified:
        warn(
            f"NOTHING PASSED the meaning check (best NLI {best.nli:.3f}). "
            f"--allow-unverified is set, so the rejected output from "
            f"'{best.label}' is printed anyway. Do not give it to a student."
        )
        return Result(best.text, "unverified", attempts, best)

    warn(
        f"NOTHING PASSED the meaning check, not even the prompt-only rewrite "
        f"(best NLI {best.nli:.3f} < {config.NLI_ENTAILMENT_MIN:.2f}). Returning "
        f"the source text UNCHANGED rather than a simplification that may have "
        f"dropped or invented facts. Re-run with --allow-unverified to inspect "
        f"the rejected output."
    )
    return Result(text, "source", attempts, None)


def calibration_path(repo: str, estimator: str, override: Path | None) -> Path:
    if override is not None:
        return override
    slug = repo.split("/")[-1]
    return config.RESULTS_DIR / f"calibration_{slug}_{estimator}.json"


def load_level(
    path: Path, level: str, repo: str, estimator: str
) -> tuple[dict, dict, dict]:
    """``(alphas, layers, max_safe)`` for ``level`` from a calibration file."""
    if not path.exists():
        raise SystemExit(
            f"[simplify] no calibration file at {path}, so --level {level} cannot "
            f"be resolved. Run\n"
            f"  python src/calibrate.py --model {repo} --estimator {estimator}\n"
            f"or pass explicit dials instead, e.g. --structural 1.0 --lexical 1.0"
        )
    payload = json.loads(path.read_text())
    levels = payload.get("levels", {})
    if level not in levels:
        raise SystemExit(
            f"[simplify] {path} has no level {level!r}; "
            f"calibrated levels are {sorted(levels)}"
        )
    entry = levels[level]
    alphas = entry.get("alphas") or {}
    missing = [a for a in config.AXES if alphas.get(a) is None]
    if missing:
        raise SystemExit(
            f"[simplify] level {level!r} is not usable: calibration status "
            f"{entry.get('status')!r} left {missing} without an alpha that both "
            f"hits the target reading level and passes the meaning and fluency "
            f"gates. See {path}. Pass explicit dials to override."
        )
    status = entry.get("status")
    if status != "ok":
        # Alphas exist but were not verified as the pair that ships, e.g. each
        # axis calibrated alone. Usable, but the reading level is not guaranteed.
        warn(
            f"level {level!r} has calibration status {status!r}, not 'ok': "
            f"{entry.get('note', 'see ' + str(path))}"
        )
    return alphas, payload.get("layers", {}), payload.get("max_safe_alpha", {})


def lookup_level(args: argparse.Namespace, repo: str) -> tuple[dict, dict, dict]:
    """Calibration lookup, kept file-only so it can run before the model loads."""
    if not args.level:
        return {}, {}, {}
    path = calibration_path(repo, args.estimator, args.calibration)
    alphas, layers, max_safe = load_level(path, args.level, repo, args.estimator)
    warn(f"level {args.level} -> {alphas} (from {path})")
    return alphas, layers, max_safe


def resolve_dials(
    args: argparse.Namespace,
    lm: SteerableLM,
    alphas: dict,
    layers: dict,
    max_safe: dict,
) -> Dials:
    # Explicit dials override the level, one axis at a time.
    structural = (
        args.structural
        if args.structural is not None
        else alphas.get("structural", 0.0)
    )
    lexical = args.lexical if args.lexical is not None else alphas.get("lexical", 0.0)

    resolved = {}
    for axis, given in (
        ("structural", args.structural_layer),
        ("lexical", args.lexical_layer),
    ):
        layer = given if given is not None else layers.get(axis)
        if layer is None:
            layer = lm.n_layers // 2
            warn(
                f"no layer for the {axis} axis (no --{axis}-layer and no "
                f"calibration file); defaulting to the middle layer {layer}"
            )
        resolved[axis] = int(layer)

    for axis, alpha in (("structural", structural), ("lexical", lexical)):
        ceiling = max_safe.get(axis) if max_safe else None
        if ceiling is not None and alpha > ceiling:
            warn(
                f"{axis} alpha {alpha:g} is above the calibrated ceiling "
                f"{ceiling:g}; outputs beyond it failed the gates during "
                f"calibration"
            )

    return Dials(
        structural=float(structural),
        lexical=float(lexical),
        structural_layer=resolved["structural"],
        lexical_layer=resolved["lexical"],
    )


def explain(source: str, result: Result, dials: Dials, chunked: bool) -> str:
    before = readability.measure(source)
    after = readability.measure(result.text)
    rows = [
        ("FKGL", f"{before.fkgl:.1f}", f"{after.fkgl:.1f}"),
        (
            "words/sentence",
            f"{before.words_per_sentence:.1f}",
            f"{after.words_per_sentence:.1f}",
        ),
        (
            "hard words",
            f"{before.pct_hard_words * 100:.1f}%",
            f"{after.pct_hard_words * 100:.1f}%",
        ),
        ("words", f"{before.n_words}", f"{after.n_words}"),
        ("sentences", f"{before.n_sentences}", f"{after.n_sentences}"),
    ]
    lines = ["", "-" * 56, f"{'':16s}{'source':>12s}{'simplified':>14s}"]
    lines += [f"{name:16s}{a:>12s}{b:>14s}" for name, a, b in rows]

    final = result.final
    if final is None:
        verdict = "no rendering passed; source returned unchanged"
    else:
        gate = "pass" if final.ok else "FAIL"
        verdict = (
            f"{final.nli:.2f} ({gate}, gate {config.NLI_ENTAILMENT_MIN:.2f}) "
            f"from '{final.label}'"
        )
    lines += [
        "",
        f"faithfulness    {verdict}",
        f"dials           {dials.label()}",
        f"mode            {'chunked' if chunked else 'whole'}",
        f"outcome         {result.outcome} after {len(result.attempts)} rendering(s)",
        "-" * 56,
    ]
    return "\n".join(lines)


def read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return args.file.read_text()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("[simplify] no input; pass --text, --file, or pipe text on stdin")


def run(args: argparse.Namespace) -> int:
    source = read_input(args).strip()
    if not source:
        raise SystemExit("[simplify] input is empty")

    repo = args.model or config.ACTIVE_MODEL

    # Both setup failures are resolved before the model load, which is the slow
    # part: a missing calibration file is a file read, a missing bank is a stat.
    alphas, layers, max_safe = lookup_level(args, repo)
    try:
        banks = load_banks(repo, args.estimator)
    except FileNotFoundError as exc:
        print(f"[simplify] {exc}", file=sys.stderr)
        return 2

    lm = load(repo)
    dials = resolve_dials(args, lm, alphas, layers, max_safe)

    sentences = split_sentences(source)
    chunked = len(sentences) > 1 if args.mode == "auto" else args.mode == "chunked"
    warn(
        f"{len(sentences)} sentence(s), mode={'chunked' if chunked else 'whole'}"
        f" ({args.mode}), dials {dials.label()}"
    )

    result = simplify(
        SteeredLM(lm, banks),
        source,
        dials,
        chunked,
        args.max_new_tokens,
        allow_unverified=args.allow_unverified,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.text + "\n")
        warn(f"wrote {args.out}")
    else:
        print(result.text)

    if args.explain:
        print(explain(source, result, dials, chunked))

    # A non-zero exit is the only machine-readable way to say "this output is not
    # a verified simplification" to a caller that is piping stdout.
    return 0 if result.outcome in ("steered", "prompt_only") else 3


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", default=None)
    src.add_argument("--file", type=Path, default=None)

    p.add_argument("--level", default=None, help=f"one of {list(config.TARGET_FKGL)}")
    p.add_argument(
        "--structural", type=float, default=None, help="alpha, overrides --level"
    )
    p.add_argument(
        "--lexical", type=float, default=None, help="alpha, overrides --level"
    )
    p.add_argument("--structural-layer", type=int, default=None)
    p.add_argument("--lexical-layer", type=int, default=None)

    p.add_argument("--model", default=None)
    p.add_argument("--estimator", default="instruction_diff")
    p.add_argument("--calibration", type=Path, default=None, help="calibration json")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--whole",
        dest="mode",
        action="store_const",
        const="whole",
        help="force one pass over the whole input",
    )
    mode.add_argument(
        "--chunked",
        dest="mode",
        action="store_const",
        const="chunked",
        help="force sentence-by-sentence rewriting",
    )
    p.set_defaults(mode="auto")

    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument(
        "--allow-unverified",
        action="store_true",
        help="print a rendering that failed the meaning check (research use)",
    )
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
