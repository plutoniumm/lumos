"""Simplify text by activation steering, a LoRA adapter, few-shot demonstrations,
a decode-time vocabulary constraint, or a composition of these -- with one
composition refused outright. report.md is the reason there is more than one:
on ASSET dSARI, lora beats steering +8.47*/+6.19* vs +0.43 ns/+2.02 ns
(Llama/Gemma) and fewshot beats it too (+3.85*/+3.44*), so a product that only
offered steering was not offering its own best-measured arm.

    python src/simplify.py --text "..." --mechanism auto
    python src/simplify.py --text "..." --structural 0.75 --structural-layer 15
    python src/simplify.py --text "..." --mechanism lora --lora models/lora/asset_r16
    python src/simplify.py --text "..." --mechanism fewshot --shots 5
    python src/simplify.py --text "..." --mechanism lora+steering --structural 0.5 --structural-layer 15
    python src/simplify.py --file in.txt --level grade4 --out out.txt --easy-words
    echo "..." | python src/simplify.py --level grade4

This is the end-user entry point, so it is the one place in the project where
being wrong has a cost outside a results table. The reader is a student who
cannot check the simplification against the source -- that is why they asked for
it. Handing them text that quietly dropped a clause or invented a fact is worse
than handing them the original, so every rendering is checked for meaning
preservation before it is printed, whatever dial the active mechanism has is
walked back down and retried if the check fails, and the tool falls back to the
plain prompted rewrite rather than shipping an unverified output. Failures are
announced on stderr; they are never silent.

--mechanism (default auto) picks lora when an adapter for the active model is
on disk, else structural steering -- see resolve_mechanism(). Name one or more
of steering, lora, fewshot, none (the bare prompted instruction, no vector, no
adapter) explicitly, joined by '+' (e.g. lora+steering). The one refusal:
none+steering, because report.md's A/B 2 measures exactly that composition --
the simplify instruction alongside vector injection -- as the single worst arm
on both models (-13.50* SARI/66% degenerate on Llama, -12.76*/28% on Gemma).
--easy-words is a separate flag layered on top of any mechanism.

stdout carries the simplified text and nothing else, in every mode -- including
``--explain`` -- so the tool composes in a pipeline
(``... | python src/simplify.py ... > out.txt`` lands exactly the rewrite in
out.txt). Warnings, progress and ``--explain``'s diagnostics all go to stderr.

Exit codes:
  0  a result was printed: from the active mechanism, a retry-ladder fallback
     (prompt-only, or the source returned unchanged), or a rejected rewrite
     shown under --allow-unverified. All are legitimate outcomes of the
     ladder; stderr says which one happened, so the exit code does not have to.
  1  the run raised before any result could be produced.
  2  a usage or configuration error the caller must fix: bad flags, no input,
     an unusable --mechanism (bad token, or none+steering), no --lora DIR and
     no adapter on disk for a mechanism that needs one, no --structural-layer/
     --lexical-layer and no calibration file when steering is in play, no
     vector bank for this model/estimator, or a bank that does not cover the
     requested layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import decoding
from lib.adapters import Adapter
from lib.chunking import split_sentences
from lib.cli import die
from lib.metrics import readability
from lib.metrics.faithfulness import faithfulness
from lib.models import load
from lib.steering import SteeredLM, SteeringSpec, load_banks

# Shot selection is evaluate.py's, not reimplemented here -- a second copy of
# "which demonstrations for which shape of input" is exactly the kind of
# divergence lib/bench.py's docstring warns cost real bugs elsewhere.
from src.evaluate import load_shots

# Each retry multiplies whichever dial the mechanism has -- steering alpha,
# LoRA's Adapter.scaled -- by this. 0.7 twice lands near half strength, which
# is far enough to matter and close enough to still be a simplification.
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


@dataclass(frozen=True)
class Mechanism:
    """Which arm(s) render the rewrite, resolved once before generation starts.

    Composable rather than exclusive: report.md measures lora_plus_steered and
    fewshot_plus_steered and finds they do not collide the way the prompted
    instruction and steering do (A/B 2). That one collision is refused in
    parse_mechanism rather than offered; everything else composes freely.
    """

    steering: bool = False
    lora: bool = False
    fewshot: bool = False
    prompted: bool = False  # the bare "simplify" instruction; --mechanism none
    easy_words: bool = False
    lora_dir: Path | None = None
    shots: int = 0

    @property
    def has_dial(self) -> bool:
        """Whether the retry ladder has anything left to turn down."""
        return self.steering or self.lora

    def instruction(self) -> str | None:
        return config.INSTRUCTIONS["simplify"] if self.prompted else None

    def label(self) -> str:
        if self.prompted:
            return "none"
        parts = [
            name
            for name, on in (
                ("steering", self.steering),
                ("lora", self.lora),
                ("fewshot", self.fewshot),
                ("easy_words", self.easy_words),
            )
            if on
        ]
        return "+".join(parts) or "none"


def _same_model(a: str, b: str) -> bool:
    """Same notion of "same model" config.base_task uses: exact id, or a
    shared trailing path segment, so unsloth/... and meta-llama/... (byte-
    identical shards, CLAUDE.md) or a resolved local models/ directory both
    match the repo id that named them."""
    return a == b or a.rstrip("/").split("/")[-1] == b.rstrip("/").split("/")[-1]


def find_lora_dir(repo: str) -> Path | None:
    """The adapter under config.LORA_DIR trained on *repo*, headline arm preferred.

    Several Llama adapters share a base model: asset_r16 is the plain-ASSET
    adapter report.md's A/B 2 measures (+8.47* SARI); asset_ctrl_r16, both_r16
    and onestop_r16* are ablations for A/B 6, 7 and 12 (control tokens, mixed
    corpus, paragraph-length training). Only the headline adapter -- the one
    train_lora.py wrote with no 'corpus' override -- is what auto-detection
    should hand back; the ablations must be named with --lora DIR explicitly.
    """
    if not config.LORA_DIR.exists():
        return None
    candidates: list[tuple[Path, dict]] = []
    for d in sorted(config.LORA_DIR.iterdir()):
        meta_path = d / "train_meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        base = meta.get("base_model")
        if base and _same_model(base, repo):
            candidates.append((d, meta))
    if not candidates:
        return None
    headline = [d for d, meta in candidates if meta.get("corpus") is None]
    return headline[0] if headline else candidates[0][0]


_MECH_TOKENS = ("steering", "lora", "fewshot", "none")


def parse_mechanism(raw: str) -> set[str]:
    tokens = {t.strip() for t in raw.split("+") if t.strip()}
    bad = tokens - set(_MECH_TOKENS)
    if bad:
        die(
            f"--mechanism has unknown token(s) {sorted(bad)}; choose from "
            f"{_MECH_TOKENS}, joined by '+' (e.g. lora+steering)"
        )
    if not tokens:
        die("--mechanism cannot be empty")
    if "none" in tokens and len(tokens) > 1:
        if "steering" in tokens:
            die(
                "--mechanism cannot combine 'none' (the prompted instruction) "
                "with 'steering': report.md's A/B 2 measures exactly that "
                "composition as the single worst arm on both models (-13.50* "
                "SARI, 66% degenerate on Llama; -12.76*, 28% degenerate on "
                "Gemma). Pick one: --mechanism steering (vector only, no "
                "instruction) or --mechanism none (instruction only, no "
                "vector)."
            )
        die(
            "--mechanism 'none' is the prompt-only arm and cannot be combined "
            "with anything else"
        )
    return tokens


def resolve_mechanism(args: argparse.Namespace, repo: str) -> Mechanism:
    """--mechanism, resolved to what actually runs. See the module docstring."""
    if args.mechanism == "auto":
        # Explicit --lora overrides the scan, same as an explicit dial
        # overrides a calibrated --level in resolve_dials below.
        lora_dir = args.lora or find_lora_dir(repo)
        if lora_dir is not None:
            warn(
                f"--mechanism auto -> lora ({lora_dir}): an adapter for "
                f"{repo} is on disk, and it is the stronger measured arm "
                f"(report.md: lora +8.47*/+6.19* SARI vs steering's own "
                f"+0.43 ns/+2.02 ns on Llama/Gemma)"
            )
            mech = Mechanism(lora=True, lora_dir=lora_dir, easy_words=args.easy_words)
        else:
            warn(f"--mechanism auto -> steering: no LoRA adapter for {repo} is on disk")
            mech = Mechanism(steering=True, easy_words=args.easy_words)
        return _check_unused(args, mech)

    tokens = parse_mechanism(args.mechanism)
    mech = Mechanism(
        steering="steering" in tokens,
        lora="lora" in tokens,
        fewshot="fewshot" in tokens,
        prompted="none" in tokens,
        easy_words=args.easy_words,
        lora_dir=args.lora,
        shots=args.shots,
    )
    if mech.lora and mech.lora_dir is None:
        mech = replace(mech, lora_dir=find_lora_dir(repo))
        if mech.lora_dir is None:
            die(
                f"--mechanism {args.mechanism!r} needs an adapter and none "
                f"under {config.LORA_DIR} has base_model matching {repo!r}; "
                f"pass --lora DIR explicitly"
            )
    if mech.fewshot and mech.shots <= 0:
        # 0 is "not overridden", not a request for zero demonstrations --
        # --mechanism fewshot with no demos is a no-op fewshot arm.
        mech = replace(mech, shots=config.FEWSHOT_N)
    return _check_unused(args, mech)


def _check_unused(args: argparse.Namespace, mech: "Mechanism") -> "Mechanism":
    """Flags the resolved mechanism cannot act on. Shared by both paths above:
    ``auto`` used to return before these ran, so ``--level`` under an
    auto-resolved LoRA was accepted and then silently dropped.
    """
    if args.lora and not mech.lora:
        warn(
            f"--lora {args.lora} given but --mechanism {args.mechanism!r} "
            f"has no 'lora'; ignored"
        )
    if args.shots > 0 and not mech.fewshot:
        warn(
            f"--shots {args.shots} given but --mechanism {args.mechanism!r} "
            f"has no 'fewshot'; ignored"
        )
    if args.level and not mech.steering:
        # Refused rather than warned, because a reading level is the one flag
        # here that states an *outcome*. src/calibrate.py maps a target grade to
        # a steering alpha and to nothing else -- LoRA's strength dial was never
        # calibrated against a grade -- so honouring --level under any other
        # mechanism would mean ignoring the only thing the caller asked for.
        die(
            f"--level {args.level!r} is calibrated against steering alpha only, "
            f"and --mechanism {args.mechanism!r} resolved to "
            f"{mech.label()!r}. Either steer:\n"
            f"  --mechanism steering --level {args.level} --structural-layer L\n"
            f"or drop --level and pick the strength directly."
        )
    return mech


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
    shots: list[tuple[str, str]] | None = None,
    logits_processor=None,
) -> str:
    kw = dict(
        instruction=instruction,
        max_new_tokens=max_new_tokens,
        shots=shots,
        logits_processor=logits_processor,
    )
    if chunked:
        return steered.generate_chunked(text, spec, **kw)
    return steered.generate(text, spec, **kw)


def _attempt_label(mech: Mechanism, dials: Dials, lora_scale: float) -> str:
    parts = []
    if mech.steering:
        parts.append(dials.label())
    if mech.lora:
        parts.append(f"lora@{lora_scale:.3g}x")
    if mech.fewshot:
        parts.append(f"fewshot={mech.shots}")
    if mech.easy_words:
        parts.append("easy_words")
    if mech.prompted:
        parts.append("prompted")
    return "+".join(parts) or "none"


def simplify(
    steered: SteeredLM,
    adapter: Adapter,
    text: str,
    mech: Mechanism,
    dials: Dials,
    chunked: bool,
    max_new_tokens: int,
    shots: list[tuple[str, str]],
    logits_processor,
    allow_unverified: bool = False,
) -> Result:
    """Render ``text`` under ``mech``, backing off whichever dial it has until
    the meaning check passes.

    Steering (alpha) and LoRA (``Adapter.scaled``) are both dials and both get
    walked down together when a mechanism composes them, e.g. lora+steering.
    Few-shot and the easy-word bias have no dial -- generation is greedy
    (``config.DO_SAMPLE`` is False), so retrying with nothing changed would
    just reproduce the same output -- so a mechanism built only from those
    goes straight to the prompt-only fallback after one attempt.

    ``adapter.scoring()`` disables the adapter for ``_check`` regardless of
    whether *mech* uses LoRA: see lib/adapters.py's docstring on why an arm
    must never rate its own output.
    """
    attempts: list[Attempt] = []
    cur_dials = dials
    cur_lora_scale = 1.0
    tries = 1 + MAX_RETRIES if mech.has_dial else 1

    for i in range(tries):
        with adapter.at(cur_lora_scale):
            out = _render(
                steered,
                text,
                cur_dials.spec() if mech.steering else SteeringSpec.off(),
                chunked,
                max_new_tokens,
                instruction=mech.instruction(),
                shots=shots if mech.fewshot else None,
                logits_processor=logits_processor if mech.easy_words else None,
            )
        with adapter.scoring():
            attempt = _check(text, out, _attempt_label(mech, cur_dials, cur_lora_scale))
        attempts.append(attempt)
        if attempt.ok:
            return Result(out, mech.label(), attempts, attempt)
        if i < tries - 1:
            cur_dials = cur_dials.scaled(RETRY_SCALE)
            cur_lora_scale *= RETRY_SCALE
            warn(
                f"attempt {i + 1} ({attempt.label}): {attempt.why()}; "
                f"retrying at {RETRY_SCALE:g}x"
            )
        else:
            warn(f"attempt {i + 1} ({attempt.label}): {attempt.why()}")

    # Every mechanism backs off to the same floor. The prompted rewrite is the
    # baseline the whole project is measured against, so it is the right thing
    # to fall back to -- but it gets checked too, and never with the adapter
    # engaged (never fall back to a mechanism-altered string).
    with adapter.engaged(False):
        fallback = _render(
            steered,
            text,
            SteeringSpec.off(),
            chunked,
            max_new_tokens,
            instruction=config.INSTRUCTIONS["simplify"],
        )
    with adapter.scoring():
        checked = _check(text, fallback, "prompt-only")
    attempts.append(checked)
    if checked.ok:
        warn(
            f"{mech.label()} could not preserve meaning at these dials; "
            f"returned the prompt-only simplification instead"
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
        die(
            f"no calibration file at {path}, so --level {level} cannot be "
            f"resolved. Run\n"
            f"  {sys.executable} src/calibrate.py --model {repo} --estimator "
            f"{estimator}\n"
            f"or pass explicit dials and a layer instead, e.g. --structural "
            f"0.75 --structural-layer 15"
        )
    payload = json.loads(path.read_text())
    levels = payload.get("levels", {})
    if level not in levels:
        die(f"{path} has no level {level!r}; calibrated levels are {sorted(levels)}")
    entry = levels[level]
    alphas = entry.get("alphas") or {}
    missing = [a for a in config.AXES if alphas.get(a) is None]
    if missing:
        die(
            f"level {level!r} is not usable: calibration status "
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
    repo: str,
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
    for axis, given, alpha in (
        ("structural", args.structural_layer, structural),
        ("lexical", args.lexical_layer, lexical),
    ):
        layer = given if given is not None else layers.get(axis)
        if layer is None:
            if alpha == 0.0:
                # alpha 0 is an exact no-op (lib/steering.py), so an axis
                # nobody dialled in doesn't need a real layer either -- only a
                # setting that actually steers does. 0 is a placeholder that
                # every bank build_vectors.py produces covers (it estimates
                # every layer by default); check_layer_coverage below only
                # enforces coverage for axes actually in play.
                layer = 0
            else:
                # This used to default to the model's middle layer and print a
                # warning. The middle layer was never swept or measured -- see
                # report.md's layer sweep, which found the effect concentrated
                # at L15 (Llama) / L31 (gemma-4), nowhere near the middle of
                # either stack -- so guessing it produced a worse rewrite
                # silently instead of refusing. Refusing is now the behaviour.
                die(
                    f"no layer for the {axis} axis: pass --{axis}-layer, or "
                    f"run src/calibrate.py first to write a calibration file "
                    f"for this model. The only validated site is structural "
                    f"steering at L15 alpha=0.75 on Llama-3.2-3B-Instruct (L31 "
                    f"alpha=0.5 on gemma-4-12B-it).\n"
                    f"  {sys.executable} src/calibrate.py --model {repo} "
                    f"--estimator {args.estimator}"
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


def check_layer_coverage(banks: dict, dials: Dials, estimator: str, repo: str) -> None:
    """Fail before generation if a resolved layer isn't in its bank.

    Same check as src/evaluate.py's EvalRun.load(): without it, a stale
    calibration file or a hand-typed --structural-layer past what
    build_vectors.py estimated surfaces three retries deep as a bare KeyError
    from inside generate(), not as a message that says what to run.
    """
    for axis, layer, alpha in (
        ("structural", dials.structural_layer, dials.structural),
        ("lexical", dials.lexical_layer, dials.lexical),
    ):
        if alpha == 0.0 or layer in banks[axis].vectors:
            continue
        held = banks[axis].layers
        die(
            f"bank {axis!r} ({estimator}) has no vector for layer {layer}; it "
            f"holds layers {held[0]}..{held[-1]}. Rebuild it with:\n"
            f"  {sys.executable} src/build_vectors.py --model {repo} "
            f"--estimator {estimator}"
        )


def explain(
    source: str, result: Result, mech: Mechanism, dials: Dials, chunked: bool
) -> str:
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
    lines += ["", f"faithfulness    {verdict}", f"mechanism       {mech.label()}"]
    if mech.steering:
        lines.append(f"dials           {dials.label()}")
    if mech.lora:
        lines.append(f"lora            {mech.lora_dir}")
    if mech.fewshot:
        lines.append(f"shots           {mech.shots}")
    lines += [
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
    die("no input; pass --text, --file, or pipe text on stdin")


def run(args: argparse.Namespace) -> int:
    source = read_input(args).strip()
    if not source:
        die("input is empty")

    repo = args.model or config.ACTIVE_MODEL
    mech = resolve_mechanism(args, repo)

    sentences = split_sentences(source)
    chunked = len(sentences) > 1 if args.mode == "auto" else args.mode == "chunked"

    # Setup failures are resolved before the model load, which is the slow part:
    # a missing calibration file, an unresolved layer, a missing bank, a bank
    # that doesn't cover the requested layer, a missing adapter directory and a
    # few-shot corpus that fails to load are all a file read, a dict lookup or
    # (for few-shot) a small download -- not a six-gigabyte weight transfer.
    dials = Dials(0.0, 0.0, 0, 0)
    banks: dict = {}
    if mech.steering:
        alphas, layers, max_safe = lookup_level(args, repo)
        dials = resolve_dials(args, repo, alphas, layers, max_safe)
        try:
            banks = load_banks(repo, args.estimator)
        except FileNotFoundError as exc:
            die(
                f"{exc}\n\nBuild the vector banks first:\n"
                f"  {sys.executable} src/build_vectors.py --model {repo} "
                f"--estimator {args.estimator}"
            )
        check_layer_coverage(banks, dials, args.estimator, repo)

    if mech.lora and not mech.lora_dir.exists():
        die(f"--lora {mech.lora_dir} does not exist; train it with src/train_lora.py")

    shots: list[tuple[str, str]] = []
    if mech.fewshot:
        # Demonstrations have to match the input's shape (load_shots's
        # docstring): chunked mode always renders one sentence at a time, and
        # whole mode on a single-sentence input does too, so both take ASSET's
        # sentence-to-sentence shots; a multi-sentence input rendered whole is
        # rewritten as one passage, which is onestop's shape.
        shot_source = "asset" if chunked or len(sentences) == 1 else "onestop"
        try:
            shots = load_shots(mech.shots, shot_source)
        except Exception as exc:
            die(
                f"could not load {shot_source} demonstrations for "
                f"--mechanism {mech.label()}: {type(exc).__name__}: {exc}\n"
                f"data/ downloads its corpus on first use; check network "
                f"access, or drop 'fewshot' from --mechanism."
            )
        if not shots:
            die(f"no {shot_source} demonstrations were returned; nothing to show the model")

    lm = load(repo)

    adapter = Adapter()
    if mech.lora:
        try:
            adapter = Adapter.load(lm, mech.lora_dir, expect_base=repo)
        except SystemExit as exc:
            die(str(exc))
        warn(f"lora: {adapter.describe()}")

    logits_processor = decoding.build(lm.tokenizer, repo_id=repo) if mech.easy_words else None

    warn(
        f"{len(sentences)} sentence(s), mode={'chunked' if chunked else 'whole'}"
        f" ({args.mode}), mechanism={mech.label()}"
        + (f", dials {dials.label()}" if mech.steering else "")
        + (f", shots={len(shots)}" if mech.fewshot else "")
    )

    result = simplify(
        SteeredLM(lm, banks),
        adapter,
        source,
        mech,
        dials,
        chunked,
        args.max_new_tokens,
        shots,
        logits_processor,
        allow_unverified=args.allow_unverified,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.text + "\n")
        warn(f"wrote {args.out}")
    else:
        print(result.text)

    if args.explain:
        # Diagnostics, not the result -- stdout stays the rewrite and nothing
        # else even in --explain mode, so a caller piping stdout never has to
        # special-case this flag.
        print(explain(source, result, mech, dials, chunked), file=sys.stderr)

    # Every branch of simplify() returns some text -- steered, the prompt-only
    # fallback, the rejected best-effort under --allow-unverified, or the
    # source unchanged -- so reaching here always means a result was produced.
    # Which branch it was is on stderr (and in --explain), not in the exit
    # code: see the module docstring for the three codes this tool uses.
    return 0


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

    p.add_argument(
        "--mechanism",
        default="auto",
        help=(
            "steering | lora | fewshot | none, joined by '+' (e.g. "
            "lora+steering, fewshot+steering). auto (default) picks lora if "
            "an adapter for this model is on disk, else steering. 'none' (the "
            "bare prompted instruction) cannot combine with 'steering' -- see "
            "the module docstring for why."
        ),
    )
    p.add_argument(
        "--lora",
        type=Path,
        default=None,
        help=(
            "adapter directory (lib/adapters.py); auto-detected under "
            "config.LORA_DIR when a mechanism needs lora and this is omitted"
        ),
    )
    p.add_argument(
        "--shots",
        type=int,
        default=0,
        help="in-context demonstrations for --mechanism fewshot (0 -> config.FEWSHOT_N)",
    )
    p.add_argument(
        "--easy-words",
        action="store_true",
        help=(
            "decode-time penalty against hard-word-initial tokens "
            "(lib/decoding.py); composes with any --mechanism"
        ),
    )

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
