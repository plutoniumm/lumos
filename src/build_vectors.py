"""Estimate and save the steering vector banks.

Three estimators, described in ``lib/vectors.py``. Two produce one bank per axis
(structural, lexical); the OneStopEnglish variant produces a single conflated
"simplify" bank that exists specifically as the control -- if the monolithic
vector matches the two-axis pair, the two-dial design buys nothing and should be
dropped in favour of one reading-level dial.

The orthogonalised variants are written alongside the raw ones rather than
replacing them, so the sweep can measure whether Gram-Schmidt actually helped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import data, vectors
from lib import jsonio
from lib.models import load
from lib.steering import SteeredLM, SteeringSpec

SYNTHETIC_CACHE = config.DATA_DIR / "synthetic_pairs.json"


def _synthetic_pairs(
    lm, axis: str, texts: list[str], batch_size: int
) -> tuple[list[str], list[str]]:
    """Axis-isolated pairs written by the model itself, prompted per axis."""
    cache: dict = {}
    if SYNTHETIC_CACHE.exists():
        cache = json.loads(SYNTHETIC_CACHE.read_text())

    key = f"{axis}:{len(texts)}"
    if key in cache and cache[key]["sources"] == texts:
        print(f"[pairs] {axis}: cache hit ({len(texts)} pairs)")
        return cache[key]["rewrites"], texts

    print(f"[pairs] {axis}: generating {len(texts)} rewrites")
    steered = SteeredLM(lm, {})
    start = time.time()
    # The sources are single sentences, so the default 320-token budget is mostly
    # spent generating trailing commentary we then average into the vector.
    rewrites = steered.generate_batch(
        texts,
        SteeringSpec.off(),
        instruction=config.INSTRUCTIONS[axis],
        max_new_tokens=config.PAIR_MAX_NEW_TOKENS,
        batch_size=batch_size,
    )
    print(f"[pairs] {axis}: {time.time() - start:.0f}s")

    cache[key] = {"sources": texts, "rewrites": rewrites}
    SYNTHETIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    jsonio.dump(SYNTHETIC_CACHE, cache)
    return rewrites, texts


def build(args: argparse.Namespace) -> int:
    lm = load(args.model)
    print(f"[model] {lm.repo_id}  layers={lm.n_layers}  hidden={lm.hidden_size}")

    texts = data.load_complex_corpus(args.n)
    print(f"[data] {len(texts)} complex texts for estimation")

    # Estimating from ASSET-domain text instead was tried and changed nothing --
    # every metric within noise, axis entanglement 0.220 against 0.223. See
    # fail.md section 3. The direction is not a property of the estimation pool.

    banks: dict[str, vectors.VectorBank] = {}

    if args.estimator == "instruction_diff":
        for axis in args.axes:
            placebo = config.PLACEBOS[axis] if args.contrast == "placebo" else None
            print(f"[build] instruction_diff / {axis} (contrast={args.contrast})")
            banks[axis] = vectors.instruction_diff(
                lm, texts, config.INSTRUCTIONS[axis], axis, placebo
            )

    elif args.estimator in ("caa", "diff_pca"):
        fn = vectors.caa if args.estimator == "caa" else vectors.diff_pca
        if args.pairs == "synthetic":
            for axis in args.axes:
                simple, cplx = _synthetic_pairs(lm, axis, texts, args.batch_size)
                print(f"[build] {args.estimator} / {axis}")
                banks[axis] = fn(lm, simple, cplx, axis)
        elif args.pairs == "onestop":
            items = data.load_onestop()[: args.n]
            print(f"[build] {args.estimator} / simplify (monolithic control)")
            banks["simplify"] = fn(
                lm,
                [i.elementary for i in items],
                [i.advanced for i in items],
                "simplify",
            )
        else:
            raise ValueError(f"unknown --pairs {args.pairs!r}")
    else:
        raise ValueError(f"unknown --estimator {args.estimator!r}")

    for name, bank in banks.items():
        path = bank.save()
        print(f"[save] {name} -> {path}")

    # Orthogonalise only when both axes exist; the monolithic control has no
    # counterpart to be orthogonal to.
    if args.orthogonalize and {"structural", "lexical"} <= set(banks):
        report = vectors.entanglement_report(banks["structural"], banks["lexical"])
        worst = max(report.items(), key=lambda kv: abs(kv[1]))
        print(
            f"[entangle] mean |cos| = "
            f"{sum(abs(v) for v in report.values()) / len(report):.3f}, "
            f"worst layer {worst[0]} at {worst[1]:+.3f}"
        )
        if abs(worst[1]) > 0.9:
            print(
                "[entangle] WARNING: the two axis vectors are nearly collinear. "
                "The model likely encodes one 'simplify' direction, not two; "
                "expect the two-dial design to collapse to a single dial."
            )
        # Löwdin, not pairwise Gram-Schmidt. Applying Gram-Schmidt in both
        # directions leaves the pair anti-correlated rather than orthogonal --
        # measured here as +0.500 becoming -0.504.
        orth_banks = vectors.symmetric_orthogonalize(
            {a: banks[a] for a in ("structural", "lexical")}
        )
        for axis, orth in orth_banks.items():
            orth.estimator = f"{args.estimator}_orth"
            print(f"[save] {axis} (orth) -> {orth.save()}")
        residual = orth_banks["structural"].cosine(
            orth_banks["lexical"], sorted(orth_banks["structural"].vectors)[0]
        )
        print(f"[orth] residual cos after Löwdin: {residual:+.2e} (should be ~0)")

        vectors.write_report(
            config.RESULTS_DIR / f"entanglement_{lm.repo_id.split('/')[-1]}"
            f"_{args.estimator}.json",
            {
                "model": lm.repo_id,
                "estimator": args.estimator,
                "n_texts": len(texts),
                "cosine_per_layer": {str(k): v for k, v in report.items()},
            },
        )

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument(
        "--estimator",
        default="instruction_diff",
        choices=("instruction_diff", "caa", "diff_pca"),
    )
    p.add_argument("--axes", nargs="+", default=list(config.AXES))
    p.add_argument("--n", type=int, default=config.VECTOR_N_TEXTS)
    p.add_argument(
        "--pairs",
        default="synthetic",
        choices=("synthetic", "onestop"),
        help="pair source for caa/diff_pca",
    )
    p.add_argument(
        "--contrast",
        default=config.DEFAULT_CONTRAST,
        choices=("placebo", "none"),
        help="what instruction_diff differences against; 'none' is known to give "
        "near-collinear axis vectors and exists for comparison",
    )
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--no-orthogonalize", dest="orthogonalize", action="store_false")
    p.set_defaults(orthogonalize=config.ORTHOGONALIZE_AXES)
    return build(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
