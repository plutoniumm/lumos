"""Layer x alpha sweep: where does each axis live, and how hard can it be pushed.

This is the experiment the project turns on. The claim under test is that
structural and lexical simplification are separable by *depth* -- that sentence
structure responds to mid-stack intervention while word choice responds late,
because middle layers carry abstract behavioural features and late layers are
close enough to the unembedding that steering there mostly reweights output
vocabulary. If both axes peak at the same layer with the same profile, the model
encodes one "simplify" direction and the honest move is to ship a single
reading-level dial instead of two.

Two stages, because the full grid is a day of wall clock and most of it is spent
on layers that do nothing:

  stage 1  every layer, one moderate alpha, few texts -> the depth profile
  stage 2  the surviving layers, full alpha grid, more texts, plus NLI

Effects are measured against each text's own *unsteered rewrite*, not against the
source. Both are rewrites, so the difference isolates what steering did rather
than what asking for a rewrite did.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import data
from lib.metrics import readability
from lib.metrics.faithfulness import faithfulness
from lib.models import load
from lib.steering import SteeredLM, SteeringSpec, load_banks


@dataclass(frozen=True)
class Cell:
    stage: int
    axis: str
    layer: int
    alpha: float

    def key(self) -> str:
        return f"{self.stage}|{self.axis}|{self.layer}|{self.alpha:g}"


def _sweep_texts(n_estimate: int, n_sweep: int) -> list[str]:
    """Texts held out from vector estimation."""
    corpus = data.load_complex_corpus(n_estimate + n_sweep)
    if len(corpus) < n_estimate + n_sweep:
        raise RuntimeError(
            f"corpus has {len(corpus)} texts, need {n_estimate + n_sweep} to keep "
            f"the sweep set disjoint from the estimation set"
        )
    return corpus[n_estimate : n_estimate + n_sweep]


def _measure_cell(
    steered: SteeredLM,
    texts: list[str],
    baselines: list[readability.Readability],
    cell: Cell,
    max_new_tokens: int,
    batch_size: int,
    content_orth: bool,
    nli_n: int,
) -> dict:
    spec = SteeringSpec.single(
        cell.axis, cell.layer, cell.alpha, content_orthogonalize=content_orth
    )
    start = time.time()
    outs = steered.generate_batch(
        texts, spec, max_new_tokens=max_new_tokens, batch_size=batch_size
    )
    elapsed = time.time() - start

    measured = [readability.measure(o) for o in outs]

    row = {
        "stage": cell.stage,
        "axis": cell.axis,
        "layer": cell.layer,
        "alpha": cell.alpha,
        "n": len(texts),
        "seconds": round(elapsed, 1),
        "degenerate_frac": sum(m.degenerate for m in measured) / len(measured),
        "empty_frac": sum(m.n_words == 0 for m in measured) / len(measured),
    }

    # Every simplicity metric is gameable by degeneration: collapsed output like
    # "The,, a a the the the" has no difficult words and short "sentences", so it
    # scores as maximally simple. Observed directly during calibration -- L14 at
    # alpha=1.2 showed pct_hard_words down 91% with degenerate_frac 1.00. So
    # simplicity is credited only on outputs that survived the guard, and the
    # number of survivors is reported alongside it.
    keep = [
        (b, m, src, out)
        for b, m, src, out in zip(baselines, measured, texts, outs)
        if not m.degenerate and m.n_words > 0
    ]
    row["n_scored"] = len(keep)

    if not keep:
        row["selectivity"] = None
        row["unusable"] = True
        return row

    deltas = [readability.relative_deltas(b, m) for b, m, _, _ in keep]
    selectivity = [readability.axis_selectivity(b, m, cell.axis) for b, m, _, _ in keep]
    row["selectivity"] = sum(selectivity) / len(selectivity)
    for key in deltas[0]:
        row[key] = sum(d[key] for d in deltas) / len(deltas)
    for key in ("fkgl", "words_per_sentence", "pct_hard_words"):
        row[f"abs_{key}"] = sum(getattr(m, key) for _, m, _, _ in keep) / len(keep)

    if nli_n:
        scores = [faithfulness(src, out).score for _, _, src, out in keep[:nli_n]]
        row["nli_mean"] = sum(scores) / len(scores)
        row["nli_pass_frac"] = sum(
            s >= config.NLI_ENTAILMENT_MIN for s in scores
        ) / len(scores)

    row["samples"] = outs[:2]  # a couple of outputs per cell, for eyeballing
    return row


def _own_effect(row: dict, axis: str) -> float:
    """Fractional reduction on the axis this vector is supposed to control."""
    key = "d_words_per_sentence" if axis == "structural" else "d_pct_hard_words"
    return -row.get(key, 0.0)


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated final line from an interrupted run
        done.add(Cell(r["stage"], r["axis"], r["layer"], r["alpha"]).key())
    return done


def _plan(args: argparse.Namespace, n_layers: int, best: dict | None) -> list[Cell]:
    axes = args.axes
    if args.stage == 1:
        layers = (
            list(range(n_layers))
            if args.layers is None
            else [int(x) for x in args.layers]
        )
        return [
            Cell(1, a, l, alpha)
            for a in axes
            for l in layers
            for alpha in args.coarse_alpha
        ]

    if not best:
        raise RuntimeError(
            "stage 2 needs stage 1 results; run --stage 1 first, then --report"
        )
    cells = []
    for axis in axes:
        for layer in best[axis]["top_layers"][: args.top_k]:
            for alpha in args.alphas:
                cells.append(Cell(2, axis, layer, alpha))
    return cells


def run(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    if args.shard_count > 1:
        out_path = out_path.with_suffix(f".shard{args.shard_index}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lm = load(args.model)
    banks = load_banks(lm.repo_id, args.estimator, tuple(args.axes))
    steered = SteeredLM(lm, banks)
    print(f"[model] {lm.repo_id}  layers={lm.n_layers}  estimator={args.estimator}")

    texts = _sweep_texts(args.n_estimate, args.n)
    print(f"[data] {len(texts)} held-out texts")

    best = None
    if args.stage == 2:
        report_path = Path(args.best_layers or _best_path(lm.repo_id, args.estimator))
        if report_path.exists():
            best = json.loads(report_path.read_text())["axes"]

    cells = _plan(args, lm.n_layers, best)
    if args.shard_count > 1:
        cells = [
            c for i, c in enumerate(cells) if i % args.shard_count == args.shard_index
        ]
    done = _load_done(out_path)
    cells = [c for c in cells if c.key() not in done]
    print(f"[plan] {len(cells)} cells to run ({len(done)} already done)")
    if not cells:
        return 0

    # One unsteered pass, reused as the reference for every cell. Recomputing it
    # per cell would triple the sweep for no information.
    print("[base] generating unsteered references")
    base_outs = steered.generate_batch(
        texts,
        SteeringSpec.off(),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    baselines = [readability.measure(o) for o in base_outs]

    with out_path.open("a") as fh:
        for i, cell in enumerate(cells, 1):
            try:
                row = _measure_cell(
                    steered,
                    texts,
                    baselines,
                    cell,
                    args.max_new_tokens,
                    args.batch_size,
                    args.content_orth,
                    args.nli_n if cell.stage == 2 else 0,
                )
            except Exception as exc:  # a bad cell must not kill a multi-hour sweep
                row = {
                    "stage": cell.stage,
                    "axis": cell.axis,
                    "layer": cell.layer,
                    "alpha": cell.alpha,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  [{i}/{len(cells)}] {cell.key()} ERROR {exc}")
            else:
                head = (
                    f"  [{i}/{len(cells)}] {cell.axis} L{cell.layer} a={cell.alpha:g}"
                )
                if row.get("selectivity") is None:
                    # Every output collapsed, so there is nothing to score. This
                    # is an ordinary sweep outcome, not an error.
                    print(
                        f"{head} UNUSABLE all {row['n']} degenerate "
                        f"({row['seconds']:.0f}s)"
                    )
                else:
                    print(
                        f"{head} sel={row['selectivity']:+.3f} "
                        f"dWPS={row['d_words_per_sentence']:+.3f} "
                        f"dHARD={row['d_pct_hard_words']:+.3f} "
                        f"degen={row['degenerate_frac']:.2f} "
                        f"kept={row['n_scored']}/{row['n']} ({row['seconds']:.0f}s)"
                    )
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return 0


def spawn_workers(args: argparse.Namespace) -> int:
    """Run the sweep across N processes, each with its own model copy."""
    procs = []
    for i in range(args.workers):
        cmd = [
            sys.executable,
            __file__,
            *_forward_args(args),
            f"--shard-index={i}",
            f"--shard-count={args.workers}",
        ]
        print(f"[spawn] worker {i}: {' '.join(cmd[-2:])}")
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]

    merged = Path(args.out)
    with merged.open("a") as out:
        for i in range(args.workers):
            shard = merged.with_suffix(f".shard{i}.jsonl")
            if shard.exists():
                out.write(shard.read_text())
                # Remove after merging, or --report globs both the shard and the
                # merged copy and every cell is counted twice.
                shard.unlink()
    print(f"[merge] -> {merged}  (worker exits: {codes})")
    return max(codes) if codes else 0


def _forward_args(args: argparse.Namespace) -> list[str]:
    out = [
        f"--stage={args.stage}",
        f"--estimator={args.estimator}",
        f"--n={args.n}",
        f"--n-estimate={args.n_estimate}",
        f"--out={args.out}",
        f"--batch-size={args.batch_size}",
        f"--max-new-tokens={args.max_new_tokens}",
        "--coarse-alpha",
        *[str(a) for a in args.coarse_alpha],
        f"--top-k={args.top_k}",
        f"--nli-n={args.nli_n}",
        "--axes",
        *args.axes,
        # Alphas and layers must be forwarded, or stage-2 workers silently fall
        # back to the defaults and sweep a different grid than was asked for.
        "--alphas",
        *[str(a) for a in args.alphas],
    ]
    if args.layers:
        out += ["--layers", *[str(l) for l in args.layers]]
    if args.model:
        out.append(f"--model={args.model}")
    if args.content_orth:
        out.append("--content-orth")
    return out


def _best_path(repo_id: str, estimator: str) -> Path:
    return config.RESULTS_DIR / f"best_layers_{repo_id.split('/')[-1]}_{estimator}.json"


def report(args: argparse.Namespace) -> int:
    """Print the depth profile and write the best (layer, alpha) per axis."""
    # Glob on the stem, not the full name: shards are written as
    # "<stem>.shard0.jsonl", so a "<stem>.jsonl*" pattern matches none of them
    # and the report would silently cover only the merged file.
    base = Path(args.out)
    rows = []
    for path in sorted(base.parent.glob(base.stem + "*.jsonl")):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in r:
                    rows.append(r)
    if not rows:
        print(f"no results at {args.out}")
        return 1

    stage1 = [r for r in rows if r["stage"] == 1]
    axes = sorted({r["axis"] for r in rows})

    if stage1:
        print("\n=== stage 1: depth profile ===")
        print(
            "effect = fractional reduction vs the unsteered rewrite; higher is simpler\n"
        )
        print(
            "cols per axis: wps = sentence-length reduction, "
            "hard = hard-word reduction, sel = selectivity for its own axis\n"
        )
        print(f"{'layer':>5}", end="")
        for axis in axes:
            tag = axis[:4]
            print(f"{tag + '.wps':>11}{tag + '.hard':>11}{tag + '.sel':>11}", end="")
        print(f"{'degen':>8}")
        for layer in sorted({r["layer"] for r in stage1}):
            print(f"{layer:>5}", end="")
            degen = 0.0
            for axis in axes:
                hit = [
                    r
                    for r in stage1
                    if r["axis"] == axis
                    and r["layer"] == layer
                    and r.get("selectivity") is not None
                ]
                if not hit:
                    # No non-degenerate output survived at this layer/alpha.
                    print(f"{'collapsed':>11}{'-':>11}{'-':>11}", end="")
                    continue
                # Best surviving alpha at this layer, since coarse runs several.
                r = max(hit, key=lambda x: x["selectivity"])
                print(
                    f"{-r['d_words_per_sentence']:>+11.3f}"
                    f"{-r['d_pct_hard_words']:>+11.3f}"
                    f"{r['selectivity']:>+11.3f}",
                    end="",
                )
                degen = max(degen, r["degenerate_frac"])
            print(f"{degen:>8.2f}")

    summary = {"axes": {}}
    for axis in axes:
        usable = [
            r
            for r in rows
            if r["axis"] == axis
            and r.get("selectivity") is not None
            and r["degenerate_frac"] <= args.max_degenerate
            # A cell that only scored a couple of survivors has a selectivity
            # estimated on too little to trust, however good it looks.
            and r.get("n_scored", 0) >= max(2, args.min_scored_frac * r["n"])
            and r.get("nli_mean", 1.0) >= args.min_nli
            # Selectivity is own_effect - |other_effect|, which scores a cell that
            # does nothing at all as 0 - 0 = 0 and thereby ranks it above a strong
            # but imperfect one. Without this floor the sweep picked layer 4 for
            # the lexical axis at sel=+0.002, where both effects were noise.
            and _own_effect(r, axis) >= args.min_own_effect
        ]
        if not usable:
            print(f"\n[{axis}] no cell passed the degeneration/NLI gates")
            summary["axes"][axis] = {"top_layers": [], "best": None}
            continue

        by_layer: dict[int, float] = {}
        for r in usable:
            by_layer[r["layer"]] = max(by_layer.get(r["layer"], -9e9), r["selectivity"])
        top = sorted(by_layer, key=lambda l: by_layer[l], reverse=True)
        best = max(usable, key=lambda r: r["selectivity"])

        summary["axes"][axis] = {
            "top_layers": top[: max(args.top_k, 3)],
            "best": {
                "layer": best["layer"],
                "alpha": best["alpha"],
                "selectivity": best["selectivity"],
                "d_words_per_sentence": best["d_words_per_sentence"],
                "d_pct_hard_words": best["d_pct_hard_words"],
                "nli_mean": best.get("nli_mean"),
            },
        }
        b = summary["axes"][axis]["best"]
        print(
            f"\n[{axis}] best L{b['layer']} alpha={b['alpha']:g}  "
            f"sel={b['selectivity']:+.3f}  top layers {top[:5]}"
        )

    # The headline result: do the two axes actually live at different depths?
    picks = {a: summary["axes"][a]["best"] for a in axes if summary["axes"][a]["best"]}
    if {"structural", "lexical"} <= set(picks):
        s, l = picks["structural"]["layer"], picks["lexical"]["layer"]
        print(
            f"\n=== axis separation: structural L{s} vs lexical L{l} "
            f"(gap {l - s:+d} layers) ==="
        )
        if abs(l - s) <= 1:
            print(
                "The two axes peak at effectively the same depth. That is evidence "
                "AGAINST the two-dial hypothesis; prefer a single reading-level dial."
            )

    path = Path(
        args.best_layers
        or _best_path(args.model or config.ACTIVE_MODEL, args.estimator)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n[write] {path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--estimator", default="instruction_diff")
    p.add_argument("--axes", nargs="+", default=list(config.AXES))
    p.add_argument("--stage", type=int, default=1, choices=(1, 2))
    p.add_argument("--layers", nargs="+", default=None)
    p.add_argument("--alphas", nargs="+", type=float, default=list(config.SWEEP_ALPHAS))
    p.add_argument(
        "--coarse-alpha",
        nargs="+",
        type=float,
        default=list(config.SWEEP_COARSE_ALPHAS),
    )
    p.add_argument("--top-k", type=int, default=config.SWEEP_TOP_K_LAYERS)
    p.add_argument("--n", type=int, default=config.SWEEP_COARSE_N_TEXTS)
    p.add_argument("--n-estimate", type=int, default=config.VECTOR_N_TEXTS)
    p.add_argument("--nli-n", type=int, default=config.SWEEP_NLI_SUBSAMPLE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=config.SWEEP_MAX_NEW_TOKENS)
    p.add_argument(
        "--content-orth",
        action="store_true",
        help="enable per-input content orthogonalisation (off in the sweep "
        "so the depth profile is not confounded by it)",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--report", action="store_true")
    p.add_argument("--best-layers", default=None)
    p.add_argument("--max-degenerate", type=float, default=0.15)
    p.add_argument(
        "--min-own-effect",
        type=float,
        default=0.05,
        help="minimum fractional reduction on a vector's own axis "
        "before its selectivity is meaningful",
    )
    p.add_argument(
        "--min-scored-frac",
        type=float,
        default=0.6,
        help="minimum fraction of texts that must survive the "
        "degeneration guard for a cell to be trusted",
    )
    p.add_argument("--min-nli", type=float, default=0.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.out is None:
        slug = (args.model or config.ACTIVE_MODEL).split("/")[-1]
        args.out = str(config.RESULTS_DIR / f"sweep_{slug}_{args.estimator}.jsonl")

    if args.report:
        return report(args)
    if args.workers > 1 and args.shard_count == 1:
        return spawn_workers(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
