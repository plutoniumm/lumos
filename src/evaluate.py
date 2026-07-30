#!/usr/bin/env python
"""The comparison the project's claim stands or falls on.

Six variants over two eval sets, one row of metrics per item, written to JSONL.
The variant set is the whole point of this file and is not optional:

    unsteered              task prompt only. The floor.
    prompted               prompt-only "simplify" instruction. The *honest bar*:
                           if steering does not beat this on the
                           simplicity/faithfulness tradeoff, the project has no
                           result. It is not a courtesy baseline, it is the
                           thing being argued against.
    steered_structural     one axis, best (layer, alpha) from the sweep
    steered_lexical        the other axis
    steered_dual           both dials, the two-dial product configuration
    prompted_plus_steered  instruction *and* steering, to find out whether they
                           compose or fight each other

For the passage eval set every variant is additionally run through
``generate_chunked`` and reported under ``<variant>_chunked``. Which of the two
paths wins is an open question -- whole-passage generation lets a constant
injected vector drift over hundreds of tokens, sentence-by-sentence generation
cannot drift but loses inter-sentence flow -- so both are measured and neither
is assumed.

Baselines and comparability
---------------------------
Every reported delta is *within-item*: an item's readability deltas and its
perplexity ratio are computed against that same item's own unsteered output, in
the same generation mode. Cross-item aggregates of raw FKGL hide the fact that
some sources are simply harder than others; the deltas do not.

The unsteered baseline is therefore generated even when ``unsteered`` is not in
``--variants``, and is cached to ``<out>.baselines.json`` so a resumed run scores
against byte-identical baselines rather than a fresh sample of them.

Perplexity ratios condition the steered output and the baseline output on the
*same* prompt -- the plain unsteered prompt -- because a ratio between two
different conditionings measures the prompts, not the outputs. That means the
``prompted`` variants are scored under a prompt they were not generated from,
which is deliberate and is the only way the ratio stays a fluency measurement.

Interruption
------------
Long MPS runs get killed. Rows are appended and flushed one at a time, and a
re-run skips any (variant, eval_set, item_index) already on disk. Resume keys
assume the underlying data order is stable, which it is as long as ``--n`` only
grows.

A row is only "done" for the metrics it records in its own ``metrics`` field. A
``--skip-nli`` row therefore does not satisfy a later run that wants NLI: it is
regenerated, the thin row stays on disk behind it, and the aggregate keeps the
last row per key. Without that, one cheap smoke run would quietly leave a full
run's output a mixture of scored and unscored items with nothing to show which.

Non-finite floats (an infinite perplexity from an empty generation) are written
as ``null``: JSON has no infinity and a strict parser must not choke on results.

Usage
-----
    python src/evaluate.py --variants unsteered,prompted --n 4 --skip-sari --skip-nli
    python src/evaluate.py --layers-from results/best_layers_<model>_<estimator>.json
    python src/evaluate.py --structural 14,1.5 --lexical 18,2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import config  # noqa: E402
from lib import data, models  # noqa: E402
from lib.metrics import faithfulness as faith_metrics  # noqa: E402
from lib.metrics import fluency, readability  # noqa: E402
from lib.metrics import sari as sari_metrics  # noqa: E402
from lib.steering import SteeredLM, SteeringSpec, load_banks  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    instruction: str | None
    axes: tuple[str, ...]  # () means no steering

    @property
    def needs_banks(self) -> bool:
        return bool(self.axes)


VARIANTS: tuple[Variant, ...] = (
    Variant("unsteered", None, ()),
    Variant("prompted", config.INSTRUCTIONS["simplify"], ()),
    Variant("steered_structural", None, ("structural",)),
    Variant("steered_lexical", None, ("lexical",)),
    Variant("steered_dual", None, ("structural", "lexical")),
    Variant(
        "prompted_plus_steered",
        config.INSTRUCTIONS["simplify"],
        ("structural", "lexical"),
    ),
)
BY_NAME = {v.name: v for v in VARIANTS}

EVAL_SETS = ("asset", "passages")
# ASSET items are single sentences, so chunking them is a no-op; passages get both.
MODES = {"asset": ("whole",), "passages": ("whole", "chunked")}


def build_spec(
    variant: Variant, settings: dict[str, tuple[int, float]]
) -> SteeringSpec:
    if not variant.axes:
        return SteeringSpec.off()
    if len(variant.axes) == 1:
        axis = variant.axes[0]
        layer, alpha = settings[axis]
        return SteeringSpec.single(axis, layer, alpha)
    return SteeringSpec.dual(settings["structural"], settings["lexical"])


def row_variant(variant: Variant, mode: str) -> str:
    return variant.name if mode == "whole" else f"{variant.name}_chunked"


@dataclass
class EvalItem:
    index: int
    source: str
    references: list[str]


def load_items(eval_set: str, n: int, asset_split: str) -> list[EvalItem]:
    """Items for one eval set, from ``lib.data``'s loaders as they are declared."""
    try:
        if eval_set == "asset":
            rows = data.load_asset(asset_split)
            if n > len(rows):
                print(
                    f"  [warn] ASSET {asset_split} holds {len(rows)} items; --n {n} "
                    f"is truncated to that (it is not padded)"
                )
            items = [
                EvalItem(i, r.source.strip(), list(r.references))
                for i, r in enumerate(rows[:n])
            ]
        else:
            passages = data.load_eval_passages(n, config.SEED)
            items = [EvalItem(i, p.strip(), []) for i, p in enumerate(passages)]
    except Exception as exc:
        die(
            f"lib.data could not supply the {eval_set!r} eval set: "
            f"{type(exc).__name__}: {exc}"
        )

    items = [it for it in items if it.source]
    if not items:
        die(f"lib.data returned no usable items for eval set {eval_set!r}")
    return items


# Which metric families a row carries is recorded in the row itself, because
# --skip-nli/--skip-sari rows are otherwise indistinguishable from fully scored
# ones and a cheap smoke run would silently satisfy the resume check for a later
# full run, leaving the file a mixture nothing flags.


def wanted_metrics(item: EvalItem, *, skip_nli: bool, skip_sari: bool) -> set[str]:
    """The metric families a fully scored row for *item* would carry."""
    metrics = {"readability", "ppl"}
    if not skip_nli:
        metrics.add("nli")
    if not skip_sari and item.references:
        metrics.add("sari")
    return metrics


_EXPECTED_SWEEP_SHAPE = (
    '{"axes": {"structural": {"top_layers": [...], '
    '"best": {"layer": 14, "alpha": 1.0, ...}}, "lexical": {...}}}'
)


def parse_layers_from(path: Path) -> dict[str, tuple[int, float]]:
    """Best (layer, alpha) per axis, as ``src/sweep_layers.py --report`` writes it."""
    payload = json.loads(path.read_text())
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
        # degeneration/NLI gates. That is a sweep outcome, not a malformed file.
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
    return out


def parse_pair(text: str, flag: str) -> tuple[int, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        die(f"{flag} wants LAYER,ALPHA (e.g. {flag} 14,1.5), got {text!r}")
    try:
        return int(parts[0]), float(parts[1])
    except ValueError:
        die(f"{flag} wants LAYER,ALPHA with an int layer and float alpha, got {text!r}")


def assert_no_injector(lm) -> None:
    """Fail loudly if a steering hook is still attached."""
    live: list[str] = []
    for i in range(lm.n_layers):
        mods = [lm.layer(i)]
        with suppress(Exception):
            mods.append(lm.mlp(i))
        for mod in mods:
            if getattr(mod, "_forward_hooks", None):
                live.append(f"L{i}/{type(mod).__name__}")
    if live:
        raise RuntimeError(
            "steering hooks still attached while scoring the unsteered reference "
            f"model: {live[:8]}{' ...' if len(live) > 8 else ''}"
        )


def _ppl_ratio(new: float, ref: float) -> float:
    """Mirror of ``fluency.ppl_ratio``'s guard, over a cached denominator."""
    if not math.isfinite(ref) or ref <= 0:
        return float("inf")
    return new / ref


def _finite(x):
    """JSON has no infinity; a strict reader must not have to cope with one."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return _finite(obj)


def read_dict(r: readability.Readability) -> dict:
    """``as_dict`` plus the derived properties ``asdict`` drops."""
    return r.as_dict() | {
        "structural_complexity": r.structural_complexity,
        "lexical_complexity": r.lexical_complexity,
        "degenerate": r.degenerate,
    }


def gen_whole(sl, texts, spec, instruction, max_new_tokens, batch_size):
    """Batched generation; on a batch failure, retry that batch item by item."""
    out: list[str | None] = [None] * len(texts)
    err: list[str | None] = [None] * len(texts)
    sec: list[float] = [0.0] * len(texts)

    for start in range(0, len(texts), batch_size):
        group = texts[start : start + batch_size]
        t0 = time.perf_counter()
        try:
            res = sl.generate_batch(
                group,
                spec,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            dt = time.perf_counter() - t0
            for k, text in enumerate(res):
                out[start + k] = text
                sec[start + k] = dt / max(len(group), 1)
        except Exception as exc:
            print(
                f"      batch at {start} failed ({type(exc).__name__}: {exc}); "
                f"falling back to per-item"
            )
            for k, src in enumerate(group):
                t1 = time.perf_counter()
                try:
                    out[start + k] = sl.generate(
                        src,
                        spec,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as inner:
                    err[start + k] = f"{type(inner).__name__}: {inner}"
                sec[start + k] = time.perf_counter() - t1
        print(f"      generated {min(start + batch_size, len(texts))}/{len(texts)}")
    return out, err, sec


def gen_chunked(sl, texts, spec, instruction, max_new_tokens):
    """Sentence-by-sentence path. Inherently unbatched: each sentence's prompt
    depends on the previous sentence's *output*."""
    out: list[str | None] = [None] * len(texts)
    err: list[str | None] = [None] * len(texts)
    sec: list[float] = [0.0] * len(texts)
    for i, passage in enumerate(texts):
        t0 = time.perf_counter()
        try:
            out[i] = sl.generate_chunked(
                passage, spec, instruction=instruction, max_new_tokens=max_new_tokens
            )
        except Exception as exc:
            err[i] = f"{type(exc).__name__}: {exc}"
        sec[i] = time.perf_counter() - t0
        print(f"      generated {i + 1}/{len(texts)} ({sec[i]:.1f}s)")
    return out, err, sec


def read_done(
    path: Path,
) -> tuple[
    dict[tuple[str, str, int], set[str] | None],
    dict[tuple[str, int], str],
    dict[tuple[str, str, int], float],
    int,
]:
    """What each key on disk already holds, its source and gen_sec, and a bad-line
    count.
    """
    done: dict[tuple[str, str, int], set[str] | None] = {}
    sources: dict[tuple[str, int], str] = {}
    secs: dict[tuple[str, str, int], float] = {}
    bad = 0
    if not path.exists():
        return done, sources, secs, bad
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (row["eval_set"], int(row["item_index"]))
                done[(row["variant"], *key)] = row_metrics(row)
                sources.setdefault(key, row.get("source") or "")
                if isinstance(row.get("gen_sec"), (int, float)):
                    secs[(row["variant"], *key)] = float(row["gen_sec"])
            except Exception:
                bad += 1
    return done, sources, secs, bad


def row_metrics(row: dict) -> set[str] | None:
    """The metric families a row carries; ``None`` for a captured failure."""
    if row.get("error"):
        return None
    return set(row.get("metrics") or ())


def is_done(
    done: dict[tuple[str, str, int], set[str] | None],
    key: tuple[str, str, int],
    wanted: set[str],
) -> bool:
    """A key is done only if its row already carries every metric now wanted."""
    have = done.get(key, set())
    return have is None or wanted <= have


def check_resume_alignment(items: list[EvalItem], eval_set: str, known: dict) -> None:
    drift = [
        it.index
        for it in items
        if (eval_set, it.index) in known
        and known[(eval_set, it.index)].strip() != it.source.strip()
    ]
    if drift:
        print(
            f"  [warn] {len(drift)} item(s) in {eval_set} no longer match the source "
            f"recorded on disk (indices {drift[:5]}). The corpus or its seed changed; "
            f"resumed rows are NOT comparable to new ones. Delete the output file "
            f"and re-run for a clean set."
        )


def read_rows(path: Path) -> tuple[list[dict], int]:
    """Rows for the aggregate, last row per key winning, plus how many it replaced."""
    latest: dict[tuple[str, str, int], dict] = {}
    superseded = 0
    if not path.exists():
        return [], superseded
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            with suppress(Exception):
                row = json.loads(line)
                key = (row["variant"], row["eval_set"], int(row["item_index"]))
                superseded += key in latest
                latest[key] = row
    return list(latest.values()), superseded


def summary_path_for(out: Path) -> Path:
    stem = out.stem
    if stem.startswith("eval_"):
        return out.parent / f"eval_summary_{stem[len('eval_') :]}.json"
    return out.parent / f"{stem}_summary.json"


def die(message: str) -> None:
    print(f"\nFATAL: {message}\n", file=sys.stderr)
    raise SystemExit(2)


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _pct(flags: list[bool]) -> float | None:
    vals = [bool(f) for f in flags if f is not None]
    return 100.0 * sum(vals) / len(vals) if vals else None


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["eval_set"], row["variant"]), []).append(row)

    order = {name: i for i, name in enumerate(BY_NAME)}

    def sort_key(key):
        eval_set, variant = key
        base = variant[: -len("_chunked")] if variant.endswith("_chunked") else variant
        return (
            EVAL_SETS.index(eval_set) if eval_set in EVAL_SETS else 99,
            order.get(base, 99),
            variant.endswith("_chunked"),
        )

    out = []
    for key in sorted(groups, key=sort_key):
        eval_set, variant = key
        group = groups[key]
        ok = [r for r in group if not r.get("error")]
        read = [r.get("readability") or {} for r in ok]
        faith = [r.get("faithfulness") for r in ok]
        out.append(
            {
                "eval_set": eval_set,
                "variant": variant,
                "mode": (group[0].get("mode") or "whole"),
                "n": len(group),
                "n_failed": len(group) - len(ok),
                "spec_label": group[0].get("spec_label"),
                "mean_fkgl": _mean([r.get("fkgl") for r in read]),
                "mean_words_per_sentence": _mean(
                    [r.get("words_per_sentence") for r in read]
                ),
                "mean_pct_hard_words": _mean([r.get("pct_hard_words") for r in read]),
                "mean_d_words_per_sentence": _mean(
                    [(r.get("deltas") or {}).get("d_words_per_sentence") for r in ok]
                ),
                "mean_d_pct_hard_words": _mean(
                    [(r.get("deltas") or {}).get("d_pct_hard_words") for r in ok]
                ),
                "mean_d_fkgl": _mean(
                    [(r.get("deltas") or {}).get("d_fkgl") for r in ok]
                ),
                "mean_nli_score": _mean([(f or {}).get("score") for f in faith]),
                "pct_nli_passed": _pct([r.get("nli_passed") for r in ok]),
                "mean_ppl": _mean([r.get("ppl") for r in ok]),
                "mean_ppl_ratio": _mean([r.get("ppl_ratio") for r in ok]),
                "mean_sari": _mean([(r.get("sari") or {}).get("sari") for r in ok]),
                "pct_degenerate": _pct([r.get("degenerate") for r in read]),
                "mean_gen_sec": _mean([r.get("gen_sec") for r in ok]),
                # Wall clock attributable to this cell, summed from the rows, so it
                # survives resume instead of resetting to zero on a skipped run.
                "gen_seconds_total": sum(
                    r["gen_sec"]
                    for r in group
                    if isinstance(r.get("gen_sec"), (int, float))
                ),
            }
        )
    return out


_COLUMNS = (
    ("eval_set", "eval_set", 9, None),
    ("variant", "variant", 26, None),
    ("n", "n", 4, "d"),
    ("fail", "n_failed", 5, "d"),
    ("FKGL", "mean_fkgl", 7, "5.2f"),
    ("w/sent", "mean_words_per_sentence", 7, "6.2f"),
    ("hard", "mean_pct_hard_words", 7, "6.3f"),
    ("dFKGL", "mean_d_fkgl", 7, "+6.2f"),
    ("NLI", "mean_nli_score", 6, "5.3f"),
    ("gate%", "pct_nli_passed", 6, "5.1f"),
    ("pplR", "mean_ppl_ratio", 6, "5.2f"),
    ("SARI", "mean_sari", 7, "6.2f"),
    ("degen%", "pct_degenerate", 7, "6.1f"),
    ("gen_s", "gen_seconds_total", 8, "7.1f"),
)


def print_summary(agg: list[dict]) -> None:
    header = "".join(f"{title:<{width}}" for title, _, width, _ in _COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    last_set = None
    for row in agg:
        if last_set is not None and row["eval_set"] != last_set:
            print("-" * len(header))
        last_set = row["eval_set"]
        cells = []
        for _, key, width, fmt in _COLUMNS:
            val = row.get(key)
            if val is None:
                cells.append(f"{'-':<{width}}")
            elif fmt is None:
                cells.append(f"{str(val):<{width}}")
            else:
                cells.append(f"{format(val, fmt):<{width}}")
        print("".join(cells))
    print("-" * len(header))
    print(
        "FKGL/w-sent/hard: absolute readability of the output.  dFKGL: fractional "
        "change vs that item's own unsteered output (negative = simpler)."
    )
    print(
        "NLI: bidirectional entailment score.  gate%: share passing "
        f"NLI>={config.NLI_ENTAILMENT_MIN}.  pplR: conditional ppl vs unsteered "
        f"(reject >{config.PPL_RATIO_MAX})."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    p.add_argument("--estimator", default="instruction_diff")
    p.add_argument("--n", type=int, default=64, help="items per eval set")
    p.add_argument(
        "--variants",
        default="all",
        help=f"comma-separated subset of {','.join(BY_NAME)} (default all)",
    )
    p.add_argument("--layers-from", type=Path, help="sweep JSON of best (layer, alpha)")
    p.add_argument("--structural", help="manual override, LAYER,ALPHA")
    p.add_argument("--lexical", help="manual override, LAYER,ALPHA")
    p.add_argument("--out", type=Path)
    p.add_argument("--skip-sari", action="store_true")
    p.add_argument("--skip-nli", action="store_true")
    # Operational knobs; every default reproduces the full evaluation.
    p.add_argument(
        "--eval-sets",
        default=",".join(EVAL_SETS),
        help=f"subset of {','.join(EVAL_SETS)}",
    )
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--asset-split", default="test")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    # Redirected stdout is block-buffered, which is exactly wrong here: a run that
    # takes hours and is expected to be killed would show no progress while it
    # worked and would lose the whole buffer when it died. The JSONL rows are
    # flushed per row for the same reason.
    with suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config.SEED)

    names = (
        list(BY_NAME)
        if args.variants == "all"
        else [s.strip() for s in args.variants.split(",") if s.strip()]
    )
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        die(f"unknown variant(s) {unknown}; choose from {list(BY_NAME)}")
    chosen = [BY_NAME[n] for n in names]

    eval_sets = [s.strip() for s in args.eval_sets.split(",") if s.strip()]
    bad_sets = [s for s in eval_sets if s not in EVAL_SETS]
    if bad_sets:
        die(f"unknown eval set(s) {bad_sets}; choose from {list(EVAL_SETS)}")

    settings: dict[str, tuple[int, float]] = {}
    if args.layers_from:
        if not args.layers_from.exists():
            die(f"--layers-from {args.layers_from} does not exist")
        settings.update(parse_layers_from(args.layers_from))
    if args.structural:
        settings["structural"] = parse_pair(args.structural, "--structural")
    if args.lexical:
        settings["lexical"] = parse_pair(args.lexical, "--lexical")

    needed_axes = sorted({a for v in chosen if v.needs_banks for a in v.axes})
    missing_axes = [a for a in needed_axes if a not in settings]
    if missing_axes:
        die(
            f"variants {[v.name for v in chosen if v.needs_banks]} need a "
            f"(layer, alpha) for {missing_axes} and none was given.\n"
            f"Pass --layers-from <sweep json>, or "
            + " ".join(f"--{a} LAYER,ALPHA" for a in missing_axes)
            + ".\nOr evaluate only the bank-free variants:\n"
            f"  {sys.executable} src/evaluate.py --variants unsteered,prompted"
        )

    slug = args.model.split("/")[-1]
    out = args.out or (config.RESULTS_DIR / f"eval_{slug}_{args.estimator}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_out = summary_path_for(out)
    baseline_cache_path = out.with_suffix(out.suffix + ".baselines.json")

    done, known_sources, known_secs, bad_lines = read_done(out)
    if bad_lines:
        print(f"[resume] ignored {bad_lines} unparseable line(s) in {out}")
    if done:
        print(f"[resume] {len(done)} row(s) already in {out}; those will be skipped")

    baselines: dict[str, str] = {}
    if baseline_cache_path.exists():
        with suppress(Exception):
            baselines = json.loads(baseline_cache_path.read_text())
        print(f"[resume] {len(baselines)} cached baseline output(s)")
    # Baseline generation happens in its own phase, so its cost has to be carried
    # over to the unsteered rows or the wall-clock column would report the floor
    # variant as free. Not cached: a resumed row keeps the gen_sec it was written
    # with, and only freshly generated baselines need attributing.
    baseline_secs: dict[str, float] = {}

    # Deliberately ahead of models.load: the banks are a filesystem check costing
    # milliseconds, and "you have not built the vectors yet" is not worth waiting
    # through six gigabytes of weights and an MPS transfer to be told.
    banks: dict = {}
    if needed_axes:
        try:
            banks = load_banks(args.model, args.estimator)
        except FileNotFoundError as exc:
            die(
                f"{exc}\n\nBuild the vector banks first:\n"
                f"  {sys.executable} src/build_vectors.py "
                f"--model {args.model} --estimator {args.estimator}\n\n"
                f"Or evaluate only the variants that need no banks:\n"
                f"  {sys.executable} src/evaluate.py --model {args.model} "
                f"--variants unsteered,prompted"
            )
        for axis in needed_axes:
            layer = settings[axis][0]
            if layer not in banks[axis].vectors:
                held = banks[axis].layers
                die(
                    f"bank {axis!r} ({args.estimator}) has no vector for layer "
                    f"{layer}; it holds layers {held[0]}..{held[-1]}"
                )
    # What the loaded banks *are*, taken from their own metadata. config's
    # ORTHOGONALIZE_AXES is build_vectors.py's default and says nothing about the
    # file this run opened, so it is not recorded anywhere as if it did.
    bank_orth = {axis: bank.orthogonalized for axis, bank in banks.items()}

    print(f"\n[load] {args.model} on {config.DEVICE} ({config.DTYPE})")
    t_load = time.perf_counter()
    lm = models.load(args.model)
    print(
        f"[load] {lm.n_layers} layers, hidden {lm.hidden_size}, "
        f"{time.perf_counter() - t_load:.1f}s"
    )

    try:
        sl = SteeredLM(lm, banks)
    except ValueError as exc:
        die(
            f"{exc}\n\nRebuild the banks against this model:\n"
            f"  {sys.executable} src/build_vectors.py "
            f"--model {args.model} --estimator {args.estimator}"
        )

    print(f"[cfg]  estimator={args.estimator} settings={settings or '(none needed)'}")
    print(f"[cfg]  banks orthogonalized={bank_orth or '(no banks loaded)'}")
    print(f"[cfg]  variants={[v.name for v in chosen]}")
    print(f"[cfg]  eval_sets={eval_sets} n={args.n} data=lib.data")
    print(f"[cfg]  max_new_tokens={args.max_new_tokens} batch_size={args.batch_size}")
    print(
        f"[cfg]  nli={'off' if args.skip_nli else 'on'} sari={'off' if args.skip_sari else 'on'}"
    )
    print(f"[out]  {out}")

    n_written = 0
    n_failed = 0
    timings: dict[str, float] = {}
    interrupted = False
    fh = out.open("a", encoding="utf-8")

    try:
        for eval_set in eval_sets:
            items = load_items(eval_set, args.n, args.asset_split)
            print(f"\n[{eval_set}] {len(items)} item(s) from lib.data")
            check_resume_alignment(items, eval_set, known_sources)

            for mode in MODES[eval_set]:
                keys = [f"{eval_set}|{mode}|{it.index}" for it in items]
                todo = [i for i, k in enumerate(keys) if k not in baselines]
                if todo:
                    print(
                        f"  [baseline {mode}] generating {len(todo)} unsteered output(s)"
                    )
                    texts = [items[i].source for i in todo]
                    off = SteeringSpec.off()
                    if mode == "whole":
                        outs, errs, bsecs = gen_whole(
                            sl, texts, off, None, args.max_new_tokens, args.batch_size
                        )
                    else:
                        outs, errs, bsecs = gen_chunked(
                            sl, texts, off, None, args.max_new_tokens
                        )
                    for j, i in enumerate(todo):
                        baseline_secs[keys[i]] = bsecs[j]
                        if errs[j] is None and outs[j] is not None:
                            baselines[keys[i]] = outs[j]
                    baseline_cache_path.write_text(json.dumps(baselines, indent=0))

                for variant in chosen:
                    rv = row_variant(variant, mode)
                    spec = build_spec(variant, settings)
                    pending = [
                        it
                        for it in items
                        if not is_done(
                            done,
                            (rv, eval_set, it.index),
                            wanted_metrics(
                                it, skip_nli=args.skip_nli, skip_sari=args.skip_sari
                            ),
                        )
                    ]
                    if not pending:
                        print(
                            f"  [{rv}] all {len(items)} row(s) already present; skipped"
                        )
                        continue

                    thin = [it for it in pending if (rv, eval_set, it.index) in done]
                    if thin:
                        print(
                            f"  [{rv}] {len(thin)} row(s) on disk are missing a metric "
                            f"this run asks for and will be redone; the thin rows stay "
                            f"in the file and the summary keeps the last row per key"
                        )
                    print(
                        f"  [{rv}] {len(pending)}/{len(items)} item(s) "
                        f"spec={spec.label()} instruction="
                        f"{'simplify' if variant.instruction else 'none'}"
                    )
                    t_variant = time.perf_counter()

                    # The unsteered variant *is* the baseline. Reusing it instead of
                    # regenerating saves a full pass and makes its own ppl_ratio
                    # exactly 1.0, which is a free self-consistency check on the row.
                    if variant.name == "unsteered":
                        outs = [
                            baselines.get(f"{eval_set}|{mode}|{it.index}")
                            for it in pending
                        ]
                        errs = [
                            None if o is not None else "baseline generation failed"
                            for o in outs
                        ]
                        # Freshly generated baselines carry their real cost; a row
                        # being rewritten for a new metric keeps the cost recorded
                        # the first time, since the text was not generated again.
                        secs = [
                            baseline_secs.get(
                                f"{eval_set}|{mode}|{it.index}",
                                known_secs.get((rv, eval_set, it.index), 0.0),
                            )
                            for it in pending
                        ]
                    elif mode == "whole":
                        outs, errs, secs = gen_whole(
                            sl,
                            [it.source for it in pending],
                            spec,
                            variant.instruction,
                            args.max_new_tokens,
                            args.batch_size,
                        )
                    else:
                        outs, errs, secs = gen_chunked(
                            sl,
                            [it.source for it in pending],
                            spec,
                            variant.instruction,
                            args.max_new_tokens,
                        )

                    assert_no_injector(lm)
                    base_ppl_cache: dict[str, float] = {}

                    for it, output, err, sec in zip(pending, outs, errs, secs):
                        bkey = f"{eval_set}|{mode}|{it.index}"
                        baseline = baselines.get(bkey)
                        row = {
                            "variant": rv,
                            "eval_set": eval_set,
                            "mode": mode,
                            "item_index": it.index,
                            "model": args.model,
                            "estimator": args.estimator,
                            "data_source": "lib.data",
                            "spec": spec.as_dict(),
                            "spec_label": spec.label(),
                            # From the banks this row actually steered with, not
                            # from config: an "_orth" filename is not evidence.
                            "bank_orthogonalized": {
                                axis: bank_orth[axis] for axis in variant.axes
                            },
                            "instruction": variant.instruction,
                            "source": it.source,
                            "n_references": len(it.references),
                            "output": output,
                            "baseline_output": baseline,
                            "baseline_variant": f"unsteered/{mode}",
                            "gen_sec": sec,
                            "error": err,
                        }
                        if err is None:
                            try:
                                row.update(
                                    score_item(
                                        lm,
                                        it,
                                        output,
                                        baseline,
                                        base_ppl_cache,
                                        bkey,
                                        skip_nli=args.skip_nli,
                                        skip_sari=args.skip_sari,
                                    )
                                )
                            except Exception as exc:
                                row["error"] = f"scoring: {type(exc).__name__}: {exc}"
                        if row["error"]:
                            n_failed += 1
                            print(f"      item {it.index} FAILED: {row['error']}")
                        fh.write(json.dumps(_clean(row), allow_nan=False) + "\n")
                        fh.flush()
                        done[(rv, eval_set, it.index)] = row_metrics(row)
                        n_written += 1

                    timings[f"{eval_set}/{rv}"] = time.perf_counter() - t_variant
                    print(f"  [{rv}] {timings[f'{eval_set}/{rv}']:.1f}s wall clock")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] rows written so far are on disk; re-run to resume")
    finally:
        fh.close()

    rows, superseded = read_rows(out)
    agg = aggregate(rows)
    print_summary(agg)

    if timings:
        print("\nwall clock this run:")
        for key, secs in timings.items():
            print(f"  {key:<44} {secs:8.1f}s")

    print(
        f"\nrows written this run: {n_written}   failures this run: {n_failed}   "
        f"rows in file: {len(rows)}"
        + (f"   superseded by a later row: {superseded}" if superseded else "")
    )

    summary_out.write_text(
        json.dumps(
            _clean(
                {
                    "model": args.model,
                    "estimator": args.estimator,
                    "data_source": "lib.data",
                    # Straight off the loaded banks' own metadata.
                    "bank_orthogonalized": bank_orth,
                    "n_requested": args.n,
                    "variants": [v.name for v in chosen],
                    "eval_sets": eval_sets,
                    "settings": {k: list(v) for k, v in settings.items()},
                    "skip_nli": args.skip_nli,
                    "skip_sari": args.skip_sari,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "interrupted": interrupted,
                    "rows_written_this_run": n_written,
                    "failures_this_run": n_failed,
                    "rows_in_file": len(rows),
                    "rows_superseded": superseded,
                    "wall_clock_this_run": timings,
                    "config": {
                        "positions": config.DEFAULT_POSITIONS,
                        "norm_preserve": config.NORM_PRESERVE,
                        "site_kind": config.DEFAULT_SITE_KIND,
                        "content_orthogonalize": config.CONTENT_ORTHOGONALIZE,
                        "nli_entailment_min": config.NLI_ENTAILMENT_MIN,
                        "ppl_ratio_max": config.PPL_RATIO_MAX,
                    },
                    "rows": agg,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    print(f"\nsummary -> {summary_out}")
    print(f"rows    -> {out}")
    return 1 if interrupted else 0


def score_item(
    lm,
    item: EvalItem,
    output: str,
    baseline: str | None,
    base_ppl_cache: dict[str, float],
    bkey: str,
    *,
    skip_nli: bool,
    skip_sari: bool,
) -> dict:
    """Every metric for one output. No steering hooks are live in here."""
    read = readability.measure(output)
    patch: dict = {"readability": read_dict(read)}
    metrics = ["readability", "ppl"]

    if baseline is not None:
        base_read = readability.measure(baseline)
        patch["baseline_readability"] = read_dict(base_read)
        patch["deltas"] = readability.relative_deltas(base_read, read)

    # conditional_ppl is the unsteered reference model by definition; see
    # assert_no_injector, called once before this scoring phase begins.
    prompt = lm.build_prompt(item.source, None, None)
    ppl = fluency.conditional_ppl(lm, prompt, output)
    patch["ppl"] = ppl
    if baseline is not None:
        if bkey not in base_ppl_cache:
            base_ppl_cache[bkey] = fluency.conditional_ppl(lm, prompt, baseline)
        patch["baseline_ppl"] = base_ppl_cache[bkey]
        patch["ppl_ratio"] = _ppl_ratio(ppl, base_ppl_cache[bkey])
        patch["ppl_ratio_ok"] = (
            math.isfinite(patch["ppl_ratio"])
            and patch["ppl_ratio"] <= config.PPL_RATIO_MAX
        )

    if not skip_nli:
        f = faith_metrics.faithfulness(item.source, output)
        patch["faithfulness"] = f.as_dict()
        patch["nli_passed"] = f.passed()
        metrics.append("nli")

    if not skip_sari and item.references:
        patch["sari"] = sari_metrics.sari(
            item.source, output, item.references
        ).as_dict()
        metrics.append("sari")

    patch["metrics"] = metrics
    return patch


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
