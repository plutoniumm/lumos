#!/usr/bin/env python
"""The core comparison: unsteered vs. prompted vs. steered, scored and written to JSONL.

Six variants: unsteered, prompted, steered_structural, steered_lexical,
steered_dual, prompted_plus_steered. Lora, fewshot and easy_words variants
run only when named explicitly. Passage items also run through
generate_chunked as <variant>_chunked.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from lib import data, models  # noqa: E402
from lib.metrics import faithfulness as faith_metrics  # noqa: E402
from lib.metrics import fluency, readability  # noqa: E402
from lib.metrics import sari as sari_metrics  # noqa: E402
from lib import cleanup, decoding, jsonio, tables  # noqa: E402
from lib.adapters import Adapter  # noqa: E402
from lib.cli import AxisSettings, die  # noqa: E402
from lib.steering import SteeredLM, SteeringSpec, load_banks  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    instruction: str | None
    axes: tuple[str, ...]
    adapter: bool = False
    shots: int = 0
    easy_words: bool = False

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

LORA_VARIANTS: tuple[Variant, ...] = (
    Variant("lora", None, (), adapter=True),
    Variant("lora_plus_steered", None, ("structural",), adapter=True),
)

FEWSHOT_VARIANTS: tuple[Variant, ...] = (
    Variant("fewshot", None, (), shots=config.FEWSHOT_N),
    Variant(
        "fewshot_plus_steered",
        None,
        ("structural",),
        shots=config.FEWSHOT_N,
    ),
)

CONSTRAINED_VARIANTS: tuple[Variant, ...] = (
    Variant("easy_words", None, (), easy_words=True),
    Variant("easy_words_plus_steered", None, ("structural",), easy_words=True),
)

BY_NAME = {v.name: v for v in VARIANTS}
ALL_BY_NAME = {
    v.name: v
    for v in VARIANTS + LORA_VARIANTS + FEWSHOT_VARIANTS + CONSTRAINED_VARIANTS
}


def load_shots(n: int, source: str = "asset") -> list[tuple[str, str]]:
    """Demonstrations for the few-shot variants, never drawn from the split
    being evaluated. ``asset`` serves sentence-level evals, ``onestop``
    serves the paragraph-length science abstracts.
    """
    if n <= 0:
        return []
    if source == "onestop":
        return _onestop_shots(n)
    return _asset_shots(n)


def _asset_shots(n: int) -> list[tuple[str, str]]:
    """ASSET validation sentences, each paired with its median-length reference."""
    pool = data.load_asset("valid")
    rng = random.Random(config.SEED)
    picked = rng.sample(range(len(pool)), min(n * 4, len(pool)))
    shots: list[tuple[str, str]] = []
    for i in picked:
        item = pool[i]
        source = item.source.strip()
        refs = [r.strip() for r in item.references if r.strip() and r.strip() != source]
        if not source or not refs:
            continue
        shots.append((source, sorted(refs, key=len)[len(refs) // 2]))
        if len(shots) == n:
            break
    return shots


def _onestop_shots(n: int) -> list[tuple[str, str]]:
    """OneStopEnglish advanced-to-elementary paragraphs, filtered for real rewrites."""
    lo, hi = config.SCIENCE_ABSTRACT_WORDS
    shots: list[tuple[str, str]] = []
    for triple in data.load_onestop():
        adv, ele = triple.advanced.strip(), triple.elementary.strip()
        n_adv, n_ele = len(adv.split()), len(ele.split())
        if not lo <= n_adv <= hi or n_ele < 20:
            continue
        ratio = n_ele / n_adv
        if not config.SHOT_COMPRESSION_RANGE[0] <= ratio <= config.SHOT_COMPRESSION_RANGE[1]:
            continue
        drop = readability.measure(adv).fkgl - readability.measure(ele).fkgl
        if drop < config.SHOT_MIN_FKGL_DROP:
            continue
        shots.append((adv, ele))
        if len(shots) == n:
            break
    return shots


EVAL_SETS = ("asset", "passages", "science")
MODES = {"asset": ("whole",), "passages": ("whole", "chunked"), "science": ("whole",)}


def build_spec(variant: Variant, settings) -> SteeringSpec:
    if not variant.axes:
        return SteeringSpec.off()
    if len(variant.axes) == 1:
        return settings.spec(variant.axes[0])
    return settings.dual_spec()


def tokens_for(eval_set: str, requested: int) -> int:
    """Generation budget for eval_set, never below what the caller asked for."""
    if eval_set == "science":
        return max(requested, config.SCIENCE_MAX_NEW_TOKENS)
    return requested


def row_variant(variant: Variant, mode: str) -> str:
    return variant.name if mode == "whole" else f"{variant.name}_chunked"


@dataclass
class EvalItem:
    index: int
    source: str
    references: list[str]


def load_items(eval_set: str, n: int, asset_split: str) -> list[EvalItem]:
    """Items for one eval set, from lib.data's loaders as they are declared."""
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
        elif eval_set == "science":
            abstracts = data.load_science_abstracts()
            if n < len(abstracts):
                print(
                    f"  [warn] the science set is one abstract per field and is "
                    f"only meaningful whole; --n {n} drops "
                    f"{len(abstracts) - n} field(s)"
                )
            items = [
                EvalItem(i, a.abstract.strip(), [])
                for i, a in enumerate(abstracts[:n])
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


def wanted_metrics(item: EvalItem, *, skip_nli: bool, skip_sari: bool) -> set[str]:
    """The metric families a fully scored row for item would carry."""
    metrics = {"readability", "ppl"}
    if not skip_nli:
        metrics.add("nli")
    if not skip_sari and item.references:
        metrics.add("sari")
    return metrics


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
    """Mirror of fluency.ppl_ratio's guard, over a cached denominator."""
    if not math.isfinite(ref) or ref <= 0:
        return float("inf")
    return new / ref


def gen_whole(
    sl, texts, spec, instruction, max_new_tokens, batch_size,
    shots=None, logits_processor=None, task=None,
):
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
                task=task,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                shots=shots,
                logits_processor=logits_processor,
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
                        task=task,
                        max_new_tokens=max_new_tokens,
                        shots=shots,
                        logits_processor=logits_processor,
                    )
                except Exception as inner:
                    err[start + k] = f"{type(inner).__name__}: {inner}"
                sec[start + k] = time.perf_counter() - t1
        print(f"      generated {min(start + batch_size, len(texts))}/{len(texts)}")
    return out, err, sec


def gen_chunked(sl, texts, spec, instruction, max_new_tokens,
                shots=None, logits_processor=None, task=None):
    """Sentence-by-sentence path; each sentence's prompt depends on the
    previous sentence's output, so it is inherently unbatched. Takes the
    same keyword arguments as gen_whole so the dispatcher can hand both the
    same call.
    """
    out: list[str | None] = [None] * len(texts)
    err: list[str | None] = [None] * len(texts)
    sec: list[float] = [0.0] * len(texts)
    for i, passage in enumerate(texts):
        t0 = time.perf_counter()
        try:
            out[i] = sl.generate_chunked(
                passage, spec, instruction=instruction, max_new_tokens=max_new_tokens,
                shots=shots, logits_processor=logits_processor,
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
    """What each key on disk already holds, its source and gen_sec, and a
    bad-line count.
    """
    done: dict[tuple[str, str, int], set[str] | None] = {}
    sources: dict[tuple[str, int], str] = {}
    secs: dict[tuple[str, str, int], float] = {}
    bad = 0
    for n, row in jsonio.read_jsonl(path):
        if row is None:
            bad += 1
            continue
        try:
            key = (row["eval_set"], int(row["item_index"]))
            done[(row["variant"], *key)] = row_metrics(row)
            sources.setdefault(key, row.get("source") or "")
            if isinstance(row.get("gen_sec"), (int, float)):
                secs[(row["variant"], *key)] = float(row["gen_sec"])
        except Exception:
            bad += 1
    return done, sources, secs, bad


def row_metrics(row: dict) -> set[str] | None:
    """The metric families a row carries; None for a captured failure."""
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
    for n, row in jsonio.read_jsonl(path):
        if row is None:
            continue
        with suppress(Exception):
            key = (row["variant"], row["eval_set"], int(row["item_index"]))
            superseded += key in latest
            latest[key] = row
    return list(latest.values()), superseded


def summary_path_for(out: Path) -> Path:
    stem = out.stem
    if stem.startswith("eval_"):
        return out.parent / f"eval_summary_{stem[len('eval_') :]}.json"
    return out.parent / f"{stem}_summary.json"


_AGG_COLUMNS = (
    "eval_set",
    "variant",
    "mode",
    "spec_label",
    "error",
    "gen_sec",
    "readability.fkgl",
    "readability.words_per_sentence",
    "readability.pct_hard_words",
    "readability.degenerate",
    "deltas.d_words_per_sentence",
    "deltas.d_pct_hard_words",
    "deltas.d_fkgl",
    "faithfulness.score",
    "nli_passed",
    "ppl",
    "ppl_ratio",
    "sari.sari",
)


def _label(col):
    """First value in col, or None if it's missing/NaN."""
    value = col.iloc[0]
    return None if pd.isna(value) else value


def _flags(col):
    """col as pass/fail flags; a missing entry counts as a failure."""
    return [False if pd.isna(v) else v for v in col]


def aggregate(rows: list[dict]) -> list[dict]:
    """One summary row per (eval_set, variant) cell, in report order."""
    if not rows:
        return []

    df = pd.json_normalize(rows).reindex(columns=list(_AGG_COLUMNS))
    df["scored"] = df["error"].isna() | (df["error"] == "")
    cells = {
        key: cell
        for key, cell in df.groupby(["eval_set", "variant"], sort=False, dropna=False)
    }

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
    for key in sorted(cells, key=sort_key):
        eval_set, variant = key
        group = cells[key]
        ok = group[group["scored"]]
        out.append(
            {
                "eval_set": eval_set,
                "variant": variant,
                "mode": _label(group["mode"]) or "whole",
                "n": len(group),
                "n_failed": len(group) - len(ok),
                "spec_label": _label(group["spec_label"]),
                "mean_fkgl": jsonio.mean(ok["readability.fkgl"]),
                "mean_words_per_sentence": jsonio.mean(
                    ok["readability.words_per_sentence"]
                ),
                "mean_pct_hard_words": jsonio.mean(ok["readability.pct_hard_words"]),
                "mean_d_words_per_sentence": jsonio.mean(
                    ok["deltas.d_words_per_sentence"]
                ),
                "mean_d_pct_hard_words": jsonio.mean(ok["deltas.d_pct_hard_words"]),
                "mean_d_fkgl": jsonio.mean(ok["deltas.d_fkgl"]),
                "mean_nli_score": jsonio.mean(ok["faithfulness.score"]),
                "pct_nli_passed": jsonio.pct(_flags(ok["nli_passed"])),
                "mean_ppl": jsonio.mean(ok["ppl"]),
                "mean_ppl_ratio": jsonio.mean(ok["ppl_ratio"]),
                "mean_sari": jsonio.mean(ok["sari.sari"]),
                "pct_degenerate": jsonio.pct(_flags(ok["readability.degenerate"])),
                "mean_gen_sec": jsonio.mean(ok["gen_sec"]),
                "gen_seconds_total": sum(group["gen_sec"].dropna()),
            }
        )
    return out


_COLUMNS = (
    ("eval_set", "eval_set", None),
    ("variant", "variant", None),
    ("n", "n", "d"),
    ("fail", "n_failed", "d"),
    ("FKGL", "mean_fkgl", ".2f"),
    ("w/sent", "mean_words_per_sentence", ".2f"),
    ("hard", "mean_pct_hard_words", ".3f"),
    ("dFKGL", "mean_d_fkgl", "+.2f"),
    ("NLI", "mean_nli_score", ".3f"),
    ("gate%", "pct_nli_passed", ".1f"),
    ("pplR", "mean_ppl_ratio", ".2f"),
    ("SARI", "mean_sari", ".2f"),
    ("degen%", "pct_degenerate", ".1f"),
    ("gen_s", "gen_seconds_total", ".1f"),
)

_TABLE_WIDTH = 160


def _cell(value, fmt: str | None) -> str:
    if value is None:
        return "-"
    return str(value) if fmt is None else format(value, fmt)


def print_summary(agg: list[dict]) -> None:
    con = tables.console(_TABLE_WIDTH)
    table = tables.table(box=tables.box.HORIZONTALS)
    for title, _, fmt in _COLUMNS:
        table.add_column(
            title, justify="left" if fmt is None else "right", no_wrap=True
        )

    last_set = None
    for row in agg:
        if last_set is not None and row["eval_set"] != last_set:
            table.add_section()
        last_set = row["eval_set"]
        table.add_row(*(_cell(row.get(key), fmt) for _, key, fmt in _COLUMNS))

    con.print()
    con.print(table)
    con.print(
        "FKGL/w-sent/hard: absolute readability of the output.  dFKGL: fractional "
        "change vs that item's own unsteered output (negative = simpler)."
    )
    con.print(
        "NLI: bidirectional entailment score.  gate%: share passing "
        f"NLI>={config.NLI_ENTAILMENT_MIN}.  pplR: conditional ppl vs unsteered "
        f"(reported, not gated; >{config.PPL_RATIO_MAX} is flagged on the row)."
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
        help=(
            f"comma-separated subset of {','.join(BY_NAME)} (default all). "
            f"Also selectable: {','.join(v.name for v in FEWSHOT_VARIANTS + CONSTRAINED_VARIANTS)}"
            f", and with --lora, {','.join(v.name for v in LORA_VARIANTS)}. "
            "'all' always means the six steering variants, so that the headline "
            "comparison does not change shape depending on which flags were passed."
        ),
    )
    p.add_argument("--layers-from", type=Path, help="sweep JSON of best (layer, alpha)")
    p.add_argument("--structural", help="manual override; LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... (shapes: lib/schedules.py)")
    p.add_argument("--lexical", help="manual override; LAYER,ALPHA | LAYER,ALPHA,SHAPE | L:A+L:A+... (shapes: lib/schedules.py)")
    p.add_argument("--out", type=Path)
    p.add_argument("--skip-sari", action="store_true")
    p.add_argument("--skip-nli", action="store_true")
    p.add_argument(
        "--eval-sets",
        default=",".join(EVAL_SETS),
        help=f"subset of {','.join(EVAL_SETS)}",
    )
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--asset-split", default="test")
    p.add_argument(
        "--task",
        help=(
            "override the task sentence for this run. Exists to separate 'this "
            "model deletes content when asked to simplify' from 'this model was "
            "asked less precisely': Gemma carries a stricter task prompt than "
            "Llama by necessity, so a cross-model prompting comparison confounds "
            "the two until the same task is put in front of both."
        ),
    )
    p.add_argument(
        "--lora",
        type=Path,
        help="adapter directory from src/train_lora.py; enables the lora variants",
    )
    return p.parse_args(argv)


class EvalRun:
    """One invocation of the comparison: every chosen variant over every eval
    set. Split into phases so cheap failures (bad args, missing banks) surface
    before the model loads, and everything after __init__ only adds rows, so
    the run is resumable.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.variants = self._choose_variants()
        self.eval_sets = self._choose_eval_sets()
        self.settings = AxisSettings.from_args(args)
        self.needed_axes = sorted(
            {a for v in self.variants if v.needs_banks for a in v.axes}
        )
        self.settings.require(
            self.needed_axes,
            needed_by=f"variants {[v.name for v in self.variants if v.needs_banks]}",
        )
        self._check_unused()

        slug = args.model.split("/")[-1]
        self.out = args.out or (
            config.RESULTS_DIR / f"eval_{slug}_{args.estimator}.jsonl"
        )
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.summary_out = summary_path_for(self.out)
        self.baseline_cache_path = self.out.with_suffix(self.out.suffix + ".baselines.json")

        self.done, self.known_sources, self.known_secs, bad_lines = read_done(self.out)
        if bad_lines:
            print(f"[resume] ignored {bad_lines} unparseable line(s) in {self.out}")
        if self.done:
            print(
                f"[resume] {len(self.done)} row(s) already in {self.out}; "
                f"those will be skipped"
            )

        self.baselines: dict[str, str] = {}
        if self.baseline_cache_path.exists():
            with suppress(Exception):
                self.baselines = json.loads(self.baseline_cache_path.read_text())
            print(f"[resume] {len(self.baselines)} cached baseline output(s)")
        self.baseline_secs: dict[str, float] = {}

        self.lm = None
        self.sl = None
        self.adapter = Adapter()
        self.banks: dict = {}
        self.bank_orth: dict = {}
        self.easy_bias = None
        self.shots_by_set: dict[str, list] = {}

        self.timings: dict[str, float] = {}
        self.n_written = 0
        self.n_failed = 0
        self.interrupted = False

    def _choose_variants(self) -> list[Variant]:
        args = self.args
        catalogue = dict(BY_NAME)
        catalogue.update({v.name: v for v in FEWSHOT_VARIANTS})
        catalogue.update({v.name: v for v in CONSTRAINED_VARIANTS})
        if args.lora:
            catalogue.update({v.name: v for v in LORA_VARIANTS})
        names = (
            list(BY_NAME)
            if args.variants == "all"
            else [s.strip() for s in args.variants.split(",") if s.strip()]
        )
        unknown = [n for n in names if n not in catalogue]
        if unknown:
            extra = (
                ""
                if args.lora
                else f" (pass --lora DIR to unlock {[v.name for v in LORA_VARIANTS]})"
            )
            die(f"unknown variant(s) {unknown}; choose from {list(catalogue)}{extra}")
        chosen = [catalogue[n] for n in names]

        if any(v.adapter for v in chosen) and not args.lora:
            die(
                "variants "
                + str([v.name for v in chosen if v.adapter])
                + " need --lora DIR"
            )
        if args.lora and not args.lora.exists():
            die(f"--lora {args.lora} does not exist; train it with src/train_lora.py")
        return chosen

    def _check_unused(self) -> None:
        """Dials the chosen variants cannot act on."""
        args = self.args
        if (args.structural or args.lexical or args.layers_from) and not self.needed_axes:
            print(
                f"  [warn] --structural/--lexical/--layers-from given but none of "
                f"the chosen variants {[v.name for v in self.variants]} steers "
                f"(axes=()); ignored"
            )
        if args.lora and not any(v.adapter for v in self.variants):
            print(
                f"  [warn] --lora {args.lora} given but none of the chosen "
                f"variants {[v.name for v in self.variants]} uses the adapter; "
                f"ignored (the adapter still loads, since --lora also unlocks "
                f"the lora variant names)"
            )

    def _choose_eval_sets(self) -> list[str]:
        sets = [s.strip() for s in self.args.eval_sets.split(",") if s.strip()]
        bad = [s for s in sets if s not in EVAL_SETS]
        if bad:
            die(f"unknown eval set(s) {bad}; choose from {list(EVAL_SETS)}")
        return sets

    def load(self) -> None:
        """Banks, then weights, then adapter. Cheapest failure first."""
        args = self.args
        if self.needed_axes:
            try:
                self.banks = load_banks(args.model, args.estimator)
            except FileNotFoundError as exc:
                die(
                    f"{exc}\n\nBuild the vector banks first:\n"
                    f"  {sys.executable} src/build_vectors.py "
                    f"--model {args.model} --estimator {args.estimator}\n\n"
                    f"Or evaluate only the variants that need no banks:\n"
                    f"  {sys.executable} src/evaluate.py --model {args.model} "
                    f"--variants unsteered,prompted"
                )
            for axis in self.needed_axes:
                held = self.banks[axis].layers
                missing = [
                    layer
                    for layer in self.settings.layers(axis)
                    if layer not in self.banks[axis].vectors
                ]
                if missing:
                    die(
                        f"bank {axis!r} ({args.estimator}) has no vector for "
                        f"layer(s) {missing}; it holds layers "
                        f"{held[0]}..{held[-1]}"
                    )
        self.bank_orth = {axis: b.orthogonalized for axis, b in self.banks.items()}

        print(f"\n[load] {args.model} on {config.DEVICE} ({config.DTYPE})")
        t_load = time.perf_counter()
        self.lm = models.load(args.model)
        print(
            f"[load] {self.lm.n_layers} layers, hidden {self.lm.hidden_size}, "
            f"{time.perf_counter() - t_load:.1f}s"
        )

        if args.lora:
            t_lora = time.perf_counter()
            self.adapter = Adapter.load(self.lm, args.lora, expect_base=args.model)
            print(f"[lora] {self.adapter.describe()}, {time.perf_counter() - t_lora:.1f}s")

        try:
            self.sl = SteeredLM(self.lm, self.banks)
        except ValueError as exc:
            die(
                f"{exc}\n\nRebuild the banks against this model:\n"
                f"  {sys.executable} src/build_vectors.py "
                f"--model {args.model} --estimator {args.estimator}"
            )

        if any(v.easy_words for v in self.variants):
            t_bias = time.perf_counter()
            self.easy_bias = decoding.build(self.lm.tokenizer, repo_id=args.model)
            n_pen = int(self.easy_bias.mask.sum())
            n_vocab = self.easy_bias.mask.shape[0]
            print(
                f"[cfg]  easy-word bias: penalty {self.easy_bias.penalty} on "
                f"{n_pen} of {n_vocab} tokens ({100 * n_pen / n_vocab:.1f}%), "
                f"{time.perf_counter() - t_bias:.1f}s"
            )

        n_shots = max((v.shots for v in self.variants), default=0)
        self.shots_by_set = {
            e: load_shots(n_shots, config.SHOT_SOURCE.get(e, "asset"))
            for e in self.eval_sets
        }
        self.announce()

    def announce(self) -> None:
        args = self.args
        print(f"[cfg]  estimator={args.estimator} settings={self.settings!r}")
        print(f"[cfg]  banks orthogonalized={self.bank_orth or '(no banks loaded)'}")
        for e, got in self.shots_by_set.items():
            if got:
                src_name = config.SHOT_SOURCE.get(e, "asset")
                words = sum(len(a.split()) for a, _ in got) / len(got)
                print(
                    f"[cfg]  {e}: {len(got)} demonstration(s) from {src_name}, "
                    f"mean source {words:.0f} words"
                )
        if args.task:
            print(f"[cfg]  task override: {args.task!r}")
        print(f"[cfg]  variants={[v.name for v in self.variants]}")
        print(f"[cfg]  eval_sets={self.eval_sets} n={args.n} data=lib.data")
        print(
            f"[cfg]  max_new_tokens={args.max_new_tokens} batch_size={args.batch_size}"
        )
        print(
            f"[cfg]  nli={'off' if args.skip_nli else 'on'} "
            f"sari={'off' if args.skip_sari else 'on'}"
        )
        print(f"[out]  {self.out}")

    def _gen(self, mode, texts, spec, instruction, max_new, **kw):
        if mode == "whole":
            return gen_whole(
                self.sl,
                texts,
                spec,
                instruction,
                max_new,
                self.args.batch_size,
                task=self.args.task,
                **kw,
            )
        return gen_chunked(
            self.sl, texts, spec, instruction, max_new, task=self.args.task, **kw
        )

    def ensure_baselines(self, eval_set, mode, items, max_new) -> None:
        """Fill any missing unsteered output for this (eval set, mode) pair."""
        keys = [f"{eval_set}|{mode}|{it.index}" for it in items]
        todo = [i for i, k in enumerate(keys) if k not in self.baselines]
        if not todo:
            return
        print(f"  [baseline {mode}] generating {len(todo)} unsteered output(s)")
        with self.adapter.engaged(False):
            outs, errs, secs = self._gen(
                mode, [items[i].source for i in todo], SteeringSpec.off(), None, max_new
            )
        for j, i in enumerate(todo):
            self.baseline_secs[keys[i]] = secs[j]
            if errs[j] is None and outs[j] is not None:
                self.baselines[keys[i]] = outs[j]
        self.baseline_cache_path.write_text(json.dumps(self.baselines, indent=0))

    def pending(self, rv, eval_set, items) -> list[EvalItem]:
        """Items with no row on disk, or a row missing a metric this run wants."""
        return [
            it
            for it in items
            if not is_done(
                self.done,
                (rv, eval_set, it.index),
                wanted_metrics(
                    it, skip_nli=self.args.skip_nli, skip_sari=self.args.skip_sari
                ),
            )
        ]

    def outputs_for(self, variant, rv, mode, spec, eval_set, pending, max_new):
        """(texts, errors, seconds) for one variant over the pending items."""
        if variant.name == "unsteered":
            keys = [f"{eval_set}|{mode}|{it.index}" for it in pending]
            outs = [self.baselines.get(k) for k in keys]
            errs = [None if o is not None else "baseline generation failed" for o in outs]
            secs = [
                self.baseline_secs.get(
                    k, self.known_secs.get((rv, eval_set, it.index), 0.0)
                )
                for k, it in zip(keys, pending)
            ]
            return outs, errs, secs

        extra = {
            "shots": self.shots_by_set[eval_set][: variant.shots],
            "logits_processor": self.easy_bias if variant.easy_words else None,
        }
        with self.adapter.engaged(variant.adapter):
            return self._gen(
                mode,
                [it.source for it in pending],
                spec,
                variant.instruction,
                max_new,
                **extra,
            )

    def row_for(
        self, variant, rv, mode, spec, eval_set, item, output, err, sec, base_ppl
    ) -> dict:
        bkey = f"{eval_set}|{mode}|{item.index}"
        baseline = self.baselines.get(bkey)
        row = {
            "variant": rv,
            "eval_set": eval_set,
            "mode": mode,
            "item_index": item.index,
            "model": self.args.model,
            "estimator": self.args.estimator,
            "data_source": "lib.data",
            "spec": spec.as_dict(),
            "spec_label": spec.label(),
            "bank_orthogonalized": {a: self.bank_orth[a] for a in variant.axes},
            "instruction": variant.instruction,
            "adapter": str(self.args.lora) if variant.adapter else None,
            "n_shots": variant.shots,
            "easy_word_penalty": (
                self.easy_bias.penalty if variant.easy_words else None
            ),
            "source": item.source,
            "n_references": len(item.references),
            "output": output,
            "output_scored": cleanup.strip_preamble(output or ""),
            "had_preamble": cleanup.had_preamble(output or ""),
            "baseline_output": baseline,
            "baseline_variant": f"unsteered/{mode}",
            "gen_sec": sec,
            "error": err,
        }
        if err is None:
            try:
                with self.adapter.scoring():
                    row.update(
                        score_item(
                            self.lm,
                            item,
                            output,
                            baseline,
                            base_ppl,
                            bkey,
                            skip_nli=self.args.skip_nli,
                            skip_sari=self.args.skip_sari,
                        )
                    )
            except Exception as exc:
                row["error"] = f"scoring: {type(exc).__name__}: {exc}"
        return row

    def run(self) -> int:
        args = self.args
        writer = jsonio.JsonlWriter(self.out)
        try:
            for eval_set in self.eval_sets:
                items = load_items(eval_set, args.n, args.asset_split)
                max_new = tokens_for(eval_set, args.max_new_tokens)
                note = (
                    ""
                    if max_new == args.max_new_tokens
                    else f" (raised from {args.max_new_tokens})"
                )
                print(
                    f"\n[{eval_set}] {len(items)} item(s) from lib.data, "
                    f"max_new_tokens={max_new}{note}"
                )
                check_resume_alignment(items, eval_set, self.known_sources)

                for mode in MODES[eval_set]:
                    self.ensure_baselines(eval_set, mode, items, max_new)
                    for variant in self.variants:
                        self._run_variant(
                            variant, mode, eval_set, items, max_new, writer
                        )
        except KeyboardInterrupt:
            self.interrupted = True
            print("\n[interrupted] rows written so far are on disk; re-run to resume")
        finally:
            writer.close()

        rows, superseded = read_rows(self.out)
        agg = aggregate(rows)
        print_summary(agg)

        if self.timings:
            print("\nwall clock this run:")
            for key, secs in self.timings.items():
                print(f"  {key:<44} {secs:8.1f}s")
        print(
            f"\nrows written this run: {self.n_written}   "
            f"failures this run: {self.n_failed}   rows in file: {len(rows)}"
            + (f"   superseded by a later row: {superseded}" if superseded else "")
        )

        jsonio.dump(self.summary_out, self.summary(agg, len(rows), superseded))
        print(f"\nsummary -> {self.summary_out}")
        print(f"rows    -> {self.out}")
        return 1 if self.interrupted else 0

    def _run_variant(self, variant, mode, eval_set, items, max_new, writer) -> None:
        rv = row_variant(variant, mode)
        spec = build_spec(variant, self.settings)
        pending = self.pending(rv, eval_set, items)
        if not pending:
            print(f"  [{rv}] all {len(items)} row(s) already present; skipped")
            return

        thin = [it for it in pending if (rv, eval_set, it.index) in self.done]
        if thin:
            print(
                f"  [{rv}] {len(thin)} row(s) on disk are missing a metric this run "
                f"asks for and will be redone; the thin rows stay in the file and "
                f"the summary keeps the last row per key"
            )
        print(
            f"  [{rv}] {len(pending)}/{len(items)} item(s) spec={spec.label()} "
            f"instruction={'simplify' if variant.instruction else 'none'}"
            f" adapter={'on' if variant.adapter else 'off'}"
        )
        t_variant = time.perf_counter()
        outs, errs, secs = self.outputs_for(
            variant, rv, mode, spec, eval_set, pending, max_new
        )

        assert_no_injector(self.lm)
        base_ppl: dict[str, float] = {}

        scoring = tables.progress(
            zip(pending, outs, errs, secs),
            total=len(pending),
            desc=f"  [{rv}] scoring",
            unit="item",
        )
        for item, output, err, sec in scoring:
            row = self.row_for(
                variant, rv, mode, spec, eval_set, item, output, err, sec, base_ppl
            )
            if row["error"]:
                self.n_failed += 1
                tables.write(f"      item {item.index} FAILED: {row['error']}")
            writer.write(row)
            self.done[(rv, eval_set, item.index)] = row_metrics(row)
            self.n_written += 1

        self.timings[f"{eval_set}/{rv}"] = time.perf_counter() - t_variant
        print(f"  [{rv}] {self.timings[f'{eval_set}/{rv}']:.1f}s wall clock")

    def summary(self, agg: list[dict], n_rows: int, superseded: int) -> dict:
        args = self.args
        return {
            "model": args.model,
            "estimator": args.estimator,
            "data_source": "lib.data",
            "bank_orthogonalized": self.bank_orth,
            "n_requested": args.n,
            "lora": str(args.lora) if args.lora else None,
            "task": args.task,
            "shots": {e: [list(x) for x in g] for e, g in self.shots_by_set.items()},
            "easy_word_penalty": self.easy_bias.penalty if self.easy_bias else None,
            "variants": [v.name for v in self.variants],
            "eval_sets": self.eval_sets,
            "settings": self.settings.as_dict(),
            "skip_nli": args.skip_nli,
            "skip_sari": args.skip_sari,
            "max_new_tokens": args.max_new_tokens,
            "max_new_tokens_used": {
                e: tokens_for(e, args.max_new_tokens) for e in self.eval_sets
            },
            "batch_size": args.batch_size,
            "interrupted": self.interrupted,
            "rows_written_this_run": self.n_written,
            "failures_this_run": self.n_failed,
            "rows_in_file": n_rows,
            "rows_superseded": superseded,
            "wall_clock_this_run": self.timings,
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


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config.SEED)

    run = EvalRun(args)
    run.load()
    return run.run()


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
    output = cleanup.strip_preamble(output)
    read = readability.measure(output)
    patch: dict = {"readability": read.as_dict()}
    metrics = ["readability", "ppl"]

    if baseline is not None:
        base_read = readability.measure(baseline)
        patch["baseline_readability"] = base_read.as_dict()
        patch["deltas"] = readability.relative_deltas(base_read, read)

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
