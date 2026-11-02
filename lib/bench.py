"""One loaded model over one item set: the spine every steering arm shares.

Invariants this file exists to hold:

1. **A baseline is unsteered, or it is not a baseline.** ``baselines()`` reuses
   the cache ``src/evaluate.py`` wrote so that arms are compared against
   byte-identical text, and generates any miss with steering off and the
   adapter off. It never falls back to a steered string.
2. **Scoring never runs under the intervention.** Every metric is taken inside
   ``Adapter.scoring()``, and ``Bench.score`` asserts no injector is live.
3. **One row schema.** ``Bench.row`` builds it, so a new arm is comparable to
   the existing ones by construction rather than by review.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import config
from lib import data, jsonio, models
from lib.adapters import Adapter
from lib.metrics import faithfulness as faith_metrics
from lib.metrics import fluency, readability
from lib.metrics import sari as sari_metrics
from lib.steering import SteeredLM, SteeringSpec, load_banks


@dataclass(frozen=True)
class Item:
    """One evaluation input. ``index`` is the join key across every result
    file, and is never renumbered."""

    index: int
    source: str
    references: list[str] = field(default_factory=list)

    @property
    def source_fkgl(self) -> float:
        return readability.measure(self.source).fkgl


def asset_items(split: str = "test", n: int | None = None) -> list[Item]:
    rows = data.load_asset(split)
    if n is not None:
        rows = rows[:n]
    return [
        Item(i, r.source.strip(), list(r.references)) for i, r in enumerate(rows)
    ]


@dataclass
class Scored:
    """Every metric for one candidate output, and the gate's verdict on it."""

    text: str
    read: readability.Readability
    nli: float
    ppl: float
    ppl_ratio: float
    sari: float | None = None
    label: str = ""

    @property
    def degenerate(self) -> bool:
        return self.read.degenerate

    @property
    def nli_passed(self) -> bool:
        return self.nli >= config.NLI_ENTAILMENT_MIN

    @property
    def ppl_ok(self) -> bool:
        """Reported, not gated. See ``passes``."""
        return math.isfinite(self.ppl_ratio) and self.ppl_ratio <= config.PPL_RATIO_MAX

    def passes(self) -> bool:
        """The shipping gate: intact language, meaning kept."""
        return not self.degenerate and self.nli_passed


class Bench:
    """Model, banks, adapter and item set, loaded once and shared by an arm."""

    def __init__(
        self,
        lm,
        items: list[Item],
        *,
        model: str,
        estimator: str,
        banks: dict | None = None,
        adapter: Adapter | None = None,
    ):
        self.lm = lm
        self.items = items
        self.model = model
        self.estimator = estimator
        self.banks = banks or {}
        self.adapter = adapter or Adapter()
        self.sl = SteeredLM(lm, self.banks)
        self._ppl_cache: dict[tuple[int, str], float] = {}

    @classmethod
    def open(
        cls,
        args,
        *,
        needs_banks: bool = True,
        items: list[Item] | None = None,
    ) -> "Bench":
        """Load items, banks, model and adapter, in the order that fails
        cheapest first."""
        if items is None:
            items = asset_items(
                getattr(args, "asset_split", "test"), getattr(args, "n", None)
            )
        print(
            f"[data] ASSET {getattr(args, 'asset_split', 'test')}, "
            f"{len(items)} item(s)"
        )

        banks = {}
        if needs_banks:
            try:
                banks = load_banks(args.model, args.estimator)
            except FileNotFoundError as exc:
                raise SystemExit(
                    f"\nFATAL: {exc}\n\nBuild the vector banks first:\n"
                    f"  python src/build_vectors.py --model {args.model} "
                    f"--estimator {args.estimator}\n"
                )

        print(f"\n[load] {args.model} on {config.DEVICE} ({config.DTYPE})")
        t = time.perf_counter()
        lm = models.load(args.model)
        print(
            f"[load] {lm.n_layers} layers, hidden {lm.hidden_size}, "
            f"{time.perf_counter() - t:.1f}s"
        )

        adapter = Adapter()
        lora = getattr(args, "lora", None)
        if lora:
            if not Path(lora).exists():
                raise SystemExit(
                    f"\nFATAL: --lora {lora} does not exist; train it with "
                    f"src/train_lora.py\n"
                )
            adapter = Adapter.load(lm, Path(lora), expect_base=args.model)
            print(f"[lora] {adapter.describe()}")

        return cls(
            lm,
            items,
            model=args.model,
            estimator=getattr(args, "estimator", "instruction_diff"),
            banks=banks,
            adapter=adapter,
        )

    @property
    def sources(self) -> list[str]:
        return [it.source for it in self.items]

    def generate(
        self,
        spec: SteeringSpec,
        sources: list[str] | None = None,
        *,
        instruction: str | None = None,
        adapter_on: bool = True,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        batch_size: int = config.BATCH_SIZE,
        **kw,
    ) -> list[str]:
        with self.adapter.engaged(adapter_on):
            outs = self.sl.generate_batch(
                sources if sources is not None else self.sources,
                spec=spec,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                **kw,
            )
        return [o.strip() for o in outs]

    def baselines(
        self,
        cache_path: Path,
        *,
        eval_set: str = "asset",
        mode: str = "whole",
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        batch_size: int = config.BATCH_SIZE,
    ) -> list[str]:
        """Unsteered output per item: cached where possible, generated where not."""
        cache: dict[str, str] = {}
        if cache_path.exists():
            try:
                import json

                cache = json.loads(cache_path.read_text())
            except Exception:
                print(f"[base] {cache_path.name} unreadable; regenerating all")
        keys = [f"{eval_set}|{mode}|{it.index}" for it in self.items]
        out: list[str | None] = [cache.get(k) for k in keys]
        print(
            f"[base] {sum(o is not None for o in out)}/{len(out)} reused "
            f"from {cache_path.name}"
        )

        missing = [i for i, o in enumerate(out) if o is None]
        if missing:
            print(f"[base] generating {len(missing)} unsteered baseline(s)")
            got = self.generate(
                SteeringSpec.off(),
                [self.items[i].source for i in missing],
                adapter_on=False,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            for i, text in zip(missing, got):
                out[i] = text
        return [o or "" for o in out]

    def prompt(self, item: Item) -> str:
        """The bare task prompt: no simplify instruction, no demonstrations."""
        return self.lm.build_prompt(item.source, None, None)

    def conditional_ppl(self, item: Item, text: str) -> float:
        key = (item.index, text)
        if key not in self._ppl_cache:
            with self.adapter.scoring():
                self._ppl_cache[key] = fluency.conditional_ppl(
                    self.lm, self.prompt(item), text
                )
        return self._ppl_cache[key]

    def score(
        self,
        item: Item,
        text: str,
        *,
        baseline: str | None = None,
        baseline_ppl: float | None = None,
        label: str = "",
        skip_sari: bool = False,
    ) -> Scored:
        """Every metric for one output, with the intervention provably off."""
        read = readability.measure(text)
        with self.adapter.scoring():
            nli = faith_metrics.faithfulness(item.source, text).score
        ppl = self.conditional_ppl(item, text)
        if baseline_ppl is None and baseline is not None:
            baseline_ppl = self.conditional_ppl(item, baseline)
        ratio = (
            ppl / baseline_ppl
            if baseline_ppl and math.isfinite(baseline_ppl)
            else float("inf")
        )
        sari = (
            sari_metrics.sari(item.source, text, item.references).sari
            if item.references and not skip_sari
            else None
        )
        return Scored(text, read, nli, ppl, ratio, sari, label)

    def with_sari(self, item: Item, scored: Scored) -> Scored:
        """*scored* with SARI filled in, if not already present."""
        if not item.references or scored.sari is not None:
            return scored
        return replace(
            scored,
            sari=sari_metrics.sari(item.source, scored.text, item.references).sari,
        )

    def row(
        self,
        item: Item,
        scored: Scored,
        baseline: str,
        *,
        variant: str,
        spec: dict,
        spec_label: str,
        gen_sec: float = 0.0,
        eval_set: str = "asset",
        mode: str = "whole",
        **extra,
    ) -> dict:
        """One result row in the schema ``src/report.py`` reads."""
        base_read = readability.measure(baseline)
        base_ppl = self.conditional_ppl(item, baseline)
        row = {
            "variant": variant,
            "eval_set": eval_set,
            "mode": mode,
            "item_index": item.index,
            "model": self.model,
            "estimator": self.estimator,
            "data_source": "lib.data",
            "spec": spec,
            "spec_label": spec_label,
            "bank_orthogonalized": {
                axis: bank.orthogonalized for axis, bank in self.banks.items()
            },
            "instruction": None,
            "adapter": str(self.adapter.path) if self.adapter else None,
            "source": item.source,
            "n_references": len(item.references),
            "output": scored.text,
            "baseline_output": baseline,
            "baseline_variant": f"unsteered/{mode}",
            "gen_sec": gen_sec,
            "error": None,
            "readability": scored.read.as_dict(),
            "baseline_readability": base_read.as_dict(),
            "deltas": readability.relative_deltas(base_read, scored.read),
            "ppl": scored.ppl,
            "baseline_ppl": base_ppl,
            "ppl_ratio": scored.ppl_ratio,
            "ppl_ratio_ok": scored.ppl_ok,
            "faithfulness": {"score": scored.nli},
            "nli_passed": scored.nli_passed,
            "metrics": ["readability", "ppl", "nli"],
        }
        if scored.sari is not None:
            row["sari"] = sari_metrics.sari(
                item.source, scored.text, item.references
            ).as_dict()
            row["metrics"].append("sari")
        row.update(extra)
        return row

    def writer(self, path: Path) -> jsonio.JsonlWriter:
        return jsonio.JsonlWriter(path)
