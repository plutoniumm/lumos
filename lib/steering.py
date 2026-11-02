"""SteeringSpec and SteeredLM. Sweep, evaluation and product share one codepath."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib.chunking import join_sentences, rolling_context, split_sentences  # noqa: E402
from lib.hooks import (  # noqa: E402
    InjectionSite,
    Positions,
    SiteKind,
    record,
    record_inputs,
    steered,
)
from lib.models import SteerableLM  # noqa: E402
from lib.vectors import VectorBank  # noqa: E402


@dataclass(frozen=True)
class AxisSetting:
    """One axis dialled to one strength at one layer."""

    axis: str
    layer: int
    alpha: float


@dataclass(frozen=True)
class SteeringSpec:
    """A complete steering configuration. Empty ``settings`` means unsteered."""

    settings: tuple[AxisSetting, ...] = ()
    positions: Positions = config.DEFAULT_POSITIONS
    norm_preserve: bool = config.NORM_PRESERVE
    kind: SiteKind = config.DEFAULT_SITE_KIND
    content_orthogonalize: bool = config.CONTENT_ORTHOGONALIZE

    @classmethod
    def off(cls, **kw) -> "SteeringSpec":
        return cls(settings=(), **kw)

    @classmethod
    def single(cls, axis: str, layer: int, alpha: float, **kw) -> "SteeringSpec":
        return cls(settings=(AxisSetting(axis, layer, alpha),), **kw)

    @classmethod
    def dual(
        cls,
        structural: tuple[int, float],
        lexical: tuple[int, float],
        **kw,
    ) -> "SteeringSpec":
        """The two-dial product configuration: (layer, alpha) for each axis."""
        return cls(
            settings=(
                AxisSetting("structural", structural[0], structural[1]),
                AxisSetting("lexical", lexical[0], lexical[1]),
            ),
            **kw,
        )

    @property
    def is_off(self) -> bool:
        return not self.settings or all(s.alpha == 0.0 for s in self.settings)

    @property
    def layers(self) -> list[int]:
        return sorted({s.layer for s in self.settings})

    def with_alpha(self, axis: str, alpha: float) -> "SteeringSpec":
        """Same spec, one axis re-dialled. Used by the alpha calibration search."""
        return replace(
            self,
            settings=tuple(
                replace(s, alpha=alpha) if s.axis == axis else s for s in self.settings
            ),
        )

    def scaled(self, factor: float) -> "SteeringSpec":
        """Same layers, every alpha multiplied."""
        return replace(
            self,
            settings=tuple(replace(s, alpha=s.alpha * factor) for s in self.settings),
        )

    def label(self) -> str:
        if self.is_off:
            return "unsteered"
        return "+".join(f"{s.axis[:4]}@L{s.layer}a{s.alpha:g}" for s in self.settings)

    def as_dict(self) -> dict:
        return {
            "settings": [
                {"axis": s.axis, "layer": s.layer, "alpha": s.alpha}
                for s in self.settings
            ],
            "positions": self.positions,
            "norm_preserve": self.norm_preserve,
            "kind": self.kind,
            "content_orthogonalize": self.content_orthogonalize,
        }


@dataclass
class SteeredLM:
    """A model plus its vector banks, generating under a ``SteeringSpec``."""

    lm: SteerableLM
    banks: dict[str, VectorBank] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, bank in self.banks.items():
            if bank.model_repo != self.lm.repo_id:
                raise ValueError(
                    f"bank {name!r} was estimated on {bank.model_repo!r} but the "
                    f"model is {self.lm.repo_id!r}; steering vectors do not "
                    f"transfer across models"
                )

    def _content_directions(
        self, prompt: str, content_text: str, layers: list[int], kind: SiteKind
    ) -> dict[int, torch.Tensor]:
        """Mean-pooled representation of the source text, per layer."""
        span = self.lm.text_token_span(prompt, content_text)
        acts = record(self.lm, prompt, layers, kind, "mean_span", span)
        return {layer: a[0] for layer, a in acts.items()}

    def _sites(
        self, spec: SteeringSpec, prompt: str, content_text: str
    ) -> list[InjectionSite]:
        if spec.is_off:
            return []
        content: dict[int, torch.Tensor] = {}
        if spec.content_orthogonalize:
            content = self._content_directions(
                prompt, content_text, spec.layers, spec.kind
            )
        return self._build_sites(spec, content)

    def _batch_sites(
        self,
        spec: SteeringSpec,
        prompts: list[str],
        texts: list[str],
        inputs: dict[str, torch.Tensor],
    ) -> list[InjectionSite]:
        """Sites for a padded batch, with one content direction per row."""
        if spec.is_off:
            return []
        content: dict[int, torch.Tensor] = {}
        if spec.content_orthogonalize:
            spans = self.lm.padded_spans(prompts, texts, inputs["attention_mask"])
            content = record_inputs(
                self.lm, inputs, spec.layers, spec.kind, "mean_span", spans
            )
        return self._build_sites(spec, content)

    def _build_sites(
        self, spec: SteeringSpec, content: dict[int, torch.Tensor]
    ) -> list[InjectionSite]:
        missing = {s.axis for s in spec.settings} - set(self.banks)
        if missing:
            raise KeyError(f"no vector bank for axis/axes {sorted(missing)}")

        sites = []
        for setting in spec.settings:
            bank = self.banks[setting.axis]
            if setting.layer not in bank.vectors:
                raise KeyError(
                    f"bank {setting.axis!r} has no vector for layer {setting.layer}"
                )
            sites.append(
                InjectionSite(
                    layer=setting.layer,
                    vector=bank.scaled(setting.layer),
                    alpha=setting.alpha,
                    kind=spec.kind,
                    positions=spec.positions,
                    norm_preserve=spec.norm_preserve,
                    content=content.get(setting.layer),
                )
            )
        return sites

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        spec: SteeringSpec | None = None,
        instruction: str | None = None,
        task: str | None = None,
        content_text: str | None = None,
        max_new_tokens: int | None = None,
        shots: list[tuple[str, str]] | None = None,
        logits_processor=None,
    ) -> str:
        """Rewrite ``text`` under ``spec``."""
        spec = spec or SteeringSpec.off()
        prompt = self.lm.build_prompt(text, instruction, task, shots)
        sites = self._sites(spec, prompt, content_text or text)

        batch = self.lm.encode(prompt)
        prompt_len = batch["input_ids"].shape[1]
        out = self._run(batch, sites, max_new_tokens, logits_processor)
        return self.lm.decode_new(out, prompt_len)[0].strip()

    @torch.inference_mode()
    def generate_batch(
        self,
        texts: list[str],
        spec: SteeringSpec | None = None,
        instruction: str | None = None,
        task: str | None = None,
        max_new_tokens: int | None = None,
        batch_size: int | None = None,
        shots: list[tuple[str, str]] | None = None,
        logits_processor=None,
    ) -> list[str]:
        """Batched ``generate`` with identical steering semantics."""
        spec = spec or SteeringSpec.off()
        size = batch_size or config.BATCH_SIZE
        results: list[str] = []

        for start in range(0, len(texts), size):
            group = texts[start : start + size]
            prompts = [self.lm.build_prompt(t, instruction, task, shots) for t in group]
            inputs = self.lm.encode_batch(prompts)
            prompt_len = inputs["input_ids"].shape[1]
            sites = self._batch_sites(spec, prompts, group, inputs)
            out = self._run(inputs, sites, max_new_tokens, logits_processor)
            results.extend(s.strip() for s in self.lm.decode_new(out, prompt_len))
        return results

    def _run(
        self,
        inputs: dict[str, torch.Tensor],
        sites: list[InjectionSite],
        max_new_tokens: int | None,
        logits_processor=None,
    ) -> torch.Tensor:
        kwargs = dict(
            max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
            # `or` would treat a pad id of 0 as missing (gemma-3n's case)
            pad_token_id=(
                self.lm.tokenizer.eos_token_id
                if self.lm.tokenizer.pad_token_id is None
                else self.lm.tokenizer.pad_token_id
            ),
        )
        if logits_processor is not None:
            kwargs["logits_processor"] = [logits_processor]
        if not sites:
            return self.lm.model.generate(**inputs, **kwargs)
        with steered(self.lm, sites):
            return self.lm.model.generate(**inputs, **kwargs)

    def generate_chunked(
        self,
        passage: str,
        spec: SteeringSpec | None = None,
        instruction: str | None = None,
        max_new_tokens: int | None = None,
        context_sentences: int | None = None,
        shots: list[tuple[str, str]] | None = None,
        logits_processor=None,
    ) -> str:
        """Sentence-by-sentence simplification of a passage, then rejoined.

        ``shots``/``logits_processor`` are forwarded per sentence; ``task`` is not
        -- chunked mode sets its own (``config.CHUNK_TASK``) once there is context.
        """
        n_ctx = (
            config.CHUNK_CONTEXT_SENTENCES
            if context_sentences is None
            else context_sentences
        )
        simplified: list[str] = []
        for sentence in split_sentences(passage):
            ctx = rolling_context(simplified, n_ctx)
            if ctx:
                text, task = f"{ctx}\n\n{sentence}", config.CHUNK_TASK
            else:
                text, task = sentence, None
            simplified.append(
                self.generate(
                    text,
                    spec,
                    instruction=instruction,
                    task=task,
                    content_text=sentence,
                    max_new_tokens=max_new_tokens,
                    shots=shots,
                    logits_processor=logits_processor,
                )
            )
        return join_sentences(simplified)

    def unsteered(self, text: str, **kw) -> str:
        """Task prompt only, no constraint, no steering."""
        return self.generate(text, SteeringSpec.off(), **kw)

    def prompted(self, text: str, axis: str = "simplify", **kw) -> str:
        """Prompt-only baseline: the honest bar steering has to clear."""
        return self.generate(
            text, SteeringSpec.off(), instruction=config.INSTRUCTIONS[axis], **kw
        )


def load_banks(
    model_repo: str, estimator: str, axes: tuple[str, ...] = config.AXES
) -> dict[str, VectorBank]:
    """Load one estimator's bank per axis from ``vectors/``."""
    banks = {}
    for axis in axes:
        path = VectorBank.default_path(model_repo, estimator, axis)
        if not path.exists():
            raise FileNotFoundError(
                f"no bank at {path}; run src/build_vectors.py --estimator {estimator}"
            )
        banks[axis] = VectorBank.load(path)
    return banks
