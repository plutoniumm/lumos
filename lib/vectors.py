"""Steering-vector estimators and the on-disk bank.

instruction_diff  mean[h(text+instruction) - h(text+placebo)]
caa               mean[h(simple)] - mean[h(complex)]
diff_pca          leading PC of the per-pair deltas

Alpha is a fraction of the layer's activation norm: norms grow 33x with depth,
so raw alpha is not comparable across layers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib import tables  # noqa: E402
from lib.hooks import Pooling, SiteKind, record  # noqa: E402
from lib.models import SteerableLM  # noqa: E402


@dataclass
class VectorBank:
    """Per-layer steering vectors for one axis, plus the scale they live at."""

    name: str
    estimator: str
    model_repo: str
    kind: SiteKind
    pooling: Pooling
    n_texts: int
    vectors: dict[int, torch.Tensor]
    ref_norms: dict[int, float]
    orthogonalized: str = "none"

    @property
    def layers(self) -> list[int]:
        return sorted(self.vectors)

    def unit(self, layer: int) -> torch.Tensor:
        v = self.vectors[layer]
        return v / v.norm().clamp_min(1e-8)

    def scaled(self, layer: int) -> torch.Tensor:
        """Unit direction rescaled to this layer's typical activation norm."""
        return self.unit(layer) * self.ref_norms[layer]

    def cosine(self, other: "VectorBank", layer: int) -> float:
        return float(torch.dot(self.unit(layer), other.unit(layer)))

    def save(self, path: Path | None = None) -> Path:
        path = path or self.default_path(self.model_repo, self.estimator, self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        layers = self.layers
        tensors = {
            "vectors": torch.stack([self.vectors[l] for l in layers]),
            "layers": torch.tensor(layers, dtype=torch.int32),
            "ref_norms": torch.tensor([self.ref_norms[l] for l in layers]),
        }
        meta = {
            "name": self.name,
            "estimator": self.estimator,
            "model_repo": self.model_repo,
            "kind": self.kind,
            "pooling": self.pooling,
            "n_texts": str(self.n_texts),
            "orthogonalized": self.orthogonalized,
        }
        save_file(tensors, str(path), metadata=meta)
        return path

    @classmethod
    def load(cls, path: Path) -> "VectorBank":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        t = load_file(str(path))
        layers = [int(x) for x in t["layers"].tolist()]
        return cls(
            name=meta.get("name", path.stem),
            estimator=meta.get("estimator", "unknown"),
            model_repo=meta.get("model_repo", "unknown"),
            kind=meta.get("kind", config.DEFAULT_SITE_KIND),
            pooling=meta.get("pooling", "last"),
            n_texts=int(meta.get("n_texts", 0)),
            orthogonalized=meta.get("orthogonalized", "none"),
            vectors={l: t["vectors"][i] for i, l in enumerate(layers)},
            ref_norms={l: float(t["ref_norms"][i]) for i, l in enumerate(layers)},
        )

    @staticmethod
    def default_path(model_repo: str, estimator: str, name: str) -> Path:
        slug = model_repo.split("/")[-1]
        return config.VECTORS_DIR / slug / f"{estimator}.{name}.safetensors"


@dataclass
class _Accum:
    """Running mean of vectors and norms, per layer, without holding every sample."""

    total: dict[int, torch.Tensor] = field(default_factory=dict)
    norms: dict[int, float] = field(default_factory=dict)
    count: int = 0

    def add(self, delta: dict[int, torch.Tensor], norms: dict[int, float]) -> None:
        for layer, vec in delta.items():
            if layer in self.total:
                self.total[layer] += vec
            else:
                self.total[layer] = vec.clone()
            self.norms[layer] = self.norms.get(layer, 0.0) + norms[layer]
        self.count += 1

    def mean(self) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
        n = max(self.count, 1)
        return (
            {l: v / n for l, v in self.total.items()},
            {l: s / n for l, s in self.norms.items()},
        )


def _pooled(acts: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Drop the batch dim from recorded activations (batch size is always 1)."""
    return {l: a[0] for l, a in acts.items()}


@torch.inference_mode()
def instruction_diff(
    lm: SteerableLM,
    texts: list[str],
    instruction: str,
    name: str,
    baseline_instruction: str | None = None,
    layers: list[int] | None = None,
    kind: SiteKind = config.DEFAULT_SITE_KIND,
    pooling: Pooling = "last",
    progress: bool = True,
) -> VectorBank:
    """Activation delta between two instructions differing on one axis. The contrast
    must be an axis-matched placebo or the axes come out collinear.
    """
    layers = list(range(lm.n_layers)) if layers is None else list(layers)
    acc = _Accum()

    for text in tables.progress(
        texts, unit="text", desc=f"vec {name}", enabled=progress
    ):
        with_instr = _pooled(
            record(lm, lm.build_prompt(text, instruction), layers, kind, pooling)
        )
        without = _pooled(
            record(
                lm, lm.build_prompt(text, baseline_instruction), layers, kind, pooling
            )
        )
        delta = {l: with_instr[l] - without[l] for l in layers}
        # ref norm from the contrast pass: the state steering is added to at inference
        norms = {l: float(without[l].norm()) for l in layers}
        acc.add(delta, norms)

    vectors, ref_norms = acc.mean()
    return VectorBank(
        name=name,
        estimator="instruction_diff",
        model_repo=lm.repo_id,
        kind=kind,
        pooling=pooling,
        n_texts=len(texts),
        vectors=vectors,
        ref_norms=ref_norms,
    )


@torch.inference_mode()
def caa(
    lm: SteerableLM,
    simple_texts: list[str],
    complex_texts: list[str],
    name: str,
    layers: list[int] | None = None,
    kind: SiteKind = config.DEFAULT_SITE_KIND,
    pooling: Pooling = "mean_prompt",
    progress: bool = True,
) -> VectorBank:
    """``mean[h(simple)] - mean[h(complex)]`` over aligned pairs."""
    if len(simple_texts) != len(complex_texts):
        raise ValueError(
            f"pairs must be aligned: {len(simple_texts)} simple vs "
            f"{len(complex_texts)} complex"
        )
    layers = list(range(lm.n_layers)) if layers is None else list(layers)
    acc = _Accum()

    for simple, cplx in tables.progress(
        list(zip(simple_texts, complex_texts)),
        unit="pair",
        desc=f"vec {name}",
        enabled=progress,
    ):
        pos = _pooled(record(lm, lm.build_prompt(simple), layers, kind, pooling))
        neg = _pooled(record(lm, lm.build_prompt(cplx), layers, kind, pooling))
        acc.add(
            {l: pos[l] - neg[l] for l in layers},
            {l: float(neg[l].norm()) for l in layers},
        )

    vectors, ref_norms = acc.mean()
    return VectorBank(
        name=name,
        estimator="caa",
        model_repo=lm.repo_id,
        kind=kind,
        pooling=pooling,
        n_texts=len(simple_texts),
        vectors=vectors,
        ref_norms=ref_norms,
    )


@torch.inference_mode()
def diff_pca(
    lm: SteerableLM,
    simple_texts: list[str],
    complex_texts: list[str],
    name: str,
    layers: list[int] | None = None,
    kind: SiteKind = config.DEFAULT_SITE_KIND,
    pooling: Pooling = "mean_prompt",
) -> VectorBank:
    """Leading principal component of the per-pair deltas."""
    layers = list(range(lm.n_layers)) if layers is None else list(layers)
    per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    norms: dict[int, list[float]] = {l: [] for l in layers}

    for simple, cplx in zip(simple_texts, complex_texts):
        pos = _pooled(record(lm, lm.build_prompt(simple), layers, kind, pooling))
        neg = _pooled(record(lm, lm.build_prompt(cplx), layers, kind, pooling))
        for l in layers:
            per_layer[l].append(pos[l] - neg[l])
            norms[l].append(float(neg[l].norm()))

    vectors: dict[int, torch.Tensor] = {}
    for l in layers:
        deltas = torch.stack(per_layer[l])
        mean = deltas.mean(0)
        centred = deltas - mean
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        pc = vh[0]
        vectors[l] = pc if torch.dot(pc, mean) >= 0 else -pc

    return VectorBank(
        name=name,
        estimator="diff_pca",
        model_repo=lm.repo_id,
        kind=kind,
        pooling=pooling,
        n_texts=len(simple_texts),
        vectors=vectors,
        ref_norms={l: sum(norms[l]) / len(norms[l]) for l in layers},
    )


def symmetric_orthogonalize(
    banks: dict[str, VectorBank],
) -> dict[str, VectorBank]:
    """Lowdin-orthogonalise axis banks. Pairwise Gram-Schmidt flips the correlation
    sign instead of removing it.
    """
    names = sorted(banks)
    if len(names) < 2:
        return {n: banks[n] for n in names}

    shared = set.intersection(*(set(b.vectors) for b in banks.values()))
    vectors_out: dict[str, dict[int, torch.Tensor]] = {n: {} for n in names}
    worst_conditioning = (None, 1.0)

    for layer in sorted(shared):
        V = torch.stack([banks[n].unit(layer) for n in names], dim=1)
        gram = V.T @ V
        evals, evecs = torch.linalg.eigh(gram)
        smallest = float(evals.min())
        if smallest < worst_conditioning[1]:
            worst_conditioning = (layer, smallest)
        # near-zero eigenvalue: axes nearly collinear here, clamp keeps inv finite
        inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.T
        rotated = V @ inv_sqrt
        for i, name in enumerate(names):
            vectors_out[name][layer] = rotated[:, i].contiguous()

    layer, smallest = worst_conditioning
    if smallest < 0.05:
        print(
            f"[orth] WARNING: Gram matrix smallest eigenvalue {smallest:.4f} at "
            f"layer {layer}. The axes are nearly collinear there, so the "
            f"orthogonalised directions are close to arbitrary."
        )

    return {
        name: VectorBank(
            name=banks[name].name,
            estimator=banks[name].estimator,
            model_repo=banks[name].model_repo,
            kind=banks[name].kind,
            pooling=banks[name].pooling,
            n_texts=banks[name].n_texts,
            vectors=vectors_out[name],
            ref_norms={l: banks[name].ref_norms[l] for l in vectors_out[name]},
            orthogonalized=f"lowdin:{'+'.join(names)}",
        )
        for name in names
    }


def orthogonalize(bank: VectorBank, against: VectorBank) -> VectorBank:
    """Gram-Schmidt bank against another. Asymmetric; use symmetric_orthogonalize for
    a set.
    """
    out: dict[int, torch.Tensor] = {}
    for layer, v in bank.vectors.items():
        if layer not in against.vectors:
            out[layer] = v.clone()
            continue
        b = against.vectors[layer]
        denom = torch.dot(b, b)
        out[layer] = v - (torch.dot(v, b) / denom) * b if denom > 0 else v.clone()

    return VectorBank(
        name=bank.name,
        estimator=bank.estimator,
        model_repo=bank.model_repo,
        kind=bank.kind,
        pooling=bank.pooling,
        n_texts=bank.n_texts,
        vectors=out,
        ref_norms=dict(bank.ref_norms),
        orthogonalized=f"gram_schmidt:{against.name}",
    )


def entanglement_report(a: VectorBank, b: VectorBank) -> dict[int, float]:
    """Per-layer cosine between two axis vectors."""
    shared = sorted(set(a.vectors) & set(b.vectors))
    return {l: a.cosine(b, l) for l in shared}


def write_report(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
