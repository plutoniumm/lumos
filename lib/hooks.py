"""Forward hooks. Recorder reads the residual stream, Injector adds alpha*v."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, get_args

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.models import SteerableLM  # noqa: E402

SiteKind = Literal["resid_post", "mlp_out"]
Positions = Literal["gen_only", "prompt_last", "prompt_all", "all"]
Pooling = Literal["last", "mean_prompt", "mean_span"]

VALID_POSITIONS = frozenset(get_args(Positions))
VALID_POOLING = frozenset(get_args(Pooling))


def resolve_module(lm: SteerableLM, layer: int, kind: SiteKind) -> nn.Module:
    if kind == "resid_post":
        return lm.layer(layer)
    if kind == "mlp_out":
        return lm.mlp(layer)
    raise ValueError(f"unknown site kind {kind!r}")


# Decoder blocks return a bare tensor in some versions and a tuple in others;
# the feed-forward always returns a tensor. Normalise both ways.
def _split(out):
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


def _rejoin(hidden, rest):
    return hidden if rest is None else (hidden,) + rest


def _to_cpu32(t: torch.Tensor) -> torch.Tensor:
    """Detach to CPU float32. Cast dtype before device: a fused .to() drops the MPS
    storage offset.
    """
    return t.detach().float().to("cpu")


def _is_altup(hidden: torch.Tensor) -> bool:
    """Gemma-3n decoder layers return [streams, batch, seq, hidden], rank 4."""
    return hidden.dim() == 4


@dataclass
class Recorder:
    """Capture pooled hidden states at a set of layers during one forward pass."""

    lm: SteerableLM
    layers: Iterable[int]
    kind: SiteKind = "resid_post"
    pooling: Pooling = "last"
    # A single tuple applies to every row; a list gives one span per batch row.
    span: tuple[int, int] | list[tuple[int, int]] | None = None
    # Needed by mean_prompt: batches are left-padded, and the mask never reaches
    # a forward hook on its own.
    attention_mask: torch.Tensor | None = None
    _store: dict[int, torch.Tensor] = field(default_factory=dict)
    _handles: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pooling not in VALID_POOLING:
            raise ValueError(
                f"unknown pooling {self.pooling!r}; expected one of "
                f"{sorted(VALID_POOLING)}"
            )
        if self.pooling == "mean_span" and self.span is None:
            raise ValueError("pooling='mean_span' requires span=(start, end)")

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        # Record from the primary altup stream; the others are corrections to it.
        if _is_altup(hidden):
            hidden = hidden[0]

        if self.pooling == "last":
            return hidden[:, -1, :]

        if self.pooling == "mean_prompt":
            if self.attention_mask is None:
                return hidden.mean(dim=1)
            mask = self.attention_mask.to(hidden.device).unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        # mean_span
        if isinstance(self.span, tuple):
            start, end = self.span
            return hidden[:, start:end, :].mean(dim=1)
        return torch.stack(
            [hidden[i, s:e, :].mean(dim=0) for i, (s, e) in enumerate(self.span)]
        )

    def _make_hook(self, layer: int) -> Callable:
        def hook(_module, _args, output):
            hidden, _ = _split(output)
            self._store[layer] = _to_cpu32(self._pool(hidden))
            return None  # read-only

        return hook

    def __enter__(self) -> "Recorder":
        self._store.clear()
        for layer in self.layers:
            mod = resolve_module(self.lm, layer, self.kind)
            self._handles.append(mod.register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get(self) -> dict[int, torch.Tensor]:
        return dict(self._store)


@torch.inference_mode()
def record(
    lm: SteerableLM,
    prompt: str,
    layers: Iterable[int] | None = None,
    kind: SiteKind = "resid_post",
    pooling: Pooling = "last",
    span: tuple[int, int] | list[tuple[int, int]] | None = None,
) -> dict[int, torch.Tensor]:
    """One forward pass, pooled activations at every requested layer."""
    return record_inputs(lm, lm.encode(prompt), layers, kind, pooling, span)


@torch.inference_mode()
def record_inputs(
    lm: SteerableLM,
    inputs: dict[str, torch.Tensor],
    layers: Iterable[int] | None = None,
    kind: SiteKind = "resid_post",
    pooling: Pooling = "last",
    span: tuple[int, int] | list[tuple[int, int]] | None = None,
) -> dict[int, torch.Tensor]:
    """As ``record``, but on already-encoded (possibly batched) inputs."""
    layers = list(range(lm.n_layers)) if layers is None else list(layers)
    rec = Recorder(
        lm, layers, kind, pooling, span, attention_mask=inputs.get("attention_mask")
    )
    with rec:
        lm.model(**inputs)
    return rec.get()


@dataclass
class InjectionSite:
    """One steering intervention."""

    layer: int
    vector: torch.Tensor  # [hidden]
    alpha: float
    kind: SiteKind = "resid_post"
    positions: Positions = "gen_only"
    norm_preserve: bool = True
    # Content direction(s) to project the vector away from, so steering cannot
    # push along the axis carrying this text's meaning. Shape [hidden] applies to
    # every row; [batch, hidden] gives one direction per row.
    content: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # Silently mis-steering on a typo is worse than crashing: 'prompt_alll'
        # used to fall through and behave as 'all'.
        if self.positions not in VALID_POSITIONS:
            raise ValueError(
                f"unknown positions {self.positions!r}; expected one of "
                f"{sorted(VALID_POSITIONS)}"
            )


@dataclass
class _SiteState:
    calls: int = 0
    prompt_len: int = 0
    saw_decode: bool = False


@dataclass
class Injector:
    """Add ``alpha * v`` to the stream at each site for the duration of a block."""

    lm: SteerableLM
    sites: list[InjectionSite]
    _state: dict[int, _SiteState] = field(default_factory=dict)

    _handles: list = field(default_factory=list)

    def reset(self) -> None:
        self._state.clear()

    def call_counts(self) -> dict[int, int]:
        """Forward-call count per site index. For tests and debugging."""
        return {i: s.calls for i, s in self._state.items()}

    def prompt_lens(self) -> dict[int, int]:
        """Prompt length each site inferred from its first call of a generate."""
        return {i: s.prompt_len for i, s in self._state.items()}

    def _effective(
        self, site: InjectionSite, dtype, device, batch_size: int
    ) -> torch.Tensor:
        """Steering direction per batch row, shape [batch, hidden]."""
        v = site.vector.to(device=device, dtype=torch.float32)
        if site.content is None:
            return v.unsqueeze(0).expand(batch_size, -1).to(dtype)

        c = site.content.to(device=device, dtype=torch.float32)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        denom = (c * c).sum(-1, keepdim=True).clamp_min(1e-8)
        v_eff = v.unsqueeze(0) - ((c @ v).unsqueeze(-1) / denom) * c
        if v_eff.shape[0] == 1 and batch_size > 1:
            v_eff = v_eff.expand(batch_size, -1)
        return v_eff.to(dtype)

    def _target_slice(
        self, site: InjectionSite, seq: int, state: _SiteState, is_prefill: bool
    ) -> slice | None:
        """Which sequence positions this call should modify, or None to skip."""
        if site.positions == "all":
            return slice(0, seq)

        if site.positions == "gen_only":
            if is_prefill:
                return None
            # Cached decode carries only the new token; uncached decode carries
            # the whole sequence, and only the tail past the prompt is new.
            return slice(state.prompt_len, seq) if seq > 1 else slice(0, seq)

        # prompt_last / prompt_all
        if not is_prefill:
            return None
        if site.positions == "prompt_last":
            return slice(state.prompt_len - 1, state.prompt_len)
        return slice(0, state.prompt_len)

    def _make_hook(self, index: int, site: InjectionSite) -> Callable:
        def hook(_module, _args, output):
            hidden, rest = _split(output)
            altup = _is_altup(hidden)
            # For altup the axes are [streams, batch, seq, hidden].
            seq = hidden.shape[-2]
            batch = hidden.shape[1] if altup else hidden.shape[0]

            state = self._state.get(index)
            if state is None:
                state = _SiteState(prompt_len=seq)
                self._state[index] = state
            elif seq > 1 and state.saw_decode:
                # A long sequence after decode steps means a new generate began
                # inside the same block; treat it as a fresh prefill.
                state = _SiteState(prompt_len=seq)
                self._state[index] = state

            is_prefill = state.calls == 0
            state.calls += 1
            if not is_prefill:
                state.saw_decode = True

            if site.alpha == 0.0:
                return None  # exact no-op, so alpha=0 is bit-identical

            target = self._target_slice(site, seq, state, is_prefill)
            if target is None or target.start >= target.stop:
                return None

            v = self._effective(site, hidden.dtype, hidden.device, batch)
            delta = torch.zeros_like(hidden)
            if altup:
                # Apply the same correction to every prediction stream.
                delta[:, :, target, :] = site.alpha * v.unsqueeze(0).unsqueeze(2)
            else:
                delta[:, target, :] = site.alpha * v.unsqueeze(1)

            if site.norm_preserve:
                before = hidden.norm(dim=-1, keepdim=True)
                shifted = hidden + delta
                after = shifted.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                hidden = shifted * (before / after)
            else:
                hidden = hidden + delta

            return _rejoin(hidden, rest)

        return hook

    def __enter__(self) -> "Injector":
        self.reset()
        for i, site in enumerate(self.sites):
            mod = resolve_module(self.lm, site.layer, site.kind)
            self._handles.append(mod.register_forward_hook(self._make_hook(i, site)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextmanager
def steered(lm: SteerableLM, sites: list[InjectionSite]):
    """``with steered(lm, sites): lm.model.generate(...)`` -- one block per call."""
    inj = Injector(lm, sites)
    with inj:
        yield inj
