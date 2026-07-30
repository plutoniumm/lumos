"""Model loading and uniform decoder-layer access across Llama and Gemma-3n."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from lib import hub  # noqa: E402

# Candidate paths to the decoder ModuleList, most specific first.
_LAYER_PATHS = (
    "model.language_model.layers",  # Gemma3nForConditionalGeneration
    "language_model.model.layers",
    "model.layers",  # Llama, Gemma3nForCausalLM
    "transformer.h",
)


def _dig(root: object, dotted: str) -> object | None:
    cur = root
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _find_layers(model: nn.Module) -> nn.ModuleList:
    for path in _LAYER_PATHS:
        found = _dig(model, path)
        if isinstance(found, nn.ModuleList) and len(found) > 0:
            return found
    raise RuntimeError(
        f"could not locate decoder layers on {type(model).__name__}; "
        f"tried {_LAYER_PATHS}"
    )


def _hidden_size(model: nn.Module) -> int:
    cfg = model.config
    for attr in ("hidden_size", "n_embd"):
        if getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None):
        return int(text_cfg.hidden_size)
    raise RuntimeError(f"could not determine hidden size for {type(model).__name__}")


@dataclass
class SteerableLM:
    """A causal LM plus uniform access to its decoder layers."""

    model: nn.Module
    tokenizer: object
    repo_id: str
    device: torch.device
    dtype: torch.dtype
    _layers: nn.ModuleList

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def hidden_size(self) -> int:
        return _hidden_size(self.model)

    def layer(self, i: int) -> nn.Module:
        """Decoder block *i*. Negative indices count from the end."""
        if not -self.n_layers <= i < self.n_layers:
            raise IndexError(f"layer {i} out of range for {self.n_layers} layers")
        return self._layers[i]

    def mlp(self, i: int) -> nn.Module:
        """The feed-forward submodule of block *i*."""
        blk = self.layer(i)
        for attr in ("mlp", "feed_forward", "ffn"):
            mod = getattr(blk, attr, None)
            if isinstance(mod, nn.Module):
                return mod
        raise RuntimeError(f"no feed-forward submodule on {type(blk).__name__}")

    def build_prompt(
        self,
        text: str,
        instruction: str | None = None,
        task: str | None = None,
    ) -> str:
        """Chat-formatted prompt, ready for generation."""
        task = config.BASE_TASK if task is None else task
        header = " ".join(p for p in (task, instruction) if p)
        content = f"{header}\n\n{text}" if header else text
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def text_token_span(self, prompt: str, text: str) -> tuple[int, int]:
        """Token index range covering ``text`` inside ``prompt``."""
        start_char = prompt.find(text)
        if start_char < 0:
            enc = self.tokenizer(prompt, add_special_tokens=False)
            return 0, len(enc["input_ids"])
        end_char = start_char + len(text)
        enc = self.tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=False
        )
        idx = [
            i
            for i, (a, b) in enumerate(enc["offset_mapping"])
            if b > start_char and a < end_char
        ]
        return (idx[0], idx[-1] + 1) if idx else (0, len(enc["input_ids"]))

    def encode(self, prompt: str) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.device) for k, v in batch.items()}

    def encode_batch(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        """Left-padded batch. Padding side matters: with right padding, generation"""
        batch = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def padded_spans(
        self,
        prompts: list[str],
        texts: list[str],
        attention_mask: torch.Tensor,
    ) -> list[tuple[int, int]]:
        """Text token spans shifted by each row's left padding."""
        total = attention_mask.shape[1]
        spans = []
        for i, (prompt, text) in enumerate(zip(prompts, texts)):
            start, end = self.text_token_span(prompt, text)
            pad = total - int(attention_mask[i].sum())
            spans.append((start + pad, end + pad))
        return spans

    def decode_new(self, sequences: torch.Tensor, prompt_len: int) -> list[str]:
        return self.tokenizer.batch_decode(
            sequences[:, prompt_len:], skip_special_tokens=True
        )


def load(repo_id: str | None = None, device: str | None = None) -> SteerableLM:
    """Load a model for steering. Resolves to the local aria2c download if present."""
    repo_id = repo_id or config.ACTIVE_MODEL
    device = torch.device(device or config.DEVICE)
    dtype = getattr(torch, config.DTYPE)
    source = hub.resolve(repo_id)

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Llama ships no pad token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return SteerableLM(
        model=model,
        tokenizer=tokenizer,
        repo_id=repo_id,
        device=device,
        dtype=dtype,
        _layers=_find_layers(model),
    )
