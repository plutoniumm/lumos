"""Perplexity of the output under the unsteered model. No hook may be attached."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.models import SteerableLM  # noqa: E402


@torch.inference_mode()
def conditional_ppl(lm: SteerableLM, prompt: str, output: str) -> float:
    """Perplexity of output given prompt. No injector may be attached; the mask uses
    the shared token prefix.
    """
    if not output.strip():
        return float("inf")

    prompt_ids = lm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    full_ids = lm.tokenizer(
        prompt + output, return_tensors="pt", add_special_tokens=False
    )["input_ids"]

    # BPE can merge across the join, so the two encodings are not guaranteed to
    # share a prefix: the Llama-3 template ends in "\n\n" and an output starting
    # "\n\n" merges into a single different token, shifting every label. Mask by
    # the actual common prefix instead of by the prompt's own length.
    n_prompt = prompt_ids.shape[1]
    shared = 0
    limit = min(n_prompt, full_ids.shape[1])
    while shared < limit and prompt_ids[0, shared] == full_ids[0, shared]:
        shared += 1
    n_prompt = shared
    if full_ids.shape[1] <= n_prompt:
        return float("inf")

    input_ids = full_ids.to(lm.device)
    labels = input_ids.clone()
    labels[:, :n_prompt] = -100  # score only the generated continuation

    loss = lm.model(input_ids=input_ids, labels=labels).loss
    return math.exp(min(float(loss), 20.0))  # cap so a blown-up run stays sortable


def ppl_ratio(lm: SteerableLM, prompt: str, steered: str, reference: str) -> float:
    """Steered output's perplexity relative to the unsteered output's."""
    ref = conditional_ppl(lm, prompt, reference)
    new = conditional_ppl(lm, prompt, steered)
    if not math.isfinite(ref) or ref <= 0:
        return float("inf")
    return new / ref
