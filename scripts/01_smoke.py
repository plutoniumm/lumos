"""First thing to run on a new machine: does the model load and generate?

    python scripts/01_smoke.py                  # ACTIVE_MODEL if complete, else small
    python scripts/01_smoke.py --small          # force the small ungated model
    python scripts/01_smoke.py org/repo         # any repo or local dir

No steering here -- this only proves the environment. ``scripts/02_hook_noop.py``
is what proves the hooks.

``hub.resolve`` treats any directory holding a config.json as a finished
download, so a half-fetched model would be loaded and fail deep inside
from_pretrained. This script checks for aria2 control files first and falls back
to the small model rather than reporting a broken environment.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from lib import hub, models
from lib.metrics import readability

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"

HARD = (
    "The proliferation of antimicrobial resistance determinants among "
    "commensal Enterobacteriaceae constitutes a substantial epidemiological "
    "concern, notwithstanding the attenuated virulence of these organisms."
)

MAX_NEW = 96


def download_state(repo_id: str) -> tuple[bool, str]:
    """Is the local copy of *repo_id* usable? Second element says why not.

    aria2 leaves a ``<file>.aria2`` control file beside anything unfinished, and
    ``hub.download`` writes ``.download_meta.json`` only after re-planning and
    finding nothing left to fetch. Both must agree before we trust the shards.
    Note ``.aria2`` is also the name of hub's workdir *directory*, hence is_file.
    """
    path = hub.local_path(repo_id)
    if not (path / "config.json").exists():
        return False, f"no local copy at {path}"
    partial = sorted(p.name for p in path.glob("*.aria2") if p.is_file())
    if partial:
        return False, f"{len(partial)} shard(s) unfinished, e.g. {partial[0]}"
    if not (path / ".download_meta.json").exists():
        return False, f"{path}: no .download_meta.json, hub.download never completed"
    if not any(path.glob("*.safetensors")):
        return False, f"{path} has config.json but no weights"
    return True, f"complete at {path}"


def pick(argv: list[str]) -> str:
    if argv and argv[0] not in ("--small",):
        return argv[0]
    if argv and argv[0] == "--small":
        print(f"[smoke] --small: using {SMALL}")
        return SMALL
    ready, why = download_state(config.ACTIVE_MODEL)
    print(f"[smoke] ACTIVE_MODEL {config.ACTIVE_MODEL}: {why}")
    if ready:
        return config.ACTIVE_MODEL
    print(f"[smoke] falling back to {SMALL}")
    return SMALL


def load(repo_id: str) -> models.SteerableLM:
    try:
        return models.load(repo_id)
    except Exception as exc:
        print(f"[smoke] {repo_id} failed to load: {type(exc).__name__}: {exc}")
        if repo_id == SMALL:
            print(f"[smoke] trying {FALLBACK}")
            return models.load(FALLBACK)
        raise


@torch.inference_mode()
def main(argv: list[str]) -> int:
    torch.manual_seed(config.SEED)

    t0 = time.perf_counter()
    lm = load(pick(argv))
    load_s = time.perf_counter() - t0

    print(
        f"\nmodel        {lm.repo_id}\n"
        f"class        {type(lm.model).__name__}\n"
        f"n_layers     {lm.n_layers}\n"
        f"hidden_size  {lm.hidden_size}\n"
        f"dtype        {lm.dtype}\n"
        f"device       {lm.device}\n"
        f"load         {load_s:.1f}s\n"
    )

    prompt = lm.build_prompt(HARD, config.INSTRUCTIONS["simplify"])
    batch = lm.encode(prompt)
    prompt_len = int(batch["input_ids"].shape[1])

    # Warm the kernels first; on MPS the first pass pays for graph compilation
    # and would make the tokens/sec number meaningless.
    lm.model.generate(
        **batch,
        max_new_tokens=4,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=lm.tokenizer.eos_token_id,
    )

    t0 = time.perf_counter()
    out = lm.model.generate(
        **batch,
        max_new_tokens=MAX_NEW,
        do_sample=config.DO_SAMPLE,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=lm.tokenizer.eos_token_id,
    )
    gen_s = time.perf_counter() - t0

    text = lm.decode_new(out, prompt_len)[0].strip()
    n_new = int(out.shape[1]) - prompt_len

    print(f"prompt ({prompt_len} tokens):\n  {HARD}\n")
    print(f"simplification ({n_new} new tokens):\n  {text}\n")

    src, dst = readability.measure(HARD), readability.measure(text)
    print(
        f"fkgl         {src.fkgl:.1f} -> {dst.fkgl:.1f}\n"
        f"words/sent   {src.words_per_sentence:.1f} -> {dst.words_per_sentence:.1f}\n"
        f"hard words   {src.pct_hard_words:.2f} -> {dst.pct_hard_words:.2f}\n"
        f"degenerate   {dst.degenerate}\n"
    )
    print(f"{n_new / gen_s:.1f} tokens/sec ({n_new} tokens in {gen_s:.2f}s, greedy)")

    if not text:
        print("\n[smoke] FAIL: empty generation")
        return 1
    print("\n[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
