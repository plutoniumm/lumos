#!/usr/bin/env python
"""Train a LoRA adapter for simplification, the weight-space arm of the
steering-vs-LoRA comparison. Trains on the bare task prompt (no simplify
instruction) against ASSET validation / OneStopEnglish, held disjoint from the
ASSET test split used for evaluation."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from lib import controls, data, models, tables  # noqa: E402
from lib import jsonio  # noqa: E402
from lib.metrics import readability  # noqa: E402


@dataclass
class Example:
    source: str
    target: str


def _asset_examples(refs_per_source: int, seed: int) -> list[Example]:
    """One example per (source, reference) pair, capped at *refs_per_source*."""
    rng = random.Random(seed)
    out: list[Example] = []
    for item in data.load_asset(config.LORA_TRAIN_SPLIT):
        refs = [r.strip() for r in item.references if r and r.strip()]
        src = item.source.strip()
        if not src or not refs:
            continue
        refs = [r for r in refs if r != src]
        if not refs:
            continue
        for ref in rng.sample(refs, min(refs_per_source, len(refs))):
            out.append(Example(src, ref))
    return out


def _onestop_examples() -> list[Example]:
    """OneStopEnglish paragraph pairs across all three level gaps."""
    lo, hi = config.SHOT_COMPRESSION_RANGE
    out: list[Example] = []
    for t in data.load_onestop():
        for src, tgt in (
            (t.advanced, t.elementary),
            (t.advanced, t.intermediate),
            (t.intermediate, t.elementary),
        ):
            src, tgt = src.strip(), tgt.strip()
            if not src or not tgt or src == tgt:
                continue
            n_src, n_tgt = len(src.split()), len(tgt.split())
            if n_src < 20 or n_tgt < 15:
                continue
            if not lo <= n_tgt / n_src <= hi:
                continue
            drop = readability.measure(src).fkgl - readability.measure(tgt).fkgl
            if drop < config.SHOT_MIN_FKGL_DROP:
                continue
            out.append(Example(src, tgt))
    return out


def build_examples(corpus: str, refs_per_source: int, seed: int) -> list[Example]:
    """Training pairs from *corpus*: ``asset``, ``onestop``, or ``both``."""
    out: list[Example] = []
    if corpus in ("asset", "both"):
        out += _asset_examples(refs_per_source, seed)
    if corpus in ("onestop", "both"):
        out += _onestop_examples()
    if not out:
        raise SystemExit(f"no training pairs for corpus {corpus!r}")
    random.Random(seed).shuffle(out)
    return out


def encode(
    lm, ex: Example, max_len: int, use_controls: bool = False
) -> dict[str, torch.Tensor] | None:
    """Prompt+target ids with the prompt masked out of the loss."""
    source = (
        controls.prefix(ex.source, ex.target) + ex.source
        if use_controls
        else ex.source
    )
    prompt = lm.build_prompt(source)
    prompt_ids = lm.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = lm.tokenizer(ex.target, add_special_tokens=False)["input_ids"]
    target_ids = target_ids + [lm.tokenizer.eos_token_id]

    ids = prompt_ids + target_ids
    if len(ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
    }


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad each example to the batch width (training is teacher-forced)."""
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = width - n
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100)]))
        mask.append(
            torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(mask),
    }


def batches(encoded: list[dict], size: int, pad_id: int, shuffle: bool, seed: int):
    order = list(range(len(encoded)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), size):
        chunk = [encoded[j] for j in order[i : i + size]]
        yield collate(chunk, pad_id)


@torch.no_grad()
def dev_loss(model, encoded: list[dict], size: int, pad_id: int, device) -> float:
    model.eval()
    total, n_tok = 0.0, 0
    for batch in tables.progress(
        list(batches(encoded, size, pad_id, shuffle=False, seed=0)),
        unit="batch",
        desc="dev",
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        counted = int((batch["labels"] != -100).sum())
        total += float(out.loss) * counted
        n_tok += counted
    model.train()
    return total / max(n_tok, 1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    p.add_argument("--name", default="asset_r16", help="adapter directory name")
    p.add_argument("--rank", type=int, default=config.LORA_RANK)
    p.add_argument("--alpha", type=int, default=config.LORA_ALPHA)
    p.add_argument("--lr", type=float, default=config.LORA_LR)
    p.add_argument("--epochs", type=int, default=config.LORA_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.LORA_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=config.LORA_GRAD_ACCUM)
    p.add_argument("--refs", type=int, default=config.LORA_REFS_PER_SOURCE)
    p.add_argument(
        "--corpus",
        default=None,
        choices=("asset", "onestop", "both"),
        help="asset = sentences, onestop = paragraphs, both = mixed lengths "
        "(default: asset). Left unset rather than defaulting to the literal "
        "string here, because src/simplify.py's find_lora_dir() tells the "
        "headline adapter from an ablation by whether train_meta.json's "
        "'corpus' is null -- a default of 'asset' would make every ordinary "
        "run indistinguishable from '--corpus asset' typed on purpose, and "
        "auto-detection would silently fall back to picking a candidate "
        "adapter alphabetically instead of the headline one.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=config.LORA_MAX_SEQ_LEN,
        help="paragraph pairs need roughly 768; the sentence default is 320",
    )
    p.add_argument("--max-examples", type=int, default=0, help="0 = all")
    p.add_argument(
        "--controls",
        action="store_true",
        help="prepend ACCESS-style control ratios (len/wps/hard) measured from "
        "each training pair, so the adapter can be asked for a target at "
        "inference instead of pushed by an opaque dial. See lib/controls.py",
    )
    p.add_argument(
        "--dev-every",
        type=int,
        default=20,
        help="evaluate dev loss every N optimizer steps (0 = endpoints only). "
        "The dev split is small, so this is a few seconds per pass and it is "
        "what separates 'still improving' from 'past the minimum'",
    )
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("peft is not installed:  pip install peft")
        return 2

    from contextlib import suppress

    with suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    corpus = args.corpus or "asset"
    print(f"[data] corpus={corpus} (asset split {config.LORA_TRAIN_SPLIT}, "
          f"{args.refs} ref(s) per source)")
    examples = build_examples(corpus, args.refs, args.seed)
    words = sum(len(e.source.split()) for e in examples) / max(len(examples), 1)
    print(f"[data] mean source length {words:.0f} words")
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(f"[data] {len(examples)} training pair(s)")

    print(f"\n[load] {args.model} on {config.DEVICE} ({config.DTYPE})")
    t0 = time.perf_counter()
    lm = models.load(args.model)
    print(f"[load] {lm.n_layers} layers, {time.perf_counter() - t0:.1f}s")

    encoded = [encode(lm, ex, args.max_seq_len, args.controls) for ex in examples]
    dropped = sum(1 for e in encoded if e is None)
    encoded = [e for e in encoded if e is not None]
    if dropped:
        print(f"[data] dropped {dropped} pair(s) over {args.max_seq_len} tokens")

    n_dev = max(16, int(len(encoded) * config.LORA_DEV_FRAC))
    dev, train = encoded[:n_dev], encoded[n_dev:]
    print(f"[data] {len(train)} train / {len(dev)} dev")

    peft_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=list(config.LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(lm.model, peft_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[lora] r={args.rank} alpha={args.alpha} "
        f"{n_trainable:,} trainable / {n_total:,} ({100 * n_trainable / n_total:.2f}%)"
    )

    pad_id = lm.tokenizer.pad_token_id
    steps_per_epoch = math.ceil(len(train) / args.batch_size / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(total_steps * config.LORA_WARMUP_FRAC))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    print(f"\n[train] {total_steps} optimizer step(s), lr={args.lr}, warmup={warmup}")
    d0 = dev_loss(model, dev, args.batch_size, pad_id, lm.device)
    print(f"[train] dev loss before training: {d0:.4f}")

    model.train()
    history = [{"step": 0, "dev_loss": d0}]
    t_train = time.perf_counter()
    step = 0
    micro = 0
    running = 0.0
    bar = tables.progress(total=total_steps, unit="step", desc="train", leave=True)
    for epoch in range(args.epochs):
        for batch in batches(train, args.batch_size, pad_id, True, args.seed + epoch):
            batch = {k: v.to(lm.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss.detach()) * args.grad_accum
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            bar.update(1)
            if step % 20 == 0 or step == total_steps:
                mean = (
                    running / (20 * args.grad_accum)
                    if step % 20 == 0
                    else running / micro
                )
                rate = (time.perf_counter() - t_train) / step
                bar.set_postfix(loss=f"{mean:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                bar.write(
                    f"  step {step:4d}/{total_steps}  loss {mean:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {rate:.2f}s/step  "
                    f"eta {(total_steps - step) * rate / 60:.1f}m"
                )
                history.append({"step": step, "train_loss": mean})
                running = 0.0

            if args.dev_every and step % args.dev_every == 0 and step != total_steps:
                d = dev_loss(model, dev, args.batch_size, pad_id, lm.device)
                history.append({"step": step, "dev_loss": d})
                bar.write(f"  step {step:4d}/{total_steps}  dev loss {d:.4f}")
    bar.close()

    d1 = dev_loss(model, dev, args.batch_size, pad_id, lm.device)
    elapsed = time.perf_counter() - t_train
    print(f"\n[train] dev loss after training:  {d1:.4f}  (was {d0:.4f})")
    print(f"[train] {elapsed / 60:.1f} minutes")
    history.append({"step": step, "dev_loss": d1})

    dest = config.LORA_DIR / args.name
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    meta = {
        "base_model": args.model,
        "corpus": args.corpus,
        "controls": args.controls,
        "train_split": f"asset/{config.LORA_TRAIN_SPLIT}",
        "max_seq_len": args.max_seq_len,
        "eval_split_reserved": "asset/test",
        "prompt": "config.BASE_TASK, no simplify instruction",
        "n_train": len(train),
        "n_dev": len(dev),
        "refs_per_source": args.refs,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": config.LORA_DROPOUT,
        "target_modules": list(config.LORA_TARGET_MODULES),
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "trainable_params": n_trainable,
        "dev_loss_before": d0,
        "dev_loss_after": d1,
        "train_minutes": elapsed / 60,
        "history": history,
    }
    jsonio.dump(dest / "train_meta.json", meta)
    print(f"[save] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
