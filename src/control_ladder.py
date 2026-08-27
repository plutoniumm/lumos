#!/usr/bin/env python
"""Ask the control-token adapter for a target, and see what comes back.

The question
------------
Every other dial here answers "how hard should I push". Steering's alpha is
non-monotone and collapses; the LoRA's scale peaks below its trained strength.
Neither answers "produce this at grade 8", which is the request an actual
deployment makes.

``src/train_lora.py --controls`` trains an adapter that sees, prepended to each
source, the ratios the reference achieved on it (``lib/controls.py``). Here the
same prefix becomes a request rather than a measurement, and the two things
worth knowing are:

  responsiveness   does the output move when the request moves, monotonically
                   and in the right direction
  calibration      does a request of 0.6 actually produce 0.6

Responsiveness without calibration is still a usable dial -- you invert it once
from a table. Calibration without responsiveness is impossible. Neither is
readability alone: an adapter can hit any ratio by deleting the sentence, so
NLI and degeneration are reported at every rung.

One axis is swept at a time with the others held at 1.0, because the point is
whether each control does what it says, not whether some combination scores
well. A combined sweep is what the dual-steering surface did, and that only
became interpretable once the marginals were in the same run.

    python src/control_ladder.py --lora models/lora/asset_ctrl_r16 --n 32
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from lib import controls, guidance, jsonio, tables  # noqa: E402
from lib.bench import Bench  # noqa: E402
from lib.metrics import readability  # noqa: E402
from lib.steering import SteeringSpec  # noqa: E402

# What each control is supposed to move, so "did it respond" is checked against
# the axis it claims rather than against FKGL for everything.
TARGET_OF = {
    "len": "n_words",
    "wps": "words_per_sentence",
    "hard": "pct_hard_words",
}


def achieved(source: str, output: str, axis: str) -> float | None:
    s, o = readability.measure(source), readability.measure(output)
    a, b = getattr(s, TARGET_OF[axis]), getattr(o, TARGET_OF[axis])
    return b / a if a else None


@torch.no_grad()
def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=config.ACTIVE_MODEL)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--asset-split", default="test")
    p.add_argument("--axes", nargs="+", default=list(controls.CONTROLS))
    p.add_argument("--requests", default="0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="classifier-free guidance on the control prefix. 1.0 is ordinary "
        "conditional decoding; higher amplifies the control's effect on the "
        "logits, which is needed because the learned conditioning is too weak "
        "to move a greedy argmax on its own",
    )
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--out", type=Path, default=config.RESULTS_DIR / "control_ladder.json")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(config.SEED)

    asked = [float(x) for x in args.requests.split(",") if x.strip()]
    bench = Bench.open(args, needs_banks=False)
    if not bench.adapter:
        raise SystemExit("--lora is required")

    base = bench.baselines(
        args.out.with_suffix(".baselines.json"),
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    base_ppl = [bench.conditional_ppl(it, b) for it, b in zip(bench.items, base)]

    report = {"model": args.model, "lora": str(args.lora), "n": len(bench.items),
              "requests": asked, "axes": args.axes, "guidance": args.guidance,
              "rungs": []}

    for axis in args.axes:
        for want in asked:
            spec_sources = [
                controls.request(**{axis: want}) + it.source for it in bench.items
            ]
            t = time.perf_counter()
            if args.guidance == 1.0:
                outs = bench.generate(
                    SteeringSpec.off(), spec_sources,
                    max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
            else:
                neutral = [controls.request() + it.source for it in bench.items]
                outs = []
                for i in range(0, len(spec_sources), args.batch_size):
                    j = i + args.batch_size
                    outs.extend(guidance.generate(
                        bench.lm, spec_sources[i:j], neutral[i:j],
                        gamma=args.guidance,
                        max_new_tokens=args.max_new_tokens))
            secs = time.perf_counter() - t
            scored = [bench.score(it, o, baseline_ppl=bp)
                      for it, o, bp in zip(bench.items, outs, base_ppl)]
            got = [achieved(it.source, o, axis)
                   for it, o, s in zip(bench.items, outs, scored) if not s.degenerate]
            rung = {
                "axis": axis, "requested": want,
                "achieved": jsonio.mean(g for g in got if g is not None),
                "fkgl": jsonio.mean(s.read.fkgl for s in scored),
                "nli": jsonio.mean(s.nli for s in scored),
                "sari": jsonio.mean(s.sari for s in scored if s.sari is not None),
                "pct_degenerate": jsonio.pct(s.degenerate for s in scored),
                "seconds": round(secs, 1),
            }
            report["rungs"].append(rung)
            tables.write(
                f"  {axis:<5} asked {want:.2f}  ->  achieved "
                f"{'--' if rung['achieved'] is None else format(rung['achieved'], '.3f')}"
                f"  fkgl {rung['fkgl']:.2f}  nli {rung['nli']:.3f}  "
                f"sari {'--' if rung['sari'] is None else format(rung['sari'], '.2f')}"
                f"  degen {rung['pct_degenerate']:.1f}%  ({secs:.0f}s)")

    for axis in args.axes:
        rows = [r for r in report["rungs"] if r["axis"] == axis]
        t = tables.table(title=f"control '{axis}' -> {TARGET_OF[axis]} ratio")
        for h in ("asked", "achieved", "error", "FKGL", "NLI", "SARI", "degen%"):
            t.add_column(h, justify="right")
        for r in rows:
            a = r["achieved"]
            t.add_row(
                f"{r['requested']:.2f}",
                "--" if a is None else f"{a:.3f}",
                "--" if a is None else f"{a - r['requested']:+.3f}",
                f"{r['fkgl']:.2f}", f"{r['nli']:.3f}",
                "--" if r["sari"] is None else f"{r['sari']:.2f}",
                f"{r['pct_degenerate']:.1f}")
        print()
        tables.show(t)

    jsonio.dump(args.out, report)
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
