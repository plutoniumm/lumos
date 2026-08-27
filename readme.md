# steering

Simplify text without changing what it says, using a continuous strength dial
instead of a prompt.

A prompt is discrete: *make this easier to read* either fires or it does not.
This repository adds a fixed direction to the model's residual stream during
generation, scaled by a strength `alpha`, so the degree of simplification is a
number set at inference. No weights are trained.

Full write-up with method, figures and derivations:
[`docs/report/steering.tex`](docs/report/steering.tex).
Every measured number: [`report.md`](report.md).
Real generated output: [`examples.md`](examples.md).

Negative results: [`fail.md`](fail.md). Contrastive decoding, dual steering,
ASSET-domain vector estimation, paragraph-trained LoRA and the perplexity gate
were built, measured and dropped. The code is removed and the measurements are
kept, so none of them needs re-running.

## Result

Structural steering at layer 15, `alpha=0.75`, on `Llama-3.2-3B-Instruct`,
ASSET test set, n=32, paired per item with 95% confidence intervals.

Two metrics carry the result. **SARI** is the standard simplification score: it
credits words the system added, kept or removed where human references made the
same edit, on a 0 to 100 scale. **Faithfulness** is entailment probability from
an NLI model run in both directions, so the source must imply the output and the
output must imply the source, on a 0 to 1 scale. The first measures whether the
text got simpler in a way people recognise; the second, whether anything was
dropped or invented.

| comparison | SARI | faithfulness (NLI) |
|---|---|---|
| vs. prompt-only | **+5.47** [+2.58, +8.48] | **+0.33** [+0.20, +0.46] |
| vs. no intervention | +0.43 [-1.71, +2.59] | -0.08 [-0.21, +0.05] |

It beats prompting on both, significantly. Against no intervention it ties on
both, while cutting sentence length 18% and reading grade 19% at zero
degeneration — simplification with nothing else changed.

Prompt-only simplification is significantly **worse** than no intervention
(-5.04 SARI, NLI 0.46, perplexity ratio 3.78). It reaches a low reading grade
by deleting facts and inventing examples; `examples.md` section 2 shows it
replacing an income statistic with an analogy.

**One dial, not two.** The design started with two: one for sentence structure,
one for word difficulty. Both exist inside the model. Their directions are
nearly perpendicular, at cosine similarity 0.223 where 1.0 would mean a single
direction doing both jobs, and they act at different depths of the 28-layer
stack (structure at 15-17, vocabulary at 19-26). But the vocabulary dial is a
SARI tie trending negative (-1.37), and running both at once breaks 34% of
outputs into loops or fragments. The two-dial hypothesis holds as a mechanism
and fails in practice.

**The result is bounded by model scale.** The prompting failure it is measured
against is a property of the 3B model. On `gemma-4-12B-it` the same instruction
*gains* SARI (+2.91\*) at no measurable faithfulness cost, and steering is a
middling arm there. Both models are reported in full in
[`report.md`](report.md).

## Setup

```bash
conda activate steer
python scripts/00_download.py     # aria2c, 16 connections per file
python scripts/01_smoke.py        # confirm generation works
python scripts/02_hook_noop.py    # MUST report 7/7 gates passed
```

`scripts/02_hook_noop.py` is the correctness gate for the hook layer. It checks
that `alpha=0` is token-for-token identical to no hooks, that norm preservation
holds, that each position mode fires where its name says, and that the recorder
matches an independent capture of the same tensor. **No number from this
repository is valid unless it reports 7/7.** The last check exists because a
combined device-and-dtype cast on Apple MPS returned wrong memory and corrupted
every extracted vector.

`meta-llama` gates Llama-3.2 behind a licence separate from Llama-3.1, so a
valid token can return HTTP 403 on every 3.2 file while repository metadata
stays readable. `config.LLAMA` therefore points at
`unsloth/Llama-3.2-3B-Instruct`, whose bfloat16 shards are byte-identical
(4965799096 and 1459729952 bytes). Switch to `config.LLAMA_CANONICAL` once your
grant is live.

## Pipeline

```bash
# 1. estimate vectors, print per-layer axis cosine, orthogonalise
python src/build_vectors.py --estimator instruction_diff --n 256

# 2. find responsive layers, then sweep strength on the survivors
python src/sweep_layers.py --stage 1
python src/sweep_layers.py --report
python src/sweep_layers.py --stage 2
python src/sweep_layers.py --report

# 3. map alpha to reading levels, or report them unreachable
python src/calibrate.py --structural-layer 15 --lexical-layer 19

# 4. six variants against the baselines, with SARI
python src/evaluate.py --estimator instruction_diff --n 32 \
    --structural 15,0.75 --lexical 19,1.0
```

## Use it

```bash
python src/simplify.py --text "..." --structural 0.75 --structural-layer 15 --explain
python src/simplify.py --file paper.txt --structural 0.75 --structural-layer 15 --out simple.txt
echo "..." | python src/simplify.py --structural 0.75 --structural-layer 15
```

Pass `--structural-layer 15` explicitly. Without it, and without a calibration
file, the tool defaults to the middle layer, which is not the validated site.

Example, verbatim:

> **in** The mitochondrion is a double-membrane-bound organelle whose principal
> function is the generation of adenosine triphosphate.
>
> **out** The mitochondrion is a specialized organelle with two membranes. Its
> main job is to produce energy for the cell by generating adenosine
> triphosphate.

FKGL 16.2 to 9.9, one sentence to two, faithfulness 0.93 against a gate of 0.50.

If a result fails the faithfulness gate or degenerates, `simplify.py` retries at
0.7x alpha up to twice, then falls back to the prompt-only rewrite, then to the
source text unchanged, so no output that dropped or invented facts is returned.
`--allow-unverified` shows what was rejected.

## Layout

```
config.py     every tunable, including all prompts
lib/          library code
  hub.py        aria2c download, gated-repo access checks
  models.py     loading, uniform decoder-layer access
  hooks.py      Recorder reads the stream, Injector writes it
  vectors.py    three estimators, orthogonalisation, on-disk banks
  steering.py   SteeringSpec and SteeredLM
  schedules.py  named depth schedules: how one axis spreads across layers
  adapters.py   Adapter: the LoRA on/off switch and strength dial
  bench.py      Bench: model + items + baselines + scoring, shared by every arm
  cli.py        AxisSettings: the depth schedule and alphas each axis runs at
  jsonio.py     result files a strict reader can open
  tables.py     rich tables and tqdm bars that survive redirection
  cleanup.py    strip the wrapper a model puts around a rewrite
  decoding.py   easy-word logit bias
  guidance.py   classifier-free guidance over a pair of prompts
  controls.py   ACCESS-style control-token prefixes
  metrics/      readability, faithfulness, fluency, sari
  data.py       ASSET, OneStopQA, arXiv abstracts, leakage-checked corpora
src/          experiment and product entry points
scripts/      download, smoke test, correctness gate
docs/report/  paper and related work
models/ data/ vectors/   generated, gitignored
results/      the paper's evidence; tracked deliberately
```

Everything an arm does besides its own idea — decide which layers and alphas
each axis runs at, hold the model and its baselines, score an output, write a
result row — lives in `lib/cli.py`, `lib/bench.py` and `lib/jsonio.py` rather
than in each entry point. The reason is not tidiness. When each arm carried its
own copy, `src/adaptive_steering.py` filled a missing baseline with the item's
own *steered* output, which pins every delta at zero and reads as data, and
`src/lora_ladder.py` measured perplexity with the adapter still live, which asks
the adapter how likely it finds its own output. A new arm imports `Bench` and
inherits both fixes.

The same applies to knobs. An axis is a *schedule*, not a layer, and every entry
point that steers accepts the same three spellings:

    --structural 31,0.5                      one layer
    --structural 31,0.5,flat3                a named shape from lib/schedules.py
    --structural 30:0.167+31:0.167+32:0.167  explicit

The first is what every published number was measured under, and its meaning is
unchanged. The others exist because spreading a fixed budget across a band
outperforms concentrating it at the swept layer (A/B 14).

Long jobs are normally started with output redirected, so `lib/tables.py` holds
the console policy in one place: tables force plain text and a fixed width when
stdout is not a tty, and progress bars disable themselves unless stderr is one.
`lib/__init__.py` blanks transformers' load-report palette for the same reason —
transformers 5 hardcodes `PALETTE['bold']` into that banner instead of going
through its own isatty check, which puts escape bytes in every log.

## Limitations

- n=32 on ASSET, 10 on the science abstracts, 32 on SQuAD passages. Two of five
  headline comparisons have intervals spanning zero and are reported as ties.
- `caa` and `diff_pca` were compared for axis separation only. Neither was
  evaluated as a steering vector.
- No reading level in `config.TARGET_FKGL` is reachable by steering alone.
  Gate-passing grade spans 12.19 to 14.76 against targets of 4, 6 and 8.
  Steering modulates; it does not replace a prompt.
- Composing steering with a prompt is destructive, not additive: that variant is
  the worst measured, at -13.50 SARI and 66% degenerate.
- Sentence-by-sentence passage simplification is unreliable at 3B, so
  `CHUNK_CONTEXT_SENTENCES` defaults to 0. The chunked passage results are
  Gemma-only.
- `gemma-3n-E4B-it` is unswept. Its decoder returns rank-4 output under
  alternating updates, so layer indices do not transfer. The second model in
  the study is `gemma-4-12B-it`.
