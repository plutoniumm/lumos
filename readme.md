# steering

Simplify text without changing what it says.

A CLI (`src/simplify.py`) and a small library (`lib/`) that make a language model
rewrite text at a lower reading level, then **check that the meaning survived**
before returning anything. Four mechanisms are implemented and measured against
each other; the tool picks the best one available by default.

The project's subject is *activation steering* — adding a fixed direction to the
residual stream during generation, scaled by an `alpha`, so the degree of
simplification is a number set at inference with no weights trained. That
mechanism works and is fully implemented, but a trained LoRA adapter measured
better, so the product defaults to whichever is on disk. Both are one flag apart.

Measurements: [`report.md`](report.md) · Real outputs: [`examples.md`](examples.md)
(recorded on Llama at L15, not the current default)
· What failed: [`fail.md`](fail.md) · Working notes: [`CLAUDE.md`](CLAUDE.md)

## Install

```bash
conda env create -f environment.yml    # creates the `steer` env
conda activate steer
python scripts/00_download.py          # weights, via aria2c
python scripts/02_hook_noop.py         # MUST report 7/7 gates passed
```

`02_hook_noop.py` is the correctness gate for the hook layer: `alpha=0` must be
token-identical to no hooks, norm preservation must hold, each position mode must
fire where its name says, and the recorder must match an independent capture.
**No number here is valid unless it reports 7/7.** Runs on a 135M model, so it
works while the real weights are still downloading.

`meta-llama` gates Llama-3.2 separately from 3.1, so a valid token can 403 on
every file. `config.LLAMA` therefore points at the ungated `unsloth` mirror,
whose bf16 shards are byte-identical.

## Quickstart

```bash
python src/simplify.py --file paper.txt --out simple.txt
```

That's the best case. It picks LoRA if an adapter for the active model is on
disk, else steering, and prints only the rewrite:

```bash
echo "The mitochondrion is a double-membrane-bound organelle whose principal
function is the generation of adenosine triphosphate." | python src/simplify.py
```
> The mitochondrion is a double-membrane-bound organelle. Its main function is
> the generation of adenosine triphosphate.

Reading grade 16.2 → 11.7, faithfulness 0.99 against a gate of 0.50, on
`gemma-4-12B-it`. Add `--explain` for the metrics, `--chunked` to rewrite
sentence by sentence, `--out FILE` to write to disk.

`--level grade4|grade6|grade8` aims at a reading level instead of a strength,
but it is calibrated against steering `alpha` only, so it needs
`--mechanism steering` and a calibration file from `src/calibrate.py` for the
model in use. Under any other mechanism the tool exits 2 rather than accept a
level it cannot act on.

stdout carries the rewrite alone — `--explain`, warnings and progress all go to
stderr, so `... | simplify > out.txt` lands exactly the text. Exit codes: `0` a
result was printed, `2` usage or config error, `1` failed before producing one.

## Mechanisms

`--mechanism`, default `auto`. ΔSARI is against no intervention on ASSET, n=32,
`*` = 95% CI excludes zero (Llama-3.2-3B / gemma-4-12B).

| value | what it does | ΔSARI | needs |
|---|---|---|---|
| `auto` | lora if an adapter is on disk, else steering | — | — |
| `lora` | trained adapter, engaged at generation | **+8.47\* / +6.19\*** | an adapter |
| `fewshot` | 5 in-context demonstrations | +3.85\* / +3.44\* | corpus (auto-downloads) |
| `steering` | `alpha * v` added at one layer | +0.43 / +2.02 | vector banks |
| `none` | the bare prompted instruction | -5.04\* / +2.91\* | — |

Compose with `+` (`lora+steering`, `fewshot+steering`). `--easy-words` adds a
decode-time penalty on hard words (-0.44 / +1.11) to any of them.

`none+steering` is **refused**: prompting alongside steering is the worst arm
measured on both models (-13.50\* / -12.76\* SARI, 66% / 28% degenerate).

Steering needs its layer named — `--structural 0.5 --structural-layer 31` on
Gemma, `0.75` / `15` on Llama. Layers do not transfer between models, so without
a layer and without a calibration file the tool exits 2 rather than guess.

## The meaning check

Every mechanism returns through the same gate, which is the point of the tool.

1. Rewrite, then score it: bidirectional NLI entailment (source must imply the
   output *and* the output imply the source) plus a degeneracy check for loops
   and fragments.
2. Below `config.NLI_ENTAILMENT_MIN` (0.50), retry at 0.7x whichever dial the
   mechanism has — steering `alpha`, LoRA strength — up to twice.
3. Still failing: fall back to the plain prompted rewrite, then to the source
   text unchanged.

So an output that dropped or invented a fact is never returned. `--explain`
names which rung produced the text; `--allow-unverified` prints what was
rejected instead.

## Library use

```python
from lib.models import load
from lib.steering import SteeredLM, SteeringSpec, load_banks

lm = load("google/gemma-4-12B-it")
sl = SteeredLM(lm, load_banks(lm.repo_id, "instruction_diff"))

spec = SteeringSpec.single("structural", layer=31, alpha=0.5)
print(sl.generate("The mitochondrion is ...", spec))
print(sl.unsteered("The mitochondrion is ..."))      # same prompt, no vector
```

`SteeringSpec` is the whole configuration: which axes at which layers and
alphas, which token positions receive the injection, and whether the stream is
renormalised after it. `SteeredLM.generate_batch` takes the same spec.
`lib.adapters.Adapter` is the LoRA equivalent, with `engaged()`, `at(scale)` and
a `scoring()` context that disables it so metrics never run under the
intervention.

## How steering works

A steering vector is the mean activation difference between the same text under
an instruction and under an axis-matched placebo (`instruction_diff`), taken at
every layer and stored in `vectors/<model>/`. At generation, that direction is
added to the residual stream at one layer, on generated tokens only, and the
hidden state is rescaled to its original norm afterwards — the single biggest
guard against degeneration. The vector is also projected orthogonal to the source
text's own representation, so steering cannot push along content.

`alpha` is a **fraction of the layer's typical activation norm**, not a raw
scale, because norms grow ~33x with depth. Responsive layers and usable alphas
are found by a two-stage sweep (`src/sweep_layers.py`) and differ per model.

## Results

Structural steering, L15 `alpha=0.75`, Llama-3.2-3B, ASSET n=32, paired:

| comparison | SARI | faithfulness |
|---|---|---|
| vs. prompt-only | **+5.47** [+2.58, +8.48] | **+0.33** [+0.20, +0.46] |
| vs. no intervention | +0.43 [-1.71, +2.59] | -0.08 [-0.21, +0.05] |

It beats prompting on both, significantly, and ties with doing nothing while
cutting sentence length 18% and reading grade 19% at zero degeneration.
Prompt-only is significantly *worse* than no intervention on this model — it
reaches a low grade by deleting facts.

**The result is bounded by model scale.** That prompting failure is a property
of the 3B model; on `gemma-4-12B-it` the same instruction *gains* SARI (+2.91\*)
and steering is a middling arm. **One dial, not two:** the vocabulary axis exists
and is nearly perpendicular to the structural one (cosine 0.223), but it is a
SARI tie trending negative, and both at once break 34% of outputs.

## Research pipeline

```bash
python src/build_vectors.py --estimator instruction_diff --n 256   # estimate
python src/sweep_layers.py --stage 1 && python src/sweep_layers.py --report
python src/sweep_layers.py --stage 2 && python src/sweep_layers.py --report
python src/calibrate.py --structural-layer 15 --lexical-layer 19   # alpha -> grade
python src/evaluate.py --n 32 --structural 15,0.75 --lexical 19,1.0
python src/report.py results/eval_main.jsonl                       # paired tables
```

These accept a *schedule*, not just a layer: `--structural 31,0.5` (one layer,
what every published number used), `31,0.5,flat3` (a named shape from
`lib/schedules.py`), or `30:0.167+31:0.167+32:0.167`.

## Layout

```
config.py     every tunable, including all prompts
lib/
  models.py     loading, uniform decoder-layer access across model families
  hooks.py      Recorder reads the stream, Injector writes it
  vectors.py    estimators, orthogonalisation, on-disk vector banks
  steering.py   SteeringSpec and SteeredLM
  adapters.py   Adapter: LoRA on/off switch and strength dial
  bench.py      model + items + baselines + scoring, shared by every arm
  cli.py        AxisSettings: the layers and alphas each axis runs at
  schedules.py  named depth schedules
  metrics/      readability, faithfulness, fluency, sari
  data.py       corpora, with leakage between splits made impossible
  hub.py jsonio.py tables.py cleanup.py decoding.py guidance.py controls.py
src/          entry points: simplify.py is the product, the rest are experiments
scripts/      download, smoke test, correctness gate
results/      the evidence behind report.md; tracked deliberately
paper/        steering.tex, the write-up (revtex4-2); images/ alongside it
models/ data/ vectors/   generated or large, gitignored
```

Everything an arm does besides its own idea — resolve layers and alphas, hold
the model and its baselines, score an output, write a result row — lives in
`lib/cli.py`, `lib/bench.py` and `lib/jsonio.py`. That is not tidiness: when each
arm carried its own copy, one filled a missing baseline with the item's own
*steered* output and another measured perplexity with the adapter still live.

## Limitations

- n=32 on ASSET, 10 on the science abstracts, 32 on SQuAD passages. Two of five
  headline comparisons have intervals spanning zero and are reported as ties.
- No reading level in `config.TARGET_FKGL` is reachable by steering alone;
  gate-passing grade spans 12.19 to 14.76 against targets of 4, 6 and 8.
  Steering modulates, it does not replace a prompt.
- Sentence-by-sentence passage rewriting is unreliable at 3B, so
  `CHUNK_CONTEXT_SENTENCES` defaults to 0; chunked results are Gemma-only.
- `caa` and `diff_pca` were compared for axis separation only, never evaluated
  as steering vectors.
- `gemma-3n-E4B-it` is unswept: its decoder returns rank-4 output, so layer
  indices do not transfer.
- Only `gemma-4-12B-it` weights are on this machine; `config.ACTIVE_MODEL`
  points there. Reproducing a Llama number needs `scripts/00_download.py llama`
  first — its vector banks are still in `vectors/`.
