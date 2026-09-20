# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for text simplification by **activation steering**. It adds `alpha * v` to a decoder layer's residual stream during generation, with no training, and compares that against prompting, LoRA, few-shot, decode-time constraints and other arms. Models are `Llama-3.2-3B-Instruct` (the model behind the headline result) and `gemma-4-12B-it`, run in bf16 on Apple MPS. Only Gemma's weights are on this machine — `config.ACTIVE_MODEL` is `GEMMA4`, and reproducing any Llama number needs `scripts/00_download.py llama` first (its vector banks are still in `vectors/`). `readme.md` has the headline result and the pipeline. `report.md` has every measured number, in numbered "A/B N" sections. `fail.md` lists approaches that were built, measured and dropped; check it before proposing a new approach. `examples.md` shows real outputs.

## Environment and commands

`python` is not on PATH outside the conda env, and shell state does not persist between tool calls. Run everything as `conda run -n steer python <script>` from the repo root — except when the command reads stdin (`echo ... | python src/simplify.py`, or a `python - <<'PY'` heredoc), because `conda run` does not forward stdin and the script silently sees empty input. For those, call the interpreter directly: `/opt/homebrew/Caskroom/miniconda/base/envs/steer/bin/python`. Every entry point in `src/` and `scripts/` puts the repo root on `sys.path` itself. Some `lib/` modules (`bench.py`, `cli.py`) import `config` bare and depend on that.

There is no pytest suite and no linter config. The checks that exist:

```bash
conda run -n steer python scripts/02_hook_noop.py   # hook-layer correctness gate; MUST print 7/7 passed
conda run -n steer python scripts/01_smoke.py       # generation works
conda run -n steer python lib/metrics/sari.py       # SARI self-test ("all assertions passed")
# cheapest end-to-end eval smoke (no NLI/SARI):
conda run -n steer python src/evaluate.py --variants unsteered,prompted --n 4 --skip-sari --skip-nli --out <scratch path>
```

Run `02_hook_noop.py` after any change to `lib/hooks.py`, `lib/models.py`, `lib/steering.py` or `lib/vectors.py`. It uses a small ungated model, so it runs without the real weights. No number from the repo is valid unless it reports 7/7.

Research pipeline, in order (see `readme.md` for full flags):

```bash
python scripts/00_download.py [llama gemma4 nli]          # aria2c into models/
python src/build_vectors.py --estimator instruction_diff --n 256
python src/sweep_layers.py --stage 1 && python src/sweep_layers.py --report
python src/sweep_layers.py --stage 2 && python src/sweep_layers.py --report   # -> results/best_layers_*.json
python src/calibrate.py --structural-layer 15 --lexical-layer 19
python src/evaluate.py --estimator instruction_diff --n 32 --structural 15,0.75 --lexical 19,1.0 --out results/eval_X.jsonl
python src/report.py results/eval_X.jsonl                 # paired bootstrap tables (markdown) for report.md
python src/rescore.py results/eval_X.jsonl --out results/eval_X_v2.jsonl   # re-score, no regeneration
```

Product entry point: `src/simplify.py`. With `ACTIVE_MODEL = GEMMA4` the validated steering setting is `--structural 0.5 --structural-layer 31`; L15/0.75 is Llama's and is not a site on Gemma. When steering, always pass `--structural-layer`: without it, and without a calibration file, the tool exits 2 rather than guessing a layer. Its exit codes are 0 (a result was printed, fallbacks included), 2 (usage/config error), 1 (failed before producing anything), and stdout carries the rewrite alone — `--explain` and every warning go to stderr.

`models/`, `data/`, `vectors/` and `docs/` are gitignored. `results/` is tracked on purpose: it holds hours of GPU work and is the evidence behind `report.md`. Keep it small, and if a large, cheap-to-rebuild file lands there, ignore that file by name.

## Architecture

The layers, bottom up. Each depends only on the ones above it:

- **`config.py`**: every tunable and **every prompt** (`BASE_TASK`, `INSTRUCTIONS`, `PLACEBOS`). It imports no project code. Per-model differences go through lookup helpers, `base_task(repo)` and `easy_word_penalty(repo)`, which match on repo id or local-path suffix. Don't branch on model names at call sites. Changing Llama's `BASE_TASK` invalidates every recorded Llama result.
- **`lib/models.py`**: `load()` returns a `SteerableLM`. It resolves to the local aria2c download via `lib/hub.py` and finds the decoder `ModuleList` by trying known attribute paths. Llama, Gemma-3n and Gemma-4 share this code, with no per-family branches.
- **`lib/hooks.py`**: `Recorder` reads activations and `Injector` writes them. The injection site is `resid_post` (layer output) or `mlp_out`. `positions` is `gen_only` (default), `prompt_last`, `prompt_all` or `all`. Norm preservation rescales back to the original `||h||` after injection.
- **`lib/vectors.py`**: `VectorBank` holds per-layer vectors for one axis plus `ref_norms`. Alpha is a **fraction of the layer's reference norm**, not a raw scale, because norms grow ~33x with depth. The on-disk path is `vectors/<model-slug>/<estimator>.<axis>.safetensors`. The estimators are `instruction_diff` (the one used for all results), `caa` and `diff_pca`.
- **`lib/steering.py`**: `AxisSetting(axis, layer, alpha)` → `SteeringSpec` (a tuple of settings plus positions/norm/site/content-orthogonalisation) → `SteeredLM` (model plus banks). The sweep, the evaluation and the product all use this one codepath. It projects each vector away from the source text's own mean-pooled representation. It raises if a bank was estimated on a different model.
- **`lib/schedules.py` + `lib/cli.py`**: an axis is a *depth schedule*, not just a single layer. `AxisSettings` parses the shared flag grammar, `--structural 15,0.75` | `15,0.75,flat3` | `14:0.25+15:0.25+16:0.25`, or `--layers-from` a sweep JSON, into a `SteeringSpec`. The single-layer form is what every published number used. Schedules are budget-normalised with `sum` by default; `l2` over-doses.
- **`lib/bench.py`**: `Bench` is the shared frame for experiment arms: model + items + baselines + scoring + row schema. Any new arm should import `Bench`, `AxisSettings` and `jsonio` rather than hand-roll them. Hand-rolled copies caused real bugs; the module docstrings explain which.
- **`lib/metrics/`**: `readability` (FKGL etc. and the **degeneracy detector**), `faithfulness` (bidirectional NLI with roberta-large-mnli, gate 0.50 in *both* directions), `fluency` (conditional perplexity ratio), `sari`.
- **`src/`**: one script per experiment arm (`evaluate`, `sweep_layers`, `calibrate`, `best_of_n`, `adaptive_steering`, `alpha_schedule`, `lora_ladder`, `control_ladder`, `train_lora`), plus `report.py`, `rescore.py` and `simplify.py`.

## Invariants that are easy to break silently

These bugs produce plausible numbers, not errors:

- **A baseline is always unsteered, with no adapter.** Baselines are cached to `<out>.baselines.json` so resumed runs and other arms score against byte-identical text. Never fall back to a steered string.
- **Scoring never runs under the intervention.** Metrics run inside `Adapter.scoring()`, and `Bench.score` asserts no injector is live. Perplexity ratios condition both outputs on the *plain unsteered prompt*.
- **Deltas are paired within item.** `item.index` is the join key across all result files and is never renumbered. `evaluate.py` resume skips `(variant, eval_set, item_index)` already on disk, so `--n` may only grow. A row counts as "done" only for the metrics it recorded.
- **Result files go through `lib/jsonio.py`.** They are append-only with a flush per row. `allow_nan=False` turns non-finite values into `null`. Never edit a results file in place; write a new file (see `rescore.py`).
- **The perplexity ratio is reported, not gated.** `PPL_RATIO_MAX` only sets a flag. Pass/fail is NLI plus non-degeneracy (`Scored.passes`).
- **MPS tensor moves: cast dtype before device** (`t.detach().float().to("cpu")`). A fused `.to(device, dtype)` dropped the storage offset and corrupted every extracted vector.
- **Sweep generation length must match eval length** (`SWEEP_MAX_NEW_TOKENS = MAX_NEW_TOKENS`). Runaway generation only shows up past ~96 tokens, and sweeps screened at 96 underreported degeneration 6x.
- **Vectors and layer indices do not transfer across models.** `gemma-3n-E4B-it` is unswept: its decoder returns rank-4 hidden states, so layer indices don't carry over.
- **`--variants all` in `evaluate.py` always means the six steering variants.** LoRA, few-shot and constrained variants must be named explicitly, so the headline comparison keeps its shape.
- **Console output goes through `lib/tables.py`** (`tables.table/show/write/progress`), not raw rich or tqdm. Runs are normally redirected to logs, so tables use no ANSI and a fixed 140-column width, and progress bars disable themselves when stderr is not a tty.

## Conventions

- Docstrings and comments record *why*, usually with the measured number that justified a choice or the bug that motivated it (see `config.py`, `lib/bench.py`). New tunables and design decisions should follow that pattern. Put prompts and thresholds in `config.py`, not inline.
- `report.md` is chronological. Each section names the command that regenerates it, and superseded numbers stay in place with the reason. `*` marks a 95% paired bootstrap CI that excludes zero. Abandoned approaches go in `fail.md`: the code is removed and the measurements are kept.
- `config.LLAMA` points at the ungated `unsloth/` mirror, whose shards are byte-identical to `meta-llama`'s. Only switch to `LLAMA_CANONICAL` once licence access is granted.
