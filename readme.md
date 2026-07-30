# steering

Meaning-preserving text simplification by activation steering, for readers who
need complex material made simple — dyslexic students, readers new to a subject,
anyone served by shorter sentences and plainer words.

Two dials, not one switch:

- **structural** — sentence length and clause depth
- **lexical** — word difficulty

Each is a steering vector added to the residual stream at its own layer with its
own strength. Because strength is continuous, the same text can be rendered at
different reading levels — which is the thing prompting cannot do.

Design and rationale: `docs/superpowers/specs/2026-07-30-llm-steering-simplification-design.md`

## Setup

```bash
conda activate steer
python scripts/00_download.py          # aria2c, 16 connections per file
python scripts/01_smoke.py             # confirm the model generates
python scripts/02_hook_noop.py         # confirm the hook layer is correct
```

`meta-llama/Llama-3.2-3B-Instruct` is gated behind a license acceptance separate
from Llama-3.1, so a working token can still 403. `config.LLAMA` therefore points
at `unsloth/Llama-3.2-3B-Instruct`, whose bf16 shards are byte-identical. Switch
to `config.LLAMA_CANONICAL` once your grant is live.

## Pipeline

```bash
# 1. estimate vectors (+ orthogonalise, + entanglement report)
python src/build_vectors.py --estimator instruction_diff

# 2. find where each axis lives, then how hard it can be pushed
python src/sweep_layers.py --stage 1
python src/sweep_layers.py --report            # prints the depth profile
python src/sweep_layers.py --stage 2
python src/sweep_layers.py --report

# 3. measure against baselines, prompt-only included
python src/evaluate.py --estimator instruction_diff

# 4. map alpha to named reading levels
python src/calibrate.py

# 5. use it
python src/simplify.py --text "..." --level grade4 --explain
python src/simplify.py --file paper.txt --structural 0.8 --lexical 1.2
```

## What the sweep decides

The project rests on one empirical claim: that structural and lexical
simplification are separable by *depth* — structure responding to mid-stack
intervention, word choice to late layers near the unembedding. `--report` prints
the per-layer profile and says plainly whether the two axes peaked at different
depths. If they didn't, the two-dial design collapses to a single reading-level
dial, and that is a finding, not a failure.

## Layout

- `config.py` — every tunable
- `lib/` — library code, stable interfaces
- `src/` — experiment and product entry points
- `scripts/` — single-file checks
- `models/ data/ vectors/ results/` — generated, gitignored
