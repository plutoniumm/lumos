# What did not work

Approaches that were built, measured, and abandoned. The code is gone; the
numbers are here, so nothing has to be re-run to know the answer, and nobody
re-proposes an idea this repo already paid for.

Result files stay in `results/` — they are evidence, not clutter. Everything
below is Llama-3.2-3B-Instruct on ASSET test unless stated. `*` = 95% paired
bootstrap CI excludes zero.

---

## 1. Contrastive decoding / DExperts

`logits_base + β·(logits_expert − logits_base)`, with `asset_r16` as the expert
and the base model as the anti-expert — no new training, both experts already
existed. Correctness was verified before the run: **β=1 reproduces plain LoRA
3/3 identical and β=0 reproduces the base model 3/3**, which are the two
algebraic identities the mix has to satisfy.

Same rungs as `src/lora_ladder.py`, same 32 items, so the two are read together.

| rung | weight FKGL | weight SARI | weight NLI | weight pplR | logit FKGL | logit SARI | logit NLI | logit pplR |
|---|---|---|---|---|---|---|---|---|
| 0 | 12.43 | 45.85 | 0.908 | 1.00 | 12.71 | 45.84 | **0.880** | 1.00 |
| 0.25 | 9.24 | 51.37 | 0.929 | 1.52 | 11.29 | 46.53 | **0.918** | 1.04 |
| 0.5 | 8.91 | 52.69 | 0.868 | 1.86 | 10.17 | 50.12 | **0.913** | 1.19 |
| 0.75 | 8.52 | 53.73 | 0.840 | 2.09 | 9.72 | 53.39 | **0.901** | 1.49 |
| 1 | 8.12 | 53.91 | 0.871 | 2.15 | 8.17 | 53.95 | **0.871** | 2.22 |
| 1.5 | 7.72 | 52.33 | 0.859 | 2.49 | 6.24 | 42.57 | **0.533** | 23.49 |
| 2 | 7.49 | 52.50 | 0.859 | 2.53 | 4.87 | 33.50 | **0.311** | 839.31 |
| 3 | 7.30 | 52.97 | 0.895 | 3.34 | 5.46 | 31.07 | **0.281** | 10860.74 |

**Dominated by the weight-space dial it was meant to improve on.** Weight-space
extrapolation holds faithfulness to s=3 (NLI 0.895, 3.1% degeneration); logit
space collapses immediately past β=1 — NLI 0.871 → 0.533 → 0.311, conditional
perplexity reaching **10,861**, degeneration 18.8%. It loses in the
interpolation region too: weight s=0.25 beats logit β=0.5 on FKGL, SARI *and*
NLI.

The motivating argument was wrong, and the way it was wrong is worth keeping.
It ran: the decode-time vocabulary constraint acts on logits and sits at 0%
degeneration at full strength, so logit space must be the safe place to
intervene. But **easy-words is a bounded penalty on 12.7% of the vocabulary and
DExperts is an unbounded linear extrapolation of the entire logit vector.** They
share an insertion point and nothing else. The perplexity column shows what
actually happens: β>1 walks the distribution off-manifold, where scaling adapter
weights stays on the model's own.

Data: `results/dexperts_ladder.json`.

---

## 2. Dual steering (structural + lexical together)

Two axes at once degenerates 34.4% of outputs on Llama and 18.8% on Gemma where
each axis alone degenerates 0%. Two candidate causes: the vectors interfere, or
they share one strength budget that was never calibrated jointly — every alpha
in this project was swept with the other axis off.

**Degeneration over the alpha plane** (12 items per cell, fractions of the
published alphas, `results/dual_surface_llama.json`):

| stru \ lexi | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| **0** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **0.25** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **0.5** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **0.75** | 0.000 | 0.000 | 0.000 | 0.167 | 0.167 |
| **1** | 0.000 | 0.000 | 0.083 | 0.250 | 0.333 |

A corner cut off the high-high end, zero everywhere else, and **both marginals
clean at full strength**. That is the shared-budget shape. No ridge along the
interior, so interference is ruled out. It is not a symmetric contour —
(1, 0.5) = 0.083 while (0.5, 1) = 0.000, so structural buys more collapse per
unit than lexical.

**Turning it down recovers safety and still loses** (32 items, full metrics):

| dual setting | degen | ΔSARI | ΔFKGL | Δw/sent | ΔNLI |
|---|---|---|---|---|---|
| published (0.75 + 1.0) | 34.4% | -4.27* | +1.23 | +10.49 | -0.296* |
| ×0.75 (0.5625 + 0.75) | 12.5% | -0.55 | -2.72* | -3.52* | -0.135 |
| ×0.5 (0.375 + 0.5) | 0.0% | -0.93 | -1.16* | -1.91* | -0.012 |
| ×0.25 (0.1875 + 0.25) | 0.0% | -1.58 | -0.77 | -1.30 | -0.000 |
| asymmetric (0.75 + 0.5) | 3.1% | -0.67 | -0.16 | +2.17 | -0.040 |
| *structural alone, α=0.75* | 0.0% | *+0.43* | *-2.34** | *-3.74** | *-0.080* |

Degeneration falls 34.4% → 12.5% → 0% and faithfulness returns with it
(ΔNLI −0.296* → −0.012). But at every safe setting **structural alone at full
alpha dominates on every readability metric and on SARI**. The lexical axis
contributes ΔSARI −1.37 by itself, so adding it spends half a shared budget for
a negative return.

The asymmetric cell was the last hope: the 12-item surface liked stru 1.0 +
lexi 0.5 at FKGL 8.98 against 10.52 for structural alone. On 32 items it gives
FKGL 12.33, ΔFKGL −0.16 (ns), and sentences get *longer*. It did not transfer.

**Conclusion: drop dual, do not re-tune it.** The finding is real and worth
keeping — steering has one narrow strength budget, and it should be spent
entirely on the axis that works.

Data: `results/dual_surface_llama.json`, `results/eval_dual_x{0.25,0.5,0.75}.jsonl`,
`results/eval_dual_asym.jsonl`.

---

## 3. Steering vectors estimated on ASSET-domain text

Every vector bank was estimated from `complex_pool`, every sweep screens on
`complex_pool`, and every evaluation runs ASSET — and the screen had failed to
predict the evaluation four times. If the pool is a bad proxy when *screening*,
maybe it is a bad proxy when *estimating*, which would explain why steering wins
on abstracts (long, pool-like) and never on ASSET.

Vectors rebuilt from ASSET-validation sources, evaluated at the **identical**
published layer and alpha so the only difference is the estimation text:

| vectors estimated from | ΔSARI | ΔFKGL | Δw/sent | ΔNLI |
|---|---|---|---|---|
| structural — complex_pool | +0.43 | -2.34* | -3.74* | -0.080 |
| structural — **ASSET-valid** | +0.36 | -2.56* | -4.20* | -0.047 |
| lexical — complex_pool | -1.37 | -0.19 | +1.48* | -0.031 |
| lexical — **ASSET-valid** | -1.28 | -0.27 | +0.95 | -0.049 |

Every number within noise of the published one, and axis entanglement reproduces
at **mean |cos| 0.220 against 0.223** from an independent corpus.

**The estimation domain does not matter.** The instruction-difference direction
captures something general about the instruction, not statistics of the pool.

Two things follow, and both are worth more than the null itself. The screen/eval
gap is **not** a domain property of the vectors — it is specific to which cell
the screening procedure picks, so the thing to distrust is the selection metric,
not the direction. And this was the one available idea that could have rescued
steering on ASSET; it doesn't, so the ASSET null is intrinsic.

Data: `results/eval_vec_asset.jsonl`,
`results/entanglement_Llama-3.2-3B-Instruct_instruction_diff_asset.json`.

---

## 4. The sweep's preferred structural setting, L14 α=0.75

Re-screening the sweep at the evaluation's real 320-token budget correctly
rejected L17 α=1.0 (degeneration 0.000 at 96 tokens, 0.167 at 320) and promoted
L14 α=0.75 in its place. Evaluated on the same 32 items:

| structural | sweep selectivity @320 | ΔSARI | ΔFKGL | Δw/sent | ΔNLI |
|---|---|---|---|---|---|
| L15 α=0.75 (published) | +0.030 | **+0.43** | -2.34* | -3.74* | -0.080 |
| L14 α=0.75 (re-screen pick) | **+0.245** | **-1.67** | -3.44* | -7.11* | -0.031 |

**Eight times the selectivity, two points worse on SARI.** L14 cuts seven words
per sentence to L15's four, and SARI charges for the over-deletion FKGL rewards.

The re-screen fixed a real defect without producing a better setting, and the
published choice survives. This is the fourth time in one session that a
setting chosen by the sweep lost to the incumbent when actually evaluated —
see `report.md`, "The sweep screens under conditions that hide what it screens
for".

Data: `results/eval_stru_L14.jsonl`,
`results/sweep_rescreen320_*_instruction_diff.jsonl`.

---

## 5. `PPL_RATIO_MAX` as a hard gate

Removed from `Scored.passes` and `src/calibrate.py`; still written to every row,
so any run can be re-filtered after the fact.

Over all 518 scored outputs per model, above vs below the 1.5 threshold:

| | share degenerate | share failing NLI |
|---|---|---|
| Llama | 11% vs 3% | — |
| **Gemma** | **4% vs 3%** | 28% vs 4% |

On Gemma it barely separates collapsed output from intact output, which was its
job. It does separate faithfulness failures — but NLI gates those directly, on
an instrument built for them. What it mostly did was reject good work: **55% of
Gemma outputs that are neither degenerate nor unfaithful sat above it**, including
67% of `steered_structural`, Gemma's best arm, at a 2% NLI-failure rate.

It is a ratio against each model's own unsteered perplexity, compared against an
absolute constant tuned on Llama. Gemma's scale is different enough that its
LoRA ladder reaches 976.

---

## 6. LoRA trained on paragraph pairs

Asked whether LoRA's abstract faithfulness is fidelity or non-engagement, since
the adapter trains on 26-token sentences and barely moves a 120-word abstract.

| adapter | epochs | train loss | dev loss | ASSET ΔSARI | science ΔFKGL |
|---|---|---|---|---|---|
| `asset_r16` 3800 sentences | 1 | 0.672 | 0.686 | +8.47* | −1.50* |
| `onestop_r16` 203 paragraphs | 8 | **0.121** | 0.582 | +3.85* | **−2.61*** |
| `onestop_r16_e3` 203 paragraphs | 3 | 0.726 | **0.559** | +9.83* | −1.45 (ns) |
| `both_r16` 4009 mixed | 1 | 0.691 | 0.647 | +8.41* | −1.02 (ns) |

The 8-epoch run engaged with abstracts 1.7× harder, which looked like
confirmation until the epoch control landed. At 3 epochs — reaching a **better**
dev loss in 5.9 minutes against 38.7 — engagement drops back to the sentence
adapter's level. **Engagement tracked memorisation, not training length.**

The observation survives: every adapter moves an abstract 1–2.6 grades against
4.4 on ASSET. The proposed cause does not.

One thing worth carrying forward, and it is not a quality result: paired on the
same 32 items `onestop_r16_e3` is indistinguishable from `asset_r16` on SARI
(+1.36, CI [−0.05, +2.87]) and NLI while simplifying significantly *less*
(ΔFKGL +1.73*). **203 pairs in 5.9 minutes buys what 3800 pairs in 45 minutes
buys** — 19× less data, 8× less compute.

---

## 7. Control tokens as a LoRA (ACCESS-style)

The last untried arm, and the only one that targets a *specified* outcome rather
than a push. `src/train_lora.py --controls` prepends the ratios the reference
achieved on each pair (`lib/controls.py`): `[len=1.20 wps=0.60 hard=0.35]`.
Three controls -- length, words-per-sentence (structural), hard-word fraction
(lexical) -- so results are comparable to everything else here.

**It trained fine and did nothing.** Asking for `wps=0.40` versus `wps=1.00`
produced byte-identical output. But the adapter had learned the mapping; it was
too weak to move a greedy argmax:

| prefix, scored on the reference | loss |
|---|---|
| the true control | **0.6229** |
| no control | 0.6971 |
| an inverted control | 0.7781 |

True beats inverted on 53 of 60 examples. Learned but inert, which needs a
different fix from not learned.

Classifier-free guidance on the prefix supplies it --
`logits_neutral + γ(logits_asked − logits_neutral)`, same weights on two prompts,
so the perturbation is bounded by what the prefix does to the distribution
rather than by extrapolating weights (fail.md §1). γ=1 reproduces ordinary
conditional decoding 6/6.

**Then calibration and faithfulness turned out to trade directly.** `wps` at a
request of 0.40:

| γ | achieved | error | NLI | SARI | degen |
|---|---|---|---|---|---|
| 2 | 0.720 | +0.320 | **0.897** | 53.22 | 0% |
| 4 | 0.665 | +0.265 | 0.764 | 52.40 | 0% |
| 8 | 0.547 | +0.147 | 0.533 | 44.66 | 0% |
| 12 | **0.481** | **+0.081** | **0.344** | 34.77 | 3.1% |

Calibration error falls fourfold and NLI falls from 0.897 to 0.344 paying for it.
At γ=12 responsiveness also stops being monotone in the request. The top of the
range is pinned at 0.823 for every γ, because a neutral request makes guidance a
no-op, so total travel is 0.823 → 0.48 and only the top of that keeps meaning.

The other two controls are worse. `len` is the most responsive and the most
destructive -- at a request of 0.40 it reaches 0.259 with **NLI 0.091, SARI
20.34 and 21.9% degeneration**. It hits the length by deleting content, which is
the exact failure this project exists to measure, and FKGL rewards it (7.77).
`hard` costs 0.57 of NLI for 0.2 of movement.

**Conclusion: a safe dial with ~12% travel, or a calibrated dial that destroys
meaning. Neither is the "ask for grade 8 and get grade 8" capability the arm was
built for.** Kept in the tree rather than deleted, because the guidance machinery
is the only working conditional-decoding path here and the adapter is a usable
artifact on its own -- at a neutral request it scores NLI 0.957 / SARI 54.27
against `asset_r16`'s 0.871 / 53.91 on the same items.

Data: `results/control_ladder_g4.json`, `results/control_wps_g{2,8,12}.json`,
`models/lora/asset_ctrl_r16`.

---

## 8. Quadrature-matched depth schedules (`l2` normalisation)

Spreading one axis's steering budget across a band of layers works, and is the
one thing in this file that is here as a *component* of a positive result
rather than a dead end (A/B 14). The part that does not work is the arithmetic
for matching the budget.

Two rules were available for scaling k layers to a budget A:

    sum   sum(alpha_i) = A     the pushes add linearly
    l2    ||alpha||_2  = A     the pushes are near-orthogonal, adding in
                               quadrature

`l2` is catastrophic. Gemma, L31 centre, budget 0.5, 32 ASSET items:

| shape | per-layer α | NLI | SARI | degenerate |
|---|---|---|---|---|
| `flat3` (sum) | 0.167 | 0.952 | 47.67 | **0.0%** |
| `flat3_l2` | 0.289 | 0.649 | 41.96 | **37.5%** |
| `flat5` (sum) | 0.100 | 0.957 | 46.87 | **0.0%** |
| `flat5_l2` | 0.224 | 0.278 | 37.54 | **78.1%** |

Quadrature matching is only correct for near-orthogonal directions; otherwise
it spends about sqrt(k) times the real budget. The collapse is therefore a
measurement of the geometry, not just a bad setting: **adjacent-layer steering
vectors are near-parallel — the same direction carried forward** — which is
also why the sweep's top band is contiguous rather than a spike. `sum` is the
default in `lib/schedules.py` for this reason.

The `l2` shapes are kept in the registry rather than deleted, because they are
the evidence for that default and it should stay reproducible rather than being
a claim in a docstring.

Do not read the `l2` collapse as "spreading is dangerous". At `sum` matching,
every schedule tested degenerates 0.0% of outputs — the same as the single-layer
incumbent.

---

## 9. Measured decay: shrinking the push by the build-up it detects

Gemma-4-12B-it, structural L31, SQuAD passages, n=32, paired against the same
cached `unsteered` baselines as A/B 13.

A fixed alpha spends the same push on the first generated token, where nothing
has accumulated, as on the three-hundredth, where the model is reading back its
own steered text. The arm measured that accumulation each step and subtracted
it, one layer above the injection site:

    carry = (h_L+1 . u) - baseline - gain * own_push_last_step
    push  = clamp(alpha*||v|| - carry, 0, 2*alpha*||v||)

`baseline` was the mean of `h . u` over unsteered generated tokens (−1.0049,
sd 1.92, 5216 tokens on passages disjoint from the eval 32). Carry is zero on
the first generated token by construction, so with no build-up the arm is
exactly the published fixed setting.

### The first version assumed `gain = 1.0` and became an over-steering arm

Only about half an injected push reaches the next layer, so subtracting all of
it left a constant negative carry, which the controller read as "the axis is
under-represented" and answered by pushing *harder*: **1.26x alpha** on average.

| vs `unsteered` | fixed α=0.5 | decay, gain assumed 1.0 |
|---|---|---|
| FKGL | -6.70\* | -8.50\* |
| words/sent | -13.60\* | -16.88\* |
| NLI | **-0.14\*** | **-0.26\*** |
| degen % | 0.0 | 3.1 |

Head to head it lost -0.12\* NLI [-0.20, -0.04] against the fixed arm. That is
the trade raising alpha already makes.

### Measured, the gain is 0.524

Estimated per row at the first decode step, where build-up is zero by the same
argument that zeroes carry there: **0.524, range 0.483–0.569 across 32 items.**
Norm preservation and the block's own writes eat the rest. Estimating it over
all steps instead gives 0.75, which is contaminated by the build-up it is meant
to exclude — the low variance across items is what makes the first-step
estimate trustworthy.

This number is the one durable result here, and it is a fact about the residual
stream at L31 rather than about this arm: **a push does not survive one layer
intact.**

### Corrected, it is a fixed alpha with extra steps

With the gain measured, the push sits at 0.874x alpha and declines 0.887 ->
0.860 across the generation. Against a fixed arm dialled to that same average
dose (α=0.437):

| decay vs fixed α=0.437 | delta | 95% CI |
|---|---|---|
| FKGL | +0.04 | [-0.14, +0.24] |
| words/sent | -0.03 | [-0.43, +0.34] |
| hard words | 0.00 | [-0.00, +0.00] |
| NLI | **0.00** | **[-0.02, +0.03]** |
| ppl ratio | -0.11 | [-0.24, +0.01] |

Levels are indistinguishable: 10.02 / 12.49 / 0.69 against 9.97 / 12.51 / 0.68.
Two dose points were tested this way — 1.26x before the fix and 0.87x after —
and both are null against a matched constant.

### Why it cannot work at this site

The build-up is real, and it is not even small. In projection units, against a
nominal push of 63.15:

| quantity | value | fraction of push |
|---|---|---|
| baseline μ₀ | -1.00 (sd 1.92) | 0.016 |
| carry, mean | 8.14 | 0.129 |
| carry, first decile | 4.61 | 0.073 |
| carry, last decile | 8.62 | 0.136 |
| **drift across the generation** | **+4.01** | **0.063** |

Most of the 13% is present by the second token and stays there, and a constant
correction *is* a smaller alpha. Only the 6% that actually accumulates over
~230 tokens is decay in the intended sense — half the size of the α 0.5 -> 0.437
dose change that was itself barely resolvable (NLI +0.06\*), so it disappears
into the noise.

Revisit only where the time-varying part gets large: generations much longer
than 230 tokens, or a site where build-up grows faster than 6% per 230 tokens.
Neither is true at L31 on passages.

`results/eval_gemma_passages.jsonl` (the gain=1.0 rows, superseded, with
per-token traces in `.decay.jsonl`), `results/eval_gemma_passages_decayg.jsonl`
(corrected), `results/eval_gemma_passages_dose.jsonl` and `_dose2.jsonl` (the
two matched-dose controls), `results/decay_baseline_gemma-4-12B-it_*.json`.

---

## Not built, and why

**Multi-adapter merge** — one LoRA per axis. It is the weight-space version of
the two dials, and section 2 removed the premise: the lexical axis has 1 of 40
cells with positive selectivity at evaluation conditions, and dual loses to
structural-alone at every scale.

**Adaptive α per layer** — a per-depth alpha schedule. It would be tuned by the
selection metric that produced four consecutive losing settings, and it refines
the arm that is null on ASSET for both models.

**Automated prompt search (DSPy/APE)** — struck by decision, not by evidence.
Off the table permanently.
