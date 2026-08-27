# Results

Measurements in the order they were taken. Each result names the command that
regenerates it. Superseded numbers are kept in place with the reason they were
superseded.

**Base model** `unsloth/Llama-3.2-3B-Instruct` (bf16, MPS)
**Eval set** ASSET test, first 32 items, 10 human references each
**Baseline** `unsteered` — the bare task prompt `"Rewrite the following text."`,
no simplify instruction, no intervention

Every arm is compared against that baseline, on the same 32 items, with the same
prompt. Only the mechanism changes.

---

## Summary

Two models at matched coverage: fourteen arms each on ASSET, six each on the
science abstracts, plus a LoRA strength ladder per model, all paired against the
same `unsteered` baseline per model. ASSET is 32 Wikipedia sentences with 10
human references, so SARI is available; science is 10 arXiv abstracts with none,
so faithfulness is the measurement. A third set, 32 SQuAD passages on Gemma,
is reported in A/B 13. `*` = 95% bootstrap CI excludes zero.

### ASSET

| arm | Llama ΔSARI | Llama ΔNLI | Gemma ΔSARI | Gemma ΔNLI |
|---|---|---|---|---|
| **lora** | **+8.47*** | -0.00 | **+6.19*** | -0.07 |
| lora_plus_steered | +5.93* | -0.24* | +1.36 | -0.44* |
| fewshot_plus_steered | +4.08* | -0.08 | +3.42* | -0.05* |
| fewshot | +3.85* | +0.06 | +3.44* | +0.03 |
| best_of_n | +0.78 | +0.05 | +1.82* | -0.01* |
| adaptive_structural | +0.46 | -0.01 | +1.10 | -0.02* |
| steered_structural | +0.43 | -0.08 | +2.02 | -0.01* |
| easy_words | -0.44 | -0.02 | +1.11 | +0.02 |
| steered_lexical | -1.37 | -0.03 | -1.54 | -0.31* |
| steered_dual | -4.27* | -0.30* | -7.77* | -0.65* |
| **prompted** | **-5.04*** | **-0.41*** | **+2.91*** | **-0.01** |
| prompted_plus_steered | -13.50* | -0.65* | -12.76* | -0.78* |

### Science abstracts

| arm | Llama ΔFKGL | Llama ΔNLI | Gemma ΔFKGL | Gemma ΔNLI |
|---|---|---|---|---|
| prompted | -9.84* | **-0.73*** | -8.01* | -0.37* |
| fewshot_plus_steered | -10.23* | -0.57* | -7.83* | -0.17* |
| **steered_structural** | -5.14* | -0.18* | **-5.36*** | **-0.05** |
| easy_words | -3.01* | -0.26* | -3.16* | -0.12* |
| fewshot | -3.18* | -0.20* | -2.06* | -0.01 |
| lora | -1.50* | **-0.03** | -2.62* | -0.11 |

### Parity note

`best_of_n` and `adaptive_structural` were the last two arms to reach Gemma.
Both behave as on Llama — small, safe, not competitive:

| | Llama ΔSARI | Gemma ΔSARI |
|---|---|---|
| best_of_n | +0.78 (ns) | **+1.82*** |
| adaptive_structural | +0.46 (ns) | +1.10 (ns) |

Best-of-n crosses into significance on Gemma and posts the lowest perplexity
ratio of any effective arm there (1.13), which is what it optimises for. It is
still beaten threefold by LoRA and by prompting, and its own gate discards the
best available candidate 28 times in 32 (A/B 6).

### Findings

**The premise holds at 3B and not at 12B.** Prompting a model to simplify makes
it delete content: on Llama-3.2-3B, ΔSARI -5.04*, and on abstracts faithfulness
falls to 0.06. On Gemma-4-12B the same instruction gains SARI, +2.91*, at ΔNLI
-0.01.

A control isolates the cause. Giving Llama Gemma's exact task sentence recovers
a quarter of the faithfulness gap on ASSET and none of it on abstracts. The
remainder is model capability, not instruction wording. The motivation for
steering is sound at 3B and expires by 12B.

**LoRA is the strongest method.** Best arm on both models on ASSET, +8.47* and
+6.19*, from 45 minutes of training and 24M trainable parameters. It is also
the only Llama arm that survives the abstracts without a significant
faithfulness loss.

**Steering wins one eval set, and the win does not survive a larger sample.**
Structural steering leads on the Gemma abstracts: -5.36 grades, sentences cut by
11 words, ΔNLI -0.05 with the interval across zero, the lowest faithfulness cost
per grade in the study. On 32 SQuAD passages of comparable length the same arm
reads ΔNLI -0.14* (A/B 13), so the abstract null reflects n=10.

Elsewhere steering is middling. Single-layer ASSET SARI is not significant on
either model and ranks sixth of twelve on Gemma. Two dials are worse than one:
the lexical axis costs faithfulness on Gemma (-0.31*), and both dials together
rank third from last.

**Injection depth is a live knob.** At a matched alpha budget, spreading the
push over three to five adjacent layers beats concentrating it at the swept
layer: `flat3` scores ΔSARI +2.70* on Gemma ASSET against the single layer's
+1.84 (ns), and the margin survives a dose control (A/B 14). It is a single
result at n=32 and is not yet replicated.

**Two structural claims replicate across architectures.** Axis separation, mean
\|cos\| 0.223 on Llama and 0.230 on Gemma; and depth ordering, structural before
lexical on both (L15/28 and L31/48). **One does not**: alpha, even
depth-normalised, does not transfer — Llama's values degenerate 25-92% of
Gemma's outputs.

**Orthogonality and efficacy do not co-locate.** On Gemma the layers where the
axes are cleanest (10-13, \|cos\| ≈ 0) are layers where steering does nothing at
all, and the layers that work are the most entangled. Llama's flat profile hides
this entirely.

### Open

- **Why LoRA under-engages on long inputs.** That it under-engages is settled:
  every adapter measured moves a 120-word abstract by 1-2.6 grades against 4.4
  on ASSET. The proposed cause, training length of 26-token sentences, is
  **refuted** (A/B 12). A/B 13 sharpens the question without answering it: the
  adapter does worst relative to the alternatives in chunked mode, which is the
  sentence-at-a-time regime it was trained for.
- **Why a setting screened as safe breaks 47% of another pool** at the same
  token budget and comparable input lengths. Token budget, input length and
  estimation corpus have each been tested and rejected as the explanation.
- **The two models' settings were not chosen alike**, so an unknown part of the
  steering gap between them is setting quality rather than model capability.
- **The depth-schedule result is one run at n=32** (A/B 14) and has not been
  replicated at larger n or on Llama.
- **n = 10 on the abstracts** is underpowered for a null. For steering this is
  now resolved against the arm by A/B 13; for the other arms it stands.

### Measurement defects

Seven times this evaluation returned a wrong answer rather than an error. Each
has the same form: **a measure built on one distribution misfires on another,
and the misfire is reported as a property of the model.**

| defect | reported | actually |
|---|---|---|
| LoRA scored its own perplexity | ratio 0.58 | 1.80 |
| preamble counted as an output sentence | NLI 0.084 | 0.990 |
| citation fragments counted as content | ” | ” |
| unmeasured sweep cells passed the NLI gate | best setting | NLI 0.26, no degeneration |
| perplexity ratio read as fluency | Gemma prompted "fails" at 11.11 | NLI 0.95, 0% degenerate |
| digit-stripped loop detection | 3 degenerate abstracts | 0 |
| **sweep screens at 96 tokens, uses at 320** | **8% degeneration** | **47%** |

The last is the most consequential. The failure mode of excessive alpha is
run-on generation, and the screen stops generating before it appears. Every
headline (layer, alpha) here was chosen by that procedure; the settings in use
survive because the evaluation caught what the screen could not.

---

## A/B 1 — steering vs no steering

`results/eval_main_v2.jsonl` · regenerate with
`python src/report.py results/eval_main_v2.jsonl --out results/ab_all_v2.json`

Scored on the corrected convention: `lib.cleanup` strips the "Here's a rewritten
version:" wrapper before metrics run, because a preamble entails nothing and
enters the per-sentence faithfulness mean as a zero. 12 of 384 ASSET rows carried
one (2.7%) and re-scoring moved only the two prompt-based arms — prompting's
ΔNLI from -0.44 to -0.41. No conclusion changed. `results/eval_main.jsonl` keeps
the pre-fix numbers.

Steering vectors from the instruction-difference estimator, axis-matched
placebo contrast, 256 texts. Structural at layer 15, α=0.75. Lexical at layer
19, α=1.0.

### Levels

| variant | SARI | FKGL | words/sent | hard words | NLI | ppl ratio | degen % |
|---|---|---|---|---|---|---|---|
| unsteered | 45.44 | 12.49 | 20.36 | 0.34 | 0.88 | 1.00 | 0.0 |
| prompted | 40.40 | 6.47 | 14.12 | 0.16 | 0.46 | 3.78 | 0.0 |
| steered_structural | **45.87** | 10.15 | 16.62 | 0.32 | 0.80 | 1.31 | 0.0 |
| steered_lexical | 44.07 | 12.30 | 21.84 | 0.32 | 0.84 | 1.20 | 0.0 |
| steered_dual | 41.17 | 13.72 | 30.85 | 0.23 | 0.58 | 2.02 | 34.4 |
| prompted_plus_steered | 31.94 | 5.21 | 15.20 | 0.09 | 0.22 | 4.99 | 65.6 |

### Paired deltas vs `unsteered`, 95% bootstrap CI over items

| variant | SARI | FKGL | words/sent | hard words | NLI | ppl ratio |
|---|---|---|---|---|---|---|
| prompted | -5.04*<br>[-8.15, -2.05] | -6.02*<br>[-7.75, -4.34] | -6.24*<br>[-9.47, -3.10] | -0.19*<br>[-0.23, -0.15] | -0.41*<br>[-0.53, -0.29] | 2.78*<br>[2.04, 3.74] |
| steered_structural | 0.43<br>[-1.71, 2.59] | -2.34*<br>[-3.40, -1.31] | -3.74*<br>[-5.90, -1.71] | -0.02*<br>[-0.04, -0.00] | -0.08<br>[-0.21, 0.05] | 0.31*<br>[0.21, 0.45] |
| steered_lexical | -1.37<br>[-3.29, 0.44] | -0.19<br>[-1.22, 0.80] | 1.48*<br>[0.12, 2.95] | -0.02*<br>[-0.04, -0.00] | -0.03<br>[-0.13, 0.07] | 0.20*<br>[0.11, 0.33] |
| steered_dual | -4.27*<br>[-7.22, -1.51] | 1.23<br>[-3.38, 7.93] | 10.49<br>[-2.35, 29.10] | -0.11*<br>[-0.17, -0.06] | -0.30*<br>[-0.45, -0.14] | 1.02*<br>[0.72, 1.37] |
| prompted_plus_steered | -13.50*<br>[-17.41, -9.64] | -7.28*<br>[-9.50, -4.72] | -5.16<br>[-10.10, 0.69] | -0.25*<br>[-0.31, -0.20] | -0.65*<br>[-0.75, -0.54] | 3.99*<br>[1.56, 8.22] |

`*` = 95% CI excludes zero. 10,000 resamples.

### Findings

**Structural steering is the only arm with no measurable cost.** Its SARI
interval contains zero while FKGL drops significantly (−2.34 grades) and
faithfulness does not move (ΔNLI −0.08, interval crosses zero). Every other
intervention trades meaning for simplicity.

**Prompting scores best on readability and worst on fidelity.** ΔFKGL −6.02
grades, ΔSARI −5.04, faithfulness 0.88 → 0.46. It deletes content, and FKGL
rewards deletion. The two metrics disagree, and only faithfulness tracks whether
the output still says what the source said.

**The lexical dial does not work.** ΔFKGL −0.19 with the interval crossing
zero, ΔSARI negative, and sentences get longer. The lexical vector does not
encode "use easier words" at any strength that can be applied safely.

**Two dials at once are worse than one.** `steered_dual` degenerates 34% of the
time. The two vectors are only approximately orthogonal (mean absolute cosine
0.223), and what is left of the overlap compounds.

**Prompt and steering collide.** Both push in the same direction, so the sum
overshoots the collapse point: 65.6% degenerate, faithfulness 0.22, worst SARI
measured (−13.50). The task prompt is present in every variant and is required;
what fails is stacking the *simplify instruction* on top of a vector at this α.

**Open:** whether a smaller α composes safely with the instruction was never
tested.

---

## A/B 2 — LoRA vs no LoRA

`results/eval_main.jsonl` · `models/lora/asset_r16/train_meta.json`

    python src/train_lora.py --name asset_r16
    python src/evaluate.py --lora models/lora/asset_r16 \
        --variants lora,lora_plus_steered --n 32 --eval-sets asset \
        --structural 15,0.75 --lexical 19,1.0 --out results/eval_main.jsonl

Rank-16 LoRA on all attention and MLP projections, 24.3M trainable (0.75% of
the model). Trained on ASSET **validation** — 2000 sources, 2 sampled human
references each, 3800 train / 200 dev — under the bare task prompt, loss on the
completion only. 45 minutes on one M-series machine. Dev loss 1.664 → 0.686.

Evaluated on ASSET **test**: the same 32 items, the same baseline strings, the
same prompt as every other arm. The adapter is disengaged for baseline
generation and for all scoring.

### Levels

| variant | SARI | FKGL | words/sent | hard words | NLI | ppl ratio | degen % |
|---|---|---|---|---|---|---|---|
| unsteered | 45.44 | 12.49 | 20.36 | 0.34 | 0.88 | 1.00 | 0.0 |
| prompted | 40.40 | 6.47 | 14.12 | 0.16 | 0.46 | 3.78 | 0.0 |
| steered_structural | 45.87 | 10.15 | 16.62 | 0.32 | 0.80 | 1.31 | 0.0 |
| **lora** | **53.91** | 8.12 | 12.76 | 0.28 | **0.87** | 2.15 | **0.0** |
| lora_plus_steered | 51.37 | 5.42 | 8.88 | 0.24 | 0.63 | 3.12 | 0.0 |

### Paired deltas vs `unsteered`, 95% bootstrap CI over items

| variant | SARI | FKGL | words/sent | hard words | NLI | ppl ratio |
|---|---|---|---|---|---|---|
| prompted | -5.04*<br>[-8.15, -2.05] | -6.02*<br>[-7.75, -4.34] | -6.24*<br>[-9.47, -3.10] | -0.19*<br>[-0.23, -0.15] | -0.41*<br>[-0.53, -0.29] | 2.78*<br>[2.04, 3.74] |
| steered_structural | 0.43<br>[-1.71, 2.59] | -2.34*<br>[-3.40, -1.31] | -3.74*<br>[-5.90, -1.71] | -0.02*<br>[-0.04, -0.00] | -0.08<br>[-0.21, 0.05] | 0.31*<br>[0.21, 0.44] |
| **lora** | **8.47***<br>[6.12, 10.97] | **-4.37***<br>[-5.59, -3.20] | -7.60*<br>[-10.41, -5.11] | -0.06*<br>[-0.09, -0.03] | **-0.00**<br>[-0.16, 0.14] | 1.15*<br>[0.80, 1.59] |
| lora_plus_steered | 5.93*<br>[1.74, 9.69] | -7.08*<br>[-8.56, -5.67] | -11.48*<br>[-14.62, -8.61] | -0.11*<br>[-0.14, -0.07] | -0.24*<br>[-0.42, -0.07] | 2.12*<br>[1.60, 2.77] |

`*` = 95% CI excludes zero.

### Findings

**LoRA is the strongest arm.** It is the only one that produces a large
simplification at no faithfulness cost:

|  | ΔFKGL | ΔSARI | ΔNLI |
|---|---|---|---|
| prompting | -6.02 | **-5.04** | **-0.41** |
| structural steering | -2.34 | +0.43 (ns) | -0.08 (ns) |
| **LoRA** | **-4.37** | **+8.47** | **-0.00** |

Steering's SARI gain is indistinguishable from zero; LoRA's is +8.47 with the
interval well clear of it. LoRA drops 1.9 grades more than steering and comes
within 1.3 grades of prompting without prompting's faithfulness loss.

**The perplexity ratio is not a fluency failure here.** LoRA scores 2.15,
above the 1.5 gate, but conditional perplexity is measured under the *base*
model, so an adapter that shifted the output distribution is penalised for doing
what it was trained to do. The direct evidence contradicts the gate: 0% of LoRA
outputs trip any degeneracy rule and faithfulness is 0.87. The ratio measures
distance from the base model's style.

**Faithfulness is unchanged on average, not item by item.** ΔNLI is -0.00, but
the interval is [-0.15, 0.14] and individual items do lose content. Item 3
("His next work, Saturday, follows an especially eventful day...") comes back as
"His next work, Saturday, is about a day in the life of a successful
neurosurgeon" — NLI 0.220, "especially eventful" dropped. LoRA deletes
occasionally; it does not delete systematically, as prompting does.

**LoRA composes with steering better than prompting does.** Both compositions
cost faithfulness, but `prompted_plus_steered` collapses (65.6% degenerate, NLI
0.22) while `lora_plus_steered` does not degenerate at all (0%, NLI 0.63) and
posts the largest non-degenerate FKGL drop measured (-7.08). The collision that
destroys the prompt-plus-vector combination is absent in the weight-plus-vector
one.

**A worked pair**, item 11, source *"It was originally thought that the debris
thrown up by the collision filled in the smaller craters."*

| variant | output | FKGL | NLI | SARI |
|---|---|---|---|---|
| prompted | A long time ago, people thought that when two big rocks crashed into each other, the pieces that flew up into the air filled in the smaller holes that were left behind. | 11.6 | 0.009 | 34.2 |
| steered_structural | Initially, it was believed that debris from the collision filled in smaller craters. | 9.4 | 0.993 | 37.6 |
| lora | It was thought that the debris thrown up by the collision filled in the smaller craters. | 6.1 | 0.993 | 54.2 |

Prompting paraphrased the craters into "smaller holes" and the collision into
"two big rocks", scored NLI 0.009, and came out harder to read than the source.
Steering deleted two articles. LoRA removed the one word that made the sentence
hard.

---

## A/B 3 — few-shot and a decode-time vocabulary constraint

    python src/evaluate.py --variants fewshot,fewshot_plus_steered,easy_words,easy_words_plus_steered \
        --n 32 --eval-sets asset --structural 15,0.75 --out results/eval_main.jsonl

**Few-shot** is 5 demonstrations from ASSET *validation*, rendered as real
prior user/assistant turns rather than pasted into one message: an instruct
model matches a completed turn it believes it produced and largely ignores the
same pairs quoted inside a single message. Each demonstration uses its source's
*median-length* reference. The shortest is usually the one that dropped a
clause, and five such examples would teach the deletion that costs `prompted`
its faithfulness score.

**Easy-words** is a logit penalty of 5.0 on word-initial tokens that no easy
Dale-Chall word starts with — 16,284 of Llama's 128,256 tokens (12.7%). The word
list is the one `textstat.is_difficult_word` reads, which is the same call that
computes `pct_hard_words`, so constraint and metric consult the same dictionary.
Capitalised tokens are exempt or every proper noun in the source would be
deleted.

### Paired deltas vs `unsteered`, 95% bootstrap CI

| variant | SARI | FKGL | hard words | NLI | degen |
|---|---|---|---|---|---|
| prompted | -5.04*<br>[-8.15, -2.05] | -5.71*<br>[-7.46, -4.03] | -0.18*<br>[-0.23, -0.14] | -0.44*<br>[-0.57, -0.32] | 0% |
| steered_structural | 0.43<br>[-1.71, 2.59] | -2.34*<br>[-3.40, -1.31] | -0.02*<br>[-0.04, -0.00] | -0.08<br>[-0.21, 0.05] | 0% |
| steered_lexical | -1.37<br>[-3.29, 0.44] | -0.19<br>[-1.22, 0.80] | -0.02*<br>[-0.04, -0.00] | -0.03<br>[-0.13, 0.07] | 0% |
| **fewshot** | **3.85***<br>[1.57, 6.19] | -2.52*<br>[-3.45, -1.68] | -0.05*<br>[-0.08, -0.02] | **+0.06**<br>[-0.04, 0.17] | 0% |
| **fewshot_plus_steered** | **4.08***<br>[1.55, 6.71] | **-5.36***<br>[-6.70, -4.09] | -0.08*<br>[-0.13, -0.05] | -0.08<br>[-0.22, 0.06] | 0% |
| easy_words | -0.44<br>[-2.04, 1.20] | -1.80*<br>[-2.55, -1.14] | **-0.10***<br>[-0.13, -0.07] | -0.02<br>[-0.09, 0.06] | 0% |
| easy_words_plus_steered | -0.58<br>[-2.53, 1.36] | -3.95*<br>[-5.16, -2.74] | **-0.12***<br>[-0.16, -0.09] | -0.05<br>[-0.17, 0.06] | 0% |

`*` = 95% CI excludes zero.

### Findings

**Few-shot is the best training-free arm and beats steering.** SARI +3.85*
against steering's +0.43 (ns), a comparable FKGL drop, and the only arm in the
study whose faithfulness point estimate exceeds the unsteered baseline (+0.06).
It requires no training, no vector bank and no sweep.

**`fewshot_plus_steered` is the strongest no-training configuration on ASSET.**
It matches prompting's readability gain (-5.36 against -5.71) with a nine-point
SARI swing in the opposite direction and no detectable faithfulness cost. A/B 7
shows this does not hold on harder text.

**The decode constraint works mechanically and does not improve quality.** It
cuts hard words by -0.10, five times what the lexical steering vector manages
(-0.02), and drops FKGL 1.80 grades at no faithfulness cost. ΔSARI is -0.44 with
the interval crossing zero: lexically easier, not a better simplification. A
twenty-line logit mask outperforming the trained lexical vector fivefold is
further evidence that the lexical dial does not encode what it claims to.

### Steering composes; it collides with the instruction specifically

Three compositions with the structural vector, all now measured:

| composition | ΔFKGL | ΔSARI | ΔNLI | degen |
|---|---|---|---|---|
| prompt + steering | -7.28* | **-13.50*** | **-0.65*** | **65.6%** |
| few-shot + steering | -5.36* | +4.08* | -0.08 (ns) | 0% |
| LoRA + steering | -7.08* | +5.93* | -0.24* | 0% |
| easy-words + steering | -3.95* | -0.58 (ns) | -0.05 (ns) | 0% |

An earlier reading of the prompt-plus-steering row — that steering does not
compose — is not supported. Steering composes with in-context supervision, with
adapted weights and with a decode constraint, in each case without a degenerate
output. It collides with the explicit *simplify instruction* and with nothing
else tested. The mechanism is consistent: two same-direction pushes summing past
the collapse point, and only the instruction pushes hard enough to overshoot.

---

## Eval set: science abstracts

`lib/data.py: load_science_abstracts()` · `data/derived/arxiv_abstracts.json`
· `--eval-sets science`

Ten abstracts, one per field, straight from the arXiv API with their IDs kept so
any of them can be checked by hand.

| field | arXiv | words | FKGL | hard % |
|---|---|---|---|---|
| cosmology | 9605151v1 | 134 | 16.7 | 43.3 |
| superconductivity | 9307029v2 | 117 | 18.2 | 44.4 |
| chemical physics | 9403007v1 | 126 | 14.6 | 43.7 |
| neuroscience | 9306039v1 | 128 | 15.4 | 35.2 |
| populations and evolution | 9303002v2 | 95 | 18.2 | 34.7 |
| computational linguistics | 9404001v1 | 99 | 23.3 | 48.5 |
| number theory | 9204234v1 | 118 | 14.0 | 29.7 |
| econometrics | 0704.3686v2 | 99 | 16.5 | 47.5 |
| geophysics | 9610011v1 | 75 | 15.9 | 45.3 |
| signal processing | 1308.0400v1 | 114 | 16.3 | 55.3 |

Mean FKGL **16.9** against ASSET test's 12.5; mean hard-word rate **0.427**
against 0.345. This set should separate the arms: an abstract is several
dependent clauses of domain vocabulary, where dropping one is easy and invisible
to a readability metric, which is the failure every arm here is tested for.
ASSET sentences are short enough to survive almost intact.

**Reproducibility.** The query is `cat:<field>` sorted by submission date
*ascending*. The newest papers in a category change daily; the oldest ones never
do, so the same query returns the same abstracts however long from now it is
re-run. The cost is an age skew — these are mostly early-1990s papers.

**SARI is unavailable on this set.** There are no human references, so the
measurement is FKGL, Dale-Chall, bidirectional NLI, conditional perplexity and
the degeneracy rules. A result here is a statement about whether meaning
survived, not about agreement with human rewrites.

**Abstracts carrying more than two inline LaTeX spans are skipped.** A rewriter
mangling `$\alpha_g$` is not the failure being looked for, and the NLI model
scores markup as noise in both directions.

**Whole-passage mode only.** The chunked path needs the model to hold "rewrite
only the final sentence", which a 3B does not, so running both would report a
failure of that constraint as a failure of the arm. Chunked mode is measured on
Gemma in A/B 13.

---

## A/B 4 — the LoRA strength ladder

`results/lora_ladder.json` ·
`python src/lora_ladder.py --lora models/lora/asset_r16 --n 32`

peft applies the adapter as `scaling · B·A·x` with `scaling = lora_alpha/r`.
Multiplying that by *s* is a continuous strength control needing no retraining.
s=0 is the base model, s=1 the adapter as trained, s>1 extrapolates past the
data. Perplexity at every rung is measured with the adapter **disabled**, since
a ratio scored under the adapter is the adapter rating its own output.

| s | FKGL | words/sent | hard words | NLI | gate % | ppl ratio | SARI | degen % |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 12.43 | 19.89 | 0.343 | 0.908 | 93.8 | 1.00 | 45.85 | 0.0 |
| 0.25 | 9.24 | 15.38 | 0.299 | **0.929** | 93.8 | **1.52** | 51.37 | 0.0 |
| 0.50 | 8.91 | 14.50 | 0.296 | 0.868 | 87.5 | 1.86 | 52.69 | 0.0 |
| 0.75 | 8.52 | 13.58 | 0.289 | 0.840 | 84.4 | 2.09 | 53.73 | 0.0 |
| 1.00 | 8.12 | 12.76 | 0.282 | 0.871 | 87.5 | 2.15 | **53.91** | 0.0 |
| 1.50 | 7.72 | 11.70 | 0.283 | 0.859 | 87.5 | 2.49 | 52.33 | 0.0 |
| 2.00 | 7.49 | 11.81 | 0.274 | 0.859 | 87.5 | 2.53 | 52.50 | 0.0 |
| 3.00 | **7.30** | 12.01 | 0.270 | 0.895 | 90.6 | 3.34 | 52.97 | 3.1 |

The s=0 rung reproduces the `unsteered` baseline it was not given — FKGL 12.43
against 12.49, SARI 45.85 against 45.44 — confirming that this harness and
`evaluate.py` agree.

### A monotone strength dial

**FKGL falls monotonically across the whole range**, 12.43 to 7.30, with no
reversal at any rung. SARI stays above 52 from s=0.5 onward. Degeneration
reaches 3.1% at three times trained strength and is zero below it.

The steering alpha ladder does not behave this way: its best FKGL cell broke two
thirds of its outputs, and alpha=2.0 collapsed 50-100% of generations. The
continuous, safe strength control this project set out to demonstrate is
supplied by the adapter, not by steering, as a property of how peft already
applies it.

**The useful region is below the trained strength, not above it.** s=0.25 holds
95% of the SARI gain and 60% of the FKGL drop while posting the *best*
faithfulness in the table (0.929, above both the base model and s=1.0) and a
perplexity ratio of 1.52 — within rounding of passing the 1.5 gate that every
other strong arm fails outright. Past s=1.0 the trade is bad: FKGL keeps
falling but SARI decays and perplexity climbs to 3.34.

**A single item is not the ladder.** An early spot-check on one abstract showed
s=0.5 through s=3.0 producing byte-identical text, suggesting a binary dial.
Across 32 items it is not. Saturation is real on individual items under greedy
decoding — the argmax path stops moving — but the item-to-item mix shifts
smoothly and the aggregate is monotone.

---

## A/B 4b — the same ladder on Gemma

`results/gemma_lora_ladder.json` ·
`python src/lora_ladder.py --model google/gemma-4-12B-it --lora models/lora/gemma4_asset_r16 --n 32`

| s | FKGL | words/sent | NLI | SARI | degen % | | Llama NLI | Llama SARI |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 11.27 | 17.06 | 0.962 | 44.98 | 0.0 | | 0.908 | 45.85 |
| 0.25 | 9.47 | 15.12 | 0.949 | 47.70 | 0.0 | | 0.929 | 51.37 |
| 0.50 | 8.57 | 14.23 | **0.955** | 51.90 | 0.0 | | 0.868 | 52.69 |
| 0.75 | 8.70 | 13.91 | 0.918 | **53.65** | 0.0 | | 0.840 | 53.73 |
| 1.00 | 7.96 | 12.05 | 0.891 | 51.16 | 0.0 | | 0.871 | **53.91** |
| 1.50 | 7.77 | 11.71 | **0.677** | 49.22 | 0.0 | | 0.859 | 52.33 |
| 2.00 | 6.88 | 10.28 | **0.381** | 44.35 | 0.0 | | 0.859 | 52.50 |
| 3.00 | 5.93 | 8.50 | **0.284** | 36.23 | 12.5 | | 0.895 | 52.97 |

The s=0 rung reproduces Gemma's `unsteered` baseline it was never shown — FKGL
11.27, SARI 44.98, NLI 0.962, all three exact.

### Correcting an earlier claim

A/B 4 concluded from Llama alone that a LoRA gives "the usable continuous
control that steering promised and did not deliver", noting that degeneration
reached only 3.1% at three times trained strength. That is true of Llama and
**not of LoRA**.

On Llama the adapter is robust to over-scaling: faithfulness sits between 0.84
and 0.93 across the whole range and ends higher than it starts. On Gemma it
collapses — 0.891 at trained strength, 0.677 at 1.5x, 0.284
at 3x, with SARI falling from 53.65 to 36.23.

| | Llama-3.2-3B | Gemma-4-12B |
|---|---|---|
| usable range (NLI within 0.05 of baseline) | **s = 0 to 3.0** | **s = 0 to 1.0** |
| NLI at s=3.0 | 0.895 | **0.284** |
| SARI at s=3.0 | 52.97 | **36.23** |
| best operating point | s ≈ 0.25 | s ≈ 0.5 |

### What survives

**Inside [0, 1] the dial works on both models**, which is the deployable range.
FKGL falls, SARI rises to a peak, degeneration stays at zero. The comparison
against the steering alpha ladder — non-monotone, best FKGL belonging to a
setting that broke two thirds of its outputs — still favours LoRA, and the peak
is below trained strength on both (s≈0.75 here, s≈0.25 for faithfulness on
Llama).

**Safe extrapolation does not survive.** Pushing an adapter past its trained
strength is harmless on one model and destructive on the other, so it is not a
property of the method. The supported claim is that a LoRA gives a usable dial
over [0, 1] and that its behaviour above 1 must be measured per model. One
ladder read as a law; two show it was a measurement.

---

## A/B 5 — adaptive alpha per item

`python src/adaptive_steering.py --n 32 --structural 15,0.75`

Alpha ramps from zero at grade 6 to the swept ceiling at grade 16, keyed on the
*source's* reading grade. No probe: the quantity the dial should respond to is a
CPU-side function of the input available before generation. Over the 32 items
this leaves 5 unsteered outright and 4 at the full 0.75, in 7 batches.

| variant | ΔSARI | ΔFKGL | ΔNLI | ppl ratio | degen |
|---|---|---|---|---|---|
| steered_structural (flat α=0.75) | 0.43<br>[-1.71, 2.59] | -2.34*<br>[-3.40, -1.31] | -0.08<br>[-0.21, 0.05] | 1.31 | 0% |
| adaptive_structural | 0.46<br>[-0.69, 1.67] | -1.82*<br>[-2.80, -0.93] | -0.01<br>[-0.10, 0.08] | **1.09** | 0% |

**The ramp does not improve quality, which was the predicted outcome.** The
ceiling is the swept best and most items sit below it, so mean alpha is well
under 0.75 and the arm lands between unsteered and flat steering on readability.
ΔSARI remains indistinguishable from zero, as for flat steering.

**It buys fluency headroom.** Perplexity ratio 1.09 against flat steering's
1.31, faithfulness essentially unchanged (-0.01), and a narrower interval on
every metric. It reaches 78% of the flat FKGL drop while five of thirty-two
items receive no intervention, so the effect comes from the hard items, as
designed.

**Not tested:** holding *mean* alpha fixed and redistributing upward, so hard
items exceed 0.75. That version could beat flat steering rather than approximate
it cheaply, at the cost of pushing the hard tail toward the collapse region.

---

## A/B 6 — best-of-n with the gate as selector

`results/best_of_n_trace.json` · `python src/best_of_n.py --n 32 --structural 15,0.75 --lexical 19,1.0`

Seven candidates per item, scored by the existing gate, shipping the lowest-FKGL
candidate that is not degenerate, entails the source in both directions, and
stays within 1.5x the baseline's perplexity. **No references are consulted** —
the rule uses only the source and the candidate, so it is deployable rather than
an oracle. Items where nothing passes fall back to the unsteered output and stay
in the average.

| variant | ΔSARI | ΔFKGL | ΔNLI | ppl ratio | degen |
|---|---|---|---|---|---|
| steered_structural | 0.43<br>[-1.71, 2.59] | -2.34*<br>[-3.40, -1.31] | -0.08<br>[-0.21, 0.05] | 1.31 | 0% |
| best_of_n | 0.78<br>[-0.85, 2.43] | -2.16*<br>[-3.01, -1.42] | +0.05<br>[-0.00, 0.12] | **1.11** | 0% |

### The gate rejected prompting, on the wrong criterion

| candidate | Llama passed | Gemma passed |
|---|---|---|
| unsteered | 29/32 | 31/32 |
| **prompted** | **0/32** | **4/32** |
| stru@0.5x | 29/32 | 29/32 |
| stru@1.0x | 20/32 | 13/32 |
| stru@1.5x | 7/32 | 6/32 |
| stru@2.0x | 2/32 | 1/32 |
| lexi@1.0x | 27/32 | 5/32 |

An earlier reading of the Llama column took this as reference-free confirmation
that prompting deletes content: a selector that has never seen a human rewrite
rejecting the prompted output on all 32 items. Running the same selector on
Gemma required checking which criterion was doing the rejecting.

| | Llama | Gemma |
|---|---|---|
| `prompted` rejected | 32/32 | 28/32 |
| failed the faithfulness floor | 22/32 | **0/28** |
| failed the perplexity ceiling | 32/32 | 28/28 |
| **rejected on perplexity alone** | **10/32** | **28/28** |

**On Gemma every rejection is the perplexity criterion, with no faithfulness
failure at all.** Gemma's prompted arm gains SARI (+2.91*), holds NLI at 0.95
and degenerates zero percent, and the gate discards it 28 times out of 32. The
consequence is visible in what best-of-n then selects: `unsteered` 15 times of
32, because the good candidates were filtered out before selection.

### Corrected reading

**The faithfulness criterion behaves correctly on both models.** It rejects
Llama's prompted output 22 times of 32, where prompting genuinely destroys
meaning, and 0 times of 28 on Gemma, where it does not. That is the criterion
generalising, and reference-free detection of the failure is a real result at
22/32 rather than 32/32.

**The perplexity criterion is wrong on both.** It fired on all 32 Llama
rejections and all 28 Gemma ones, including 10 Llama outputs whose faithfulness
was fine. Conditional perplexity is computed under the *base* model, so any arm
that legitimately shifts the output distribution is penalised for succeeding;
`PPL_RATIO_MAX = 1.5` then converts that into a hard rejection.

The claim "32/32, no references needed" rested on the defective half of the
gate. The supported claim is narrower: *the bidirectional-NLI floor detects
prompting's faithfulness failure without references on the model where that
failure exists, and declines to flag it on the model where it does not.*

`PPL_RATIO_MAX` should not be a hard gate in its present form. Every arm it
rejects here is one the degeneracy rules and NLI both pass.

### Change made: the perplexity criterion no longer gates

`Scored.passes` in `lib/bench.py` is now degeneracy and NLI only, and
`src/calibrate.py` no longer refuses an alpha on perplexity. `ppl_ratio` and
`ppl_ratio_ok` are still written to every row, so any run on disk can be
re-filtered after the fact — the threshold is reporting, not gating.

The case, measured over all 518 scored outputs per model rather than the single
Gemma arm above:

| | above the 1.5 threshold | below |
|---|---|---|
| Llama, share degenerate | 11% | 3% |
| **Gemma, share degenerate** | **4%** | **3%** |
| Gemma, share failing NLI | 28% | 4% |

On Gemma it barely separates collapsed output from intact output, which is the
job it was doing. It does separate faithfulness failures — but NLI gates those
directly, on an instrument built for them. What it mostly does is reject good
work: **55% of Gemma outputs that are neither degenerate nor unfaithful sit
above the threshold** (33% on Llama), including 67% of `steered_structural`,
Gemma's best arm, at a 2% NLI-failure rate.

Re-run on both models (`results/eval_bon_nogate_{llama,gemma}.jsonl`), paired
against the same frozen baselines:

| | Llama gated | Llama not gated | Gemma gated | Gemma not gated |
|---|---|---|---|---|
| ΔSARI | +0.78 | +0.38 | **+1.82\*** | **+1.89** (ns) |
| ΔFKGL | -2.16* | **-3.95\*** | -0.73* | **-5.39\*** |
| Δwords/sent | -3.11* | -5.90* | -0.55 | -6.48* |
| ΔNLI | +0.05 | +0.02 | -0.01* | -0.03 |
| ppl ratio | +0.11* | +1.15* | +0.13* | +9.23* |
| fell back | 1 | **0** | 1 | **0** |
| `unsteered` chosen | 8 | 5 | **15** | **2** |
| `prompted` chosen | **0** | 5 | 2 | **12** |

**What it buys is readability, not reference-matching.** Gemma's ΔFKGL goes from
-0.73 to -5.39 grades and Llama's from -2.16 to -3.95, at no measurable
faithfulness cost — both ΔNLI intervals now cross zero. But ΔSARI does not
improve on either model, and on Gemma it *loses* significance: the point
estimate barely moves (+1.82 to +1.89) while the interval widens to
[-0.25, +4.17], because per-item selections became more varied once the
criterion stopped forcing them toward `unsteered`.

The corrected claim is narrower than "best-of-n was being held back". The gate
suppressed a real readability gain and one item's fallback, and it was the only
thing keeping `prompted` out of the pool — now chosen 12 times of 32 on Gemma.
It did not suppress a SARI gain, because there was none.

---

## A/B 7 — Llama on the science abstracts

`results/eval_science.jsonl` · 10 arXiv abstracts, one per field, mean FKGL 16.7
· no references, so **no SARI**: faithfulness is the measurement.

    python src/evaluate.py --lora models/lora/asset_r16 --n 10 --eval-sets science \
        --variants unsteered,prompted,steered_structural,lora,fewshot,fewshot_plus_steered,easy_words

### Levels

| variant | FKGL | words/sent | hard words | NLI | ppl ratio | degen % |
|---|---|---|---|---|---|---|
| unsteered | 17.28 | 25.41 | 0.48 | **0.79** | 1.02 | 0.0 |
| prompted | 7.44 | 15.57 | 0.17 | **0.06** | 1.38 | 0.0 |
| steered_structural | 12.14 | 16.91 | 0.40 | 0.61 | 1.48 | **10.0** |
| lora | 15.78 | 22.79 | 0.42 | **0.76** | 1.69 | 0.0 |
| fewshot | 14.10 | 20.74 | 0.38 | 0.58 | 1.30 | 0.0 |
| fewshot_plus_steered | 7.05 | 14.04 | 0.22 | 0.22 | 2.69 | 0.0 |
| easy_words | 14.27 | 24.74 | 0.30 | 0.53 | 1.81 | 0.0 |

### Paired deltas vs `unsteered`, 95% bootstrap CI over 10 items

| variant | ΔFKGL | Δhard words | ΔNLI | NLI cost per grade |
|---|---|---|---|---|
| prompted | -9.84*<br>[-11.24, -8.60] | -0.31*<br>[-0.35, -0.26] | **-0.73***<br>[-0.83, -0.59] | 0.074 |
| fewshot_plus_steered | -10.23*<br>[-11.61, -8.81] | -0.26*<br>[-0.31, -0.21] | **-0.57***<br>[-0.69, -0.40] | 0.056 |
| easy_words | -3.01*<br>[-4.01, -2.13] | -0.18*<br>[-0.21, -0.16] | -0.26*<br>[-0.43, -0.08] | 0.086 |
| fewshot | -3.18*<br>[-4.39, -2.15] | -0.11*<br>[-0.15, -0.06] | -0.20*<br>[-0.33, -0.07] | 0.063 |
| steered_structural | -5.14*<br>[-6.80, -3.50] | -0.08*<br>[-0.13, -0.03] | -0.18*<br>[-0.27, -0.09] | 0.035 |
| **lora** | -1.50*<br>[-2.85, -0.14] | -0.06*<br>[-0.09, -0.04] | **-0.03**<br>[-0.20, 0.11] | **0.020** |

`*` = 95% CI excludes zero.

### What changed from ASSET

**On ASSET, four arms looked free. On abstracts, one does.** Every arm except
LoRA loses faithfulness significantly. The arms whose ASSET faithfulness
intervals comfortably straddled zero — few-shot at +0.06, few-shot plus steering
at -0.08 — are at -0.20 and -0.57 here, both significant.

| arm | ΔNLI on ASSET | ΔNLI on abstracts |
|---|---|---|
| lora | -0.00 (ns) | **-0.03 (ns)** |
| steered_structural | -0.08 (ns) | -0.18* |
| fewshot | +0.06 (ns) | -0.20* |
| easy_words | -0.02 (ns) | -0.26* |
| fewshot_plus_steered | -0.08 (ns) | **-0.57*** |
| prompted | -0.44* | -0.73* |

**The best training-free arm on ASSET is the second-worst here.**
`fewshot_plus_steered` gave prompting-level readability at no detectable
faithfulness cost on ASSET. On abstracts it buys 10.2 grades at a cost of two
thirds of the meaning, marginally less than prompting. The ASSET result does not
transfer to text with more to lose.

**Simplification and fidelity trade off on hard text.** Ordering the arms by how
much they simplify orders them almost exactly by how much meaning they lose. The
last column gives the exchange rate between a reading grade and faithfulness:
0.020 to 0.086, and every arm pays it. On ASSET the relationship was weak enough
not to be visible.

**Prompting an abstract replaces it rather than simplifying it.** The output
averages 245 words against a 124-word source, nearly doubling the length while
dropping FKGL to 7.4 and faithfulness to 0.06: a fresh, easy, largely unrelated
paragraph.

**The vocabulary constraint is worse here than steering.** On Wikipedia a hard
word usually has an easy synonym, so banning it costs nothing. In an abstract
the hard words are the content — there is no easy synonym for "oxidative
phosphorylation" — so forcing the model around them costs meaning (-0.26) for
three grades.

### Principal caveat

**LoRA's safety may be non-engagement rather than fidelity.** It was trained on
ASSET *sentences*, mean 26 tokens. A 120-word abstract is out of distribution
for it, and it barely moves: -1.50 grades against -4.37 on ASSET, and its output
averages 103 words against the source's 124. An adapter that mostly declines to
act will always look faithful.

The test is to train a LoRA on OneStop advanced-to-elementary *paragraph* pairs
and re-run this table. Run as A/B 12: the hypothesis is refuted. Engagement
tracks memorisation, not training length, and the cause of the under-engagement
remains open.

### Instrument note

50% of `unsteered` outputs on this set carried a preamble, against 3% on ASSET.
Before `lib.cleanup` stripped them the baseline scored NLI 0.64; after, 0.79.
The remaining gap to ASSET's 0.88 is real -- abstracts are harder to preserve --
but it is 0.09, not 0.24. The pre-fix run is kept at
`results/eval_science_prefix.jsonl` so the size of that artifact is checkable
rather than asserted.

Ten items. Intervals are wide: this set can falsify a strong claim but not
confirm a subtle one. `config.SCIENCE_PER_CATEGORY` raises it.

---

## A/B 8 — Gemma-4-12B: the depth result replicates, the alphas do not

`results/sweep_gemma-4-12B-it_instruction_diff.jsonl` ·
`results/best_layers_gemma-4-12B-it_instruction_diff.json` ·
`results/entanglement_gemma-4-12B-it_instruction_diff.json`

Second model, different family, 4x the parameters, 48 layers against 28, 3840
hidden against 3072, different tokenizer. Adapter trained (dev 3.25 → 0.66),
vectors built, stage-1 sweep run over all 48 layers x 2 coarse alphas x 2 axes.

### Two things replicate

**Axis separation.** Mean absolute cosine between the structural and lexical
vectors:

| model | mean \|cos\| |
|---|---|
| Llama-3.2-3B | 0.223 |
| **Gemma-4-12B** | **0.230** |

The instruction-difference estimator produces the same separation on an
architecture that shares nothing with the one it was tuned on. 0.223 was a
property of the method, not of Llama.

**Depth ordering.** Structural responds *earlier* than lexical on both models:

| model | structural | lexical | relative depth |
|---|---|---|---|
| Llama-3.2-3B | L15 of 28 | L19 of 28 | 54% → 68% |
| **Gemma-4-12B** | **L31 of 48** | **L34 of 48** | **65% → 71%** |

This is the headline structural claim, and it holds on a model it was not
fitted to.

### One thing does not

**The alphas do not transfer, despite depth normalisation.** Steering strength
is applied as `α · unit(v) · mean_norm[layer]` precisely so it can move between
models of different width. It does not survive the trip. Degeneration across
Gemma's responsive band, at the same α = 0.5 and 1.0 that were safe on Llama:

| layer band | Gemma degeneration |
|---|---|
| 0–11 | 0.00 — and zero effect |
| 12–19 | 0.58 – **0.92** |
| 20–26 | 0.33 – 0.92 |
| 31 (best structural) | 0.25 |
| 34 (best lexical) | 0.58 |

Two cells were unusable outright, all twelve of their texts degenerate. The
chosen (layer, α) pairs sit at 25% and 58% degeneration, against roughly zero
for Llama's chosen pair. **These settings are not usable as they stand**: they
need a stage-2 sweep at lower α before any Gemma arm can be evaluated.

### Orthogonality and efficacy do not co-locate

**Gemma's cleanest axes are its most useless layers.** The entanglement profile
says layers 10–13 have near-zero cosine between the axes — orthogonal by
measurement, not by Löwdin — while layers 15–22 sit at 0.42–0.59.

The sweep says layers 0–11 have *no effect whatsoever*: every cell reads ±0.008
with zero degeneration, then layer 12 jumps to a 0.50–0.73 effect at 92%
degeneration. A cliff, not a ramp.

The band where the two dials are genuinely independent is the band where
neither does anything, and the band where they work is the most entangled one.
**Axis orthogonality and steering efficacy are different properties and do not
co-locate.** Llama's flatter profile — 0.17 to 0.33 throughout — hides this; it
takes a model with structure in its entanglement profile to see that the two
trade against each other.

"Are the axes separable" is therefore a question about *where*. On Gemma,
separable and effective are mutually exclusive.

### Not comparable to Llama yet

Gemma's LoRA trains on the same 3800 ASSET pairs but under a different task
prompt — `BASE_TASK` alone makes it answer with a menu of options rather than a
rewrite, so `config.BASE_TASK_OVERRIDES` gives it "Reply with the rewritten text
only, nothing else." Two variables move between the models: adequate for "does
the method work on a second model", inadequate for "which model simplifies
better".

Also unresolved: `EASY_WORD_PENALTY = 5.0` was tuned against Llama logit gaps of
1–3 nats. Gemma applies `final_logit_softcapping: 30.0`, which compresses those
gaps, so the same penalty is materially harsher there.

---

## A/B 9 — Gemma-4-12B on ASSET, all twelve arms

`results/eval_gemma_asset.jsonl` · structural L31 alpha=0.5, lexical L34
alpha=0.5, both chosen from the stage-2 grid by the project's own gate.
384 rows, 0 failures.

| variant | SARI | ΔSARI | ΔFKGL | ΔNLI | degen % |
|---|---|---|---|---|---|
| **lora** | **51.16** | **+6.19***<br>[2.81, 9.32] | -3.31* | -0.07 | 0.0 |
| fewshot | 48.41 | +3.44*<br>[1.61, 5.35] | -1.51* | +0.03 | 0.0 |
| fewshot_plus_steered | 48.40 | +3.42*<br>[0.88, 5.98] | -4.42* | -0.05* | 0.0 |
| **prompted** | 47.88 | **+2.91***<br>[0.81, 4.96] | -3.97* | **-0.01** | 0.0 |
| easy_words_plus_steered | 46.75 | +1.78<br>[-0.31, 3.88] | -3.86* | -0.06 | 0.0 |
| steered_structural | 46.99 | +2.02<br>[-0.32, 4.32] | -2.47* | -0.01* | 0.0 |
| lora_plus_steered | 46.33 | +1.36<br>[-2.26, 4.78] | -5.07* | -0.44* | 0.0 |
| easy_words | 46.09 | +1.11<br>[-0.39, 2.62] | -1.08* | +0.02 | 0.0 |
| unsteered | 44.98 | — | — | — | 0.0 |
| steered_lexical | 43.44 | -1.54<br>[-4.66, 1.59] | -1.83* | **-0.31*** | 3.1 |
| steered_dual | 37.21 | -7.77*<br>[-10.68, -5.00] | -5.39* | -0.65* | 18.8 |
| prompted_plus_steered | 32.22 | -12.76*<br>[-15.44, -10.15] | -8.63* | -0.78* | 28.1 |

### The premise does not replicate

The project premise is that prompting a 3B model to simplify makes it delete
content. On Llama it does: ΔSARI **-5.04***, ΔNLI **-0.41***. On Gemma-4-12B
the same instruction gains SARI, **+2.91***, at ΔNLI -0.01 with the interval
across zero.

| | Llama-3.2-3B | Gemma-4-12B |
|---|---|---|
| prompted ΔSARI | **-5.04*** | **+2.91*** |
| prompted ΔNLI | **-0.41*** | -0.01 (ns) |

The failure that motivates steering is a property of the smaller model, not of
prompting. Whether the cause is scale or the stricter task sentence Gemma
requires is settled by the control in A/B 11.

### What replicates

Everything except prompting transfers. On both models steering is a middling arm
that no metric singles out:

| finding | Llama | Gemma |
|---|---|---|
| LoRA is the best arm | +8.47* | **+6.19*** |
| few-shot is a real free gain | +3.85* | +3.44* |
| structural steering's SARI is **not significant** | +0.43 | +2.02 |
| lexical steering does not help | -1.37 | -1.54 |
| dual-axis steering collapses | -4.27*, 34% degen | -7.77*, 19% degen |
| prompt + steering is the worst arm | -13.50*, 66% degen | -12.76*, 28% degen |

On Gemma, structural steering ranks **sixth of twelve** and its SARI interval
contains zero, while three training-free or cheap alternatives beat it
significantly. The two-dial claim fares worse: the lexical axis costs
faithfulness here (-0.31*) where on Llama it did nothing, and running both dials
together ranks third from last.

### The perplexity gate is not measuring fluency

`prompted` on Gemma has a perplexity ratio of **11.11** — seven times the 1.5
gate — with NLI 0.95 and zero degenerate outputs. `lora` scores 5.55 with NLI
0.89 and zero degeneration. The ratio is computed under the *base* model, so any
arm that legitimately shifts the output distribution is penalised for doing what
it was asked. Gemma's baseline is terse (17 words), which amplifies it.

This is the same failure identified for LoRA on Llama, at a magnitude that
rules the ratio out as a fluency criterion on this model. Degeneracy rules and
NLI carry the measurement.

---

## A/B 10 — Gemma on the science abstracts

`results/eval_gemma_science.jsonl` · 10 arXiv abstracts, no references, so
faithfulness is the measurement. Structural L31 alpha=0.5.

| variant | ΔFKGL | Δwords/sent | ΔNLI | NLI cost per grade |
|---|---|---|---|---|
| prompted | -8.01* | -4.73* | **-0.37*** | 0.046 |
| **steered_structural** | **-5.36*** | **-10.95*** | **-0.05 (ns)** | **0.009** |
| fewshot_plus_steered | -7.83* | -12.45* | -0.17* | 0.022 |
| easy_words | -3.16* | +1.20 | -0.12* | 0.038 |
| lora | -2.62* | -1.62 | -0.11 (ns) | 0.042 |
| fewshot | -2.06* | -2.04* | -0.01 (ns) | 0.005 |

### Structural steering is the best arm here

5.4 reading grades and a 10.95-word cut in sentence length, at a faithfulness
delta whose interval contains zero. It beats LoRA on readability by more than
double and matches it on meaning. This is the only eval set steering wins, and
it is hard technical prose on the larger model.

**A/B 13 qualifies this result.** On 32 SQuAD passages of comparable length the
same arm reads ΔNLI -0.14*, so the null below reflects n=10 rather than a
property of the arm.

### The Llama trade-off does not generalise

On Llama's abstracts every arm except LoRA lost faithfulness significantly, and
the arms ordered by simplification ordered by damage. On Gemma, three arms —
structural steering, LoRA, few-shot — have ΔNLI intervals containing zero, and
the ordering breaks: the arm with the *second-largest* readability gain has the
*smallest* faithfulness cost.

| | Llama-3.2-3B | Gemma-4-12B |
|---|---|---|
| arms with ΔNLI not significant | 1 of 6 | **3 of 6** |
| steered_structural ΔNLI | -0.18* | **-0.05 (ns)** |
| fewshot ΔNLI | -0.20* | -0.01 (ns) |
| best NLI cost per grade | 0.020 (lora) | **0.009 (steering)** |

"Simplifying hard text costs meaning" is therefore a statement about a 3B
model, not about the task. A 12B model simplifies the same abstracts
substantially without a measurable meaning loss at n=10. Read with the prompting
result — Llama ΔSARI -5.04*, Gemma +2.91* — the pattern is that **model
capability, not method, decides whether simplification is destructive.**

### Two caveats

**n = 10 is underpowered for a null.** ΔNLI -0.05 with interval [-0.14, 0.03] is
consistent with a real loss of up to 0.14. "Not significant" here means "not
detected at this sample size"; the set exists to falsify strong claims, not to
establish nulls. `config.SCIENCE_PER_CATEGORY` raises it. A/B 13 runs the
equivalent comparison at n=32 and finds -0.14*.

**The settings were not chosen alike.** Gemma's L31 alpha=0.5 came from a
two-stage sweep filtered by the corrected gate (NLI measured at 0.888,
degeneration 0.00). Llama's L15 alpha=0.75 came from the older process. Some of
the gap between the two models' steering results is setting quality rather than
model capability, and the amount is not known.

**Degeneration figures on this table predate the loop-detector fix** — the
10% shown for `unsteered`, `steered_structural` and `fewshot` includes the
silicon-cluster false positive. Re-scored numbers land in
`results/eval_gemma_science_v2.jsonl`.

---

## A/B 11 — control: model or prompt?

`results/eval_llama_stricttask.jsonl` ·
`python src/evaluate.py --task "Rewrite the following text. Reply with the rewritten text only, nothing else."`

Gemma's prompted arm gains SARI and holds faithfulness where Llama's collapses.
But Gemma also carries a stricter task sentence out of necessity — `BASE_TASK`
alone makes it answer with a menu of options — so "Gemma does not delete
content" and "Gemma was asked more precisely" were the same observation. This
puts the identical sentence in front of Llama.

| | ΔSARI | ΔNLI | ΔFKGL |
|---|---|---|---|
| Llama, plain `BASE_TASK` | -5.04* | **-0.41*** | -6.02* |
| **Llama, Gemma's strict task** | **-4.96*** | **-0.32*** | -4.62* |
| Gemma, strict task | **+2.91*** | **-0.01 (ns)** | -3.97* |

**It is the model.** The stricter sentence recovers 0.09 of the 0.40
faithfulness gap — about a quarter — and leaves SARI essentially untouched
(-4.96 against -5.04). Three quarters of the difference is Llama-3.2-3B.

On the abstracts it recovers nothing at all:

| | prompted NLI | ΔNLI |
|---|---|---|
| Llama, plain `BASE_TASK` | 0.06 | -0.73* |
| **Llama, strict task** | **0.07** | -0.78* |
| Gemma, strict task | **0.53** | -0.37* |

A quarter of the gap on Wikipedia sentences, **none** on technical abstracts.
The harder the source, the more of the difference is model capability rather
than instruction wording — the opposite of what a prompt-engineering explanation
predicts.

The premise should therefore be stated with a scale bound. Prompting a model to
simplify does make it delete content, and this is not an artifact of a loosely
worded instruction; it is a property of a 3B instruct model and does not survive
to 12B.

---

## A/B 12 — is LoRA's abstract faithfulness non-engagement?

`results/eval_{science,asset}_paralora.jsonl` · `_paralora_e3` · `_bothlora` ·
`models/lora/{onestop_r16,onestop_r16_e3,both_r16}`

The adapter in A/B 2 trains on ASSET sentences of mean 26 tokens and barely
engages with a 120-word abstract — 1.5 grades against 4.4 on ASSET. An adapter
that mostly declines to act scores well on faithfulness for a reason that has
nothing to do with being faithful, so the claim "LoRA preserves meaning on hard
text" could not be separated from "LoRA did not touch the hard text" until one
was trained at the right length.

OneStopEnglish supplies 219 aligned paragraph pairs (mean 119 source words, 96
target) after the same filtering the demonstrations use. Three adapters, all
scored against the byte-identical baselines A/B 2 used:

| adapter | corpus | epochs | train loss | dev loss | ASSET ΔSARI | science ΔFKGL |
|---|---|---|---|---|---|---|
| `asset_r16` | 3800 sentences | 1 | 0.672 | 0.686 | +8.47* | -1.50* |
| `onestop_r16` | 203 paragraphs | 8 | **0.121** | 0.582 | +3.85* | **-2.61*** |
| `onestop_r16_e3` | 203 paragraphs | 3 | 0.726 | **0.559** | +9.83* | -1.45 |
| `both_r16` | 4009 mixed | 1 | 0.691 | 0.647 | +8.41* | -1.02 |

### The first run supported the hypothesis and was overfit

`onestop_r16` engaged with abstracts 1.7x harder than the sentence adapter,
which is what the hypothesis predicted, and sentence length moved significantly
for the first time. But it ran 8 epochs over 203 examples drawn from 162
articles, and ended at a training loss of 0.121 against a dev loss of 0.582. It
had memorised the corpus.

The epoch control settles it. `onestop_r16_e3` runs the same data for 3 epochs,
reaches a **better** dev loss (0.559 against 0.582) in 5.9 minutes rather than
38.7, and engages with abstracts **no more than the sentence adapter does**:
ΔFKGL -1.45, interval [-2.88, +0.08], against the sentence adapter's -1.50*.
Mixing both corpora is worse still, at -1.02.

**Engagement tracked memorisation, not training length.** Only the overfit run
engaged more, and the two properly regularised paragraph adapters did not. The
proposed cause is refuted.

The observation it was meant to explain survives untouched: every adapter here
moves an abstract by 1-2.6 grades and ASSET by 4.4. Something makes LoRA
under-engage on long inputs, and it is not the length of what it trained on.

### The paragraph adapter buys efficiency, not quality

Paired on the same 32 ASSET items, `onestop_r16_e3` is indistinguishable from
`asset_r16`:

| | vs `asset_r16`, paired, 10k bootstrap |
|---|---|
| SARI | +1.36 [-0.05, +2.87] |
| NLI | +0.07 [-0.01, +0.17] |
| FKGL | **+1.73*** [+0.88, +2.64] — significantly *less* simplified |

Reading the unpaired +9.83 against +8.47 as an improvement is not supported:
the intervals overlap and the paired difference crosses zero. The comparable
SARI also comes with significantly less simplification, partly SARI rewarding a
conservative rewrite.

The result is a cost one: **203 pairs in 5.9 minutes matches 3800 pairs in 45
minutes** — 19x less data, 8x less compute, no measurable loss on either metric.
That is an efficiency finding, not a quality one, and it is the only claim this
section supports.

---

## A/B 13 — SQuAD passages

`python src/evaluate.py --model google/gemma-4-12B-it --eval-sets passages --n 32 --structural 31,0.5 --lexical 34,0.5`

The third eval set, and the one that settles A/B 10. On ten abstracts, Gemma
structural steering held faithfulness at ΔNLI −0.045 (ns) and was the best arm
in the study. Thirty-two SQuAD passages is three times the items at comparable
length, and the same arm reads **−0.14\***. The abstract result was a null the
sample could not resolve, not a win.

No SARI: SQuAD carries no simplification references, so this set separates arms
on readability and faithfulness only.

### Whole-passage mode, paired against `unsteered`

| variant | FKGL | words/sent | hard words | NLI | ppl ratio |
|---|---|---|---|---|---|
| steered_structural | -6.70\* | -13.60\* | -0.05\* | **-0.14\*** [-0.20, -0.07] | 1.81\* |
| lora | -3.68\* | -5.25\* | -0.07\* | **-0.02** [-0.09, +0.04] | 0.76\* |
| prompted | -6.64\* | -7.21\* | -0.18\* | -0.16\* [-0.25, -0.08] | 1.52\* |
| fewshot | -2.72\* | -3.67\* | -0.04\* | -0.03 [-0.08, +0.01] | 0.43\* |

LoRA's null is now measured rather than undetected: the interval is
`[-0.09, +0.04]` on 32 items, where A/B 10's was wide enough to hide a real
loss. It is the only arm that keeps meaning on long input, and it buys the least
readability doing so.

### Chunked mode, paired against `unsteered_chunked`

Sentence-at-a-time with rolling context. Pairing against the *chunked* baseline
is the comparison that isolates the arm, because chunking on its own is not
neutral: `unsteered_chunked` scores ΔNLI **+0.06\*** and ΔFKGL −1.44\* against
whole-passage `unsteered`. Against the whole-mode baseline instead,
`steered_structural_chunked` reads ΔNLI −0.00 — but that number is the chunking
credited to the arm, and it is not what the arm does.

| variant | FKGL | words/sent | hard words | NLI | ppl ratio |
|---|---|---|---|---|---|
| steered_structural_chunked | -3.99\* | -8.46\* | -0.03\* | **-0.07\*** [-0.10, -0.02] | 0.40\* |
| lora_chunked | -3.99\* | -6.36\* | -0.06\* | **-0.11\*** [-0.16, -0.06] | 0.17\* |
| prompted_chunked | -5.84\* | -5.74\* | -0.18\* | -0.23\* [-0.32, -0.15] | 0.64\* |
| fewshot_chunked | -1.50\* | -1.27\* | -0.04\* | **-0.02** [-0.07, +0.01] | 0.14\* |

Two results:

**Steering loses meaning on long text in both modes.** No reading of this set
holds it at zero.

**LoRA and few-shot swap places under chunking.** Whole-passage, LoRA is the
only faithful arm and few-shot is second; chunked, LoRA is −0.11\* and few-shot
is the only arm whose interval still covers zero. Sentence-at-a-time is the
regime the adapter was trained for — 26-token ASSET pairs — so chunking should
favour it, and it is where it does worst relative to the alternatives. This is
the A/B 12 question from the other side, and it remains open.

### One cell was measured twice

`fewshot_chunked` was initially byte-identical to `unsteered_chunked` on 32 of
32 passages. The dispatcher built its `shots` and `logits_processor` keywords
under `if mode == "whole"`, so the chunked path was handed an empty dict and
generated the plain baseline while the row recorded `n_shots: 5` from the
variant's declaration. The number above is the re-run. `MODES` gives chunked
mode only to `passages`, and no other chunked row in `results/` uses shots or
the easy-words processor, so nothing else was affected — but the row recorded
what was *configured* rather than what *ran*, which is the same defect the sweep
had with `max_new_tokens`, and it is why both now record the latter.

---

## A/B 14 — depth schedules

`python src/alpha_schedule.py --model google/gemma-4-12B-it --structural 31,0.5 --n 32 --baselines results/eval_gemma_asset.jsonl.baselines.json`

Every steered arm above injects at one layer, because the sweep is a grid over
(layer, alpha) whose best cell is a single pair. That is not an argument for
depth-one injection. The question here is the one the sweep cannot ask: at a
fixed budget, is it better to concentrate or to spread?

Budget matching is the experiment. Three layers at the incumbent alpha against
one layer at the incumbent alpha is "more steering vs less steering", which A/B
1 already answers. Two normalisations, because the right one is not obvious a
priori: `sum` spreads so that the alphas add to the budget, `l2` so that they
add in quadrature.

### The ladder, 32 ASSET items, paired against `unsteered`

| arm | layers x alpha | ΔFKGL | ΔNLI | ΔSARI | degen % |
|---|---|---|---|---|---|
| `single` (incumbent) | L31:0.500 | -2.57\* | -0.02\* | +1.84 [-0.51, +4.15] | 0.0 |
| `flat3` | L30-32:0.167 | -1.78\* | -0.01\* | **+2.70\*** [+0.42, +4.98] | 0.0 |
| `tent5` | peak 0.167 | -1.95\* | -0.00 | **+2.61\*** [+0.26, +5.04] | 0.0 |
| `ramp5_deep` | 0.033 -> 0.167 | -1.18\* | -0.01 | **+2.52\*** [+0.13, +5.01] | 0.0 |
| `ramp5_shallow` | 0.167 -> 0.033 | -1.93\* | -0.01\* | **+2.45\*** [+0.16, +4.76] | 0.0 |
| `flat5` | L29-33:0.100 | -1.52\* | -0.00\* | +1.89 [-0.38, +4.25] | 0.0 |
| `flat3_l2` | L30-32:0.289 | -5.31\* | -0.31\* | -3.01\* | **37.5** |
| `flat5_l2` | L29-33:0.224 | -2.05 | -0.68\* | -7.44\* | **78.1** |

This is the first significant SARI gain steering has produced on ASSET in this
study. The incumbent single-layer setting does not have one.

### Quadrature is the wrong model of how the layers add

The two `l2` arms carry the mechanism. Matching in quadrature is correct only
for near-orthogonal directions; otherwise it spends about sqrt(k) times the
intended budget. At the same nominal budget of 0.5, quadrature matching
degenerates 37.5% and 78.1% of outputs where every sum-matched schedule
degenerates none. **Adjacent-layer steering vectors are near-parallel** — the
same direction carried forward — which is also why the sweep's top band is
contiguous rather than a spike. `sum` is therefore the default in
`lib/schedules.py`, measured rather than assumed.

### Dose control

Norm preservation rescales the stream back to its original norm after each
injection, so a later layer partly undoes an earlier one and a sum-matched
spread lands a weaker push than the single layer it is matched to — ΔFKGL −1.78
against −2.57. That makes the headline ambiguous: a schedule scoring better on
SARI might only sit at a gentler point on the same dial, and A/B 4 shows the
best trade is below the swept alpha.

So the single layer was walked down to the spread's realised dose:

| arm | L31 alpha | ΔFKGL | ΔSARI |
|---|---|---|---|
| `single` | 0.500 | -2.57\* | +1.84 |
| `single_x0.7` | 0.350 | -1.57\* | +0.61 |
| `single_x0.6` | 0.300 | -1.24\* | +0.36 |
| `single_x0.5` | 0.250 | -0.85\* | +1.20\* |
| `single_x0.35` | 0.175 | -0.57\* | +0.24 |
| `flat3` | 3 x 0.167 | -1.78\* | **+2.70\*** |

Head to head, paired within item:

| comparison | ΔFKGL (is the dose matched?) | ΔSARI |
|---|---|---|
| `flat3` vs `single_x0.7` | -0.21 [-0.58, +0.19] — matched | **+2.09\*** [+0.98, +3.41] |
| `flat3` vs `single_x0.6` | -0.54 [-1.27, +0.21] — matched | **+2.34\*** [+1.02, +3.83] |
| `tent5` vs `single_x0.7` | -0.38 [-0.78, +0.05] — matched | **+2.01\*** [+0.56, +3.52] |
| `flat3` vs `single` | +0.79 [+0.33, +1.30] — *not* matched | +0.86\* [+0.12, +1.66] |

At a dose statistically indistinguishable from the spread, the single layer
scores +0.61 SARI and the spread +2.70. No point on the single-layer dose curve
reaches the spread's +2.6. The confound is refuted: the effect is the shape, not
the strength.

### Cost and confidence

The change is a different tuple in the same spec: no retraining, no extra
compute per token, reachable as `--structural 31,0.5,flat3` from any entry
point.

**It requires replication.** n=32, and the single-layer dose curve is
non-monotone (+1.84, +0.61, +0.36, +1.20, +0.24), which puts SARI noise here at
roughly ±1.5 — the same order as the effect. The iso-dose margin (+2.09\*) is
the solid number; the `flat3`-vs-`single` margin (+0.86\*) is too thin to carry
an argument alone. A single ASSET result reversing a study-long null should be
repeated at larger n, and on Llama, before it is stated as a claim.

---

## Instrument defect: the sweep's selector could not fail its own gate

`src/sweep_layers.py` picks the best (layer, alpha) per axis subject to a
degeneration cap and an NLI floor. The NLI filter read

    r.get("nli_mean", 1.0) >= args.min_nli

Stage 1 does not run NLI, so its rows carry no ``nli_mean`` key at all, and the
absent-key default made every unmeasured cell score as perfectly faithful. All
192 of Gemma's stage-1 cells cleared any faithfulness gate that could be asked
for. **"Not measured" was being read as "passed".**

Fixed: a cell now has to be *known* to clear the floor.

### What it cost

**On Gemma, it chose a setting whose faithfulness had never been measured.**
Stage-1 L31 alpha=1.0 (selectivity 0.352, degeneration 0.083, NLI never
measured) was preferred over stage-2 L31 alpha=0.5 (selectivity 0.269,
degeneration 0.000, NLI 0.888). The corrected selector picks alpha=0.5, which is
what the Gemma evaluation was re-run with.

An earlier version of this paragraph said alpha=1.0 "breaks a quarter of its
outputs" at degeneration 0.25. That number is not in the evidence:
`results/sweep_gemma-4-12B-it_instruction_diff.jsonl` records 0.0833 for the
only row that cell has, and no field on it holds 0.25. The correction does not
change the conclusion, because degeneration was never what excluded alpha=1.0 --
the missing NLI was. Replaying the current selector over the tracked sweep shows
it directly: at `--min-nli 0` it still picks alpha=1.0, and only at `--min-nli
0.6` and above does it fall to alpha=0.5.

### Measured: what alpha=1.0 actually does

The cell has now been re-screened at the evaluation's 320-token budget with NLI
measured (`results/sweep_rescreen320_gemma-4-12B-it_instruction_diff.jsonl`):

| L31 | degeneration @96 | degeneration @320 | selectivity @320 | **NLI @320** |
|---|---|---|---|---|
| alpha=0.5 | 0.000 | 0.000 | +0.281 | **0.858** |
| alpha=1.0 | 0.083 | **0.000** | **+0.370** | **0.260** |

**alpha=1.0 does not degenerate at all.** It produces intact, well-formed,
maximally "selective" text that no longer means what the source meant — NLI 0.26
against 0.86. The original claim (degeneration 0.25) and the guess that replaced
it (that 0.0833 was a truncation artifact hiding worse) were both wrong in the
same direction: both looked for the failure in *fluency*. The failure is in
*fidelity*, the one thing stage 1 never measured.

This is the direct argument for the fix. A selector that treats "not measured"
as "passed" does not merely risk a bad setting: run at `--min-nli 0` over the
re-screened data it still picks alpha=1.0, on a cell now known to score NLI
0.26. Only at `--min-nli 0.6` does it fall to alpha=0.5, which is what the
evaluation used.

It also disposes of a claim made while investigating this: that Gemma's
structural alpha was under-set, because selectivity kept climbing to alpha=1.0
at a degeneration of only 0.083. Selectivity does keep climbing and degeneration
is near zero, but the setting is not under-set: the metric that made it look
attractive is blind to the way it fails.

**On Llama it chose the lexical setting used in every result above.**

| axis | published setting | stage | NLI at selection |
|---|---|---|---|
| structural | L15 alpha=0.75 | 2 | 0.765, measured |
| **lexical** | **L19 alpha=1.0** | **1** | **never measured** |

The structural setting was legitimately chosen. The lexical one was not, and a
better lexical cell existed:

| setting | hard-word reduction | degen | NLI |
|---|---|---|---|
| L19 alpha=1.0 (used) | -0.139 | 0.00 | 0.830 |
| L26 alpha=1.5 (corrected pick) | **-0.223** | 0.08 | 0.665 |

### A second instance, in the degeneracy rule

`_WORD` drops digits, which is right for counting words and wrong for deciding
whether text repeats. "Si33, Si39 and Si45" tokenises to the same word three
times and reads as an immediate n-gram loop, so a clean Gemma rewrite of a
silicon-cluster abstract was flagged `loop_2gram_x2` and marked degenerate. The
model had not looped; the tokeniser had erased what distinguished the terms.

Degeneracy is a **hard gate** -- `best_of_n` discards any candidate that trips
it -- so a false positive here silently removes good outputs from contention.

Fixed by giving the loop detector its own alphanumeric tokenisation. Re-scoring
every completed run changed 2 rows out of 932, both of them this false positive:

| file | degenerate before → after |
|---|---|
| Llama ASSET (448 rows) | 32 → 32 |
| Llama abstracts (70) | 1 → 1 |
| Gemma ASSET (384) | 16 → 16 |
| Gemma abstracts (30) | 2 → **0** |

Narrow, but it is the same shape as the rest: a measure built on one text
distribution misfiring on another, and reporting the misfire as a property of
the output.

### What this does and does not invalidate

It does **not** invalidate any measurement. Every arm's faithfulness was scored
properly at evaluation time.

It also, on testing, did **not** cost anything. Re-running the lexical arm at
both settings against identical frozen baselines:

| setting | SARI | FKGL | words/sent | NLI | degen |
|---|---|---|---|---|---|
| L19 alpha=1.0 (published, chosen through the bug) | 44.28 | 12.32 | 21.9 | 0.84 | 0% |
| L26 alpha=1.5 (chosen by the corrected gate) | **32.84** | **66.31** | **157.5** | 0.41 | **46.9%** |

The setting the fixed selector prefers is catastrophic — 157-word sentences,
half the outputs degenerate. The buggy selector happened to pick the better one.
The re-run also reproduces the published L19 numbers almost exactly (ΔSARI -1.16
against -1.37, ΔFKGL -0.17 against -0.19), which is a reproducibility check on
everything that changed in the scoring path since.

**The lexical-axis conclusion stands.** It was stated on one setting, and the
alternative is unusable, so "the lexical dial does not work" is not an artifact
of a bad choice.

## The sweep screens under conditions that hide what it screens for

The re-test exposed a larger problem than either bug. The sweep's estimate for
L26 alpha=1.5, against what the evaluation found:

| | sweep | evaluation |
|---|---|---|
| degeneration | **0.08** | **0.47** |
| NLI | 0.665 | 0.407 |
| generation budget | 96 tokens (`SWEEP_MAX_NEW_TOKENS`) | 320 (`MAX_NEW_TOKENS`) |
| observed output length | capped | 159 words, about 210 tokens |

The failure mode of a too-high alpha is run-on generation. The sweep cuts
generation at 96 tokens, which truncates the runaway before it becomes
measurable, so the cell records 8% degeneration where the real figure is 47% —
wrong by a factor of six. A screening procedure that stops generating before
the failure appears cannot screen for it.

The text distribution differs too — 12 complex-pool passages in the sweep, 32
ASSET sentences in the evaluation — so the discrepancy is not attributable to
the token budget alone. Both cut the same way: **the conditions under which a
setting is chosen are more forgiving than the conditions it is used in.**

A sweep-selected setting is therefore a *candidate*, not a choice, and has to
be validated at evaluation conditions. L19 alpha=1.0 survives that; L26
alpha=1.5 does not. Every headline (layer, alpha) in this project was picked by
the unvalidated procedure, and the settings in use survive only because the
evaluation checked them downstream.

### Re-screened: the budget matters, but it is not the explanation

Both sweeps have been re-run over their top candidate cells at 320 tokens
(`results/sweep_rescreen320_*.jsonl`). The token budget hides real degeneration,
and it hides it precisely where it does damage — on cells that read exactly
0.000 at the short budget:

| model | cells changed | worst |
|---|---|---|
| Llama | 3 of 44 | structural L15 and L17 alpha=1.0, 0.000 → **0.167** |
| Gemma | 9 of 32 | structural L18 alpha=0.5, 0.083 → **0.333** |

That is enough to change the recommendation. **L17 alpha=1.0 — the structural
setting this project's own best-layers file names — exceeds the 0.15
degeneration cap once measured at 320 and is now correctly rejected.** The
selector instead picks L14 alpha=0.75 (selectivity +0.245, degeneration 0.000,
NLI 0.675), a cell the coarse stage-1 alpha grid never evaluated.

**That new pick was then evaluated, and it is worse than the published
setting.** On the same 32 ASSET items (`results/eval_stru_L14.jsonl`):

| structural | sweep selectivity @320 | ASSET dSARI | dFKGL | dwords/sent | dNLI |
|---|---|---|---|---|---|
| L15 alpha=0.75 (published) | +0.030 | **+0.43** | -2.34* | -3.74* | -0.080 |
| L14 alpha=0.75 (re-screen's pick) | **+0.245** | **-1.67** | -3.44* | -7.11* | -0.031 |

Eight times the selectivity, and two points less SARI. L14 cuts harder — seven
words per sentence against four — and SARI charges for the over-deletion FKGL
rewards. The re-screen fixed a real defect (L17 alpha=1.0 is unusable) without
producing a better setting; the published choice stands.

**But the budget does not explain the case that started this.** Re-measured at
320 tokens on the sweep's own texts, L26 alpha=1.5 still degenerates 0.083 —
identical to its 96-token figure, against 0.469 on ASSET. So the sentence above
that attributes the six-fold gap to truncation is wrong, and the alternative
that suggested itself — that the sweep pool is long passages and ASSET is short
sentences — does not survive contact with the data either. The sweep pool runs
20-43 words (median 29) against ASSET's 6-34 (median 17), and *within* the ASSET
run the degenerate items are no shorter than the intact ones (median source 16
against 17 words).

What the data does settle is what the failure is. Degenerate ASSET outputs run
to a median of **314 words against 25** for intact ones: generation to the cap.
The failure mode is run-on. Why one twelve-text pool tolerates a setting that
breaks 47% of another, at the same budget and comparable input lengths, is
**not known**, and no number in this report should be read as explaining it.

One thing the re-screen does settle for the lexical axis: at 320 tokens **1 of
40 candidate cells has positive selectivity at all**, and it degenerates 0.167.
L26 alpha=1.5 goes from +0.021 to -0.044. The axis's only apparently-selective
cell was an artifact of the short screen.

---

## Data index

`results/` is tracked in git. It is regenerable, but it is also hours of GPU on
one machine and the evidence behind every number above. Under 2 MB total.

### Paper data

| file | what it is |
|---|---|
| `sweep_Llama-3.2-3B-Instruct_instruction_diff.jsonl` | the steering layer x alpha sweep |
| `lora_ladder.json` | the LoRA strength sweep, 8 rungs x 32 items, per-item outputs kept |
| `eval_main.jsonl` | every arm, one row per (variant, item), with the generated text |
| `ab_all.json` | paired bootstrap deltas behind every A/B table above |
| `calibration_Llama-3.2-3B-Instruct_instruction_diff.json` | alpha calibration against target grades |
| `contrast_none_vs_placebo.json` | axis cosine 0.542 (no-instruction) vs 0.223 (placebo) |
| `entanglement_Llama-3.2-3B-Instruct_*.json` | axis cosines per estimator |
| `paper_examples.json` | the worked before/after pairs |
| `best_layers_Llama-3.2-3B-Instruct_instruction_diff.json` | what the selector *recommends* per axis — **not** what the runs used, see below |
| `eval_main.jsonl.baselines.json` | the frozen unsteered outputs every delta is paired against |

**Read the Llama setting off the rows, not off the best-layers file.** The file
names structural L17 alpha=1.0 and lexical L26 alpha=1.5; every Llama result row
above was generated at L15 alpha=0.75 and L19 alpha=1.0. The file is not stale —
replaying the current selector over the tracked sweep reproduces it exactly — the
runs deliberately differ from it, and `eval_lexretest_*` shows they were right to:
L26 alpha=1.5 degenerates 46.9% of its outputs at evaluation conditions against
the sweep's 8%. Gemma's file agrees with Gemma's runs, but only because those
were selected under `--min-nli 0.6`; at the default of 0 the selector wants
Gemma structural alpha=1.0 as well.

### Added after the first write-up

| file | what it is |
|---|---|
| `eval_{science,asset}_paralora.jsonl` | paragraph adapter, 8 epochs — the overfit one (A/B 12) |
| `eval_{science,asset}_paralora_e3.jsonl` | paragraph adapter, 3 epochs — the epoch control |
| `eval_{science,asset}_bothlora.jsonl` | ASSET + OneStop mixed, 1 epoch |
| `eval_bon_nogate_{llama,gemma}.jsonl` | best-of-n with perplexity no longer gating |
| `bon_trace_nogate_{llama,gemma}.json` | per-candidate scores for the above |
| `sweep_rescreen320_*_instruction_diff.jsonl` | top candidate cells re-screened at the evaluation's 320-token budget |
| `eval_gemma_passages.jsonl` | SQuAD passages, whole and chunked, 10 cells x 32 items (A/B 13) |
| `eval_gemma_passages.jsonl.baselines.json` | the frozen unsteered passage outputs, per mode |
| `alpha_schedule_gemma.json` | the depth-schedule ladder, 8 shapes x 32 items (A/B 14) |
| `alpha_schedule_dose_gemma.json` | the iso-dose control for the above |
| `control_wps_g{2,4,8,12}.json`, `control_ladder_g4.json` | control-token ladders at each guidance strength (fail.md section 7) |
| `dexperts_ladder.json` | logit-space extrapolation against the weight-space ladder (fail.md section 1) |
| `dual_surface_llama.json` | the 5x5 structural x lexical alpha surface (fail.md section 2) |

**Both sweeps are paper data.** How the layers were *chosen* — the stage-1 scan
that narrows 28 layers to 3 — is method, not result, and does not need to appear
in the paper. The alpha and strength sweeps that run on the chosen layers do.

### Regenerating

    python src/report.py results/eval_main.jsonl --out results/ab_all.json
    python src/lora_ladder.py --lora models/lora/asset_r16 --n 32
    python src/report.py results/eval_gemma_passages.jsonl --eval-set passages \
        --baseline unsteered_chunked
    python src/alpha_schedule.py --model google/gemma-4-12B-it --structural 31,0.5 \
        --n 32 --baselines results/eval_gemma_asset.jsonl.baselines.json
    python src/alpha_schedule.py --model google/gemma-4-12B-it --structural 31,0.5 \
        --n 32 --schedules dose \
        --baselines results/eval_gemma_asset.jsonl.baselines.json \
        --out results/alpha_schedule_dose_gemma.json

Not in git: `models/` (weights and adapters), `data/` (corpus caches), and
`vectors/` (the steering banks). All rebuild from
`scripts/00_download.py` and `src/build_vectors.py`.

---

## Notes

- Model weights are not in the repo. `python scripts/00_download.py llama nli`
  pulls them with aria2c into `models/`.
- `src/report.py` produces every table above. It pairs within item and
  bootstraps over items; unpaired intervals on 32 ASSET sentences are dominated
  by which sentences were drawn.
