# Progress report

Running log of what has been measured, in the order it was measured. Every
result here is reproducible from a command recorded next to it. When a number
changes, the old one stays in place with a note saying why, because a report
that silently rewrites its own history is not evidence of anything.

Plan and remaining knobs: [todo.md](todo.md).

**Base model** `unsloth/Llama-3.2-3B-Instruct` (bf16, MPS)
**Eval set** ASSET test, first 32 items, 10 human references each
**Baseline** `unsteered` — the bare task prompt `"Rewrite the following text."`,
no simplify instruction, no intervention

Every arm below is compared against that same baseline, on those same 32 items,
with the same prompt. Only the mechanism changes.

---

## Where this stands

Two models at full parity: fourteen arms each on ASSET, six each on the science
abstracts, plus a LoRA strength ladder per model. All paired against the same
`unsteered` baseline per model. ASSET is 32 Wikipedia sentences with 10 human
references (SARI available); science is 10 arXiv abstracts with none, so
faithfulness is the measurement. `*` = 95% bootstrap CI excludes zero.

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
ratio of any effective arm there (1.13), which is what it is for — safety, not
quality. It is still beaten threefold by LoRA and by prompting, and it now does
so while its own gate discards the best available candidate 28 times in 32 (see
A/B 6).

### The result

**The premise is real, and it has a scale boundary.** This project exists
because prompting a model to simplify makes it delete content. On
Llama-3.2-3B that is emphatic — ΔSARI -5.04*, and on abstracts faithfulness
falls to 0.06. On Gemma-4-12B the same instruction *gains* SARI, +2.91*, at
ΔNLI -0.01.

A direct control settles what causes the difference. Giving Llama Gemma's exact
task sentence recovers a quarter of the faithfulness gap on ASSET and **none of
it** on abstracts. Three quarters on easy text and all of it on hard text is
the model, not the wording. The motivation for steering is sound at 3B and
expires by 12B.

**LoRA is the strongest method.** Best arm on both models on ASSET, +8.47* and
+6.19*, from 45 minutes of training and 24M trainable parameters. It is also
the only Llama arm that survives the abstracts without a significant
faithfulness loss.

**Steering wins exactly once, and the once is informative.** Structural steering
takes the Gemma abstracts: -5.36 grades, sentences cut by 11 words, ΔNLI -0.05
with the interval across zero — the best simplicity-per-unit-meaning of any arm
anywhere in the study. It wins where the model is capable enough to simplify
hard text unaided, so a gentle nudge lands; on Llama every intervention is
fighting a model that sheds content under any pressure.

Everywhere else steering is middling. Its ASSET SARI is not significant on
either model, it ranks sixth of twelve on Gemma, and the two-dial claim is worse
off than the one-dial one: the lexical axis costs faithfulness on Gemma
(-0.31*), and both dials together is the third-worst arm on the board.

**Two structural claims replicate across architectures.** Axis separation, mean
\|cos\| 0.223 on Llama and 0.230 on Gemma; and depth ordering, structural before
lexical on both (L15/28 and L31/48). **One does not**: alpha, even
depth-normalised, does not transfer — Llama's values degenerate 25-92% of
Gemma's outputs.

**Orthogonality and efficacy do not co-locate.** On Gemma the layers where the
axes are cleanest (10-13, \|cos\| ≈ 0) are layers where steering does nothing at
all, and the layers that work are the most entangled. Llama's flat profile hides
this entirely.

### Not settled

- **Whether LoRA's abstract faithfulness is fidelity or non-engagement.** It
  trains on 26-token sentences and barely moves a 120-word abstract. An adapter
  that declines to act always looks faithful. `--corpus onestop` is built and
  unrun.
- **n = 10 on the abstracts** is underpowered for a null. ΔNLI -0.05 with
  interval [-0.14, 0.03] means "not detected", not "not there".
- **The two models' settings were not chosen alike**, so some of the steering
  gap between them is setting quality by an unknown amount.

### The methods finding

Seven times this evaluation produced a confident wrong answer rather than an
error. Every one had the same shape: **a measure built on one distribution
misfiring on another, and reporting the misfire as a property of the model.**

| defect | reported | actually |
|---|---|---|
| LoRA scored its own perplexity | ratio 0.58 | 1.80 |
| preamble counted as an output sentence | NLI 0.084 | 0.990 |
| citation fragments counted as content | ” | ” |
| unmeasured sweep cells passed the NLI gate | best setting | breaks 25% of outputs |
| perplexity ratio read as fluency | Gemma prompted "fails" at 11.11 | NLI 0.95, 0% degenerate |
| digit-stripped loop detection | 3 degenerate abstracts | 0 |
| **sweep screens at 96 tokens, uses at 320** | **8% degeneration** | **47%** |

The last is the one to lead with. The failure mode of too much alpha is run-on
generation, and the screen stops generating before it appears. Every headline
(layer, alpha) in this project was chosen by that procedure; the ones in use
survive because the evaluation caught what the screen could not, not because
the screen worked.

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
| prompted | -5.04*<br>[-8.12, -2.08] | -6.02*<br>[-7.80, -4.31] | -6.24*<br>[-9.44, -3.18] | -0.19*<br>[-0.23, -0.15] | -0.41*<br>[-0.53, -0.29] | 2.78*<br>[2.02, 3.74] |
| steered_structural | 0.43<br>[-1.68, 2.58] | -2.34*<br>[-3.43, -1.30] | -3.74*<br>[-5.91, -1.62] | -0.02*<br>[-0.04, -0.00] | -0.08<br>[-0.21, 0.05] | 0.31*<br>[0.21, 0.44] |
| steered_lexical | -1.37<br>[-3.35, 0.47] | -0.19<br>[-1.23, 0.79] | 1.48*<br>[0.11, 3.03] | -0.02*<br>[-0.04, -0.00] | -0.03<br>[-0.13, 0.07] | 0.20*<br>[0.12, 0.32] |
| steered_dual | -4.27*<br>[-7.13, -1.51] | 1.23<br>[-3.38, 8.25] | 10.49<br>[-2.19, 29.66] | -0.11*<br>[-0.17, -0.06] | -0.30*<br>[-0.45, -0.14] | 1.02*<br>[0.71, 1.37] |
| prompted_plus_steered | -13.50*<br>[-17.45, -9.67] | -7.28*<br>[-9.55, -4.69] | -5.16<br>[-10.12, 0.90] | -0.25*<br>[-0.30, -0.20] | -0.65*<br>[-0.75, -0.55] | 3.99*<br>[1.56, 8.31] |

`*` = 95% CI excludes zero. 10,000 resamples.

### What this says

**Structural steering is the only arm that does not cost anything.** It is the
only variant whose SARI interval contains zero while FKGL drops significantly
(−2.34 grades) and faithfulness does not move (ΔNLI −0.08, CI crosses zero).
Every other intervention buys simplicity with meaning.

**Prompting reads better and is worse.** It wins FKGL by a mile (−5.71 grades)
and loses SARI (−5.04) while faithfulness collapses from 0.88 to 0.43 — it is
deleting content, and FKGL rewards deletion. This is the central result: the
readability metric and the faithfulness metric disagree, and only one of them
is measuring what a dyslexic reader actually needs.

**The lexical dial does not work.** ΔFKGL −0.19 with the interval straddling
zero, ΔSARI negative, and sentences get *longer*. Whatever the lexical vector
encodes, it is not "use easier words" at any strength we can apply safely.

**Two dials at once are worse than one.** `steered_dual` degenerates 34% of the
time. The two vectors are only approximately orthogonal (mean absolute cosine
0.223), and what is left of the overlap compounds.

**Prompt + steering collide.** Both push the same direction, so the sum
overshoots the collapse point: 65.6% degenerate, faithfulness 0.22, the worst
SARI measured (−13.50). This is not an argument for dropping the prompt — the
task prompt is present in every variant and is required — it is an argument
against stacking the *simplify instruction* on top of a vector at this α.

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
| prompted | -5.04*<br>[-8.12, -2.08] | -6.02*<br>[-7.80, -4.31] | -6.24*<br>[-9.44, -3.18] | -0.19*<br>[-0.23, -0.15] | -0.41*<br>[-0.53, -0.29] | 2.78*<br>[2.02, 3.74] |
| steered_structural | 0.43<br>[-1.68, 2.58] | -2.34*<br>[-3.43, -1.30] | -3.74*<br>[-5.91, -1.62] | -0.02*<br>[-0.04, -0.00] | -0.08<br>[-0.21, 0.05] | 0.31*<br>[0.21, 0.44] |
| **lora** | **8.47***<br>[6.05, 10.98] | **-4.37***<br>[-5.60, -3.17] | -7.60*<br>[-10.40, -5.08] | -0.06*<br>[-0.09, -0.03] | **-0.00**<br>[-0.15, 0.14] | 1.15*<br>[0.80, 1.60] |
| lora_plus_steered | 5.93*<br>[1.80, 9.67] | -7.08*<br>[-8.60, -5.66] | -11.48*<br>[-14.70, -8.59] | -0.24*<br>[-0.41, -0.08] | -0.24*<br>[-0.41, -0.08] | 2.12*<br>[1.61, 2.79] |

`*` = 95% CI excludes zero.

### What this says

**LoRA wins, and not narrowly.** It is the only arm that buys a large
simplification without paying for it in meaning:

|  | ΔFKGL | ΔSARI | ΔNLI |
|---|---|---|---|
| prompting | -6.02 | **-5.04** | **-0.41** |
| structural steering | -2.34 | +0.43 (ns) | -0.08 (ns) |
| **LoRA** | **-4.37** | **+8.47** | **-0.00** |

Steering's SARI gain is indistinguishable from zero. LoRA's is +8.47 with the
interval nowhere near it. LoRA drops 1.9 grades more than steering does, and
gets within 1.3 grades of prompting while prompting is destroying its own
faithfulness score.

**The perplexity ratio should not be read as a fluency failure here.** LoRA
scores 2.15, above the 1.5 gate. But conditional perplexity is measured under
the *base* model, and an adapter that legitimately changed the output
distribution is penalised by that construction for doing exactly what it was
trained to do. The gate was built to catch steering-induced degeneration, and
the direct evidence disagrees with it in this case: 0% of LoRA outputs trip any
degeneracy rule, and faithfulness is 0.87. The metric is measuring distance from
the base model's style, which is the thing we asked for.

**Faithfulness is unchanged on average, not item by item.** ΔNLI is -0.00, but
the interval is [-0.15, 0.14] and individual items do lose content. Item 3
("His next work, Saturday, follows an especially eventful day...") comes back as
"His next work, Saturday, is about a day in the life of a successful
neurosurgeon" — NLI 0.220, "especially eventful" dropped. LoRA deletes
sometimes; it just does not delete *systematically* the way prompting does.

**LoRA composes with steering far better than prompting does.** Both
compositions cost faithfulness, but `prompted_plus_steered` collapses (65.6%
degenerate, NLI 0.22) while `lora_plus_steered` does not degenerate at all
(0%, NLI 0.63) and posts the largest non-degenerate FKGL drop measured (-7.08).
Whatever collision destroys the prompt-plus-vector combination is not present
in the weight-plus-vector one.

**A worked pair**, item 11, source *"It was originally thought that the debris
thrown up by the collision filled in the smaller craters."*

| variant | output | FKGL | NLI | SARI |
|---|---|---|---|---|
| prompted | A long time ago, people thought that when two big rocks crashed into each other, the pieces that flew up into the air filled in the smaller holes that were left behind. | 11.6 | 0.009 | 34.2 |
| steered_structural | Initially, it was believed that debris from the collision filled in smaller craters. | 9.4 | 0.993 | 37.6 |
| lora | It was thought that the debris thrown up by the collision filled in the smaller craters. | 6.1 | 0.993 | 54.2 |

Prompting paraphrased the craters into "smaller holes" and the collision into
"two big rocks", scored NLI 0.009, and came out *harder* to read than the
source. Steering deleted two articles. LoRA removed the one word that made the
sentence hard.

---

## A/B 3 — the free arms: few-shot and a decode-time vocabulary constraint

    python src/evaluate.py --variants fewshot,fewshot_plus_steered,easy_words,easy_words_plus_steered \
        --n 32 --eval-sets asset --structural 15,0.75 --out results/eval_main.jsonl

**Few-shot** is 5 demonstrations from ASSET *validation*, rendered as real prior
user/assistant turns rather than pasted into one message, because an instruct
model matches a completed turn it believes it produced and largely ignores the
same pairs quoted inside a single message. Each demonstration uses its source's
*median-length* reference: the shortest is usually the one that dropped a
clause, and five examples of dropping a clause would teach exactly the deletion
that costs `prompted` its faithfulness score.

**Easy-words** is a logit penalty of 5.0 on word-initial tokens that no easy
Dale-Chall word starts with — 16,284 of Llama's 128,256 tokens (12.7%). The word
list is the one `textstat.is_difficult_word` reads, which is the same call that
computes `pct_hard_words`, so constraint and metric consult the same dictionary.
Capitalised tokens are exempt or every proper noun in the source would be
deleted.

### Paired deltas vs `unsteered`, 95% bootstrap CI

| variant | SARI | FKGL | hard words | NLI | degen |
|---|---|---|---|---|---|
| prompted | -5.04*<br>[-8.12, -2.08] | -5.71*<br>[-7.51, -3.99] | -0.18*<br>[-0.23, -0.14] | -0.44*<br>[-0.57, -0.32] | 0% |
| steered_structural | 0.43<br>[-1.68, 2.58] | -2.34*<br>[-3.43, -1.30] | -0.02*<br>[-0.04, -0.00] | -0.08<br>[-0.21, 0.05] | 0% |
| steered_lexical | -1.37<br>[-3.35, 0.47] | -0.19<br>[-1.23, 0.79] | -0.02*<br>[-0.04, -0.00] | -0.03<br>[-0.13, 0.07] | 0% |
| **fewshot** | **3.85***<br>[1.59, 6.20] | -2.52*<br>[-3.46, -1.67] | -0.05*<br>[-0.08, -0.02] | **+0.06**<br>[-0.04, 0.17] | 0% |
| **fewshot_plus_steered** | **4.08***<br>[1.45, 6.65] | **-5.36***<br>[-6.75, -4.06] | -0.08*<br>[-0.13, -0.05] | -0.08<br>[-0.23, 0.05] | 0% |
| easy_words | -0.44<br>[-2.06, 1.15] | -1.80*<br>[-2.58, -1.14] | **-0.10***<br>[-0.13, -0.07] | -0.02<br>[-0.09, 0.07] | 0% |
| easy_words_plus_steered | -0.58<br>[-2.50, 1.34] | -3.95*<br>[-5.20, -2.74] | **-0.12***<br>[-0.16, -0.09] | -0.05<br>[-0.17, 0.06] | 0% |

`*` = 95% CI excludes zero.

### What this says

**Few-shot is the best free arm, and it beats steering outright.** Higher SARI
(+3.85 significant, against steering's +0.43 that is not), a comparable FKGL
drop, and it is the only arm in the whole study whose faithfulness point
estimate is *above* the unsteered baseline (+0.06). It needs no training, no
vector bank, and no sweep — five examples in the context window.

**`fewshot_plus_steered` is the strongest no-training configuration measured.**
It matches prompting's readability gain (-5.36 against -5.71) with a nine-point
SARI swing in the opposite direction and no detectable faithfulness cost. If
nothing may be trained, this is the arm to ship.

**The decode constraint works mechanically and does not help.** It cuts hard
words by -0.10, five times what the *lexical steering vector* manages (-0.02),
and drops FKGL by 1.80 grades — all with no faithfulness cost. But ΔSARI is
-0.44 with the interval straddling zero: it makes text lexically easier without
making it a better simplification by human standards. A twenty-line logit mask
beating the trained lexical vector by 5x is the clearest evidence yet that the
lexical dial is not encoding what it claims to.

### Steering composes; it collides with the instruction specifically

Three compositions with the structural vector, all now measured:

| composition | ΔFKGL | ΔSARI | ΔNLI | degen |
|---|---|---|---|---|
| prompt + steering | -7.28* | **-13.50*** | **-0.65*** | **65.6%** |
| few-shot + steering | -5.36* | +4.08* | -0.08 (ns) | 0% |
| LoRA + steering | -7.08* | +5.93* | -0.24* | 0% |
| easy-words + steering | -3.95* | -0.58 (ns) | -0.05 (ns) | 0% |

An earlier reading of the prompt-plus-steering row — that steering does not
compose — does not survive these. Steering composes with in-context
supervision, with adapted weights, and with a decode constraint, in every case
without a single degenerate output. It collides with the explicit *simplify
instruction* and with nothing else tested. The mechanism still fits: two
same-direction pushes summing past the collapse point. Only the instruction
pushes hard enough to overshoot.

---

## The hard eval set: science abstracts

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
against 0.345. This is the set that should separate the arms, because an
abstract is several dependent clauses of domain vocabulary where dropping one is
both easy and invisible to a readability metric — which is the exact failure
every arm here is being tested for. ASSET sentences are short enough to carry
almost intact.

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

**Not yet run.** Whole-passage mode only: the chunked path needs the model to
hold "rewrite only the final sentence", which a 3B does not, so running both
would report a failure of that constraint as a failure of the arm.

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

The s=0 rung reproduces the `unsteered` baseline it was never told about — FKGL
12.43 against 12.49, SARI 45.85 against 45.44 — which is a free check that this
harness and `evaluate.py` agree.

### This is the dial steering never delivered

**FKGL falls monotonically across the whole range**, 12.43 to 7.30, with no
reversal at any rung. SARI stays above 52 from s=0.5 onward. Degeneration
reaches 3.1% at *three times* trained strength and is zero everywhere below it.

Set against the steering alpha ladder, where the best FKGL in the entire sweep
belonged to a setting that broke two thirds of its outputs and alpha=2.0
collapsed 50-100% of generations, the contrast is the point of this table. The
project's headline claim was a continuous, safe strength control. Steering does
not provide one. A rank-16 adapter does, for free, as a property of how peft
already applies it.

**The useful region is below the trained strength, not above it.** s=0.25 holds
95% of the SARI gain and 60% of the FKGL drop while posting the *best*
faithfulness in the table (0.929, above both the base model and s=1.0) and a
perplexity ratio of 1.52 — within rounding of passing the 1.5 gate that every
other strong arm fails outright. Past s=1.0 the trade is bad: FKGL keeps
falling but SARI decays and perplexity climbs to 3.34.

**A single item is not the ladder.** An early spot-check on one abstract showed
s=0.5 through s=3.0 producing byte-identical text, which looked like a binary
dial. Across 32 items it is nothing of the kind. The saturation is real on
*individual* items under greedy decoding — the argmax path stops moving — but
the item-to-item mix shifts smoothly, and the aggregate is monotone.

---

## A/B 4b — the same ladder on Gemma. The dial's safety is model-specific.

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

On Llama the adapter is extraordinarily robust to over-scaling: faithfulness
sits between 0.84 and 0.93 across the entire range and *ends higher than it
starts*. On Gemma it collapses — 0.891 at trained strength, 0.677 at 1.5x, 0.284
at 3x, with SARI falling from 53.65 to 36.23.

| | Llama-3.2-3B | Gemma-4-12B |
|---|---|---|
| usable range (NLI within 0.05 of baseline) | **s = 0 to 3.0** | **s = 0 to 1.0** |
| NLI at s=3.0 | 0.895 | **0.284** |
| SARI at s=3.0 | 52.97 | **36.23** |
| best operating point | s ≈ 0.25 | s ≈ 0.5 |

### What survives

**Inside [0, 1] the dial works on both models**, and that is the range anyone
would ship. FKGL falls, SARI rises to a peak, degeneration stays at zero. Set
against the steering alpha ladder — non-monotone, best FKGL belonging to a
setting that broke two thirds of its outputs — the comparison still favours
LoRA, and the peak is below trained strength on both (s≈0.75 here, s≈0.25 for
faithfulness on Llama).

**What does not survive is safe extrapolation.** Pushing an adapter past what it
was trained on is fine on one model and destroys the other, so "scale it up if
you want more" is not a property of the method. The honest claim is: *a LoRA
gives a usable dial over [0, 1]; its behaviour above 1 has to be measured per
model.*

That the two models disagree at all is the useful part. One ladder looked like a
law; two ladders show it was a measurement.

---

## A/B 5 — adaptive alpha per item

`python src/adaptive_steering.py --n 32 --structural 15,0.75`

Alpha ramps from zero at grade 6 to the swept ceiling at grade 16, keyed on the
*source's* reading grade. No probe: the quantity the dial should respond to is a
CPU-side function of the input available before generation. Over the 32 items
this leaves 5 unsteered outright and 4 at the full 0.75, in 7 batches.

| variant | ΔSARI | ΔFKGL | ΔNLI | ppl ratio | degen |
|---|---|---|---|---|---|
| steered_structural (flat α=0.75) | 0.43<br>[-1.68, 2.58] | -2.34*<br>[-3.43, -1.30] | -0.08<br>[-0.21, 0.05] | 1.31 | 0% |
| adaptive_structural | 0.46<br>[-0.68, 1.67] | -1.82*<br>[-2.79, -0.94] | -0.01<br>[-0.10, 0.08] | **1.09** | 0% |

**The ramp does not pay for itself in quality, and this was the predicted
outcome.** Because the ceiling is the swept best and most items sit below it,
mean alpha is well under 0.75, so the arm lands between unsteered and flat
steering on readability. ΔSARI remains indistinguishable from zero, as it is for
flat steering.

**What it does buy is fluency headroom.** Perplexity ratio 1.09 against flat
steering's 1.31 — comfortably inside the 1.5 gate rather than near its edge —
with faithfulness essentially untouched (-0.01) and a narrower interval on every
metric. It reaches 78% of the flat FKGL drop while five of the thirty-two items
receive no intervention at all, which is the design working: the effect is
coming from the hard items, where it should.

**Not tested:** holding *mean* alpha fixed and redistributing upward, so hard
items exceed 0.75. That is the version that could beat flat steering rather than
approximate it cheaply, and it pushes the hard tail toward the collapse region.

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
| steered_structural | 0.43<br>[-1.68, 2.58] | -2.34*<br>[-3.43, -1.30] | -0.08<br>[-0.21, 0.05] | 1.31 | 0% |
| best_of_n | 0.78<br>[-0.85, 2.40] | -2.16*<br>[-3.00, -1.42] | +0.05<br>[-0.00, 0.12] | **1.11** | 0% |

### The gate never once chose prompting — but not for the reason claimed

| candidate | Llama passed | Gemma passed |
|---|---|---|
| unsteered | 29/32 | 31/32 |
| **prompted** | **0/32** | **4/32** |
| stru@0.5x | 29/32 | 29/32 |
| stru@1.0x | 20/32 | 13/32 |
| stru@1.5x | 7/32 | 6/32 |
| stru@2.0x | 2/32 | 1/32 |
| lexi@1.0x | 27/32 | 5/32 |

An earlier reading of the Llama column claimed this was a reference-free
confirmation that prompting deletes content: a selector that has never seen a
human rewrite rejecting the prompted output on all 32 items. Running the same
selector on Gemma forced a look at *which* criterion was doing the rejecting.

| | Llama | Gemma |
|---|---|---|
| `prompted` rejected | 32/32 | 28/32 |
| failed the faithfulness floor | 22/32 | **0/28** |
| failed the perplexity ceiling | 32/32 | 28/28 |
| **rejected on perplexity alone** | **10/32** | **28/28** |

**On Gemma every rejection is the perplexity criterion, with no faithfulness
failure at all.** Gemma's prompted arm gains SARI (+2.91*), holds NLI at 0.95
and degenerates zero percent — and the gate discards it 28 times out of 32. The
consequence is visible in what best-of-n then ships: `unsteered` is chosen 15
times of 32, because the good candidates were filtered out before selection.

### What the corrected reading is

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

The headline "32/32, no references needed" was carried by the broken half of the
gate. The defensible claim is narrower and still worth having: *the
bidirectional-NLI floor detects prompting's faithfulness failure without
references, on the model where that failure exists, and correctly declines to
flag it on the model where it does not.*

`PPL_RATIO_MAX` should not be a hard gate in its present form. Every arm it
rejects here is one the degeneracy rules and NLI both pass.

---

## A/B 7 — the science abstracts. This changes the answer.

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
| prompted | -9.84*<br>[-11.28, -8.59] | -0.31*<br>[-0.36, -0.26] | **-0.73***<br>[-0.83, -0.59] | 0.074 |
| fewshot_plus_steered | -10.23*<br>[-11.60, -8.78] | -0.26*<br>[-0.31, -0.21] | **-0.57***<br>[-0.69, -0.40] | 0.056 |
| easy_words | -3.01*<br>[-4.05, -2.12] | -0.18*<br>[-0.21, -0.16] | -0.26*<br>[-0.43, -0.09] | 0.086 |
| fewshot | -3.18*<br>[-4.35, -2.16] | -0.11*<br>[-0.15, -0.06] | -0.20*<br>[-0.33, -0.07] | 0.063 |
| steered_structural | -5.14*<br>[-6.79, -3.49] | -0.08*<br>[-0.13, -0.03] | -0.18*<br>[-0.27, -0.09] | 0.035 |
| **lora** | -1.50*<br>[-2.88, -0.16] | -0.06*<br>[-0.09, -0.04] | **-0.03**<br>[-0.20, 0.11] | **0.020** |

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

**The best free arm on ASSET is the second-worst arm here.**
`fewshot_plus_steered` was the strongest no-training configuration measured —
prompting-level readability at no detectable faithfulness cost. On abstracts it
buys 10.2 grades by destroying two thirds of the meaning, which is what
prompting does, only slightly less of it. Its ASSET result did not survive
contact with text that has something to lose.

**There is no free lunch on hard text.** Order the arms by how much they
simplify and they order almost perfectly by how much meaning they destroy. The
last column makes it explicit: the exchange rate between a reading grade and
faithfulness runs from 0.020 to 0.086 and every arm pays it. On ASSET the
relationship was weak enough that "simplify more, lose nothing" looked
achievable.

**Prompting on an abstract does not simplify it, it replaces it.** The output
averages 245 words against a 124-word source -- it nearly doubles the length
while dropping FKGL to 7.4 and faithfulness to 0.06. That is not compression,
it is the model writing a fresh, easy, largely unrelated paragraph.

**The vocabulary constraint is worse here than steering.** Predictably: on
Wikipedia a hard word usually has an easy synonym, so banning it costs nothing.
In an abstract the hard words *are* the content -- there is no easy synonym for
"oxidative phosphorylation" -- so forcing the model around them costs meaning
(-0.26) for three grades.

### The caveat that matters most

**LoRA's safety may be non-engagement rather than fidelity.** It was trained on
ASSET *sentences*, mean 26 tokens. A 120-word abstract is out of distribution
for it, and it barely moves: -1.50 grades against -4.37 on ASSET, and its output
averages 103 words against the source's 124. An adapter that mostly declines to
act will always look faithful.

That reading is testable and the test is obvious: **train a LoRA on OneStop
advanced-to-elementary *paragraph* pairs** and re-run this table. If paragraph
LoRA keeps the faithfulness and gains the readability, LoRA is genuinely the
answer. If it starts losing meaning like everything else, then the honest
conclusion is that nothing here simplifies hard technical prose without damaging
it, and the project's real finding is the measurement rather than the method.

### Instrument note

50% of `unsteered` outputs on this set carried a preamble, against 3% on ASSET.
Before `lib.cleanup` stripped them the baseline scored NLI 0.64; after, 0.79.
The remaining gap to ASSET's 0.88 is real -- abstracts are harder to preserve --
but it is 0.09, not 0.24. The pre-fix run is kept at
`results/eval_science_prefix.jsonl` so the size of that artifact is checkable
rather than asserted.

Ten items. Intervals are wide and this set can falsify a strong claim but not
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

This is the paper's headline structural claim, and it holds on a model it was
never fitted to.

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

Two cells were unusable outright, every one of their twelve texts degenerate.
The chosen (layer, α) pairs sit at 25% and 58% degeneration, against roughly
zero for Llama's chosen pair. **These settings are not usable as they stand** —
they need a stage-2 sweep at lower α before any Gemma arm can be evaluated.
Reporting a Gemma-versus-Llama comparison from them would be reporting the
bracket, not the model.

### The finding neither model's evaluation would have produced alone

**Gemma's cleanest axes are its most useless layers.** The entanglement profile
says layers 10–13 have near-zero cosine between the axes — orthogonal by
measurement, not by Löwdin — while layers 15–22 sit at 0.42–0.59.

The sweep says layers 0–11 have *no effect whatsoever*: every cell reads ±0.008
with zero degeneration, then layer 12 jumps to a 0.50–0.73 effect at 92%
degeneration. A cliff, not a ramp.

So the band where the two dials are genuinely independent is the band where
neither dial does anything, and the band where they work is the band where they
are most entangled. **Axis orthogonality and steering efficacy are different
properties and they do not co-locate.** Llama's flatter profile — 0.17 to 0.33
throughout — hides this entirely; it takes a model with structure in its
entanglement profile to see that the two can be traded against each other.

That reframes "are the axes separable" from a yes/no into a question about
*where*, and the answer on Gemma is that you may pick separable or effective but
not both.

### Not comparable to Llama yet

Gemma's LoRA trains on the same 3800 ASSET pairs but under a different task
prompt — `BASE_TASK` alone makes it answer with a menu of options rather than a
rewrite, so `config.BASE_TASK_OVERRIDES` gives it "Reply with the rewritten text
only, nothing else." Two variables move between the models, which is fine for
"does the method work on a second model" and not fine for "which model
simplifies better."

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
| **lora** | **51.16** | **+6.19***<br>[2.81, 9.30] | -3.31* | -0.07 | 0.0 |
| fewshot | 48.41 | +3.44*<br>[1.62, 5.34] | -1.51* | +0.03 | 0.0 |
| fewshot_plus_steered | 48.40 | +3.42*<br>[0.82, 5.92] | -4.42* | -0.05* | 0.0 |
| **prompted** | 47.88 | **+2.91***<br>[0.87, 5.00] | -3.97* | **-0.01** | 0.0 |
| easy_words_plus_steered | 46.75 | +1.78<br>[-0.32, 3.89] | -3.86* | -0.06 | 0.0 |
| steered_structural | 46.99 | +2.02<br>[-0.26, 4.36] | -2.47* | -0.01* | 0.0 |
| lora_plus_steered | 46.33 | +1.36<br>[-2.24, 4.84] | -5.07* | -0.44* | 0.0 |
| easy_words | 46.09 | +1.11<br>[-0.34, 2.64] | -1.08* | +0.02 | 0.0 |
| unsteered | 44.98 | — | — | — | 0.0 |
| steered_lexical | 43.44 | -1.54<br>[-4.68, 1.57] | -1.83* | **-0.31*** | 3.1 |
| steered_dual | 37.21 | -7.77*<br>[-10.64, -5.04] | -5.39* | -0.65* | 18.8 |
| prompted_plus_steered | 32.22 | -12.76*<br>[-15.45, -10.08] | -8.63* | -0.78* | 28.1 |

### The premise does not replicate

This project exists because prompting a 3B model to simplify makes it delete
content. On Llama that is emphatic: ΔSARI **-5.04***, ΔNLI **-0.41***. On
Gemma-4-12B the same instruction *gains* SARI, **+2.91***, at ΔNLI -0.01 with
the interval across zero.

| | Llama-3.2-3B | Gemma-4-12B |
|---|---|---|
| prompted ΔSARI | **-5.04*** | **+2.91*** |
| prompted ΔNLI | **-0.41*** | -0.01 (ns) |

The failure that motivates steering is a property of the smaller model, not of
prompting. Whether it is *scale* or the stricter task sentence Gemma requires is
the open question the strict-task control answers.

### What replicates, and it is bad for steering

Everything except prompting transfers, and on both models steering is a
middling arm that no metric singles out:

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
significantly. The two-dial claim fares worse still: the lexical axis costs
faithfulness here (-0.31*) where on Llama it merely did nothing, and running
both dials together is the third-worst arm on the board.

### The perplexity gate is not measuring fluency

`prompted` on Gemma has a perplexity ratio of **11.11** — seven times the 1.5
gate — with NLI 0.95 and zero degenerate outputs. `lora` scores 5.55 with NLI
0.89 and zero degeneration. The ratio is computed under the *base* model, so any
arm that legitimately shifts the output distribution is penalised for doing what
it was asked. Gemma's baseline is terse (17 words), which amplifies it.

This is the same failure identified for LoRA on Llama, and it is now large
enough that the gate should not be read as a fluency criterion at all on this
model. Degeneracy rules and NLI are doing the real work.

---

## A/B 10 — Gemma on the abstracts. Steering's best result, and a reversal.

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
double and matches it on meaning. Nowhere else in this project does steering win
an eval set.

That it happens *here* is the interesting part: hard technical prose, on the
larger model. Steering does best exactly where the task is hardest.

### The Llama "no free lunch" finding does not generalise

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

So "simplifying hard text costs meaning" is a statement about a 3B model, not
about the task. A 12B model simplifies the same abstracts substantially without
a measurable meaning loss. Read together with the prompting result — Llama
ΔSARI -5.04*, Gemma +2.91* — the pattern is that **model capability, not
method, decides whether simplification is destructive.**

### Two caveats that matter

**n = 10 is underpowered for a null.** ΔNLI -0.05 with interval [-0.14, 0.03] is
consistent with a real loss of up to 0.14. "Not significant" here means "not
detected at this sample size", and the set exists to falsify strong claims, not
to establish nulls. `config.SCIENCE_PER_CATEGORY` raises it.

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

## A/B 11 — control: is Gemma's prompting advantage the model or the prompt?

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

A quarter of the gap on Wikipedia sentences, **none** of it on technical
abstracts. The harder the source, the more completely the difference is model
capability rather than instruction wording — which is the opposite of what a
prompt-engineering explanation would predict, and about as clean a separation as
this eval can give.

This matters for how the project's premise should be stated. Prompting a model
to simplify really does make it delete content, and that is not an artifact of a
loosely worded instruction. It is a property of a 3B instruct model, and it does
not survive to 12B. The motivation for steering is sound *at the scale it was
developed*, and expires above it.

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

**On Gemma, it chose a setting that breaks a quarter of its outputs.** Stage-1
L31 alpha=1.0 (selectivity 0.352, degeneration 0.25, NLI never measured) was
preferred over stage-2 L31 alpha=0.5 (selectivity 0.269, degeneration 0.00, NLI
0.888). The corrected selector picks alpha=0.5, which is what the Gemma
evaluation was re-run with.

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

**So the lexical-axis conclusion stands, and stands harder.** It was stated on
one setting; the alternative turns out to be unusable, so "the lexical dial does
not work" is not an artifact of a bad choice.

## The sweep screens under conditions that hide what it screens for

The re-test exposed something larger than either bug. The sweep's estimate for
L26 alpha=1.5 and what the evaluation found:

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

The practical consequence is that a sweep-selected setting is a *candidate*, not
a choice, and has to be validated at evaluation conditions before it is
believed. L19 alpha=1.0 survives that; L26 alpha=1.5 does not. Every headline
(layer, alpha) in this project was picked by the unvalidated procedure, and the
ones in use survive only because they were checked downstream by the evaluation
itself.

---

## Data index

`results/` is tracked in git as of this run. It had been ignored as
"regenerable", which is true and beside the point: these are hours of GPU on one
machine and they are the evidence behind every number above. Under 2 MB total.

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
| `best_layers_Llama-3.2-3B-Instruct_instruction_diff.json` | chosen (layer, alpha) per axis |
| `eval_main.jsonl.baselines.json` | the frozen unsteered outputs every delta is paired against |

**Both sweeps are paper data.** How the layers were *chosen* — the stage-1 scan
that narrows 28 layers to 3 — is method, not result, and does not need to appear
in the paper. The alpha and strength sweeps that run on the chosen layers do.

### Regenerating

    python src/report.py results/eval_main.jsonl --out results/ab_all.json
    python src/lora_ladder.py --lora models/lora/asset_r16 --n 32

Not in git: `models/` (weights and adapters), `data/` (corpus caches), and
`vectors/` (the steering banks). All rebuild from
`scripts/00_download.py` and `src/build_vectors.py`.

---

## Notes

- Model weights are not in the repo. `python scripts/00_download.py llama nli`
  pulls them with aria2c into `models/`.
- `src/report.py` is the only thing that produces the tables above. It pairs
  within item and bootstraps over items; unpaired intervals on 32 ASSET
  sentences are dominated by which sentences were drawn.
