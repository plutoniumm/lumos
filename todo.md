# Knobs for text simplification

Everything we could pull on next, filtered to what is worth pulling. Three filters
applied: no large training runs (small ones are fine), no negligible effects, and
nothing that fails to scale.

Status: `[x]` built and measured, `[~]` built but not yet measured, `[ ]` not
started. Measurements live in [report.md](report.md).

## No training at all

- [x] **Steering** — `α·v` on the residual stream. The config sub-knobs (layer band,
  position mode, norm preservation, content-orthogonalization) are part of this one
  dial, not separate ones.
- [x] **Conditional / adaptive steering** — `α` ramps from 0 to the swept ceiling with
  the *source's* reading grade, so an already-readable sentence gets no push. No probe
  needed: the quantity the dial should respond to is a CPU-side function of the input.
  Over the 32 eval items this leaves 5 unsteered and 4 at the full 0.75.
  Measured: 78% of the flat-alpha FKGL drop at a much better perplexity ratio
  (1.09 vs 1.31), but SARI still indistinguishable from zero.
  `src/adaptive_steering.py`
- [x] **Best-of-n + gate rerank** — generate N variants across `α` or prompts, ship the
  simplest that passes NLI + perplexity + degeneracy, falling back to unsteered when
  none do. The gate already existed; this turns it from a reporting filter into a
  selector. Subsumes "NLI filter and regenerate". Measured: buys safety, not
  quality — best perplexity ratio of any effective arm (1.11) and faithfulness
  above baseline, but ΔSARI crosses zero. Its real value is that the gate
  rejected prompting 32/32 times using no references at all.
  `src/best_of_n.py`
- [x] **Decode-time vocabulary constraint** — logit penalty on word-initial tokens no
  easy Dale-Chall word starts with (12.7% of the vocabulary). Largest single lever on
  lexical difficulty, costs one vector add per step. `lib/decoding.py`,
  `--variants easy_words,easy_words_plus_steered`
- [ ] **Contrastive decoding / DExperts** — `logits(simple) − logits(complex)` at each
  step. 2× compute, no training, works at any model scale.
- [x] **Few-shot exemplars** — 5 median-length ASSET-valid pairs as real prior chat
  turns. The strongest prompt-side lever by a wide margin, and it raises the baseline
  steering must beat. `--variants fewshot,fewshot_plus_steered`
- [ ] **Automated prompt search** — DSPy/APE over instruction wordings. One-time cost.

## Small training only

- [x] **LoRA on graded pairs** — ASSET validation, rank 16, ~1.5 h on one machine.
  The only approach that plausibly hits a *target* reading grade, which steering
  demonstrably cannot reach. Measured: +8.47 SARI, −4.37 FKGL, no faithfulness
  cost — the best arm in the study. `src/train_lora.py`, evaluated via
  `src/evaluate.py --lora`
- [x] **LoRA scale as a dial** — the adapter's contribution multiplied by s at
  inference, laddered in the same format as the steering α sweep. It answers
  yes: FKGL falls monotonically 12.43 → 7.30 across s = 0 → 3 with ≤3.1%
  degeneration, where the steering ladder is non-monotone and collapses
  50–100% at α=2.0. Best trade is *below* trained strength, at s≈0.25.
  `src/lora_ladder.py`
- [ ] **Adaptive α per layer** — one α per layer across a band rather than one
  scalar for the whole injection. The per-*item* version is measured (above);
  this is the per-*depth* version, and it is the natural next dial now that the
  layer-band result exists.
- [ ] **LoRA on paragraph pairs** — the current adapter trains on ASSET
  sentences (mean 26 tokens) and barely engages with a 120-word abstract, so its
  faithfulness there may be non-engagement rather than fidelity. OneStop
  advanced→elementary paragraphs are the length-matched training set. This is
  the single most informative experiment left.
- [ ] **Multi-adapter merge** — one LoRA per axis, mixed with per-adapter weights. The
  weight-space version of the two dials in this repo.
- [ ] **Control tokens as a LoRA** — ACCESS-style conditioning on length ratio, word
  rank and dependency-tree depth, trained as an adapter rather than a full seq2seq run.

## Model coverage

- [x] **Llama-3.2-3B-Instruct** — everything above is measured on this.
      Full parity with Gemma: 14 ASSET arms, 6 science arms, LoRA ladder.
- [~] **Gemma-4-12B-it** — smoke passes 8/8 on MPS, so the kernel worry is
  closed. LoRA trained (dev 3.25 → 0.66) and steering vectors built (axis
  cosine 0.230, replicating Llama's 0.223 on a different architecture). Layer
  **at full parity with Llama** — 14 ASSET arms, 6 science arms, LoRA ladder,
  best-of-n and adaptive α all run.
  Sweep: structural L31, lexical L34 — the depth ordering replicates
  (structural before lexical, as on Llama). But α=0.5/1.0 leaves 25–92%
  degeneration across the responsive band, so the settings are unusable until a
  stage-2 sweep at lower α. Needs its own task prompt
  (`config.BASE_TASK_OVERRIDES`), and `EASY_WORD_PENALTY` re-tuned against
  `final_logit_softcapping: 30.0`.
- [ ] **Gemma stage-2 sweep at lower α** — the blocker on every Gemma arm.
  Try α ∈ {0.1, 0.2, 0.3} on layers 31/18/30 (structural) and 34/38/32
  (lexical).

## Eval sets

- [x] **ASSET test** — 32 single Wikipedia sentences, 10 human references each,
  so SARI is available. Mean FKGL 12.5.
- [x] **arXiv abstracts** — 10 abstracts, one per field, mean FKGL 16.7. It did
  separate the arms, and it overturned ASSET: only LoRA holds faithfulness
  (ΔNLI -0.03 ns), every other arm loses it significantly, and the best *free*
  ASSET arm (few-shot + steering) drops to -0.57. Raise
  `config.SCIENCE_PER_CATEGORY` for more.
- [ ] **SQuAD passages** — already in the harness, not part of the current
  comparison.

## Open questions the instrument raised

- [x] **Re-test the lexical axis at L26 α=1.5** — done, and it settles the
  question against the alternative: L26 α=1.5 gives 157-word sentences and 47%
  degeneration, against L19 α=1.0's clean 0%. The published setting was the
  better one; "the lexical dial does not work" is not an artifact of a bad pick.
- [x] **Raise `SWEEP_MAX_NEW_TOKENS` to the eval budget** — done, it now tracks
  `MAX_NEW_TOKENS`. Existing sweep files were collected at 96 and stay
  optimistic about degeneration at high α; their low-α cells are unaffected,
  since a setting that terminates inside 96 tokens terminates the same at 320.
  Original note: the sweep generates
  96 tokens where the eval generates 320, so a run-on failure is truncated
  before it can be measured. It recorded 8% degeneration for a setting that
  degenerates 47% in use. Screening cells cost ~3× more this way, which is the
  price of the screen meaning anything.
- [x] **Is Gemma's prompting advantage the model or the prompt?** — the model.
  Giving Llama Gemma's exact task sentence recovers a quarter of the
  faithfulness gap on ASSET and none of it on abstracts. The premise (prompting
  deletes content) is real at 3B and expires by 12B.
- [ ] **Re-check the Llama structural setting under the corrected selector** —
  it was legitimately chosen (stage 2, NLI 0.765 measured), but the corrected
  report with a degeneration cap prefers L17 α=1.0. Worth knowing whether the
  published L15 α=0.75 is still the best available.

- [ ] **Steered generations do not reproduce across sessions.** Re-running
  `steered_lexical` L19 a1.0 over ASSET test reproduces 25/32 of the archived
  generations byte for byte and differs on the other 7. It is deterministic --
  two fresh runs agree 32/32, and the same 7 items differ every time -- and it
  reproduces with `src/evaluate.py` bypassed entirely, generating straight
  through `lib.steering`, so it lives at or below the injection path. The
  unsteered path is unaffected. Nothing in `lib/{steering,hooks,models,vectors}.py`
  or the vector banks has changed since the archived run; `config.py` has, 39
  minutes after it. Worth finding before the next full re-run, because it sets
  how much a published steered number can be trusted to the last decimal.

- [ ] **Drop or recalibrate `PPL_RATIO_MAX` as a hard gate** — it is computed
  under the base model, so it punishes any arm that legitimately shifts the
  output distribution. It rejected Gemma's `prompted` 28/28 times on perplexity
  alone, with zero faithfulness failures, and made best-of-n ship `unsteered`
  15/32 times. Every arm it rejects passes both the degeneracy rules and NLI.

## Not doing, and why

**Encoder-decoder (Flan-T5 and similar).** Architecturally it fits simplification well
— a seq2seq model is built to transduce a source, and encoder-vs-decoder injection
would separate meaning from expression by construction. Ruled out anyway: essentially
every model that will matter going forward is decoder-only, so a method that depends
on an encoder does not carry forward. Anything we build has to work in one residual
stream.

**Large training.** Full SFT, continued pretraining, distillation, RLHF/DPO, RL against
a programmatic reward.

**Does not scale.** PPLM and FUDGE — both need gradient steps or a classifier call per
generated token.

**Negligible, or dominated by something already listed.** Sampling parameters, persona
framing, untrained prompt-side control tokens, prefix/prompt/IA³ tuning, constrained
beam search.

**Analysis tools, not controls.** Activation patching, ablation, function vectors.

**Blocked on missing artifacts.** SAE feature clamping — no usable sparse autoencoder
exists for Llama-3.2-3B, and training one is a large training run.

**Brittle or already measured to fail here.** Dictionary lexical substitution, glossary
expansion, sentence-level chunking (a 3B model does not hold the "rewrite only the
final sentence" constraint), a user-facing dial (a product decision, not a method).
