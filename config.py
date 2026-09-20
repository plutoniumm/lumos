"""Every tuning parameter in the project. Nothing here should import project code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
VECTORS_DIR = ROOT / "vectors"
RESULTS_DIR = ROOT / "results"


# meta-llama gates Llama-3.2 behind a license acceptance separate from the one
# covering Llama-3.1, so a token can reach 3.1 and 403 on 3.2. The mirror is
# ungated and its bf16 shards are byte-for-byte identical (4965799096 +
# 1459729952). Point LLAMA at the canonical repo once the grant is live.
LLAMA_CANONICAL = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_MIRROR = "unsloth/Llama-3.2-3B-Instruct"
LLAMA = LLAMA_MIRROR

GEMMA = "google/gemma-3n-E4B-it"

# The second architecture, so that a steering result is not a fact about Llama.
# 12B is the only Gemma 4 that is *encoder-free*: it is
# Gemma4UnifiedForConditionalGeneration, whose vision and audio inputs go
# through 52M of embedders rather than towers, so loading it for a text-only job
# wastes ~0.1 GB. The 31B/26B/E4B siblings are Gemma4ForConditionalGeneration
# and do carry real towers. All four hang the text stack off the same
# model.language_model.layers, so lib/models.py needs no per-family branch.
GEMMA4 = "google/gemma-4-12B-it"

NLI_MODEL = "FacebookAI/roberta-large-mnli"

# Gemma, not Llama, because Llama's weights are no longer on this machine and
# hub.resolve falls back to the repo id -- leaving this at LLAMA would make
# every default command silently re-download six gigabytes instead of failing.
# Llama remains the model behind report.md's headline result; its vector banks
# are still in vectors/, so pass --model to reproduce those numbers once the
# weights are back.
ACTIVE_MODEL = GEMMA4
DTYPE = "bfloat16"
DEVICE = "mps"


# HF's client opens one connection per file; aria2c saturates the link instead.
ARIA2_CONNECTIONS_PER_FILE = 16
ARIA2_CONCURRENT_FILES = 4
ARIA2_MIN_SPLIT = "8M"

# Weights we want, and the ones we must not pull: meta-llama ships a duplicate
# 6 GB consolidated .pth under original/ that transformers never reads.
#
# .jinja is here because transformers 5 moved chat templates out of
# tokenizer_config.json into chat_template.jinja, and gemma-4 ships only the new
# location -- filtered out, the model loads fine and then apply_chat_template
# raises, i.e. there is no prompt at all. Llama ships both and its 3827-byte
# .jinja is byte-identical to its embedded chat_template, so the extra suffix
# cannot change what Llama renders.
DOWNLOAD_ALLOW_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".jinja")
DOWNLOAD_DENY_SUBSTRINGS = (
    "original/",
    "consolidated",
    "onnx/",
    ".pth",
    ".bin",
    ".gguf",
    ".h5",
    ".msgpack",
    ".tflite",
)


AXES = ("structural", "lexical")

# The task stays in the prompt at all times; steering supplies the *degree*, not
# the task. This mirrors llm-steer-instruct, where a base instruction remains and
# only the additional constraint becomes a vector. Drop the task and an instruct
# model handed a bare sentence has no reason to rewrite it -- it will comment on
# it or ask a question instead, and the whole measurement collapses.
BASE_TASK = "Rewrite the following text."

# The same string is not the same *treatment* across models. Llama-3.2-3B answers
# BASE_TASK with a rewrite; Gemma-4-12B answers it with a menu -- "Depending on
# the tone you want to achieve, here are a few ways to rewrite the text:" plus
# two or three markdown-headed options. Every readability number computed on that
# is measuring the preamble. The override buys back the behaviour BASE_TASK is
# supposed to elicit, so the models stay comparable in what they were asked to
# do rather than in which bytes they were sent.
#
# Llama is deliberately left on the bare BASE_TASK: it already complies, and
# changing its prompt would invalidate every result already measured with it.
BASE_TASK_OVERRIDES = {
    "google/gemma-4-12B-it": (
        "Rewrite the following text. Reply with the rewritten text only, "
        "nothing else."
    ),
}


def base_task(repo_id: str | None = None) -> str:
    """BASE_TASK, or this model's override if it needs one to produce a rewrite."""
    if repo_id is None:
        return BASE_TASK
    # Accept a local directory as well as a repo id, since models.load resolves
    # to a path and callers pass whichever they have.
    for key, task in BASE_TASK_OVERRIDES.items():
        if repo_id == key or repo_id.endswith(key.split("/")[-1]):
            return task
    return BASE_TASK


# Used by the chunked passage path, where only the last sentence is the target.
CHUNK_TASK = (
    "Rewrite only the final sentence of the following text. "
    "Keep it consistent with the sentences before it."
)

# Axis-specific constraints, appended after BASE_TASK. The instruction-difference
# estimator separates the two axes by construction: each vector is the activation
# delta caused by *only* its own constraint, so whatever the model encodes for
# "shorter sentences" never enters the lexical vector.
INSTRUCTIONS = {
    "structural": (
        "Use short, simple sentences with one idea each. "
        "Keep every fact exactly as it is."
    ),
    "lexical": (
        "Replace difficult or technical words with common, everyday words. "
        "Keep the sentence structure the same."
    ),
    # Monolithic control, and the prompt-only baseline that steering must beat.
    "simplify": (
        "Make it easy to read for a young student. "
        "Do not change the meaning and do not leave out any information."
    ),
}

# Axis-matched placebos: same topic, same register, similar length, differing from
# the real instruction only in its direction along that one axis.
#
# These exist because contrasting against a *no-instruction* prompt entangles the
# axes. Part of the delta is then caused by one prompt being shorter, and that part
# is common to both axes. Re-measured on Llama-3.2-3B over the same 256 texts with
# a correct recorder: no-instruction contrast gives mean absolute axis cosine
# 0.542, axis-matched placebos give 0.223. Differencing two instructions of
# comparable length cancels the shared length component and leaves the part
# specific to the axis.
#
# The 0.997 previously recorded here was measured before the recorder was fixed and
# was an artifact of that defect, as was the claim that the two norms matched.
# Real per-axis mean norms under the no-instruction contrast are 8.95 and 7.33.
PLACEBOS = {
    "structural": (
        "Keep the original sentence structure and sentence length exactly as they are."
    ),
    "lexical": ("Keep the original wording and vocabulary exactly as they are."),
    "simplify": ("Keep the text exactly as it is."),
}

# "placebo" contrasts against PLACEBOS; "none" contrasts against no instruction at
# all. Kept switchable because the difference between them is a finding worth
# reporting, not an implementation detail.
DEFAULT_CONTRAST = "placebo"

# Which token positions receive the injection.
#   gen_only    - newly generated tokens only (CAA convention)
#   prompt_last - the final prompt token only (llm-steer-instruct convention)
#   all         - prompt and generation
DEFAULT_POSITIONS = "gen_only"

# Rescale the hidden state to its original norm after adding alpha*v. This is
# the single biggest guard against degeneration at high alpha.
NORM_PRESERVE = True

# Gram-Schmidt the axis vectors against each other so the two dials are
# genuinely independent rather than both encoding "be terse".
ORTHOGONALIZE_AXES = True

# Project the steering vector orthogonal to the source text's own mean-pooled
# representation, so steering cannot push along the content axis.
CONTENT_ORTHOGONALIZE = True

# Where the vector is read from / added to. "resid_post" is the decoder layer
# output (the residual stream after the block); "mlp_out" is the feed-forward
# output only, which is what a "steer the feed-forward" reading implies.
DEFAULT_SITE_KIND = "resid_post"


# Alpha is a fraction of the layer's typical activation norm. Re-measured with a
# correct recorder (every earlier figure came from the MPS storage-offset bug):
# 0.25 barely moves, 1.0 gives large clean effects at 0-12% degeneration, and 2.0
# collapses 50-100% of outputs. Negative alphas are kept as a sign check -- they
# should make text *harder*, and a vector that simplifies in both directions is
# not measuring what it claims to.
SWEEP_ALPHAS = (-1.0, -0.5, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)

# The full grid (every layer x every alpha x both axes) is around a day of wall
# clock even batched, and most of it is spent on layers that do nothing. Stage 1
# scans all layers at one moderate alpha to find where each axis responds; stage 2
# spends the alpha grid only on the layers that survived.
# Two coarse alphas, not one. Alpha tolerance is strongly depth-dependent:
# measured on Llama-3.2-3B, layer 10 collapses at 0.6 while layer 22 is still
# clean at 1.2. A single coarse alpha would break the early layers and barely
# move the late ones, excluding both from the profile for opposite reasons.
SWEEP_COARSE_ALPHAS = (0.5, 1.0)
SWEEP_COARSE_N_TEXTS = 12
SWEEP_TOP_K_LAYERS = 3
SWEEP_NLI_SUBSAMPLE = 12  # NLI is the expensive metric; sample it in stage 2

# Number of texts used to estimate each steering vector.
VECTOR_N_TEXTS = 256


NLI_ENTAILMENT_MIN = 0.50  # required in BOTH directions

# Reporting threshold, NOT a gate. Every row still carries ppl_ratio and a
# ppl_ratio_ok flag against this number, so any run can be re-filtered after the
# fact -- but nothing rejects an output for exceeding it. Measured across both
# models it does not separate collapsed output from intact output (above vs
# below the threshold, share degenerate: Gemma 4% / 3%, Llama 11% / 3%), while
# rejecting 55% of the Gemma outputs that are neither degenerate nor unfaithful.
# The ratio is relative to each model's own unsteered perplexity, and 1.50 was
# picked on Llama; Gemma's scale is different enough that its LoRA ladder
# reaches 976. See Scored.passes in lib/bench.py.
PPL_RATIO_MAX = 1.50  # vs unsteered output on the same source
TARGET_FKGL = {"grade4": 4.0, "grade6": 6.0, "grade8": 8.0}
# Upper bound tracks the measured collapse point (~2.0), not an optimistic guess:
# with bounds to 6.0 an evenly spaced ladder puts most of its rungs in the
# degenerate region and resolves nothing.
CALIBRATE_ALPHA_BOUNDS = (0.0, 2.0)
CALIBRATE_MAX_ITERS = 8


MAX_NEW_TOKENS = 320
DO_SAMPLE = False
SEED = 0

# Batched generation is what makes the sweep tractable: unbatched, a full
# layer x alpha x axis grid is roughly a day of wall clock.
BATCH_SIZE = 8

# The sweep used to stop at 96 on the reasoning that it only needs enough tokens
# to measure sentence length and word difficulty, not a complete answer. That
# reasoning is wrong, and it cost a real result.
#
# The failure mode of too much alpha is run-on generation. Cutting generation at
# 96 tokens truncates the runaway before it becomes measurable, so the screen
# cannot see the thing it screens for. Measured on lexical L26 alpha=1.5: the
# sweep recorded 8% degeneration and NLI 0.665, and the evaluation -- same
# vectors, same model, 320 tokens -- found 47% degeneration, NLI 0.407 and
# 157-word sentences. Wrong by a factor of six, in the direction that makes a
# broken setting look usable.
#
# Matching MAX_NEW_TOKENS costs roughly 3x the sweep wall clock. That is the
# price of the screen meaning anything.
#
# Sweeps already on disk were collected at 96 and are therefore optimistic about
# degeneration at high alpha; their low-alpha cells are unaffected, because a
# setting that terminates well inside 96 tokens terminates the same way at 320.
SWEEP_MAX_NEW_TOKENS = MAX_NEW_TOKENS

# Synthetic CAA pairs are rewrites of single sentences; anything longer is
# trailing commentary that would be averaged into the vector.
PAIR_MAX_NEW_TOKENS = 128


# Rolling previous *simplified* sentences, passed back in so pronouns and
# referents stay coherent. Defaults to 0 -- i.e. off -- because prepending context
# requires switching to CHUNK_TASK ("Rewrite only the final sentence...") and a 3B
# instruct model follows that unreliably: measured on a SQuAD passage it rewrote
# the context instead, emitting the same sentence 6 of 9 times and inventing a
# date the source did not contain ("concluded on May 14, 2014" for a source
# reading "premiered on January 15, 2014"). The whole-passage path handled the
# same item cleanly. Raise this only with a model large enough to hold the
# constraint, and check faithfulness when you do.
CHUNK_CONTEXT_SENTENCES = 0


# --- LoRA -------------------------------------------------------------------
# The weight-space arm of the A/B. Steering adds a vector at inference and
# trains nothing; LoRA trains a small adapter and adds nothing at inference.
# Both are compared against the same unsteered baseline, on the same items,
# with the same prompt, so the only difference is the mechanism.

LORA_DIR = MODELS_DIR / "lora"

# ASSET *validation* only. The eval set is ASSET test; training on it would make
# the comparison meaningless, so the split is not a parameter here.
LORA_TRAIN_SPLIT = "valid"

# ASSET ships 10 human references per source. Using all of them is ten near-
# duplicate gradients per example, which mostly buys optimizer noise. Two gives
# the model a spread of valid rewrites for the same input without that.
LORA_REFS_PER_SOURCE = 2

# r=16 is the smallest rank that reliably moves a 3B model on a rewriting task;
# r=8 underfits the lexical substitutions and r=32 costs twice the memory for a
# change we cannot resolve at this eval size.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Attention *and* MLP. Attention-only adapters move phrasing but leave word
# choice alone, and word choice is half of what we are trying to control.
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

LORA_LR = 1e-4
LORA_EPOCHS = 1
LORA_BATCH_SIZE = 4
LORA_GRAD_ACCUM = 4  # effective batch 16
LORA_WARMUP_FRAC = 0.03
LORA_MAX_SEQ_LEN = 320  # ASSET is single sentences; 320 covers prompt + target
LORA_DEV_FRAC = 0.05  # held out to detect overfitting, not to tune anything


# --- Few-shot ---------------------------------------------------------------
# Demonstrations come from ASSET *validation*; the eval split is test, and the
# two are disjoint in the corpus. Five is where in-context learning on a 3B
# model stops improving on this task and starts eating context that the source
# sentence needs.
FEWSHOT_N = 5


# --- Decode-time vocabulary constraint --------------------------------------
# Logit penalty on word-initial tokens that no easy Dale-Chall word starts with.
# Not a ban: an infinite penalty forces the model to paraphrase around technical
# terms that have no easy synonym, and it paraphrases them wrongly. 5.0 is large
# against typical Llama logit gaps (1-3 nats between plausible continuations)
# without being absolute.
EASY_WORD_PENALTY = 5.0

# Not portable across models. Gemma-4 applies final_logit_softcapping=30.0,
# which squashes logits through c*tanh(x/c) and compresses the very gaps this
# penalty is measured against, so the same 5.0 is a materially harsher
# intervention there. What has to stay constant for the arms to be comparable is
# the penalty's size relative to the decision it perturbs, not its raw value.
# Filled from scripts/05_logit_gaps.py; empty means "use the reference value",
# which is only correct for a model with Llama-like logit spread.
EASY_WORD_PENALTY_OVERRIDES: dict[str, float] = {"google/gemma-4-12B-it": 11.39}


def easy_word_penalty(repo_id: str | None = None) -> float:
    if repo_id is None:
        return EASY_WORD_PENALTY
    for key, value in EASY_WORD_PENALTY_OVERRIDES.items():
        if repo_id == key or repo_id.endswith(key.split("/")[-1]):
            return value
    return EASY_WORD_PENALTY


# --- Adaptive steering ------------------------------------------------------
# A flat alpha spends the same push on a sentence that is already grade 6 as on
# one at grade 18. The first does not need it and degrades under it; the second
# is barely moved. Alpha is therefore scaled by how hard the *source* is, which
# is measurable on CPU before generation and needs no probe.
#
# Below the floor the source is already readable and gets no push at all. Above
# the ceiling it gets the full swept alpha. Between, it ramps linearly.
ADAPTIVE_FKGL_FLOOR = 6.0
ADAPTIVE_FKGL_CEIL = 16.0
# Alphas are snapped to this grid so that 32 items collapse into a handful of
# batches instead of 32 single-item generations.
ADAPTIVE_ALPHA_STEP = 0.125


# --- Science abstracts: the hard eval set -----------------------------------
# ASSET is single Wikipedia sentences a rewriter can carry almost intact. An
# abstract is several dependent clauses of domain vocabulary where dropping one
# is easy and invisible to a readability metric, which is the failure this
# project exists to catch. Ten fields, deliberately far apart, so a result is
# not an artifact of one discipline's house style.
SCIENCE_CATEGORIES = (
    "astro-ph.CO",      # cosmology
    "cond-mat.supr-con",  # superconductivity
    "physics.chem-ph",  # chemical physics
    "q-bio.NC",         # neuroscience
    "q-bio.PE",         # populations and evolution
    "cs.CL",            # computational linguistics
    "math.NT",          # number theory
    "econ.EM",          # econometrics
    "physics.geo-ph",   # geophysics
    "eess.SP",          # signal processing
)

# Long enough to contain the multi-clause structure we want to stress, short
# enough that prompt plus rewrite stays inside MAX_NEW_TOKENS.
SCIENCE_ABSTRACT_WORDS = (60, 140)

# Candidates pulled per category before the length filter picks the first fit.
SCIENCE_FETCH_PER_CATEGORY = 25

# Abstracts need a bigger generation budget than the 320 that covers an ASSET
# sentence. Structural simplification *expands* token count -- splitting one
# sentence into three repeats the subject each time -- and the longest abstract
# in the set is 247 source tokens, so a 1.8x expansion needs 444. Truncating
# there would emit an unterminated sentence, which the degeneracy rules would
# correctly flag and which would then be reported as the arm failing rather
# than the harness running out of room.
SCIENCE_MAX_NEW_TOKENS = 512


# Which demonstration corpus serves which eval set. Demonstrations must match the
# *shape* of the target: sentence-length demos in front of a paragraph-length
# source teach the model that the answer is one sentence, and it will discard the
# rest of the input to produce one.
SHOT_SOURCE = {"asset": "asset", "passages": "onestop", "science": "onestop"}

# OneStopEnglish pairs are kept only when the easier side is a real rewrite.
# Some aligned pairs differ by a single edit that changes a fact -- one differs
# only in a date -- and demonstrating that teaches drift.
SHOT_COMPRESSION_RANGE = (0.55, 0.95)
SHOT_MIN_FKGL_DROP = 1.0

# Abstracts kept per field. Ten items total is enough to falsify a strong claim
# -- an arm that collapses on abstracts is obvious at n=10 -- but not to confirm
# a subtle one, since a paired interval on ten items is wide. Raising this is the
# honest response to a borderline result, rather than reading the ten harder.
# The set stays interleaved by field, so --n 20 is two per field and not the
# first two fields.
SCIENCE_PER_CATEGORY = 1
