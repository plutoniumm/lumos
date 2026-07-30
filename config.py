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
NLI_MODEL = "FacebookAI/roberta-large-mnli"

ACTIVE_MODEL = LLAMA
DTYPE = "bfloat16"
DEVICE = "mps"


# HF's client opens one connection per file; aria2c saturates the link instead.
ARIA2_CONNECTIONS_PER_FILE = 16
ARIA2_CONCURRENT_FILES = 4
ARIA2_MIN_SPLIT = "8M"

# Weights we want, and the ones we must not pull: meta-llama ships a duplicate
# 6 GB consolidated .pth under original/ that transformers never reads.
DOWNLOAD_ALLOW_SUFFIXES = (".safetensors", ".json", ".txt", ".model")
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

# The sweep only needs enough tokens to measure sentence length and word
# difficulty, not a complete long-form answer.
SWEEP_MAX_NEW_TOKENS = 96

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
