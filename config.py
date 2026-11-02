"""Every tuning parameter in the project. Nothing here should import project code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
VECTORS_DIR = ROOT / "vectors"
RESULTS_DIR = ROOT / "results"


LLAMA_CANONICAL = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_MIRROR = "unsloth/Llama-3.2-3B-Instruct"
LLAMA = LLAMA_MIRROR

GEMMA = "google/gemma-3n-E4B-it"
GEMMA4 = "google/gemma-4-12B-it"

NLI_MODEL = "FacebookAI/roberta-large-mnli"

ACTIVE_MODEL = GEMMA4
DTYPE = "bfloat16"
DEVICE = "mps"


ARIA2_CONNECTIONS_PER_FILE = 16
ARIA2_CONCURRENT_FILES = 4
ARIA2_MIN_SPLIT = "8M"

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

BASE_TASK = "Rewrite the following text."

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
    for key, task in BASE_TASK_OVERRIDES.items():
        if repo_id == key or repo_id.endswith(key.split("/")[-1]):
            return task
    return BASE_TASK


CHUNK_TASK = (
    "Rewrite only the final sentence of the following text. "
    "Keep it consistent with the sentences before it."
)

INSTRUCTIONS = {
    "structural": (
        "Use short, simple sentences with one idea each. "
        "Keep every fact exactly as it is."
    ),
    "lexical": (
        "Replace difficult or technical words with common, everyday words. "
        "Keep the sentence structure the same."
    ),
    "simplify": (
        "Make it easy to read for a young student. "
        "Do not change the meaning and do not leave out any information."
    ),
}

PLACEBOS = {
    "structural": (
        "Keep the original sentence structure and sentence length exactly as they are."
    ),
    "lexical": ("Keep the original wording and vocabulary exactly as they are."),
    "simplify": ("Keep the text exactly as it is."),
}

DEFAULT_CONTRAST = "placebo"

DEFAULT_POSITIONS = "gen_only"

NORM_PRESERVE = True

ORTHOGONALIZE_AXES = True

CONTENT_ORTHOGONALIZE = True

DEFAULT_SITE_KIND = "resid_post"


SWEEP_ALPHAS = (-1.0, -0.5, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)

SWEEP_COARSE_ALPHAS = (0.5, 1.0)
SWEEP_COARSE_N_TEXTS = 12
SWEEP_TOP_K_LAYERS = 3
SWEEP_NLI_SUBSAMPLE = 12

VECTOR_N_TEXTS = 256


NLI_ENTAILMENT_MIN = 0.50

PPL_RATIO_MAX = 1.50
TARGET_FKGL = {"grade4": 4.0, "grade6": 6.0, "grade8": 8.0}
CALIBRATE_ALPHA_BOUNDS = (0.0, 2.0)
CALIBRATE_MAX_ITERS = 8


MAX_NEW_TOKENS = 320
DO_SAMPLE = False
SEED = 0

BATCH_SIZE = 8

SWEEP_MAX_NEW_TOKENS = MAX_NEW_TOKENS

PAIR_MAX_NEW_TOKENS = 128


CHUNK_CONTEXT_SENTENCES = 0


# --- LoRA -------------------------------------------------------------------

LORA_DIR = MODELS_DIR / "lora"

LORA_TRAIN_SPLIT = "valid"

LORA_REFS_PER_SOURCE = 2

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

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
LORA_GRAD_ACCUM = 4
LORA_WARMUP_FRAC = 0.03
LORA_MAX_SEQ_LEN = 320
LORA_DEV_FRAC = 0.05


# --- Few-shot ---------------------------------------------------------------

FEWSHOT_N = 5


# --- Decode-time vocabulary constraint --------------------------------------

EASY_WORD_PENALTY = 5.0

EASY_WORD_PENALTY_OVERRIDES: dict[str, float] = {"google/gemma-4-12B-it": 11.39}


def easy_word_penalty(repo_id: str | None = None) -> float:
    if repo_id is None:
        return EASY_WORD_PENALTY
    for key, value in EASY_WORD_PENALTY_OVERRIDES.items():
        if repo_id == key or repo_id.endswith(key.split("/")[-1]):
            return value
    return EASY_WORD_PENALTY


# --- Adaptive steering ------------------------------------------------------

ADAPTIVE_FKGL_FLOOR = 6.0
ADAPTIVE_FKGL_CEIL = 16.0
ADAPTIVE_ALPHA_STEP = 0.125


# --- Science abstracts: the hard eval set -----------------------------------

SCIENCE_CATEGORIES = (
    "astro-ph.CO",
    "cond-mat.supr-con",
    "physics.chem-ph",
    "q-bio.NC",
    "q-bio.PE",
    "cs.CL",
    "math.NT",
    "econ.EM",
    "physics.geo-ph",
    "eess.SP",
)

SCIENCE_ABSTRACT_WORDS = (60, 140)

SCIENCE_FETCH_PER_CATEGORY = 25

SCIENCE_MAX_NEW_TOKENS = 512


SHOT_SOURCE = {"asset": "asset", "passages": "onestop", "science": "onestop"}

SHOT_COMPRESSION_RANGE = (0.55, 0.95)
SHOT_MIN_FKGL_DROP = 1.0

SCIENCE_PER_CATEGORY = 1
