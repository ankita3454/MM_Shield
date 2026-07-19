"""Central configuration for the typographic module: paths, dataset sources, and fixed constants.

Every other module imports paths/constants from here rather than hardcoding them,
so the dataset location or sample size only ever needs to change in one place.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
OUTPUTS_DIR = REPO_ROOT / "typographic" / "outputs"

SAMPLED_IMAGES_PATH = DATASETS_DIR / "sampled_images.json"
ATTACK_METADATA_PATH = DATASETS_DIR / "attack_metadata.json"
ATTACK_TEMPLATES_PATH = REPO_ROOT / "typographic" / "dataset" / "attack_templates.json"

# --- Training dataset sources (Hugging Face Hub, locked — see MEMORY) ------
# Do not add dataset-specific branching anywhere outside dataset/download.py.
# Every module downstream of download.py consumes only the unified record format:
#   {"text": str, "bbox": [xmin, ymin, xmax, ymax], "page_width": int, "page_height": int, "dataset": str}

HF_DATASET_SOURCES = {
    "FUNSD": "nielsr/funsd",
    "CORD": "naver-clova-ix/cord-v2",
    "SROIE": "rth/sroie-2019-v2",
}

# FinInject is the external benchmark only — never used for training/fine-tuning.
TRAINING_DATASETS = ("FUNSD", "CORD", "SROIE")
BENCHMARK_DATASET = "FinInject"

# --- Sampling ----------------------------------------------------------------

SAMPLES_PER_DATASET = 50
RANDOM_SEED = 42

# --- Split -------------------------------------------------------------------

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
