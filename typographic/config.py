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

DOCLAYNET_SAMPLED_PATH = DATASETS_DIR / "doclaynet_sampled_images.json"
DOCLAYNET_ATTACK_METADATA_PATH = DATASETS_DIR / "doclaynet_attack_metadata.json"

# --- Training dataset sources (Hugging Face Hub, locked — see MEMORY) ------
# Do not add dataset-specific branching anywhere outside dataset/download.py.
# Every module downstream of download.py consumes only the unified record format:
#   {"text": str, "bbox": [xmin, ymin, xmax, ymax], "page_width": int, "page_height": int, "dataset": str}

HF_DATASET_SOURCES = {
    "FUNSD": "nielsr/funsd",
    "CORD": "naver-clova-ix/cord-v2",
    "SROIE": "rth/sroie-2019-v2",
    "FigStep": "AngelAlita/FigStep",
    "DocLayNet": "docling-project/DocLayNet-v1.1",
}

# FigStep is an external zero-shot benchmark — never used for training/
# fine-tuning. It is 100% attack images (no clean counterpart class), so the
# only meaningful metric on it is detection rate (recall), not accuracy/
# precision/ROC-AUC. See typographic/training/benchmark.py.
TRAINING_DATASETS = ("FUNSD", "CORD", "SROIE")
BENCHMARK_DATASET = "FigStep"

# DocLayNet is a second external zero-shot benchmark, with attacks generated
# on it the same way as the training datasets (so it DOES have a clean
# counterpart class, unlike FigStep — full accuracy/precision/recall/F1/
# ROC-AUC apply). ~32GB full size, so it is stream-sampled rather than fully
# downloaded — see dataset/download.py:download_external_sample().
DOCLAYNET_SAMPLE_SIZE = 200
DOCLAYNET_SPLIT = "test"

# --- Sampling ----------------------------------------------------------------

SAMPLES_PER_DATASET = 50
RANDOM_SEED = 42

# --- Split -------------------------------------------------------------------

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
