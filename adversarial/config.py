"""Central configuration for the adversarial patch module: paths, dataset
sources, and fixed constants.

Reuses the Typographic module's dataset sourcing/sampling wholesale rather
than duplicating it: FUNSD/CORD/SROIE/DocLayNet are the same HF-sourced,
already-downloaded, already-normalized datasets regardless of which attack
type is applied to them, and per an explicit decision this module reuses
Typographic's exact frozen 150-image clean sample and 200-page DocLayNet
sample rather than drawing independent ones (same seed/dataset/counts would
produce an identical result anyway, and re-sampling DocLayNet in particular
means a multi-hour stream+OCR run for no benefit).
"""

from pathlib import Path

from typographic.config import (
    DATASETS_DIR,
    DOCLAYNET_SAMPLE_SIZE,
    DOCLAYNET_SAMPLED_PATH,
    DOCLAYNET_SPLIT,
    HF_DATASET_SOURCES,
    RANDOM_SEED,
    REPO_ROOT,
    SAMPLED_IMAGES_PATH,
    TEST_FRACTION,
    TRAIN_FRACTION,
    TRAINING_DATASETS,
    VALIDATION_FRACTION,
)

# --- Paths -----------------------------------------------------------------

OUTPUTS_DIR = REPO_ROOT / "adversarial" / "outputs"

ADVERSARIAL_DATASETS_DIR = DATASETS_DIR / "adversarial"
ATTACK_METADATA_PATH = ADVERSARIAL_DATASETS_DIR / "attack_metadata.json"
ATTACKS_DIR = ADVERSARIAL_DATASETS_DIR / "attacks"
CANDIDATES_DIR = ADVERSARIAL_DATASETS_DIR / "candidates"

PATCH_LIBRARY_DIR = Path(__file__).resolve().parent / "dataset" / "patches"

DOCLAYNET_ATTACK_METADATA_PATH = ADVERSARIAL_DATASETS_DIR / "doclaynet_attack_metadata.json"
DOCLAYNET_ATTACKS_DIR = ADVERSARIAL_DATASETS_DIR / "doclaynet_attacks"
DOCLAYNET_CANDIDATES_DIR = ADVERSARIAL_DATASETS_DIR / "doclaynet_candidates"

# --- Dataset sources (reused from Typographic — see MEMORY) -----------------
# Training: FUNSD, CORD, SROIE. Never train on anything else.
# External benchmark: DocLayNet only (layout generalization). APRICOT and
# ImageNet-Patch were both confirmed NOT available on Hugging Face under any
# searchable name (2026-07-19) - do not re-search for them without new
# information; patch-generalization benchmarking is deferred until a real
# HF-hosted patch benchmark surfaces.

BENCHMARK_DATASET = "DocLayNet"

# --- Sampling (reused from Typographic's frozen sample) ---------------------

SAMPLES_PER_DATASET = 50

# --- Candidate generation -----------------------------------------------------
# Candidate generation MUST run on both clean and attacked pages (clean ->
# negative samples, attacked -> positive samples) - the old implementation's
# core flaw was only ever running it on attacked pages. See MEMORY.
CANDIDATES_PER_IMAGE = 3
