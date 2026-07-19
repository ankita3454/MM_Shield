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
# Candidate generation MUST run on both clean and attacked pages - the old
# implementation's core flaw was only ever running it on attacked pages.
#   clean page    -> every candidate            -> negative
#   attacked page -> patch_coverage >= threshold -> positive
#   attacked page -> patch_coverage <  threshold -> hard_negative (label 0,
#                    tagged distinctly in metadata - kept, not discarded)
# patch_coverage = intersection_area / ground_truth_patch_area, NOT standard
# IoU - an oversized-but-correct proposal shouldn't be penalized for a large
# union.
#
# Region proposal is connected components on the thresholded heatmap, PLUS
# an explicit max-area cap - a hybrid reached after two empirical rounds:
# a pure sliding-window search (fixed square windows, scored by mean heatmap
# value) was tried first and performed worse (13% genuine recall on a 30-
# image test vs. 30% here) because a fixed square window's mean score gets
# diluted by non-patch background pixels whenever a patch is rotated
# (frequently diamond-shaped within its bbox) or doesn't match the window's
# size/aspect exactly - connected components trace whatever shape the actual
# signal forms, without that dilution. But connected components alone
# degenerate at low score thresholds into whole-page-covering blobs (up to
# ~90% of the page) that trivially "contain" the patch without meaningfully
# localizing it - MAX_CANDIDATE_AREA_FRACTION blocks that failure mode
# (0.20 is deliberately close to the largest a patch can plausibly be: even
# at attack_generator's max 35%-of-shorter-side scale, worst-case rotation-
# bbox expansion tops out around 18-19% of page area, so this cap doesn't
# exclude genuine patch-sized detections). MAX_CANDIDATES_PER_IMAGE is a cap,
# not a fixed count: a page with only one strong candidate keeps one.
MAX_CANDIDATES_PER_IMAGE = 12
CANDIDATE_SCORE_THRESHOLD = 0.3
MAX_CANDIDATE_AREA_FRACTION = 0.20
PATCH_COVERAGE_THRESHOLD = 0.5
