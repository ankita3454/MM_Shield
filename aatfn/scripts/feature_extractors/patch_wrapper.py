"""Document-level pooling on top of the adversarial/patch module.

adversarial/ only has CANDIDATE-level features (27 handcrafted + 1280 CNN =
1307-dim per proposed region, up to MAX_CANDIDATES_PER_IMAGE=12 candidates
per page) -- there is no existing function anywhere in the repo that
collapses a page's candidates into one fixed-length document vector
(confirmed: adversarial/inference/ is an empty stub, and train.py/
evaluate.py both operate strictly at candidate-row level). This module adds
that missing step, purpose-built for AATFN fusion:

  1. propose up to 12 candidate regions per page (unsupervised heatmap,
     works on any image)
  2. extract the full 1307-dim feature vector per candidate (handcrafted +
     CNN, both from the existing adversarial/features/*.py code, unchanged)
  3. MAX-POOL across candidates, dim-by-dim, into one 1307-dim document
     vector -- standard multiple-instance-learning pooling: for each
     feature dimension, keep whichever candidate scored highest on it, on
     the premise that a genuine patch (if present) will dominate at least
     one candidate. A page with zero candidates yields all-zeros.
  4. append num_candidates as one extra diagnostic dimension -> 1308-dim
     total document-level patch vector.
"""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import setup_sys_path

setup_sys_path()
from adversarial.dataset import candidate_generator  # noqa: E402
from adversarial.features.cnn_features import CNN_EMBEDDING_DIM, embed_crops  # noqa: E402
from adversarial.features.handcrafted_features import (  # noqa: E402
    RAW_FEATURE_NAMES as HANDCRAFTED_FEATURE_NAMES,
    extract_crop_handcrafted_features,
)

CANDIDATE_DIM = len(HANDCRAFTED_FEATURE_NAMES) + CNN_EMBEDDING_DIM  # 27 + 1280 = 1307
PATCH_DIM = CANDIDATE_DIM + 1  # + num_candidates = 1308

PATCH_FEATURE_NAMES = (
    [f"patch_max_{n}" for n in HANDCRAFTED_FEATURE_NAMES]
    + [f"patch_max_cnn_{i}" for i in range(CNN_EMBEDDING_DIM)]
    + ["patch_num_candidates"]
)


def _candidate_vector(crop: Image.Image, crop_path: str) -> np.ndarray:
    handcrafted = extract_crop_handcrafted_features(crop)
    handcrafted_vec = np.array([handcrafted[n] for n in HANDCRAFTED_FEATURE_NAMES], dtype=np.float64)
    cnn_vec = np.array(embed_crops([crop_path])[0], dtype=np.float64)
    return np.concatenate([handcrafted_vec, cnn_vec])


def extract_patch_features(image_path: str) -> np.ndarray:
    """-> np.ndarray shape (1308,): max-pooled candidate features + count."""
    image = Image.open(image_path).convert("RGB")
    candidates = candidate_generator.propose_candidates(image)

    if not candidates:
        return np.zeros(PATCH_DIM, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        vectors = []
        for i, c in enumerate(candidates):
            crop = image.crop(tuple(c["bbox"]))
            if crop.width == 0 or crop.height == 0:
                continue
            crop_path = str(Path(tmpdir) / f"crop_{i}.png")
            crop.save(crop_path)
            vectors.append(_candidate_vector(crop, crop_path))

    if not vectors:
        return np.zeros(PATCH_DIM, dtype=np.float64)

    stacked = np.stack(vectors, axis=0)  # (num_candidates, 1307)
    pooled = stacked.max(axis=0)  # (1307,) -- MIL-style max pooling
    return np.concatenate([pooled, [float(len(candidates))]])  # (1308,)
