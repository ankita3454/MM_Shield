"""Phase 2: candidate region generator.

Fuses five unsupervised signal maps - edge density, local contrast, texture
(Laplacian) variance, gradient magnitude, and spectral-residual saliency -
into one threat heatmap, thresholds it, extracts connected components, and
keeps up to MAX_CANDIDATES_PER_IMAGE regions by score (a cap, not a fixed
count - a page with only one strong candidate keeps one), after dropping any
component whose area exceeds MAX_CANDIDATE_AREA_FRACTION of the page.

Region-proposal method reached after two empirical rounds (see MEMORY / the
comment in config.py for full detail): a pure multi-scale sliding-window
search (fixed square windows scored by mean heatmap value) was tried and
performed worse (13% genuine recall on a 30-image test vs. 30% here),
because a fixed window's mean score gets diluted by non-patch background
pixels whenever a patch is rotated (often diamond-shaped within its bbox)
or doesn't match the window's size/aspect exactly. Connected components
trace whatever shape the actual signal forms without that dilution - but
need an explicit area cap, since without one they degenerate at low score
thresholds into whole-page-covering blobs that trivially "contain" the
patch without meaningfully localizing it.

THE key fix (see MEMORY - this is the module's central methodological
change from the old reference implementation): candidate generation runs on
BOTH clean and attacked pages, not attacked pages only.

    clean page    -> every candidate                         -> negative
    attacked page -> patch_coverage >= PATCH_COVERAGE_THRESHOLD -> positive
    attacked page -> patch_coverage <  PATCH_COVERAGE_THRESHOLD -> hard_negative

patch_coverage = intersection_area / ground_truth_patch_area, deliberately
NOT standard IoU, so an oversized-but-correct proposal that fully contains
the patch isn't penalized for a large union. hard_negative candidates get
label=0 same as clean-page negatives for the classifier, but are tagged
distinctly (never discarded) since a visually complex region from a page
that has a patch elsewhere is valuable precision signal.
"""

import cv2
import numpy as np
from PIL import Image

from adversarial.config import (
    CANDIDATE_SCORE_THRESHOLD,
    MAX_CANDIDATE_AREA_FRACTION,
    MAX_CANDIDATES_PER_IMAGE,
    PATCH_COVERAGE_THRESHOLD,
)

_WORKING_SIZE = 512  # downsample for fast heatmap computation, upsample result back
_MIN_COMPONENT_AREA = 16


def _resize_for_processing(gray: np.ndarray):
    h, w = gray.shape
    scale = _WORKING_SIZE / max(h, w)
    if scale < 1:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small, scale = gray, 1.0
    return small, scale


def _normalize(map_: np.ndarray) -> np.ndarray:
    lo, hi = float(map_.min()), float(map_.max())
    if hi - lo < 1e-6:
        return np.zeros_like(map_, dtype=np.float32)
    return ((map_ - lo) / (hi - lo)).astype(np.float32)


def _edge_density(gray, ksize=15):
    edges = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
    return cv2.boxFilter(edges, -1, (ksize, ksize))


def _local_contrast(gray, ksize=15):
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, -1, (ksize, ksize))
    mean_sq = cv2.boxFilter(gray_f**2, -1, (ksize, ksize))
    variance = np.clip(mean_sq - mean**2, 0, None)
    return np.sqrt(variance)


def _texture_variance(gray, ksize=15):
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return cv2.boxFilter(laplacian**2, -1, (ksize, ksize))


def _gradient_magnitude(gray, ksize=15):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return cv2.boxFilter(magnitude, -1, (ksize, ksize))


def _saliency(gray):
    saliency_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency_obj.computeSaliency(gray)
    if not success:
        return np.zeros_like(gray, dtype=np.float32)
    return saliency_map.astype(np.float32)


def _compute_heatmap(image: Image.Image) -> np.ndarray:
    gray_full = np.array(image.convert("L"))
    small, _ = _resize_for_processing(gray_full)

    signals = [
        _normalize(_edge_density(small)),
        _normalize(_local_contrast(small)),
        _normalize(_texture_variance(small)),
        _normalize(_gradient_magnitude(small)),
        _normalize(_saliency(small)),
    ]
    fused = np.mean(signals, axis=0)

    h, w = gray_full.shape
    return cv2.resize(fused, (w, h), interpolation=cv2.INTER_LINEAR)


def propose_candidates(image: Image.Image) -> list[dict]:
    """Up to MAX_CANDIDATES_PER_IMAGE candidate {"bbox", "score"} dicts,
    ranked by score, excluding any connected component covering more than
    MAX_CANDIDATE_AREA_FRACTION of the page. Works on any image regardless
    of clean/attacked - labeling happens downstream in
    generate_candidates_for_{clean,attacked}."""
    heatmap = _compute_heatmap(image)
    H, W = heatmap.shape
    page_area = H * W
    binary = (heatmap >= CANDIDATE_SCORE_THRESHOLD).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    for label in range(1, num_labels):  # label 0 is background
        x, y, w, h, area = stats[label]
        if area < _MIN_COMPONENT_AREA or (w * h) / page_area > MAX_CANDIDATE_AREA_FRACTION:
            continue
        region_score = float(heatmap[labels == label].mean())
        candidates.append({"bbox": [int(x), int(y), int(x + w), int(y + h)], "score": region_score})

    candidates.sort(key=lambda c: -c["score"])
    return candidates[:MAX_CANDIDATES_PER_IMAGE]


def _patch_coverage(candidate_bbox, patch_bbox) -> float:
    cx0, cy0, cx1, cy1 = candidate_bbox
    px0, py0, px1, py1 = patch_bbox
    ix0, iy0 = max(cx0, px0), max(cy0, py0)
    ix1, iy1 = min(cx1, px1), min(cy1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    patch_area = max((px1 - px0) * (py1 - py0), 1)
    return intersection / patch_area


def generate_candidates_for_clean(image: Image.Image) -> list[dict]:
    candidates = propose_candidates(image)
    for c in candidates:
        c["source_type"] = "negative"
        c["label"] = 0
    return candidates


def generate_candidates_for_attacked(image: Image.Image, patch_bbox: list) -> list[dict]:
    candidates = propose_candidates(image)
    for c in candidates:
        coverage = _patch_coverage(c["bbox"], patch_bbox)
        c["patch_coverage"] = coverage
        if coverage >= PATCH_COVERAGE_THRESHOLD:
            c["source_type"] = "positive"
            c["label"] = 1
        else:
            c["source_type"] = "hard_negative"
            c["label"] = 0
    return candidates
