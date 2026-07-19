"""Phase: typography feature extraction.

Two levels of output, matching both the literal per-region request and the
locked self-relative-outlier aggregation strategy:

  - extract_all_region_features(): one raw 12-dim feature dict per OCR region.
  - aggregate_document_features(): collapses those into a fixed 12-dim
    document-level vector, where each dimension is the largest deviation
    (in MADs) of any single region from the document's own median for that
    property. This keeps an anomalous region's signal from being averaged
    away by dozens of normal ones, and is scale-invariant across documents.

Consumes only the unified region format {text, bbox, confidence,
rotation_degrees, page_width, page_height} produced by ocr.py - no
dataset-specific logic here.
"""

import math
import string

import cv2
import numpy as np
from PIL import Image

RAW_FEATURE_NAMES = [
    "height_norm",
    "width_norm",
    "aspect_ratio",
    "stroke_density",
    "luminance",
    "saturation",
    "rotation_degrees",
    "avg_char_width_norm",
    "capitalization_ratio",
    "punctuation_ratio",
    "local_density",
    "whitespace_margin_norm",
]

_EPS = 1e-6


def _crop(image: Image.Image, bbox):
    W, H = image.size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(math.ceil(x1)), W), min(int(math.ceil(y1)), H)
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def _stroke_density(crop: Image.Image) -> float:
    gray = np.array(crop.convert("L"))
    if gray.size == 0:
        return 0.0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu splits into two classes; treat the minority class as ink/text strokes.
    ink_fraction = min(np.mean(binary == 0), np.mean(binary == 255))
    return float(ink_fraction)


def _luminance(crop: Image.Image) -> float:
    gray = np.array(crop.convert("L"), dtype=np.float64)
    if gray.size == 0:
        return 0.0
    return float(gray.mean() / 255.0)


def _saturation(crop: Image.Image) -> float:
    hsv = np.array(crop.convert("HSV"), dtype=np.float64)
    if hsv.size == 0:
        return 0.0
    return float(hsv[:, :, 1].mean() / 255.0)


def _center(bbox):
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def extract_region_typography_features(image: Image.Image, region: dict, all_regions: list[dict]) -> dict:
    """Raw typography features for a single OCR region, given its siblings on the page."""
    bbox = region["bbox"]
    page_w, page_h = region["page_width"], region["page_height"]
    page_diag = math.hypot(page_w, page_h) + _EPS

    x0, y0, x1, y1 = bbox
    width, height = max(x1 - x0, _EPS), max(y1 - y0, _EPS)

    crop = _crop(image, bbox)
    if crop is None or crop.width == 0 or crop.height == 0:
        stroke_density = luminance = saturation = 0.0
    else:
        stroke_density = _stroke_density(crop)
        luminance = _luminance(crop)
        saturation = _saturation(crop)

    text = region.get("text", "")
    char_count = max(len(text), 1)
    alpha_chars = [c for c in text if c.isalpha()]
    punct_chars = [c for c in text if c in string.punctuation]

    capitalization_ratio = (sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)) if alpha_chars else 0.0
    punctuation_ratio = len(punct_chars) / char_count

    this_center = _center(bbox)
    other_centers = [_center(r["bbox"]) for r in all_regions if r is not region]
    neighbor_radius = 0.1 * page_diag
    if other_centers:
        distances = [math.hypot(cx - this_center[0], cy - this_center[1]) for cx, cy in other_centers]
        local_density = sum(1 for d in distances if d <= neighbor_radius) / len(other_centers)
        whitespace_margin_norm = min(distances) / page_diag
    else:
        local_density = 0.0
        whitespace_margin_norm = 1.0

    return {
        "height_norm": height / page_h,
        "width_norm": width / page_w,
        "aspect_ratio": width / height,
        "stroke_density": stroke_density,
        "luminance": luminance,
        "saturation": saturation,
        "rotation_degrees": region.get("rotation_degrees", 0.0),
        "avg_char_width_norm": (width / char_count) / page_w,
        "capitalization_ratio": capitalization_ratio,
        "punctuation_ratio": punctuation_ratio,
        "local_density": local_density,
        "whitespace_margin_norm": whitespace_margin_norm,
    }


def extract_all_region_features(image: Image.Image, regions: list[dict]) -> list[dict]:
    return [extract_region_typography_features(image, region, regions) for region in regions]


def aggregate_document_features(region_features: list[dict]) -> dict:
    """Collapse per-region raw features into 12 document-level self-relative
    outlier scores: log1p(max(|x - median| / MAD)) across regions, per feature."""
    if not region_features:
        return {name: 0.0 for name in RAW_FEATURE_NAMES}

    # When a document's regions are near-uniform for a given property (e.g.
    # rotation ~0 everywhere on an axis-aligned page), MAD collapses to ~0 and
    # dividing by a bare epsilon blows the z-score up to the millions. A hard
    # cap would fix the blowup but saturates: a "very", "extremely", and
    # "absurdly" unusual region would all clip to the same value, destroying
    # the ordering the classifier needs. log1p is monotonic and never
    # saturates, so it compresses heavy-tailed blowups into a sane range while
    # still preserving the relative ranking between different anomaly magnitudes.
    _MAD_EPS = 1e-6

    doc_features = {}
    for name in RAW_FEATURE_NAMES:
        values = np.array([rf[name] for rf in region_features], dtype=np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        z_scores = np.abs(values - median) / (mad + _MAD_EPS)
        doc_features[name] = float(np.log1p(np.max(z_scores)))
    return doc_features


def extract_typography_features(image: Image.Image, regions: list[dict]) -> dict:
    """Convenience entry point: raw per-region features -> 12-dim document vector."""
    region_features = extract_all_region_features(image, regions)
    return aggregate_document_features(region_features)
