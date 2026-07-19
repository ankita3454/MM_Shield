"""Phase 3: handcrafted (non-CNN) feature extraction for candidate crops.

Deliberately NOT a reuse of typographic/features/typography_features.py:
that module scores OCR text regions relative to their document siblings
(self-relative outlier z-scores, because a typographic attack is defined by
looking anomalous next to normal text on the same page). A candidate crop
here has no siblings to compare against and isn't text - it's a proposed
image region that may or may not contain a visual adversarial patch. So
every feature below is an absolute, patch-local descriptor computed purely
from pixels inside the crop itself: geometry, edge/gradient statistics,
texture, color/HSV statistics, and connected-component/contour shape - the
kind of signal that separates a synthetic patch (checkerboard, noise block,
QR-like grid, warning block, geometric logo - see attack_generator.py) from
ordinary document content (paragraphs, tables, logos) at the crop level.
"""

import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from adversarial.config import OUTPUTS_DIR
from adversarial.dataset.dataset_builder import CANDIDATE_LABELS_PATH
from typographic.config import DATASETS_DIR

HANDCRAFTED_FEATURES_PATH = OUTPUTS_DIR / "handcrafted_features.csv"

RAW_FEATURE_NAMES = [
    "width",
    "height",
    "aspect_ratio",
    "area_px",
    "edge_density",
    "gradient_mean",
    "gradient_std",
    "gradient_max",
    "laplacian_variance",
    "entropy",
    "r_mean", "g_mean", "b_mean",
    "r_std", "g_std", "b_std",
    "hue_mean", "hue_std",
    "saturation_mean", "saturation_std",
    "value_mean", "value_std",
    "cc_count",
    "cc_area_frac_mean",
    "largest_contour_area_frac",
    "largest_contour_perimeter_norm",
    "compactness",
]

_EPS = 1e-6


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _connected_component_stats(gray: np.ndarray, page_area: float) -> tuple:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:  # only background
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background label 0
    cc_count = num_labels - 1
    cc_area_frac_mean = float(np.mean(areas) / page_area)
    return cc_count, cc_area_frac_mean


def _largest_contour_stats(gray: np.ndarray, page_area: float) -> tuple:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    area_frac = float(area / page_area)
    perimeter_norm = float(perimeter / (2 * math.sqrt(page_area) + _EPS))
    compactness = float(4 * math.pi * area / (perimeter ** 2 + _EPS)) if perimeter > 0 else 0.0
    return area_frac, perimeter_norm, compactness


def extract_crop_handcrafted_features(image: Image.Image) -> dict:
    """Absolute, patch-local feature dict for a single candidate crop."""
    rgb = np.array(image.convert("RGB"), dtype=np.float64)
    gray = np.array(image.convert("L"))
    hsv = np.array(image.convert("HSV"), dtype=np.float64)

    height, width = gray.shape
    page_area = max(width * height, 1)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges) / 255.0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    cc_count, cc_area_frac_mean = _connected_component_stats(gray, page_area)
    contour_area_frac, contour_perimeter_norm, compactness = _largest_contour_stats(gray, page_area)

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / max(height, _EPS)),
        "area_px": float(page_area),
        "edge_density": edge_density,
        "gradient_mean": float(gradient_magnitude.mean()),
        "gradient_std": float(gradient_magnitude.std()),
        "gradient_max": float(gradient_magnitude.max()),
        "laplacian_variance": float(laplacian.var()),
        "entropy": _entropy(gray),
        "r_mean": float(rgb[:, :, 0].mean()), "g_mean": float(rgb[:, :, 1].mean()), "b_mean": float(rgb[:, :, 2].mean()),
        "r_std": float(rgb[:, :, 0].std()), "g_std": float(rgb[:, :, 1].std()), "b_std": float(rgb[:, :, 2].std()),
        "hue_mean": float(hsv[:, :, 0].mean()), "hue_std": float(hsv[:, :, 0].std()),
        "saturation_mean": float(hsv[:, :, 1].mean()), "saturation_std": float(hsv[:, :, 1].std()),
        "value_mean": float(hsv[:, :, 2].mean()), "value_std": float(hsv[:, :, 2].std()),
        "cc_count": float(cc_count),
        "cc_area_frac_mean": cc_area_frac_mean,
        "largest_contour_area_frac": contour_area_frac,
        "largest_contour_perimeter_norm": contour_perimeter_norm,
        "compactness": compactness,
    }


def extract_handcrafted_features(force: bool = False) -> pd.DataFrame:
    if HANDCRAFTED_FEATURES_PATH.exists() and not force:
        print(f"{HANDCRAFTED_FEATURES_PATH} already exists - not regenerating (pass force=True to override deliberately)")
        return pd.read_csv(HANDCRAFTED_FEATURES_PATH)

    df = pd.read_csv(CANDIDATE_LABELS_PATH)
    rows = []
    total = len(df)
    for i, candidate in df.iterrows():
        image = Image.open(DATASETS_DIR / candidate["crop_file"])
        features = extract_crop_handcrafted_features(image)
        row = {
            "candidate_id": candidate["candidate_id"],
            "source_image_id": candidate["source_image_id"],
            "dataset": candidate["dataset"],
            "label": candidate["label"],
            "source_type": candidate["source_type"],
        }
        row.update(features)
        rows.append(row)

        done = i + 1
        if done % 500 == 0 or done == total:
            print(f"processed {done}/{total} candidates")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(HANDCRAFTED_FEATURES_PATH, index=False)
    print(f"wrote {len(out_df)} rows -> {HANDCRAFTED_FEATURES_PATH}")
    return out_df


if __name__ == "__main__":
    extract_handcrafted_features()
