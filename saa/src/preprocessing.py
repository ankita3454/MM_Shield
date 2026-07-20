"""
Preprocessing: compute + cache residual images.

The noise residual (median-filter based, see noise.py) and the SRM residual
bank (see srm_features.py) are both re-derivable from the source image, but
recomputing them on every experiment (baseline run, feature-selection sweep,
improvement experiments) is wasteful and makes it easy to introduce subtle
inconsistencies between runs. This module computes both once per image and
caches them as .npy arrays under datasets/residuals/, keyed off the source
image's filename.

Layout written per source image `<stem>.<ext>`:
  datasets/residuals/<stem>__noise.npy   -> 2D float64 array (noise residual)
  datasets/residuals/<stem>__srm_h.npy   -> 2D float64 array
  datasets/residuals/<stem>__srm_v.npy   -> 2D float64 array
  datasets/residuals/<stem>__srm_d1.npy  -> 2D float64 array
  datasets/residuals/<stem>__srm_d2.npy  -> 2D float64 array
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

from noise import compute_noise_residual
from srm_features import compute_srm_residuals


def _load_grayscale(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float64)


def compute_and_cache_residuals(image_path: str, residuals_dir: str, overwrite: bool = False) -> dict:
    """
    Compute the noise + SRM residuals for one image and cache them as .npy
    files under `residuals_dir`. Returns a dict of {name: np.ndarray} for
    immediate in-memory use as well (so callers don't have to re-read disk).
    """
    os.makedirs(residuals_dir, exist_ok=True)
    stem = Path(image_path).stem
    out_paths = {
        "noise": os.path.join(residuals_dir, f"{stem}__noise.npy"),
        "srm_h": os.path.join(residuals_dir, f"{stem}__srm_h.npy"),
        "srm_v": os.path.join(residuals_dir, f"{stem}__srm_v.npy"),
        "srm_d1": os.path.join(residuals_dir, f"{stem}__srm_d1.npy"),
        "srm_d2": os.path.join(residuals_dir, f"{stem}__srm_d2.npy"),
    }

    if not overwrite and all(os.path.exists(p) for p in out_paths.values()):
        return {name: np.load(p) for name, p in out_paths.items()}

    gray = _load_grayscale(image_path)
    noise_residual = compute_noise_residual(gray)
    srm_residuals = compute_srm_residuals(gray)

    results = {
        "noise": noise_residual,
        "srm_h": srm_residuals["h"],
        "srm_v": srm_residuals["v"],
        "srm_d1": srm_residuals["d1"],
        "srm_d2": srm_residuals["d2"],
    }

    for name, arr in results.items():
        np.save(out_paths[name], arr)

    return results


def batch_precompute(image_paths, residuals_dir: str, overwrite: bool = False) -> None:
    """Precompute + cache residuals for a list of image paths."""
    for i, path in enumerate(image_paths):
        compute_and_cache_residuals(path, residuals_dir, overwrite=overwrite)
        if (i + 1) % 25 == 0:
            print(f"  precomputed residuals for {i + 1}/{len(image_paths)} images")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        compute_and_cache_residuals(sys.argv[1], sys.argv[2], overwrite=True)
        print("done")
