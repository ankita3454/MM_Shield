"""
Group 2: Noise Stats (3 features)
-----------------------------------
noise_mean, noise_std, noise_abs_mean

The "noise" here is the high-frequency residual left after removing the
image's smooth structure with a median filter. LSB and other steganographic
payloads live in exactly this residual, so its statistics are a core
steganalysis signal. This residual is also reused by srm_features.py's
sibling computation (see preprocessing.py for the cached version).
"""
import numpy as np
from scipy.ndimage import median_filter
from PIL import Image


def _load_grayscale(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.float64)
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return arr.astype(np.float64)


def compute_noise_residual(image_path_or_array, size: int = 3) -> np.ndarray:
    """
    Denoise with a median filter and return the residual (original - denoised).
    A 3x3 median filter removes shot/LSB-scale noise while preserving edges
    much better than a mean/Gaussian filter, which is what we want here:
    we want the residual to be dominated by fine-grained noise, not by
    legitimate edge structure.
    """
    gray = _load_grayscale(image_path_or_array)
    denoised = median_filter(gray, size=size)
    residual = gray - denoised
    return residual


def noise_mean(residual: np.ndarray) -> float:
    return float(np.mean(residual))


def noise_std(residual: np.ndarray) -> float:
    return float(np.std(residual))


def noise_abs_mean(residual: np.ndarray) -> float:
    return float(np.mean(np.abs(residual)))


def extract_noise_features(image_path_or_array) -> np.ndarray:
    """Returns [noise_mean, noise_std, noise_abs_mean], in frozen order."""
    residual = compute_noise_residual(image_path_or_array)
    return np.array(
        [noise_mean(residual), noise_std(residual), noise_abs_mean(residual)],
        dtype=np.float64,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_noise_features(sys.argv[1]))
