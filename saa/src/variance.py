"""
Group 5: Variance / Histogram (5 features)
---------------------------------------------
local_variance_mean, local_variance_std, local_variance_max,
hist_skewness, hist_kurtosis

Local variance (computed over sliding windows) captures texture regularity;
LSB embedding tends to smooth out or perturb the natural variance landscape
of flat document regions. Histogram skewness/kurtosis capture higher-order
shape changes in the global intensity distribution that raw entropy misses.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import skew, kurtosis
from PIL import Image


def _load_grayscale(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.float64)
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return arr.astype(np.float64)


def local_variance_map(gray: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Sliding-window local variance via the standard E[X^2] - E[X]^2 trick,
    computed efficiently with uniform_filter instead of a Python loop.
    """
    mean = uniform_filter(gray, size=window)
    mean_sq = uniform_filter(gray ** 2, size=window)
    variance = mean_sq - mean ** 2
    return np.clip(variance, 0, None)  # guard tiny negative floating error


def extract_variance_features(image_path_or_array) -> np.ndarray:
    """
    Returns [local_variance_mean, local_variance_std, local_variance_max,
    hist_skewness, hist_kurtosis], in frozen order.
    """
    gray = _load_grayscale(image_path_or_array)

    var_map = local_variance_map(gray)
    lv_mean = float(np.mean(var_map))
    lv_std = float(np.std(var_map))
    lv_max = float(np.max(var_map))

    flat = gray.ravel()
    h_skew = float(skew(flat))
    h_kurt = float(kurtosis(flat))  # excess kurtosis (Fisher), matches scipy default

    return np.array([lv_mean, lv_std, lv_max, h_skew, h_kurt], dtype=np.float64)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_variance_features(sys.argv[1]))
