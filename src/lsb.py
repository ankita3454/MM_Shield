"""
Group 3: LSB (2 features)
---------------------------
lsb_ratio, lsb_entropy

Classic LSB-plane statistics. A clean image's LSB plane looks close to
random noise already (~50/50 split, high entropy), which is exactly why
naive "does the LSB plane look random" detectors fail on natural images.
These two features are kept as weak/contributing signals for the fusion
network rather than standalone detectors -- chi_square.py carries the
stronger, structure-aware LSB signal (pairs-of-values test).
"""
import numpy as np
from PIL import Image


def _load_grayscale_uint8(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.uint8)
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def get_lsb_plane(image_path_or_array) -> np.ndarray:
    gray = _load_grayscale_uint8(image_path_or_array)
    return gray & 1


def lsb_ratio(lsb_plane: np.ndarray) -> float:
    """Fraction of LSB bits equal to 1. A random/embedded plane sits near 0.5."""
    total = lsb_plane.size
    if total == 0:
        return 0.0
    return float(np.sum(lsb_plane) / total)


def lsb_entropy(lsb_plane: np.ndarray) -> float:
    """Binary Shannon entropy (base 2) of the LSB plane's bit distribution."""
    p1 = lsb_ratio(lsb_plane)
    p0 = 1.0 - p1
    entropy = 0.0
    for p in (p0, p1):
        if p > 0:
            entropy -= p * np.log2(p)
    return float(entropy)


def extract_lsb_features(image_path_or_array) -> np.ndarray:
    """Returns [lsb_ratio, lsb_entropy], in frozen order."""
    plane = get_lsb_plane(image_path_or_array)
    return np.array([lsb_ratio(plane), lsb_entropy(plane)], dtype=np.float64)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_lsb_features(sys.argv[1]))
