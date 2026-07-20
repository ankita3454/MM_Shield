"""
Group 1: Entropy (1 feature)
-----------------------------
entropy_manual: Shannon entropy of the grayscale pixel-intensity histogram,
computed manually (no scipy.stats.entropy) so the base and epsilon handling
are explicit and reproducible.
"""
import numpy as np
from PIL import Image


def _load_grayscale(image_path_or_array):
    if isinstance(image_path_or_array, (str,)):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.float64)
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        # simple luminosity conversion if a color array was passed directly
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return arr.astype(np.float64)


def entropy_manual(image_path_or_array) -> float:
    """
    Shannon entropy (base 2) of the 8-bit grayscale intensity histogram.

    H = -sum(p_i * log2(p_i)) for i in 0..255, p_i = count_i / N
    """
    gray = _load_grayscale(image_path_or_array)
    gray_uint8 = np.clip(np.round(gray), 0, 255).astype(np.uint8)

    hist = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0

    probs = hist / total
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    return float(entropy)


def extract_entropy_features(image_path_or_array) -> np.ndarray:
    """Returns the 1-element entropy feature vector, in frozen order."""
    return np.array([entropy_manual(image_path_or_array)], dtype=np.float64)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(entropy_manual(sys.argv[1]))
