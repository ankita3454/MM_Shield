"""
Group 6: Edge Stats (2 features)
------------------------------------
edge_density, edge_mean_strength

Sobel-gradient based edge statistics. Document images (forms, receipts,
invoices) have very structured, high-contrast edges from text and rules;
steganographic payloads embedded in flat background regions can subtly
shift this edge landscape, and this also serves as a sanity/texture
covariate alongside the LSB-specific features.
"""
import numpy as np
from skimage.filters import sobel
from PIL import Image


def _load_grayscale(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.float64) / 255.0
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    arr = arr.astype(np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def edge_map(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude map, normalized to [0, 1] input range."""
    return sobel(gray)


def edge_density(edges: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of pixels whose gradient magnitude exceeds `threshold`."""
    total = edges.size
    if total == 0:
        return 0.0
    return float(np.sum(edges > threshold) / total)


def edge_mean_strength(edges: np.ndarray) -> float:
    """Mean Sobel gradient magnitude across the whole image."""
    return float(np.mean(edges))


def extract_edge_features(image_path_or_array) -> np.ndarray:
    """Returns [edge_density, edge_mean_strength], in frozen order."""
    gray = _load_grayscale(image_path_or_array)
    edges = edge_map(gray)
    return np.array([edge_density(edges), edge_mean_strength(edges)], dtype=np.float64)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_edge_features(sys.argv[1]))
