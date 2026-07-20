"""
Group 4: Frequency (2 features)
----------------------------------
total_frequency_energy, high_freq_ratio

Computed from the 2D FFT magnitude spectrum. LSB and scattered payloads
inject energy into high spatial frequencies that isn't explained by the
document's natural structure (text edges, table lines), so the ratio of
high- to total-frequency energy is a useful, resolution-independent signal.
"""
import numpy as np
from PIL import Image


def _load_grayscale(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.float64)
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return arr.astype(np.float64)


def _magnitude_spectrum(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    return np.abs(fshift)


def total_frequency_energy(magnitude: np.ndarray) -> float:
    """Sum of squared magnitudes across the whole spectrum (Parseval energy)."""
    return float(np.sum(magnitude.astype(np.float64) ** 2))


def high_freq_ratio(magnitude: np.ndarray, radius_frac: float = 0.25) -> float:
    """
    Fraction of total spectral energy lying outside a low-frequency disk
    centered on the DC component. radius_frac is the disk radius as a
    fraction of the smaller image dimension's half-width.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = radius_frac * min(cy, cx)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    energy = magnitude.astype(np.float64) ** 2
    total = np.sum(energy)
    if total == 0:
        return 0.0
    high_freq_energy = np.sum(energy[dist > radius])
    return float(high_freq_energy / total)


def extract_frequency_features(image_path_or_array) -> np.ndarray:
    """Returns [total_frequency_energy, high_freq_ratio], in frozen order."""
    gray = _load_grayscale(image_path_or_array)
    magnitude = _magnitude_spectrum(gray)
    return np.array(
        [total_frequency_energy(magnitude), high_freq_ratio(magnitude)],
        dtype=np.float64,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_frequency_features(sys.argv[1]))
