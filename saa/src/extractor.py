"""
SAA public interface: stego_analyzer(image_path) -> np.ndarray[21]

This is the frozen contract the rest of MMShield (AATFN fusion network,
Keertana's typography/adversarial-patch modules) is written against. Do not
reorder or resize the output vector without updating FEATURE_SPEC.md and
every downstream consumer.

Frozen 21-feature order:
  0  entropy_manual                  (Group 1: Entropy)
  1  noise_mean                      (Group 2: Noise Stats)
  2  noise_std
  3  noise_abs_mean
  4  lsb_ratio                       (Group 3: LSB)
  5  lsb_entropy
  6  total_frequency_energy          (Group 4: Frequency)
  7  high_freq_ratio
  8  local_variance_mean             (Group 5: Variance/Histogram)
  9  local_variance_std
  10 local_variance_max
  11 hist_skewness
  12 hist_kurtosis
  13 edge_density                    (Group 6: Edge Stats)
  14 edge_mean_strength
  15 chisq_max_pvalue                (Group 7: Chi-Square)
  16 chisq_mean_pvalue
  17 chisq_fraction_high
  18 srm_diagonal_ratio              (Group 8: SRM)
  19 srm_entropy
  20 srm_energy
"""
import numpy as np

from entropy import entropy_manual
from noise import compute_noise_residual, noise_mean, noise_std, noise_abs_mean
from lsb import get_lsb_plane, lsb_ratio, lsb_entropy
from frequency import extract_frequency_features
from variance import extract_variance_features
from edge import extract_edge_features
from chi_square import extract_chisq_features
from srm_features import compute_srm_residuals, srm_diagonal_ratio, srm_entropy, srm_energy
from PIL import Image

FEATURE_NAMES = [
    "entropy_manual",
    "noise_mean", "noise_std", "noise_abs_mean",
    "lsb_ratio", "lsb_entropy",
    "total_frequency_energy", "high_freq_ratio",
    "local_variance_mean", "local_variance_std", "local_variance_max",
    "hist_skewness", "hist_kurtosis",
    "edge_density", "edge_mean_strength",
    "chisq_max_pvalue", "chisq_mean_pvalue", "chisq_fraction_high",
    "srm_diagonal_ratio", "srm_entropy", "srm_energy",
]

assert len(FEATURE_NAMES) == 21


def _load_grayscale(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float64)


def stego_analyzer(image_path: str) -> np.ndarray:
    """
    Extract the frozen 21-element SAA feature vector for a single image.

    Parameters
    ----------
    image_path : str
        Path to an image file readable by PIL (png/jpg/etc).

    Returns
    -------
    np.ndarray, shape (21,), dtype float64
    """
    gray = _load_grayscale(image_path)

    # Group 1: Entropy
    f_entropy = np.array([entropy_manual(gray)])

    # Group 2: Noise Stats
    noise_residual = compute_noise_residual(gray)
    f_noise = np.array([
        noise_mean(noise_residual),
        noise_std(noise_residual),
        noise_abs_mean(noise_residual),
    ])

    # Group 3: LSB
    lsb_plane = get_lsb_plane(gray)
    f_lsb = np.array([lsb_ratio(lsb_plane), lsb_entropy(lsb_plane)])

    # Group 4: Frequency
    f_freq = extract_frequency_features(gray)

    # Group 5: Variance/Histogram
    f_var = extract_variance_features(gray)

    # Group 6: Edge Stats
    f_edge = extract_edge_features(gray)

    # Group 7: Chi-Square
    f_chisq = extract_chisq_features(gray)

    # Group 8: SRM
    srm_residuals = compute_srm_residuals(gray)
    f_srm = np.array([
        srm_diagonal_ratio(srm_residuals),
        srm_entropy(srm_residuals),
        srm_energy(srm_residuals),
    ])

    vector = np.concatenate([f_entropy, f_noise, f_lsb, f_freq, f_var, f_edge, f_chisq, f_srm])
    vector = vector.astype(np.float64)

    assert vector.shape == (21,), f"expected 21 features, got {vector.shape}"
    return vector


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vec = stego_analyzer(sys.argv[1])
        for name, val in zip(FEATURE_NAMES, vec):
            print(f"{name:28s} {val:.6f}")
