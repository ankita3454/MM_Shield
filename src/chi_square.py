"""
Group 7: Chi-Square (3 features)
------------------------------------
chisq_max_pvalue, chisq_mean_pvalue, chisq_fraction_high

Classic Pairs-of-Values (PoV) chi-square LSB steganalysis test
(Westfeld & Pfitzmann, 2000). LSB embedding equalizes the frequencies
within each pair of values that differ only in their LSB (2k, 2k+1),
which the chi-square goodness-of-fit test is sensitive to. We slide the
test over successive prefixes of the image (in raster order) to get a
distribution of p-values rather than a single global one, since scattered/
partial payloads only equalize pair statistics in the embedded region.

NOTE: Experiment 005 (see EXPERIMENTS.md) tried replacing this growing-
cumulative-prefix windowing with fixed-size, non-overlapping 10000-pixel
blocks, on the theory that mixing wildly different sample sizes into one
p-value distribution was adding noise. Result: chi-square's per-feature AUC
collapsed from 0.674 (SROIE) to 0.506 (essentially random) -- a clear,
uniform failure across all three datasets, not just SROIE. Rejected and
reverted. This file reflects the original Experiment 002 (growing-prefix)
implementation. The fixed-window version and its frozen (worse) results are
preserved in outputs/model_exp005_rf_chisq_frozen.pkl for reference, not as
the active implementation.
"""
import numpy as np
from scipy.stats import chisquare
from PIL import Image


def _load_grayscale_uint8(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.uint8)
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def _pov_chisquare_pvalue(values: np.ndarray) -> float:
    """
    Chi-square PoV test on a flat array of pixel values.
    Pairs are (0,1), (2,3), ..., (254,255). Under the H1 (LSB-embedded)
    hypothesis, the two frequencies in each pair should be equal, so we
    compare observed counts to an "equalized" expected distribution built
    from the pair averages.
    """
    hist = np.bincount(values, minlength=256).astype(np.float64)
    observed = hist[0::2]  # counts for even values 2k
    observed_odd = hist[1::2]  # counts for odd values 2k+1
    pair_sum = observed + observed_odd
    expected = pair_sum / 2.0

    # drop pairs with no support to keep the test well-defined
    mask = pair_sum > 0
    obs = observed[mask]
    exp = expected[mask]
    if obs.size < 2:
        return 1.0

    # normalize observed sum to match expected sum exactly (chisquare requires this)
    obs_scaled = obs  # observed counts for the "2k" bin only, tested against exp
    total_obs = np.sum(obs_scaled)
    total_exp = np.sum(exp)
    if total_exp == 0:
        return 1.0
    exp_scaled = exp * (total_obs / total_exp)

    try:
        _, pvalue = chisquare(f_obs=obs_scaled, f_exp=exp_scaled)
    except ValueError:
        return 1.0
    if np.isnan(pvalue):
        return 1.0
    return float(pvalue)


def sliding_pov_pvalues(image_path_or_array, n_windows: int = 20, min_window_frac: float = 0.05) -> np.ndarray:
    """
    Compute PoV chi-square p-values over `n_windows` growing prefixes of the
    flattened (raster-order) pixel stream, from min_window_frac of the image
    up to the full image. High p-values (fail to reject H0 of "no embedding
    equalization detected... actually here high p-value under our expected
    construction indicates values ARE close to equalized, i.e. more
    stego-like) are the signal of interest.
    """
    gray = _load_grayscale_uint8(image_path_or_array)
    flat = gray.ravel()
    n = flat.size

    start = max(int(n * min_window_frac), 64)
    checkpoints = np.linspace(start, n, n_windows, dtype=int)

    pvalues = []
    for end in checkpoints:
        window = flat[:end]
        pvalues.append(_pov_chisquare_pvalue(window))
    return np.array(pvalues, dtype=np.float64)


def chisq_max_pvalue(pvalues: np.ndarray) -> float:
    return float(np.max(pvalues)) if pvalues.size else 0.0


def chisq_mean_pvalue(pvalues: np.ndarray) -> float:
    return float(np.mean(pvalues)) if pvalues.size else 0.0


def chisq_fraction_high(pvalues: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of windows whose p-value exceeds `threshold`."""
    if pvalues.size == 0:
        return 0.0
    return float(np.sum(pvalues > threshold) / pvalues.size)


def extract_chisq_features(image_path_or_array) -> np.ndarray:
    """Returns [chisq_max_pvalue, chisq_mean_pvalue, chisq_fraction_high], in frozen order."""
    pvalues = sliding_pov_pvalues(image_path_or_array)
    return np.array(
        [chisq_max_pvalue(pvalues), chisq_mean_pvalue(pvalues), chisq_fraction_high(pvalues)],
        dtype=np.float64,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_chisq_features(sys.argv[1]))
