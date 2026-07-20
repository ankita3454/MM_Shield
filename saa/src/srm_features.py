"""
Group 8: SRM (3 features)
----------------------------
srm_diagonal_ratio, srm_entropy, srm_energy

Spatial Rich Model (SRM) residual features (Fridrich & Kodovsky, 2012),
computed with classical fixed high-pass kernels -- no learned weights
required, which matches the project's "no deep learning" constraint
(SRNet/YeNet/XuNet have no downloadable weights, and DiT was proven in
prior experiments to destroy the LSB signal via resizing).

Kernel bank (7 kernels; expanded from an original 4-kernel 2nd-order-only
bank after baseline validation showed near-zero signal on natural/photo
images like CORD, where marginal LSB statistics are structurally blind --
see FEATURE_SPEC.md "Known issues"):
  - h, v, d1, d2:  original 3x3 2nd-order residuals (horizontal, vertical,
                   both diagonals)
  - h1, v1:        3x3 1st-order residuals (simple local differences),
                   complementary to the 2nd-order kernels above
  - kv:            the 5x5 "KV" kernel (Kodovsky & Fridrich) -- the single
                   most discriminative classical high-pass filter in the
                   steganalysis literature, included because it captures
                   pixel *correlations* rather than marginal bit statistics,
                   which is exactly the signal that survives on natural
                   images even when the LSB plane already looks random.

The interface stays 3 output features (frozen contract, see extractor.py) --
this only enriches the underlying residual bank those 3 stats are computed
over.

  - srm_energy:          mean squared residual magnitude across the full bank
                          (overall "richness"/roughness of the noise floor)
  - srm_entropy:          Shannon entropy of the pooled, per-kernel-normalized
                          residual histogram (higher = more random/noise-like,
                          which is what embedding tends to push toward)
  - srm_diagonal_ratio:   energy in the diagonal-kernel residuals relative
                          to the horizontal+vertical residuals (still based
                          on the original 4 kernels only, so its meaning is
                          unchanged); embedding that is not edge-aware tends
                          to shift this ratio away from the natural-image
                          baseline

NOTE: Experiment 003 (see EXPERIMENTS.md) tried switching srm_energy and
srm_diagonal_ratio to a 90th-percentile-across-32x32-patches aggregate
instead of the whole-image mean below, on the theory that averaging over
the whole image washes out scattered embedding concentrated in a minority
of it. Result: overall accuracy -1.1pp, CORD -4.3pp, SROIE unchanged --
a real negative result, not adopted. This file reflects the reverted
Experiment 002 (whole-image mean) implementation; the patch-based version
and its frozen results are preserved in outputs/model_v3_rf_patchsrm_frozen.pkl
for reference, not as the active implementation.
"""
import numpy as np
from scipy.signal import convolve2d
from PIL import Image

# Classical 2nd-order SRM-style high-pass kernels (un-normalized "spam"
# style residual kernels), horizontal / vertical / diag1 / diag2.
_KERNEL_H = np.array([[0, 0, 0],
                       [1, -2, 1],
                       [0, 0, 0]], dtype=np.float64)

_KERNEL_V = np.array([[0, 1, 0],
                       [0, -2, 0],
                       [0, 1, 0]], dtype=np.float64)

_KERNEL_D1 = np.array([[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]], dtype=np.float64)

_KERNEL_D2 = np.array([[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]], dtype=np.float64)

# 1st-order simple-difference kernels (complementary to the 2nd-order ones
# above -- cheap to add, and first- and second-order residuals pick up
# different aspects of local correlation).
_KERNEL_H1 = np.array([[0, 0, 0],
                        [-1, 1, 0],
                        [0, 0, 0]], dtype=np.float64)

_KERNEL_V1 = np.array([[0, -1, 0],
                        [0, 1, 0],
                        [0, 0, 0]], dtype=np.float64)

# The classic 5x5 "KV" (Kodovsky-Fridrich minmax) kernel -- the single
# strongest fixed high-pass filter used across the steganalysis literature,
# normalized by 12 per the standard definition.
_KERNEL_KV = np.array([
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1],
], dtype=np.float64) / 12.0


def _load_grayscale(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("L")
        return np.array(img, dtype=np.float64)
    arr = np.asarray(image_path_or_array)
    if arr.ndim == 3:
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return arr.astype(np.float64)


def compute_srm_residuals(gray: np.ndarray) -> dict:
    """Convolve with the full 7-kernel SRM bank, returning each residual map."""
    residuals = {
        "h": convolve2d(gray, _KERNEL_H, mode="same", boundary="symm"),
        "v": convolve2d(gray, _KERNEL_V, mode="same", boundary="symm"),
        "d1": convolve2d(gray, _KERNEL_D1, mode="same", boundary="symm"),
        "d2": convolve2d(gray, _KERNEL_D2, mode="same", boundary="symm"),
        "h1": convolve2d(gray, _KERNEL_H1, mode="same", boundary="symm"),
        "v1": convolve2d(gray, _KERNEL_V1, mode="same", boundary="symm"),
        "kv": convolve2d(gray, _KERNEL_KV, mode="same", boundary="symm"),
    }
    return residuals


def srm_energy(residuals: dict) -> float:
    """Mean squared residual value, averaged across the full kernel bank."""
    values = [np.mean(r ** 2) for r in residuals.values()]
    return float(np.mean(values))


def srm_entropy(residuals: dict, bins: int = 101, clip_std: float = 5.0) -> float:
    """
    Shannon entropy (base 2) of the pooled residual histogram, computed
    across the full kernel bank. Each kernel's residual map has a very
    different natural scale (the 5x5 KV kernel in particular), so each map
    is first z-scored (divided by its own std) before pooling -- otherwise
    the widest-range kernel would dominate the combined histogram and the
    others would just look like a spike near zero. Clipped to +-clip_std
    standard deviations and binned into `bins` bins.
    """
    normalized = []
    for r in residuals.values():
        std = np.std(r)
        if std > 0:
            normalized.append((r / std).ravel())
    if not normalized:
        return 0.0
    pooled = np.concatenate(normalized)
    pooled = np.clip(pooled, -clip_std, clip_std)
    hist, _ = np.histogram(pooled, bins=bins, range=(-clip_std, clip_std))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def srm_diagonal_ratio(residuals: dict) -> float:
    """
    Ratio of diagonal-kernel energy to axis-aligned (h+v) kernel energy.
    """
    diag_energy = np.mean(residuals["d1"] ** 2) + np.mean(residuals["d2"] ** 2)
    axis_energy = np.mean(residuals["h"] ** 2) + np.mean(residuals["v"] ** 2)
    if axis_energy == 0:
        return 0.0
    return float(diag_energy / axis_energy)


def extract_srm_features(image_path_or_array) -> np.ndarray:
    """Returns [srm_diagonal_ratio, srm_entropy, srm_energy], in frozen order."""
    gray = _load_grayscale(image_path_or_array)
    residuals = compute_srm_residuals(gray)
    return np.array(
        [srm_diagonal_ratio(residuals), srm_entropy(residuals), srm_energy(residuals)],
        dtype=np.float64,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_srm_features(sys.argv[1]))
