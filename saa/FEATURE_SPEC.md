# SAA Feature Specification

Steganographic Analysis (SAA) module for MMShield. Owner: ankita (SAA + AATFN fusion, current scope SAA only). Teammate Keertana owns typography + adversarial patch.

## Interface (frozen)

```python
stego_analyzer(image_path: str) -> np.ndarray  # shape (21,), dtype float64
```

Defined in `src/extractor.py`. Do not reorder, add, or remove elements without updating this doc and every downstream consumer (AATFN fusion network).

## Constraints

- No deep learning. SRNet/YeNet/XuNet have no downloadable pretrained weights; a ViT/DiT-based approach was tested in 2 prior experiments and proven to destroy the LSB signal via internal resizing. Classical signal-processing + SRM (fixed kernels, no training) only.
- `StandardScaler` is mandatory before any classifier — feature magnitudes span orders of magnitude (e.g. `total_frequency_energy` ~1e14 vs `lsb_ratio` ~0.3).

## 21-Feature Vector (frozen order)

| Index | Feature | Group | Module |
|---|---|---|---|
| 0 | `entropy_manual` | Entropy (1) | `entropy.py` |
| 1 | `noise_mean` | Noise Stats (3) | `noise.py` |
| 2 | `noise_std` | Noise Stats | `noise.py` |
| 3 | `noise_abs_mean` | Noise Stats | `noise.py` |
| 4 | `lsb_ratio` | LSB (2) | `lsb.py` |
| 5 | `lsb_entropy` | LSB | `lsb.py` |
| 6 | `total_frequency_energy` | Frequency (2) | `frequency.py` |
| 7 | `high_freq_ratio` | Frequency | `frequency.py` |
| 8 | `local_variance_mean` | Variance/Histogram (5) | `variance.py` |
| 9 | `local_variance_std` | Variance/Histogram | `variance.py` |
| 10 | `local_variance_max` | Variance/Histogram | `variance.py` |
| 11 | `hist_skewness` | Variance/Histogram | `variance.py` |
| 12 | `hist_kurtosis` | Variance/Histogram | `variance.py` |
| 13 | `edge_density` | Edge Stats (2) | `edge.py` |
| 14 | `edge_mean_strength` | Edge Stats | `edge.py` |
| 15 | `chisq_max_pvalue` | Chi-Square (3) | `chi_square.py` |
| 16 | `chisq_mean_pvalue` | Chi-Square | `chi_square.py` |
| 17 | `chisq_fraction_high` | Chi-Square | `chi_square.py` |
| 18 | `srm_diagonal_ratio` | SRM (3) | `srm_features.py` |
| 19 | `srm_entropy` | SRM | `srm_features.py` |
| 20 | `srm_energy` | SRM | `srm_features.py` |

### Method notes

- **Entropy**: manual Shannon entropy (base 2) of the 8-bit grayscale histogram.
- **Noise Stats**: residual = original − 3×3 median-filtered image; mean/std/mean(abs) of that residual.
- **LSB**: ratio of 1-bits and binary Shannon entropy of the LSB bit-plane. Weak/contributing signal on their own — natural images already have near-random LSB planes.
- **Frequency**: 2D FFT magnitude spectrum; total squared-magnitude energy, and the fraction of that energy outside a low-frequency disk (`radius_frac=0.25` of half-width).
- **Variance/Histogram**: sliding-window (5×5) local variance map (mean/std/max), plus skewness and excess kurtosis of the global intensity histogram.
- **Edge Stats**: Sobel gradient magnitude map; density of pixels above a 0.1 threshold, and mean gradient strength.
- **Chi-Square**: classic Pairs-of-Values LSB test (Westfeld & Pfitzmann 2000), computed over 20 growing raster-order prefixes of the image to get a p-value distribution rather than one global value. `chisq_fraction_high` = fraction of windows with p > 0.5.
- **SRM** (v2, updated 2026-07-19): 7-kernel classical residual bank — the original 4 (horizontal, vertical, 2 diagonals, all 2nd-order) plus 2 first-order simple-difference kernels and the classic 5×5 KV kernel (Kodovsky-Fridrich minmax), the single strongest fixed high-pass filter in the steganalysis literature. Still no learned weights. `srm_energy` = mean squared residual across the full 7-kernel bank; `srm_entropy` = Shannon entropy of the pooled residual histogram, each kernel's residual z-scored before pooling (since the KV kernel's natural scale differs a lot from the 3×3 kernels) and clipped to ±5 std devs, 101 bins; `srm_diagonal_ratio` = diagonal-kernel energy / axis-aligned-kernel energy, still based on only the original 4 kernels so its meaning is unchanged. See `src/srm_features.py` docstring for the full rationale.

## Known issues / open questions

- **Chi-square saturation**: on a synthetic smoke test, `chisq_*` p-values collapsed to 0.0 for both clean and stego images — likely float64 underflow from the chi-square statistic at large N (full-image pixel counts). May need per-block windowing at fixed small N, or a log-p transform, rather than growing-prefix windows over the whole image. Still not diagnosed on real data — chi-square's low RF feature importance (see Baseline validation results v2) is consistent with this being a real, not just synthetic, issue.
- **CORD is structurally hard for marginal-statistic features, confirmed on real data**: CORD's clean images already have an LSB plane at ~49.3% ones (real photos have enough natural sensor/JPEG noise that the LSB plane looks random *before* any embedding), vs. FUNSD's clean images at ~95.1% ones (mostly flat white background, so `11111111` dominates). This means `lsb_ratio`/`lsb_entropy`-style features are structurally blind on CORD — embedding more random bits onto an already-random plane doesn't produce a detectable shift, so higher payload density won't fix this specific failure mode. Correlation-based features (SRM, chi-square) are the only ones with a chance of catching it, which is why the v2 SRM expansion (below) helped CORD specifically.

## Dataset (v2)

- 150 clean + 150 stego: 50 images each from FUNSD, CORD, SROIE.
- HF dataset ids used by `src/download_datasets.py`: `nielsr/funsd-layoutlmv3` (FUNSD), `naver-clova-ix/cord-v2` (CORD), `Voxel51/scanned_receipts` (SROIE — the original `darentang/sroie` has no portable image source, see script comments). Confirmed working against real HF downloads on 2026-07-19: 50/50 FUNSD, 50/50 CORD, 50/50 SROIE.
- Stego images: 15% scattered LSB payload (`src/embed_lsb_scattered.py`), matching literature-realistic sparse-payload density.

## Baseline validation results

**Canonical experiment ledger is `EXPERIMENTS.md`** — full config, confusion matrices, precision/recall/F1, feature importances, and reproduce commands for every numbered experiment (frozen models saved to `outputs/model_*.pkl`). The summary below is kept for quick reference but may lag; EXPERIMENTS.md is the source of truth.

### v1 — first real run, 2026-07-19

`src/validate.py` with StandardScaler + SVM(rbf), single stratified 70/30 split, seed=42, on the full 150 clean + 150 stego set (300 images, all real FUNSD/CORD/SROIE + 15% scattered-LSB stego). Not cross-validated — single split, so treat as a first read rather than a tight estimate.

| Dataset | Accuracy | Prior (deleted codebase) result | Notes |
|---|---|---|---|
| FUNSD | 90.6% | 100% | Strong — clean scanned forms give the clearest LSB/SRM signal, as expected. |
| SROIE | 62.9% | 80% | Weaker than prior run; SROIE here is a different HF mirror (`Voxel51/scanned_receipts`, not `darentang/sroie` — see Dataset section) so isn't a strict apples-to-apples reproduction. |
| CORD | 43.5% | 56.67% | Weakest of the three — actually *below* chance, diagnosed below. |
| BOSSBase | — | 58.33% | Out of current v2 scope (not in the 3-dataset plan). |
| **Overall** | **67.8%** | — | 90 held-out test images (45 clean / 45 stego). |

**Diagnosis of the CORD/SROIE weakness** (per-dataset clean-image feature stats, `outputs/features_v1_svm_baseline.csv`): CORD's clean images have a baseline `lsb_ratio` of ~49.3% (already looks statistically random — real photos carry enough sensor/JPEG noise that the LSB plane is close to 50/50 before any embedding), vs. FUNSD's ~95.1% (flat white background pixels are `11111111`, so the LSB plane is heavily skewed and embedding produces an easy, large shift). SROIE sits in between (~78.3%). This means marginal-statistic features (`lsb_ratio`, `lsb_entropy`, `entropy_manual`) are structurally blind on CORD regardless of payload density — there's no "before" state to disturb. The below-chance CORD accuracy suggests the SVM's decision boundary, fit across all three datasets together, was dominated by FUNSD's large easy signal and actively misfired on CORD's very different feature distribution.

### v2 — richer SRM + Random Forest, 2026-07-19

Two changes, both non-breaking (21-feature interface unchanged): (1) expanded `srm_features.py`'s kernel bank from 4 to 7 kernels, adding 2 first-order difference kernels and the classic 5×5 KV kernel — designed specifically to catch pixel-correlation signal that survives even when marginal LSB stats don't; (2) swapped the classifier from SVM(rbf) to Random Forest (300 trees, `class_weight="balanced"`, same train/test split and seed for comparability).

| Dataset | v1 (SVM, 4-kernel SRM) | v2 (RF, 7-kernel SRM) | Change |
|---|---|---|---|
| FUNSD | 90.6% | 96.9% | +6.3pp |
| SROIE | 62.9% | 62.9% | unchanged |
| CORD | 43.5% | 56.5% | **+13.0pp, and now above chance** |
| **Overall** | **67.8%** | **73.3%** | **+5.5pp** |

Top RF feature importances: `lsb_entropy` (0.268) and `lsb_ratio` (0.199) still dominate overall (they carry FUNSD's huge signal), followed by `srm_entropy` (0.062), `entropy_manual` (0.058), `noise_std` (0.039) — the richer SRM bank picked up meaningfully more weight than the old 4-kernel version did. `chisq_*` features are notably absent from the top 10, consistent with the chi-square saturation issue flagged above.

SROIE not moving at all across this change is itself informative — its weakness likely isn't a correlation-feature gap the same way CORD's was; next best guess is real classifier confusion from mixing all three datasets in one training set, which patch-based aggregation or a higher payload density might address differently than it did for CORD.

Full per-image feature tables: `outputs/features.csv` (current, v2 SRM) and `outputs/features_v1_svm_baseline.csv` (archived, pre-SRM-change, for comparison).

## Planned improvement experiments

1. ~~Richer multi-kernel SRM~~ done (v2, 2026-07-19) — 4→7 kernels, see above.
2. ~~Random Forest~~ done (v2, 2026-07-19). XGBoost not yet tried.
3. Patch-based max/percentile feature aggregation instead of whole-image stats — best next candidate for SROIE specifically, since the v2 SRM/RF change didn't move it.
4. Feature selection — chi-square group looks like noise not signal (near-zero RF importance in v2); worth dropping and re-measuring rather than assuming.
5. Higher payload density: confirmed *not* useful for CORD's `lsb_ratio`/`lsb_entropy` features specifically (see diagnosis above — embedding more random bits on an already-random plane doesn't help), but may still help correlation-based features (SRM/chi-square) on both CORD and SROIE. Worth testing in isolation from the v2 SRM change to see how much of the remaining gap it closes.

## Repo layout

```
SAA/
  src/
    entropy.py            # Group 1
    noise.py               # Group 2
    lsb.py                  # Group 3
    frequency.py           # Group 4
    variance.py             # Group 5
    edge.py                  # Group 6
    chi_square.py           # Group 7
    srm_features.py         # Group 8
    preprocessing.py        # residual caching (noise + SRM)
    extractor.py             # stego_analyzer() — the frozen 21-d interface
    download_datasets.py    # pull 50 FUNSD/CORD/SROIE images each
    embed_lsb_scattered.py  # generate 15% scattered-LSB stego images
    validate.py              # StandardScaler + SVM baseline, per-dataset accuracy
  datasets/
    clean/<FUNSD|CORD|SROIE>/    # populated by download_datasets.py
    stego/<FUNSD|CORD|SROIE>/    # populated by embed_lsb_scattered.py
    residuals/                    # cached .npy residuals from preprocessing.py
  outputs/                        # features.csv, trained scaler/model artifacts
  FEATURE_SPEC.md
```

## Immediate next steps (from planning session)

1. ~~`mkdir -p SAA/src SAA/datasets SAA/datasets/residuals SAA/outputs`~~ done
2. ~~Install deps~~ done (numpy, scipy, scikit-image, scikit-learn, pandas, matplotlib, pillow, datasets, huggingface_hub)
3. ~~Write modules~~ done — all 8 feature groups + preprocessing + extractor implemented and smoke-tested
4. ~~Run `python src/download_datasets.py`~~ done — 50/50/50 real FUNSD/CORD/SROIE images
5. ~~Run `python src/embed_lsb_scattered.py`~~ done — 150 real stego images (15% scattered LSB)
6. ~~Run `python src/validate.py`~~ done — real baseline: 67.8% overall (FUNSD 90.6% / SROIE 62.9% / CORD 43.5%)
7. ~~Fill in the results table above with genuine numbers~~ done
8. Next: improvement experiments (see above) — CORD is the clear priority given it's the weakest link
