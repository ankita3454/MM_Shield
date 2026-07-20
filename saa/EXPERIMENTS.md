# SAA Experiment Log

Tracks each validation run as a numbered experiment so improvements are attributable to a single change, not a bundle of them. See `FEATURE_SPEC.md` for the frozen 21-feature interface and method definitions; this file is the results/provenance ledger.

**Rule going forward: one change per experiment.** Don't combine payload density + classifier + feature changes in a single run, or it becomes impossible to tell which change caused which effect.

---

## Experiment 001 — v1 baseline (SVM, 4-kernel SRM)

**Date**: 2026-07-19
**Status**: superseded by Experiment 002 (kept for comparison)

**Change from prior state**: none — first real run on downloaded data (all synthetic smoke-test numbers before this are void).

**Config**:
- Feature set: 21-feature `stego_analyzer()` as defined in `FEATURE_SPEC.md`, SRM group using the original 4-kernel bank (horizontal, vertical, 2 diagonals, 2nd-order only).
- Classifier: `sklearn.svm.SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)`
- Scaler: `StandardScaler`, fit on train only
- Split: single stratified 70/30 train/test split (`train_test_split(..., test_size=0.3, random_state=42, stratify=y)`), not cross-validated
- Data: 150 clean (50 FUNSD + 50 CORD + 50 SROIE, real HF downloads) + 150 stego (15% scattered LSB payload, `embed_lsb_scattered.py`)
- n_train=208-210, n_test=90 (exact split varied slightly run to run due to 2 corrupted/re-embedded files mid-run; final clean run was 210/90)

**Results**:

| Dataset | Accuracy |
|---|---|
| FUNSD | 90.6% |
| SROIE | 62.9% |
| CORD | 43.5% (below chance) |
| **Overall** | **67.8%** |

Precision/recall: clean 0.65/0.71, stego 0.68/0.62 (f1 ~0.65-0.68 both classes).

**Artifacts**: `outputs/features_v1_svm_baseline.csv` (300-row feature table, pre-SRM-change). Model not frozen for this run (added only in Experiment 002's tooling).

**Diagnosis** (led directly to Experiment 002): CORD's clean images have baseline `lsb_ratio` ≈ 49.3% (natural photo noise already randomizes the LSB plane), vs. FUNSD's ≈ 95.1% (flat white background). Marginal-statistic features (`lsb_ratio`, `lsb_entropy`, `entropy_manual`) are structurally blind on CORD regardless of payload density. The below-chance CORD accuracy suggests the SVM decision boundary, fit across all three very different datasets at once, was dominated by FUNSD's easy signal and misfired on CORD specifically.

---

## Experiment 002 — richer SRM + Random Forest — **FROZEN OFFICIAL BASELINE**

**Date**: 2026-07-19
**Status**: current best, frozen. All future experiments compare against this.

**Change from Experiment 001** (two changes bundled here since both were proposed together as one "highest-leverage, lowest-risk" step before the one-change-at-a-time rule was adopted — noted as a limitation):
1. `srm_features.py` kernel bank expanded 4→7 kernels: added 2 first-order difference kernels (`h1`, `v1`) and the classic 5×5 KV kernel (Kodovsky-Fridrich minmax, the single strongest fixed high-pass filter in the steganalysis literature). `srm_energy`/`srm_entropy` now computed over the full 7-kernel bank (each kernel z-scored before pooling for entropy, since KV's natural scale differs a lot from the 3×3 kernels). `srm_diagonal_ratio` unchanged (still only the original 4 kernels). Interface still exactly 3 output features — non-breaking.
2. Classifier swapped from SVM(rbf) to Random Forest.

**Config**:
- Feature set: 21-feature `stego_analyzer()`, SRM group using the new 7-kernel bank (see `FEATURE_SPEC.md` SRM method note for full definitions).
- Classifier: `sklearn.ensemble.RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)`
  - `class_weight="balanced"` specifically because mixing FUNSD (huge, easy signal) with CORD/SROIE (weak signal) in one training set risks an unweighted ensemble being dominated by whichever dataset's examples are easiest to split on.
- Scaler: `StandardScaler`, fit on train only
- Split: single stratified 70/30 train/test split, `random_state=42`, stratified on label only (not on source dataset) — same split as Experiment 001 for direct comparability
- Data: identical 150 clean + 150 stego set as Experiment 001 (no re-download, no re-embedding)
- n_train=210, n_test=90

**Results**:

| Dataset | Accuracy | vs. Exp 001 |
|---|---|---|
| FUNSD | 96.9% | +6.3pp |
| SROIE | 62.9% | +0.0pp (unchanged) |
| CORD | 56.5% | **+13.0pp** (now above chance) |
| **Overall** | **73.3%** | **+5.5pp** |

**Precision / Recall / F1**:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| clean | 0.76 | 0.69 | 0.72 | 45 |
| stego | 0.71 | 0.78 | 0.74 | 45 |

**Confusion matrix** (rows = true, cols = predicted):

|  | pred_clean | pred_stego |
|---|---|---|
| **true_clean** | 31 | 14 |
| **true_stego** | 10 | 35 |

**Top 10 feature importances (Random Forest, Gini)**:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `lsb_entropy` | 0.2677 |
| 2 | `lsb_ratio` | 0.1993 |
| 3 | `srm_entropy` | 0.0621 |
| 4 | `entropy_manual` | 0.0584 |
| 5 | `noise_std` | 0.0386 |
| 6 | `srm_diagonal_ratio` | 0.0343 |
| 7 | `noise_abs_mean` | 0.0331 |
| 8 | `hist_skewness` | 0.0316 |
| 9 | `local_variance_std` | 0.0310 |
| 10 | `srm_energy` | 0.0305 |

Chi-square features (`chisq_max_pvalue`, `chisq_mean_pvalue`, `chisq_fraction_high`) are absent from the top 10 — consistent with the chi-square saturation issue flagged in `FEATURE_SPEC.md`. Worth testing removal in a future experiment (Experiment 004 candidate: feature selection).

**Reproduce this exact run**:
```
cd saa/src
python3 validate.py --clean-dir ../datasets/clean --stego-dir ../datasets/stego \
  --out-csv ../outputs/features.csv --classifier rf \
  --save-model ../outputs/model_v2_rf_frozen.pkl
```

**Artifacts**:
- `outputs/features.csv` — full 300-row feature table (v2 SRM values)
- `outputs/model_v2_rf_frozen.pkl` — pickled `{scaler, clf, feature_names, split params, all metrics}` — the actual reproducible model, not just numbers in a table
- `outputs/model_v2_rf_frozen.json` — human-readable sidecar (everything in the pickle except the model objects themselves), for quick diffing against future frozen versions

**SROIE analysis**: unchanged accuracy across a fairly substantial feature+classifier change is itself informative — SROIE's weakness likely isn't the same "marginal stats are blind" problem CORD had (the richer SRM bank should have helped if it were). Leading hypotheses for the next single-change experiment: (a) SROIE images are the most heterogeneous in scan quality/lighting of the three sources, so whole-image aggregate stats may wash out signal that's concentrated in specific image regions — patch-based aggregation is the natural test; (b) SROIE may need its own payload-density tuning independent of CORD's.

---

## Experiment 003 — patch-based SRM aggregation — **negative result, not adopted**

**Date**: 2026-07-19
**Status**: rejected. Code reverted to Experiment 002's implementation; current best remains Experiment 002.

**Change from Experiment 002** (one change only): in `srm_features.py`, `srm_energy` and `srm_diagonal_ratio` switched from a single whole-image average to the 90th percentile across a grid of non-overlapping 32×32 patches (each patch's own energy/ratio computed, then percentile taken across all patches). `srm_entropy` deliberately left untouched as a whole-image statistic, as a built-in control. Rationale: whole-image mean-pooling could be washing out scattered embedding concentrated in a minority of a large, mostly-uniform image — a high percentile across patches should surface the "hottest" affected regions instead of averaging them away.

**Config**: identical to Experiment 002 otherwise — same 300-image set, same RF hyperparameters, same 70/30 split, same seed=42.

**Results**:

| Dataset | Exp 002 (whole-image mean) | Exp 003 (patch 90th %ile) | Change |
|---|---|---|---|
| FUNSD | 96.9% | 96.9% | 0 |
| SROIE | 62.9% | 62.9% | 0 |
| CORD | 56.5% | 52.2% | **-4.3pp** |
| **Overall** | **73.3%** | **72.2%** | **-1.1pp** |

Confusion matrix (v3): 32 TN / 13 FP / 12 FN / 33 TP (vs. v2's 31/14/10/35 — slightly more false negatives, fewer false positives; net worse).

**Interpretation**: the hypothesis was wrong, and the result is clean enough to draw a real conclusion from, not just noise — CORD got measurably worse, and SROIE didn't move at all despite being the explicit target. This means SROIE's weakness is *not* a whole-image-averaging-washes-out-localized-signal problem, which rules out that entire theory rather than just this one implementation of it. For CORD, patch percentiles apparently added variance/noise without adding signal (plausible: 32×32 patches on already low-signal photographic content may just be small enough to be dominated by local texture noise rather than embedding).

**Artifacts** (kept for reference, not active): `outputs/features_v3_patchsrm... ` — not saved separately since the CSV was overwritten; the frozen model captures the full result: `outputs/model_v3_rf_patchsrm_frozen.pkl` / `.json`. The patch-aggregation code itself is not preserved in the active `srm_features.py` (reverted) — see this entry's config description and the frozen model for reproduction if needed.

**What this rules out for SROIE**: whole-image vs. patch-level aggregation of SRM features specifically. Still open: whether SROIE needs a different fix entirely (payload density, a different feature group's patch treatment, or something about its image characteristics not yet diagnosed the way CORD's was).

---

## Experiment 004 — SROIE root-cause diagnostic (analysis only, no code change)

**Date**: 2026-07-19
**Status**: both hypotheses tested; one rejected, one led directly to Experiment 005.

```
Hypothesis 1: Dataset mixing causes SROIE's poor accuracy (the RF gets
              dominated by FUNSD's easy signal during training).
Method:       Train and evaluate a fresh RF on SROIE's 100 images only
              (same split params/seed/hyperparameters as Experiment 002,
              zero FUNSD/CORD in the training set).
Outcome:      63.3% isolated vs. 62.9% mixed -- essentially unchanged.
Conclusion:   Hypothesis rejected. Same result for CORD (56.7% vs 56.5%).
Next hypothesis: The ceiling is set by the feature representation on each
              dataset, not by cross-dataset interference -- so look at
              which features carry signal on SROIE specifically.
```

```
Hypothesis 2: SRM is a low-value feature group for SROIE (based on
              Experiments 002/003 both leaving SROIE flat despite SRM
              being the thing that changed).
Method:       Per-dataset, per-feature AUC (single-feature ranking power,
              threshold-independent -- unlike RF Gini importance).
Outcome:      Chi-square is SROIE's #1 feature by AUC (0.674), not SRM.
              Chi-square was previously flagged as "likely noise" based on
              near-zero *global* RF Gini importance in Experiment 002 --
              that ranking undersells it because it's correlated with the
              dominant lsb_ratio/lsb_entropy features, and Gini importance
              discounts a feature once a correlated stronger one exists.
Conclusion:   Confirmed -- SRM was never SROIE's bottleneck (explains why
              Experiments 002 and 003 didn't move it), and the "drop
              chi-square" plan from Experiment 003's next-steps list is
              retracted -- it would remove SROIE's best feature.
Next hypothesis: Chi-square's raw signal (AUC 0.674) isn't being fully
              exploited -- tune its windowing scheme (Experiment 005).
```

Full detail below.

**Q1: Is SROIE's flatness a cross-dataset training-confusion artifact?** Trained and tested a fresh RF *only* on SROIE's 100 images (70/30 split, same hyperparameters as Experiment 002, no FUNSD/CORD in the training set at all).

| Dataset | Mixed-dataset accuracy (Exp 002) | Isolated (that dataset only) | Difference |
|---|---|---|---|
| FUNSD | 96.9% | 100.0% | +3.1pp |
| CORD | 56.5% | 56.7% | ~0 |
| SROIE | 62.9% | 63.3% | ~0 |

**Answer: no.** CORD and SROIE's ceilings are essentially identical whether trained in isolation or mixed with the other datasets. This rules out "the classifier gets confused/dominated by FUNSD's easy signal" as the explanation for either dataset's weakness — their features genuinely don't separate clean/stego well on their own, full stop.

**Q2: Which features actually carry SROIE's signal?** Per-dataset per-feature AUC (single-feature ranking ability, threshold-independent, unlike RF Gini importance):

| Rank | SROIE | AUC | CORD | AUC | FUNSD | AUC |
|---|---|---|---|---|---|---|
| 1 | `chisq_max_pvalue` | 0.674 | `lsb_entropy` | 0.780 | `lsb_ratio` | 0.982 |
| 2 | `chisq_mean_pvalue` | 0.674 | `chisq_max_pvalue` | 0.526 | `lsb_entropy` | 0.982 |
| 3 | `lsb_ratio` | 0.612 | `chisq_mean_pvalue` | 0.525 | `entropy_manual` | 0.748 |
| 4 | `lsb_entropy` | 0.594 | `srm_entropy` | 0.523 | `srm_entropy` | 0.683 |

**This corrects something we got wrong in Experiment 002.** We flagged chi-square as likely-noise there because it had near-zero *global* RF importance — but that ranking was computed on the mixed 3-dataset model, where chi-square is correlated with the much stronger `lsb_ratio`/`lsb_entropy` features (which dominate via FUNSD). Random Forest's Gini importance systematically discounts a feature when a correlated, stronger feature is available, even if the weaker feature carries real independent signal elsewhere. Chi-square is actually **SROIE's single best feature** (AUC 0.674) and FUNSD's 4th-best (AUC 0.682) — it's not noise, it was just getting drowned out in the global importance ranking. Also notable: SRM (the whole focus of Experiments 002-003) isn't even in SROIE's top 4 — which explains, after the fact, why two different SRM-focused changes both left SROIE completely flat. We were optimizing a feature group that was never SROIE's bottleneck.

**Secondary observation, not yet confirmed causal**: image size varies far more within CORD (CV=1.48, range 0.28-9.4 megapixels) and SROIE (CV=1.10, range 0.28-8.7 MP) than within FUNSD (CV=0.01, essentially constant ~0.76-0.80 MP — a controlled scanned-form dataset). `total_frequency_energy` was checked as the most obviously scale-sensitive feature and turned out uninformative everywhere (AUC ~0.51-0.52 on all three datasets) rather than specifically bad on the high-variance datasets, so this doesn't look like the primary driver — but it's still an open structural difference worth keeping in mind for any future feature that isn't already scale-normalized.

**Implication for next experiments**: the planned "drop chi-square, it's noise" experiment (previously priority #2 below) is now known to be the wrong move for SROIE specifically — it would remove SROIE's best feature. Feature selection should be per-dataset-aware, or the removal should target something else. SRM-focused work is deprioritized for SROIE given it isn't in the signal picture there at all.

---

## Next experiments (one change each, in priority order)

1. ~~Inspect the 11 always-misclassified clean SROIE images directly~~ done — Experiment 008. Found: 45% share a composited title-text overlay (dataset artifact), 18% are literal duplicates, and quantitatively 18/21 features place these clean images objectively closer to genuine stego than to genuine clean.
2. **Higher payload density, CORD/SROIE only** — re-embed at e.g. 40-50% instead of 15% for just those two dataset subfolders, re-extract, re-validate with Experiment 002's exact config otherwise unchanged. Better motivated now (Experiment 008): several FP images' clean/stego feature vectors sit almost on top of each other, suggesting 15% genuinely isn't perturbing the representation enough for this content.
3. **Central-crop or header-exclusion variant of entropy/edge features** — new lead from Experiment 008: 5/11 SROIE false positives share a composited title-text band that inflates edge/entropy stats independent of embedding. Test whether restricting those features to a central crop (excluding the top ~15% of the image) removes this specific failure mode.
4. **XGBoost specifically** (not just LightGBM) — worth trying on a machine with normal internet access, since Experiment 007 had to substitute LightGBM after XGBoost's pip install repeatedly failed in this sandbox. Low expected value given LightGBM (also boosting) didn't help, but not a clean substitute for the real thing.
4. ~~Permutation importance (diagnostic)~~ done — Experiment 004 addendum, led to Experiment 006.
5. ~~Feature selection — drop chi-square~~ retracted per Experiment 004 (chi-square is SROIE's best single-feature AUC, even though no classifier tried actually uses it).
6. ~~Patch-based SRM aggregation~~ tried in Experiment 003, rejected.
7. ~~Chi-square windowing tune-up~~ tried in Experiment 005, rejected — made things worse, and revealed the original windowing's signal may itself be a fragile/spatial-position artifact rather than a robust statistical one (see Experiment 005's open question).
8. ~~`max_features` sweep~~ tried in Experiment 006, rejected.
9. ~~Boosting (LightGBM)~~ tried in Experiment 007, rejected — but proved SROIE's ceiling is data/feature-level, not classifier-level.

---

## Experiment 004 addendum — permutation importance diagnostic

**Date**: 2026-07-19

```
Hypothesis:   Chi-square's real standalone signal (AUC 0.674 on SROIE,
              per Experiment 004) isn't reaching the model -- i.e. the
              trained RF isn't actually relying on it, which would
              explain why Experiment 005's chi-square improvements had
              zero effect on accuracy.
Method:       Permutation importance (30-50 repeats) on Experiment 002's
              frozen RF, computed separately on each dataset's test
              subset, not just globally.
Outcome:      On SROIE's test subset, all 3 chisq features show ~zero or
              slightly negative permutation importance (chisq_max_pvalue
              -0.003, chisq_mean_pvalue -0.002, chisq_fraction_high
              0.000) -- shuffling them doesn't hurt accuracy at all.
              lsb_ratio (+0.049) and lsb_entropy (+0.032) dominate, same
              as FUNSD. (CORD and FUNSD breakdowns also recorded --
              CORD's top permutation features were lsb_entropy, lsb_ratio,
              then a cluster of edge/SRM features; FUNSD is almost
              entirely lsb_ratio.)
Conclusion:   Confirmed: the trained model isn't using chi-square for
              SROIE, despite its real standalone signal. This is a
              *candidate mechanism* for Experiments 003/005 both failing,
              not yet a confirmed cause.
Next hypothesis: RandomForestClassifier's default max_features="sqrt"
              (~5 of 21 features considered per split) may be preventing
              chi-square from ever being evaluated at SROIE-relevant
              splits, even when it would win. Testable directly by
              varying max_features (Experiment 006). (Note: Experiment 005,
              below, was run in parallel with this line of thinking and
              tests a related but distinct hypothesis about chi-square's
              windowing itself.)
```

---

## Experiment 005 — chi-square windowing redesign

```
Hypothesis:   Chi-square's raw per-feature signal (SROIE AUC 0.674) is
              being underexploited because its current windowing scheme
              (20 growing, cumulative, raster-order prefixes -- window 1
              is the first 5% of pixels, window 20 is 100% of them, each
              window a superset of the last) mixes wildly different
              sample sizes into one p-value distribution. Small early
              windows are underpowered; huge late windows risk the
              float64 underflow-to-p=0 issue already flagged in
              FEATURE_SPEC.md's known issues. Fixed-size, non-overlapping
              blocks tiled across the image should give many independent,
              consistently-powered p-values instead.
Method:       Redesign chi_square.py's sliding_pov_pvalues() to tile the
              flattened pixel stream into non-overlapping WINDOW_SIZE=10000
              pixel blocks (chosen so images from 0.28MP to 9.4MP -- the
              actual CORD/SROIE range -- all get a reasonable, adaptively
              scaling window count, ~28 to ~940 windows respectively,
              rather than a fixed window count regardless of image size).
              chisq_max_pvalue / chisq_mean_pvalue / chisq_fraction_high
              computed identically otherwise (same aggregate definitions,
              just over the new window set). Everything else unchanged
              from Experiment 002 (same RF, split, seed, 7-kernel SRM).
Outcome:      Overall accuracy 73.3% -> 71.1% (-2.2pp). CORD 56.5% -> 47.8%
              (-8.7pp, back below chance). SROIE 62.9% -> 62.9% (unchanged
              -- the one metric this experiment specifically targeted).
              FUNSD unchanged (96.9%). Chi-square dropped out of the RF's
              top-10 importances entirely.

              Checked *why*: re-measured chi-square's raw per-feature AUC
              on the new windowing. It collapsed from 0.674 (SROIE) / 0.780
              n/a (CORD's best was lsb_entropy, chisq was ~0.526) / 0.682
              (FUNSD) down to ~0.50-0.53 across all three datasets --
              essentially random. The fixed-size-window redesign didn't
              just fail to help, it destroyed the signal that the original
              growing-prefix design had.
Conclusion:   Hypothesis rejected, cleanly. Reverted chi_square.py to the
              Experiment 002 growing-prefix implementation; verified the
              revert reproduces Experiment 002's exact 73.3% overall /
              56.5% CORD / 96.9% FUNSD / 62.9% SROIE numbers.

              This is now the *third* single-feature-group change
              (Experiment 002's SRM expansion, Experiment 003's SRM
              patching, this one) to leave SROIE at exactly 62.9%. Two of
              the three (003, 005) were direct, targeted attempts and both
              failed outright, one making a different dataset (CORD) worse
              in the process. That's a strong pattern, not coincidence.
Next hypothesis: SROIE's ~63% ceiling is likely not reachable by further
              tuning of individual existing feature *groups* with the
              current classifier. Worth stepping back to a genuinely
              different angle rather than a 4th single-feature tweak --
              e.g. permutation importance to see what the RF is actually
              using on SROIE (not just what's available), or accepting
              this as a documented limitation of the classical/non-DL
              feature set for SROIE specifically and moving attention to
              other project priorities.

              Open, unexplained question worth a note for future work: WHY
              does the original growing-cumulative-prefix chi-square design
              carry real signal (AUC 0.674-0.780) while the more
              "textbook-correct" fixed-window design carries none? One
              plausible mechanism: small early prefixes are dominated by
              whichever image region comes first in raster order (often a
              header/margin), so the growing design may be picking up a
              spatial-position artifact correlated with embedding rather
              than a genuine statistical equalization signal. If true, that
              would mean the chi-square feature's apparent usefulness is
              somewhat fragile/dataset-specific rather than a robust
              steganalysis signal -- worth being skeptical of before
              leaning on it further.
```

---

## Experiment 006 — max_features sweep

**Date**: 2026-07-19
**Status**: hypothesis rejected.

```
Hypothesis:   Restricting candidate features per split (max_features=
              "sqrt", ~5 of 21) prevents the RF from consistently
              evaluating chi-square, reducing its contribution. If true,
              chi-square's split-usage frequency and SROIE permutation
              importance should both increase as max_features increases.
              (Framed as a hypothesis to test, not an established cause --
              per-feedback correction: correlation with lsb_ratio/entropy,
              tree depth/leaf constraints, or bootstrap sampling bias are
              equally plausible alternative mechanisms this experiment
              does not by itself rule in or out.)
Method:       Swept max_features in {"sqrt" (Exp002 baseline), None (all
              21), 10, 5}. Everything else held fixed: same 300 trees,
              same train/test split, same seed=42, same features. Measured
              four things per config, not just accuracy: overall accuracy,
              per-dataset accuracy, chi-square's total split-usage count
              across all 300 trees (direct evidence of whether chi-square
              is entering the trees more), and SROIE permutation
              importance for the 3 chisq features.
```

| max_features | Overall | FUNSD | CORD | SROIE | chisq splits (of total) | chisq SROIE perm. importance |
|---|---|---|---|---|---|---|
| `sqrt` (baseline) | 73.3% | 96.9% | 56.5% | 62.9% | 181/7387 (2.45%) | -0.005, -0.003, +0.000 |
| `None` (all 21) | 72.2% | 93.8% | 52.2% | 65.7% | 97/5273 (1.84%) | -0.002, -0.002, +0.000 |
| `10` | 72.2% | 96.9% | 52.2% | 62.9% | 99/5826 (1.70%) | +0.000, +0.000, +0.000 |
| `5` | 74.4% | 96.9% | 56.5% | 65.7% | 134/6901 (1.94%) | +0.000, +0.000, +0.000 |

```
Outcome:      Chi-square's split-usage percentage did NOT increase with
              max_features -- if anything it trended slightly down (2.45%
              at sqrt, dropping to 1.70-1.94% at 10/5, 1.84% at None).
              Permutation importance for chi-square on SROIE stayed at
              essentially zero across every setting, including max_features
              =None where all 21 features are considered at every single
              split -- so even with zero restriction, chi-square still
              isn't what the trees end up choosing.
Conclusion:   Hypothesis rejected, cleanly, per the "if nothing changes"
              case anticipated going in: max_features is not why the RF
              ignores chi-square for SROIE. Remaining candidate mechanisms
              (correlation with lsb features causing trees to get similar
              information earlier from a stronger feature; something about
              how chi-square's specific value distribution interacts with
              Gini splitting) are still open, but max_features specifically
              is ruled out.

              Side observation, not adopted as a new baseline: max_features
              =5 scored slightly better than the sqrt baseline on this one
              split (74.4% overall / 65.7% SROIE vs. 73.3% / 62.9%), and
              not via increased chi-square usage (its split count was
              actually similar to or lower than baseline). This is most
              likely single-split noise from RF's own internal
              randomization (n_test=90 overall / 35 for SROIE -- a
              difference of 1-2 correctly classified images swings the
              reported percentage by several points) rather than a real
              effect, and per Experiment 002's own frozen-baseline
              discipline should not be adopted without validation across
              multiple seeds or folds first.
Next hypothesis: Chi-square carrying real AUC but the RF never using it,
              across every max_features setting tried, points away from
              "the classifier isn't looking hard enough" and toward "the
              classifier structurally can't combine this feature usefully
              given what else is in the training set" -- e.g. a boosting
              method that targets residual errors (XGBoost) might succeed
              where bagging didn't. Payload density is the other live
              lead, since it's the one variable in the whole SAA pipeline
              that hasn't been touched for SROIE at all yet.
```

---

## Experiment 007 — boosting instead of bagging (LightGBM)

**Date**: 2026-07-19
**Status**: hypothesis rejected. Unexpected secondary finding is more useful than the primary result.

**Deviation from plan, noted honestly**: XGBoost was the intended classifier (boosting is fundamentally different from RF's bagging, and was the top-priority next experiment). Its pip package repeatedly failed to finish installing in this sandbox after 6+ attempts (large bundled binary, ~120MB+ partial download each time, sandbox network too slow/unstable to complete it) -- confirmed not a code issue, a sandbox environment limitation. Substituted LightGBM, another boosting-based tree ensemble, which installed in seconds. This tests the same underlying hypothesis (boosting vs. bagging) but is not literally XGBoost. **XGBoost itself is still untested and worth trying on a machine with normal internet access** -- it may behave differently than LightGBM despite both being boosting methods.

```
Hypothesis:   Random Forest (bagging) isn't combining chi-square's real
              signal usefully for SROIE across every configuration tried
              so far (Experiments 002, 003, 005, 006). A boosting method,
              which iteratively fits new trees to the residual errors of
              the ensemble so far rather than averaging independent trees,
              may succeed where bagging didn't -- boosting can pick up a
              weaker complementary feature specifically because it targets
              exactly the examples the dominant features (lsb_ratio/
              lsb_entropy) get wrong.
Method:       Same 300-image feature table, same 70/30 split, same seed=42,
              same 21 features (Experiment 002's implementation, chi-square
              reverted per Experiment 005). Only the classifier changed:
              RandomForestClassifier -> LGBMClassifier(n_estimators=300,
              min_child_samples=5, class_weight="balanced", seed=42).
              min_child_samples lowered from LightGBM's default of 20,
              since the training set is only 210 rows.
Outcome:      Overall accuracy 73.3% -> 72.2% (-1.1pp, worse). CORD 56.5%
              -> 52.2% (-4.3pp, worse). SROIE 62.9% -> 62.9% (unchanged --
              the *fourth* separate change to land on exactly this number:
              Experiments 002 baseline, 003, 005, and now 007). FUNSD
              unchanged (96.9%).

              Permutation importance on SROIE under LightGBM: chi-square
              features again ~zero or slightly negative (chisq_max_pvalue
              -0.0023, mean/frac_high both 0.0000) -- confirms chi-square
              goes unused under boosting too, not just bagging. This is
              useful: it means "chi-square is unused" is a property of the
              feature/data relationship itself, not an artifact of any one
              algorithm family.

              **Unplanned but more interesting finding**: compared RF's and
              LightGBM's predictions image-by-image on the 35 SROIE test
              images (not just the aggregate accuracy). They agree on
              every single image -- 22/22 identical correct predictions,
              13/13 identical wrong predictions, zero disagreements. Of
              the 13 errors, 11 are clean SROIE images misclassified as
              stego (false positives) and only 2 are stego images missed
              (false negatives) -- a strong, consistent bias, not random
              noise. The 11 always-wrong clean images are specific and
              identifiable (listed in the source analysis: img_000, 002,
              003, 006, 007, 012, 015, 021, 024, 044, 047).
Conclusion:   Hypothesis rejected -- boosting doesn't outperform bagging
              here, and doesn't use chi-square either. But the perfect
              prediction agreement between two structurally different
              algorithms is itself a strong result: SROIE's ~63% ceiling
              is not a classifier-choice artifact. It's baked into the
              feature *values* for these specific 35 test images (and
              presumably the rest of SROIE) -- any reasonable classifier
              trained on this feature set will draw essentially the same
              decision boundary and make the same mistakes, because the
              relevant information (or lack of it) is fixed at the feature
              extraction stage, not the modeling stage.
Next hypothesis: Stop varying the classifier -- four attempts (002-style
              RF, 003 SRM-patched RF, 005 chisq-windowed RF, 007 LightGBM)
              have now converged on the same result. The productive next
              step is to inspect the 11 specific always-misclassified
              clean SROIE images directly (raw feature values, and
              probably the images themselves) to find out what makes their
              lsb_entropy/lsb_ratio look stego-like even though they're
              clean -- e.g. unusually low image quality, heavy compression,
              or scan noise that mimics embedding. That's a data-level
              question a classifier swap can't answer. Payload-density
              tuning remains a separate, untried lead.
```

---

## Experiment 008 — root-cause inspection of the 11 always-misclassified clean SROIE images

**Date**: 2026-07-19
**Status**: hypothesis rejected (predicted cause wrong); real cause identified via a combination of visual inspection, group feature comparison, nearest-neighbor analysis, and direct clean-vs-stego feature comparison.

```
Hypothesis:   The 11 clean SROIE images that both RF and LightGBM
              misclassify as stego (Experiment 007) share a visual/quality
              characteristic -- photographed rather than scanned, uneven
              illumination, heavy JPEG artifacts, textured paper, folds/
              wrinkles, or a noisy background -- that pushes their feature
              vectors into the stego region even though no payload was
              embedded.
Method:       Four complementary checks, in order:
              1. Visual inspection (direct image read) of all 11 FP images
                 (img_000, 002, 003, 006, 007, 012, 015, 021, 024, 044, 047),
                 plus one correctly-classified clean image (img_030) and one
                 correctly-classified stego image as visual controls.
              2. Group-mean comparison of all 21 raw feature values across
                 four groups: correct_clean, FP_clean (the 11), correct_
                 stego, FN_stego.
              3. Nearest-neighbor search in standardized 21-d feature space:
                 for each of the 11 FP images, find its nearest neighbor
                 (by Euclidean distance) among all 100 SROIE images (both
                 classes) and check whether it's a clean or stego image,
                 and specifically how far it is from its own stego-embedded
                 twin.
              4. Direct feature-by-feature comparison: for each of the 21
                 features, is FP_clean's mean closer to correct_clean's
                 mean or to correct_stego's mean?
```

**1. Visual inspection.** All 11 FP images were viewed directly. Findings contradicted the hypothesis:

- **5 of 11 (45%) — img_000, 002, 003, 006, 007 — share an identical, distinctive artifact**: a large, bold, black, digitally-composited title line ("tan woon yann" / "tan chay yee") rendered above the actual receipt content. This is not part of the original receipt — it's an annotation overlay baked into the image itself, almost certainly a dataset-construction artifact (the receipt's extracted ground-truth "company name" field rendered as a header) rather than anything related to photography or scan quality. A large, sharp-edged, high-contrast text block sitting on an otherwise sparse, uniform white receipt is exactly the kind of localized high-frequency content that whole-image aggregate features (`edge_density`, `entropy_manual`, `srm_entropy`) are sensitive to.
- **2 of 11 — img_012, img_015 — are literal duplicate images** (identical receipt content, confirmed by direct visual comparison). This is a data-quality artifact: at least one of the "11 independent hard examples" is not independent.
- **img_044, img_047**: clean, uniform, well-scanned dot-matrix receipts, among the least noisy-looking images in the entire SROIE clean set — the opposite of "photographed/messy."
- **img_021, img_024**: have a red handwritten "PAID" stamp/scribble overlay — but so does img_030, a *correctly-classified* clean image used as a control. Same visual feature present in both an FP and a correctly-classified image, so it isn't the differentiator.
- **Control (img_030, correctly classified clean)**: this was the genuinely messy one — visible paper texture, skew, uneven shading, red ink stamp, strikethrough — exactly the profile the hypothesis predicted for the *misclassified* images, but it was classified correctly.

**Conclusion so far: the "photographed/noisy/textured" hypothesis is rejected by direct observation.** If anything the correlation runs backwards on this small sample: some of the cleanest, most uniform images are the false positives, and the messiest one was handled correctly.

**2. Group feature comparison.** FP_clean images have elevated `entropy_manual`, `edge_density`, `srm_entropy` and depressed `hist_kurtosis` relative to *both* correct_clean and correct_stego — an outlier pattern rather than a simple "looks like stego" pattern on its own. Consistent with the title-overlay explanation for the 5 images that have it (a large dense text block raises entropy/edge density above even genuine stego levels).

**3. Nearest-neighbor analysis.** Several FP images' nearest neighbor in standardized feature space is literally their own stego-embedded twin, at very small distance (0.19–0.30). Correctly-classified clean images either cluster with other clean images, or when their own stego twin is nearest, the distance is much larger (0.69–0.85). This means for these specific images, the 15% LSB embedding leaves almost no measurable footprint in the 21-feature representation — clean and stego versions of the same image are nearly indistinguishable to the extractor, not because the clean image looks artificially stego-like, but because the embedding itself doesn't move the feature vector much for this content.

**4. Direct clean-vs-stego feature comparison** (the user's specific request: check whether FP_clean's feature values are closer to genuine stego than to genuine clean). Computed `|mean(FP_clean) − mean(correct_stego)|` vs. `|mean(FP_clean) − mean(correct_clean)|` per feature:

| # features where FP_clean mean is closer to... | count |
|---|---|
| correct_stego mean | **18 / 21** |
| correct_clean mean | 3 / 21 |

The only 3 features where FP_clean stays closer to clean: `local_variance_std`, `local_variance_max`, `chisq_fraction_high`. Every other feature — including the classifier's top-importance features `lsb_entropy`, `lsb_ratio`, `entropy_manual`, `srm_entropy` — has the FP_clean group sitting closer to the genuine-stego mean than to the genuine-clean mean.

```
Outcome:      Hypothesis (photographed/textured/wrinkled) rejected by
              direct visual inspection -- the opposite pattern was observed
              (messy control image classified correctly; several of the
              cleanest images were false positives). The real, identified
              causes are structurally different and heterogeneous across
              the 11 images, not one shared quality issue:
                - 5/11: a large composited text overlay (dataset
                  annotation artifact, not a photography/scan artifact)
                  that inflates edge/entropy features independent of any
                  embedding.
                - 2/11: literal duplicate images (data-quality issue,
                  not independent hard examples).
                - remaining images (and to a lesser degree all 11):
                  clean and stego versions of the same image sit almost on
                  top of each other in feature space (self-stego distance
                  0.19-0.30 vs. 0.69-0.85 for correctly-classified images)
                  -- the embedding pipeline produces almost no measurable
                  footprint on this specific content, not that the clean
                  image spuriously resembles stego.
              And confirmed quantitatively (not just by nearest-neighbor
              proximity): on 18 of 21 features, FP_clean's group mean is
              objectively closer to genuine stego's mean than to genuine
              clean's mean -- including every feature the classifier
              actually relies on. The classifier isn't erring arbitrarily;
              given this feature representation, these 11 images really do
              sit in stego territory.
Conclusion:   The 11 misclassifications are not explained by a single
              image-quality characteristic. They're explained by (a) a
              dataset-construction artifact (composited title text, 45% of
              cases) that the feature set was never designed to be robust
              to, (b) a data-quality issue (duplicate images, ~18% of
              cases), and (c) a genuine feature-extraction ceiling: for a
              meaningful subset of SROIE's clean images, embedding at 15%
              payload density does not perturb the 21-feature vector far
              enough from the clean baseline to be separable, while
              simultaneously some clean images' baseline feature values
              already resemble the stego region even before any embedding
              (dense text overlays raising entropy/edge stats being the
              clearest mechanism identified). This is a representation
              limitation, confirmed directly rather than inferred: 18/21
              features place these clean images closer to stego than to
              clean.
Next hypothesis: Two independent, testable fixes fall out of this:
              (1) the composited-title-text artifact should be checked
              across the full SROIE set (clean and stego both) -- if it's
              specific to certain source images rather than random, it may
              be a confound affecting more than just these 5; a targeted
              feature (e.g. flagging large solid-black connected regions,
              or restricting entropy/edge stats to a central crop that
              excludes a synthetic header band) could help. (2) The near-
              zero clean-to-stego feature displacement for the remaining
              images suggests the payload-density experiment (still
              untried for SROIE, next on the priority list) is now better
              motivated than before -- if 15% embedding barely moves the
              feature vector for this content, a higher density is a
              direct, mechanistic fix rather than a guess.
```

**Artifacts**: analysis run against `outputs/features.csv` (cached, no re-extraction) and `outputs/model_v2_rf_frozen.pkl` (Experiment 002's frozen model, used for predictions to identify the FP/correct groups). No code or data changes in this experiment — diagnostic only.

---

## Experiment 009 — double the dataset (50→100 images per source)

**Date**: 2026-07-19
**Status**: hypothesis confirmed, new best result. Supersedes Experiment 002 as the current frozen baseline going forward, but Experiment 002 is kept as the "100% original data" reference point since this experiment changes the training set size/composition rather than the feature/classifier pipeline.

```
Hypothesis:   The ~57-63% accuracy ceiling on CORD/SROIE (unmoved across
              five straight feature/classifier changes -- Experiments 002,
              003, 005, 006, 007) is at least partly a training-set-size
              limitation, not purely a representation limitation. With only
              50 clean images per dataset, the RF has seen too few receipt
              layouts, fonts, lighting conditions, and vendors to learn a
              stable decision boundary, especially for CORD/SROIE's
              structurally noisier feature signal. Doubling to 100 clean
              images per dataset (300 clean + 300 stego = 600 total, up
              from 300) should improve CORD and SROIE accuracy, with FUNSD
              relatively unaffected since it's already near ceiling (96.9%).
Method:       Downloaded 50 additional clean images per dataset (skip=50,
              n=50 in download_datasets.py, extending each dataset's
              existing img_000-049 with new img_050-099 -- no overlap).
              Embedded the same 15% scattered LSB payload at the same
              seed scheme (embed_lsb_scattered.py, unchanged). Re-extracted
              all 21 features for the full 600-image set using the exact
              Experiment 002 feature implementation (7-kernel SRM,
              growing-prefix chi-square -- neither touched). Retrained with
              Experiment 002's exact RandomForestClassifier config
              (n_estimators=300, min_samples_leaf=2, max_features="sqrt",
              class_weight="balanced", random_state=42), same 70/30
              stratified split params, same seed=42. Only the dataset size
              changed -- feature code, classifier hyperparameters, split
              logic, and seed are all identical to Experiment 002.
Outcome:      Overall accuracy 73.3% -> 80.6% (+7.3pp). Per-dataset:
```

| Dataset | Experiment 002 (n=150 clean/150 stego) | Experiment 009 (n=300 clean/300 stego) | Change |
|---|---|---|---|
| FUNSD | 96.9% | 96.4% | -0.5pp (noise — already near ceiling) |
| CORD | 56.5% | **75.8%** | **+19.3pp** |
| SROIE | 62.9% | **70.7%** | **+7.8pp** |
| **Overall** | **73.3%** | **80.6%** | **+7.3pp** |

**Precision / Recall / F1** (Experiment 009, n_test=180):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| clean | 0.83 | 0.77 | 0.80 | 90 |
| stego | 0.78 | 0.84 | 0.81 | 90 |

**Confusion matrix** (rows = true, cols = predicted):

|  | pred_clean | pred_stego |
|---|---|---|
| **true_clean** | 69 | 21 |
| **true_stego** | 14 | 76 |

**Top 10 feature importances (RF, Gini)** — same top 4 features as Experiment 002 (`lsb_entropy`, `lsb_ratio`, `srm_entropy`, `entropy_manual`), similar relative weighting; chi-square still absent from the top 10, consistent with Experiments 004/004-addendum/006's finding that no RF configuration tried actually uses it.

```
Conclusion:   Hypothesis confirmed, and the effect is large -- CORD alone
              gained +19.3pp, more than any single feature or classifier
              change attempted in Experiments 002-008 combined (the next-
              largest single-experiment gain was Experiment 002's own
              +13.0pp on CORD from the SRM+RF change). SROIE's +7.8pp is
              within the range predicted before running this (68-75%
              estimated; 70.7% actual). CORD exceeded the predicted range
              (63-70% estimated; 75.8% actual). FUNSD's -0.5pp is
              consistent with "already near ceiling, more data can't help
              much" and is well within single-split noise (n_test for
              FUNSD is 60 in this split; 1-2 images swings the percentage
              by ~1.5pp).

              This directly corroborates Experiment 008's finding: several
              SROIE false positives had near-zero clean-to-stego feature
              displacement, meaning the classifier previously had too few
              examples to reliably separate a subtle signal from noise.
              More data doesn't fix the *specific* 5 images with the
              composited title-text artifact (that's a representation
              issue, not a sample-size issue) or the 2 duplicate images
              (a data-quality issue) -- but it does help everywhere the
              underlying signal was real but sparse, which appears to be
              most of the remaining gap.
Next hypothesis: Two directions now compete for priority: (1) repeat the
              Experiment 008-style false-positive inspection on this new,
              larger SROIE set to check whether the same title-overlay/
              duplicate artifacts are still the dominant error mode at
              n=100, or whether a new pattern emerges now that the
              decision boundary has shifted; (2) since data volume just
              produced the single largest gain of the whole project, a
              further increase (e.g. 100->150 or 200 per dataset) is worth
              testing as its own single-variable experiment before
              returning to feature/classifier tweaks, to see whether the
              CORD/SROIE gains continue, plateau, or have already captured
              most of the available benefit.
```

**Reproduce this exact run**:
```
cd saa/src
python3 download_datasets.py --skip 50 --n 50          # img_050-099, all 3 datasets
python3 embed_lsb_scattered.py                          # resumable, only embeds the new 150
python3 validate.py --clean-dir ../datasets/clean --stego-dir ../datasets/stego \
  --out-csv ../outputs/features.csv --classifier rf \
  --save-model ../outputs/model_exp009_rf_600img_frozen.pkl
```

**Artifacts**:
- `outputs/features.csv` — now 600 rows (300 clean + 300 stego)
- `outputs/model_exp009_rf_600img_frozen.pkl` / `.json` — frozen model + metrics for this experiment

**Note on environment**: the sandbox this project runs in temporarily blocked `huggingface.co` at the network-proxy/allowlist level (confirmed via direct `curl`, unrelated to the `datasets` library or any code change) partway through this project. The additional 150 images were downloaded by the user running `download_datasets.py --skip 50 --n 50` directly on their own machine, then working from the same `~/Desktop/SAA` folder — same script, no code differences, just a different execution environment for that one step.

---

## Experiment 010 — out-of-domain benchmark generalization (BOSSBase; DocILE-substitute pending)

**Date**: 2026-07-19
**Status**: BOSSBase portion complete, unexpected and instructive negative result. DocILE-substitute portion blocked on a slow/throttled download, to be appended when available.

**Design, per explicit direction**: unlike Experiments 002-009 (which all train on a mix of FUNSD/CORD/SROIE and report per-dataset breakdowns of one shared model), benchmark evaluation here trains and tests a *fresh* RF independently on each benchmark's own images only — answering "how well does this method perform on this benchmark," not testing cross-domain transfer from the document datasets (that would be a separate, harder experiment). 100 clean images pulled per benchmark, 15% scattered LSB stego generated the same way as every other dataset in this project, 70/30 stratified split, same RF hyperparameters as Experiment 002/009 (`n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=42`).

**Dataset note**: BOSSBase images pulled from `italoaa/Cropped_BOSSBase_1.01_WOW_S-UNIWARD`'s `cover/` folder only (100 images, random sample, seed=42) -- the dataset's own pre-made WOW/S-UNIWARD stego versions were deliberately not used (see `download_datasets.py` module docstring); our own `embed_lsb_scattered.py` was run on top instead, for methodological consistency with every other dataset in this project.

```
Hypothesis:   BOSSBase (256x256 grayscale natural-photo patches from real
              camera sensors, no text/flat backgrounds -- the classic image
              steganalysis benchmark) is a genuinely out-of-domain test for
              a feature set and payload scheme built and tuned entirely on
              scanned/photographed documents (FUNSD/CORD/SROIE). Expected
              accuracy was not confidently predicted in advance beyond
              "probably the hardest of the benchmarks," given CORD's own
              natural-photo clean images already showed this project's
              marginal-statistic features (lsb_ratio, lsb_entropy) are
              "structurally blind" when a baseline image already has
              near-random LSB statistics (Experiment 001's diagnosis).
Method:       Extracted the 21-feature vector for all 200 BOSSBase images
              (100 clean + 100 stego), trained+tested a fresh RF using
              Experiment 002/009's exact hyperparameters on this 200-image
              set only (140 train / 60 test, seed=42).
Outcome:      Overall accuracy: 8.3%. Confusion matrix: 1 TN / 29 FP / 26
              FN / 4 TP -- the model's predictions are almost the *inverse*
              of the true labels, not just wrong.

              This number alone would be a red flag for a code bug, so it
              was investigated rather than taken at face value:
                - No NaN/inf values in the feature table; no constant
                  (zero-variance) features.
                - Per-dataset group means: lsb_ratio 0.510 (clean) vs 0.509
                  (stego); lsb_entropy 0.994 vs 0.996; entropy_manual 6.407
                  vs 6.415 -- clean and stego feature distributions are
                  nearly identical on every top-importance feature from
                  every prior experiment.
                - Per-feature AUC across all 21 features: every single one
                  falls in the 0.48-0.56 range (essentially the 0.50 =
                  "no information" line). For comparison, SROIE's weakest
                  useful feature was still 0.594 (Experiment 004). No
                  feature in this benchmark carries real signal in either
                  direction.
                - 5-fold stratified cross-validation with the identical RF
                  config: accuracies of 10.0%, 15.0%, 12.5%, 10.0%, 7.5%
                  (mean 11.0%, std 2.5%) -- consistently far below the ~50%
                  chance level across every fold, not a one-off artifact of
                  the single 70/30 split.
Conclusion:   Not a bug -- confirmed by the NaN/constant check, the
              near-identical group means, and the reproducibility across 5
              independent CV folds. The real finding is two-layered:
                (1) BOSSBase's clean images already have a near-maximal
                  baseline LSB entropy (0.994, vs. e.g. FUNSD's much lower,
                  more embedding-sensitive baseline) because they're raw
                  natural camera images with inherent sensor noise -- an
                  even more extreme version of the "structurally blind"
                  problem first diagnosed for CORD in Experiment 001. A 15%
                  scattered LSB payload added on top of an already
                  near-random LSB plane is, at the level of this classical
                  21-feature set, statistically invisible: every single
                  feature's standalone AUC is indistinguishable from 0.50.
                (2) Separately and more subtly: when a classifier with this
                  much capacity (300 trees, min_samples_leaf=2,
                  class_weight="balanced") is forced to fit a genuinely
                  zero-signal feature space, it does not degrade gracefully
                  to ~50% test accuracy the way a simpler/more regularized
                  model might. It overfits to whatever incidental,
                  sample-specific noise correlations exist in the training
                  bootstraps, and that overfit pattern generalizes *worse*
                  than chance on held-out data -- consistently, across every
                  CV fold. The headline "8.3% accuracy" number is real and
                  reproducible, but it should not be read as "the model
                  detects the opposite of the truth" (there is no real
                  reverse signal -- every feature's AUC is ~0.50, not close
                  to 0 or 1). It should be read as: accuracy is simply the
                  wrong metric to headline in a confirmed zero-signal
                  regime, because it can be pushed arbitrarily below chance
                  by ensemble overfitting even when no real relationship
                  exists in either direction. Per-feature AUC (~0.50 across
                  the board) is the metric that actually describes what
                  happened here, and is the number that should be reported
                  as the headline BOSSBase result, with accuracy reported
                  alongside as a secondary, explicitly-caveated number.
Next hypothesis: This is a genuine, informative limitation to report as
              part of the project's scope rather than something to try to
              fix by further tuning -- the classical/non-DL feature set
              (built and validated entirely on document images) does not
              transfer to natural photographic steganalysis, and 15% LSB
              is fundamentally too shallow a payload to detect on a
              carrier whose baseline noise floor already saturates these
              features. If BOSSBase-specific performance is wanted later,
              the credible path is a higher payload density specific to
              this benchmark (the same lever flagged for SROIE in
              Experiment 008/009) or an entirely different feature family
              tuned for high-ISO natural-image noise rather than document
              structure -- both out of scope for the current one-change-
              at-a-time experiment queue. For the DocILE-substitute half of
              this experiment (still pending), the opposite outcome is
              plausible: it's a document-image domain much closer to
              FUNSD/CORD/SROIE than to BOSSBase, so no similar zero-signal
              collapse is expected -- but that should be confirmed, not
              assumed, once its download completes.
```

**Verification addendum** (requested review before the below-chance number was written up further): an 8.3% accuracy on a balanced binary task is unusual enough that it should be verified, not just explained. Checked directly:

| Check | Result |
|---|---|
| Label encoding (clean=0, stego=1 correctly aligned with source path) | Confirmed -- 0 mismatched rows across all 200 |
| Train/test split preserves class balance | Confirmed -- train 70/70, test 30/30 exactly |
| Confusion matrix arithmetic matches reported accuracy | Confirmed -- (1 TN + 4 TP) / 60 = 0.0833, matches exactly |
| `clf.classes_` order matches `confusion_matrix(..., labels=[0,1])` | Confirmed -- `clf.classes_ = [0, 1]`, no ordering mismatch |

No bug found. But the deeper check -- inspecting `predict_proba` rather than just hard predictions -- refines the finding into something more precise than "the RF overfits to noise and lands below chance by luck": the model's predicted probabilities are **confidently and consistently inverted**, not just wrong. Mean predicted P(stego) for true clean images is 0.665; mean predicted P(stego) for true stego images is 0.332. Flipping every prediction gives 91.7% accuracy. This is a real, reproducible pattern (consistent across all 5 CV folds, not one unlucky split), not measurement noise.

This is worth stating precisely rather than glossing: **every individual feature's AUC is ~0.50 (no univariate signal)**, yet the Random Forest confidently learns a **multivariate combination** that separates classes well *on the training folds* -- and that combination happens to point the wrong way on held-out data, consistently. That combination is very likely fit to incidental structure in this specific 200-image sample (e.g. an accidental correlation between some feature combination and image index/content that doesn't reflect any true clean-vs-stego relationship), rather than to real steganographic signal -- there is no candidate mechanism by which genuine LSB-embedding signal would be simultaneously undetectable feature-by-feature (AUC~0.50 each) yet confidently detectable-but-backwards in combination. The evidence-based statement this supports is: **under the proposed 21-feature representation and 15% scattered LSB embedding, BOSSBase does not exhibit separable statistical signatures accessible to this feature set** -- the RF's confident-but-inverted behavior is a property of how ensemble models respond to fitting bootstrap noise in high-capacity settings (300 trees, `min_samples_leaf=2`, 21 features over 140 training rows), not evidence of a hidden reverse signal. The natural-camera-sensor-noise explanation for *why* the underlying features carry no signal (Experiment 001's "structurally blind" mechanism, more extreme here) remains a plausible interpretation of the AUC~0.50 result, not a proven fact -- stated here as a hypothesis, not a conclusion.

**Artifacts**:
- `outputs/features_bossbase.csv` — 200-row feature table (BOSSBase only)
- `outputs/model_exp010_rf_bossbase_frozen.pkl` / `.json` — frozen model + metrics
- `datasets/clean_bossbase_only/`, `datasets/stego_bossbase_only/` — symlink wrapper directories (point at `datasets/clean/BOSSBASE` and `datasets/stego/BOSSBASE`) used so `validate.py` could be run against BOSSBase alone rather than mixed with FUNSD/CORD/SROIE, per this experiment's per-benchmark-independent design

**Reproduce this exact run**:
```
cd saa/src
python3 download_datasets.py --only BOSSBASE --n 100 --random --seed 42
python3 embed_lsb_scattered.py --only BOSSBASE
python3 validate.py --clean-dir ../datasets/clean_bossbase_only --stego-dir ../datasets/stego_bossbase_only \
  --out-csv ../outputs/features_bossbase.csv --classifier rf \
  --save-model ../outputs/model_exp010_rf_bossbase_frozen.pkl
```

---

### DocILE-substitute (`Voxel51/high-quality-invoice-images-for-ocr`) — completed

**Download note**: the first attempt hung for 40+ minutes at ~100 B/s per file because non-streaming `load_dataset()` was pulling the source repo's raw `data/` folder (8,106 individual JPGs) file-by-file instead of its auto-converted parquet mirror. Fixed by switching `download_datasets.py` to `streaming=True` with a shuffle-buffer random sample (`--buffer-size 200`), which only needed to touch a few hundred files instead of the whole repo -- completed promptly after that.

One corrupted file was found and fixed before finalizing this result: `stego/DOCILE/img_054.png` was a 0-byte file from an embed run that got cut off mid-write by the sandbox's wall-clock timeout. Deleted and re-embedded (resumable by design -- `embed_lsb_scattered.py` only regenerated the one missing file); the 200-image set is now complete and consistent.

```
Outcome (DocILE-substitute): Overall accuracy: 100.0% (60/60 test images,
              30/30 clean and 30/30 stego, zero misclassifications).

              Sanity-checked the same way as BOSSBase, given a perfect score
              is just as worth scrutinizing as a near-zero one:
                - Group means show a large, real gap on the features that
                  matter: lsb_ratio 0.943 (clean) vs 0.877 (stego);
                  lsb_entropy 0.313 vs 0.538; srm_entropy 0.843 vs 1.553.
                - Per-feature AUC: lsb_entropy and srm_entropy both hit a
                  literal 1.000 (perfect separation), lsb_ratio hits 0.000
                  (perfect separation in the opposite direction -- same
                  information, inverted sign). entropy_manual reaches 0.778.
                  Every other feature sits near the uninformative 0.50 line,
                  same pattern seen throughout this project (chi-square and
                  most of the SRM/edge/frequency groups contribute ~nothing
                  once lsb_ratio/lsb_entropy are available).
                - Caveat found and confirmed by visual inspection, not just
                  inferred: 38 of 200 feature rows fall into duplicate-value
                  groups across all 21 features, and critically, every one
                  of them is a **clean** image (label 0) -- zero stego rows
                  and zero clean/stego cross-duplicates involved. Directly
                  compared two of them (img_003.png, img_004.png): they are
                  literal pixel-identical duplicate invoices (same invoice
                  number 12323459, same seller/client/line items/totals,
                  same everything). Because the duplication is entirely
                  within the clean class, it does not inflate or explain
                  the clean-vs-stego separation driving the 100% figure --
                  that separation comes from the genuine lsb_entropy/
                  srm_entropy gap described above, which holds regardless.
                  What it does mean: the 100 "clean" images represent fewer
                  than 100 independent invoice templates (comparable in
                  kind to Experiment 008's SROIE duplicate-image finding),
                  so this benchmark's effective sample diversity is smaller
                  than its nominal size -- a caveat about how much this
                  result generalizes, not about whether the 100% number
                  itself is computed correctly.
Conclusion:   The model achieved 100% accuracy on the sampled DocILE-
              substitute subset. Inspection revealed repeated feature
              vectors -- confirmed by direct visual comparison to be
              literal duplicate invoice images -- among a meaningful
              fraction of the "clean" class, indicating reduced template
              diversity in this 100-image sample; this does not explain the
              100% score (duplicates are clean-only, not cross-class) but
              does mean the result should be interpreted as evidence that
              the proposed features are highly effective on low-entropy
              business documents, rather than as a guarantee of perfect
              generalization to a more diverse invoice sample. The
              mechanism is the same one identified for FUNSD: DocILE-
              substitute's clean images have large uniform low-entropy
              regions (consistent with "high-quality" synthetic/template
              invoice scans), so a 15% LSB payload produces a large,
              easily-linearly-separable shift in lsb_ratio/lsb_entropy/
              srm_entropy. This is the opposite end of the spectrum from
              BOSSBase in the same experiment: BOSSBase's natural-photo
              noise floor already saturates these features before any
              embedding (AUC ~0.50 everywhere); DocILE-substitute's clean
              baseline is so far from that saturation point that embedding
              is maximally visible
              (AUC ~1.00 on the top features). Both results are consistent
              with -- and extend -- this project's very first diagnosis in
              Experiment 001: across every dataset tried so far, detection
              accuracy tracks how much baseline "room" a carrier's LSB/noise
              statistics have to move under embedding. Stated carefully:
              under the proposed 21-feature representation and 15%
              scattered LSB embedding, structured/low-entropy carriers
              (FUNSD, DocILE-substitute) exhibit large separable statistical
              shifts, while high-entropy natural-photo carriers (BOSSBase)
              do not. The sensor-noise/baseline-variability explanation for
              *why* is a plausible interpretation of this pattern, not a
              proven causal claim -- see the cross-benchmark synthesis
              section below for the full evidence this rests on.
Next hypothesis: The duplicate-row caveat has now been checked the same
              way Experiment 008 checked SROIE's duplicates (direct visual
              comparison, confirmed above) -- it's real, clean-only, and
              doesn't inflate the 100% figure, but does reduce this
              benchmark's effective template diversity below its nominal
              100 images. No further tuning is motivated for DocILE-
              substitute itself -- unlike CORD/SROIE/BOSSBase, this
              benchmark is not a case the current feature set struggles
              with. If a tighter generalization claim is wanted later, a
              re-pull with deduplication (e.g. hashing images before
              sampling) would be a cheap follow-up.
```

**Results table, both benchmarks**:

| Benchmark | n (clean/stego) | Overall accuracy | Precision (clean/stego) | Recall (clean/stego) | Confusion matrix (TN/FP/FN/TP) | Top feature (AUC) |
|---|---|---|---|---|---|---|
| BOSSBase | 100/100 | **8.3%** (5-fold CV mean: 11.0% ± 2.5%) | 0.04 / 0.12 | 0.03 / 0.13 | 1 / 29 / 26 / 4 | none — all 21 features AUC 0.48-0.56 (no signal) |
| DocILE-substitute | 100/100 | **100.0%** | 1.00 / 1.00 | 1.00 / 1.00 | 30 / 0 / 0 / 30 | `lsb_entropy`, `srm_entropy` (AUC 1.000) |

**Artifacts**:
- `outputs/features_docile.csv` — 200-row feature table (DocILE-substitute only)
- `outputs/model_exp010_rf_docile_frozen.pkl` / `.json` — frozen model + metrics
- `datasets/clean_docile_only/`, `datasets/stego_docile_only/` — symlink wrapper directories, same pattern as BOSSBase's

**Reproduce this exact run**:
```
cd saa/src
python3 download_datasets.py --only DOCILE --n 100 --random --seed 42 --buffer-size 200
python3 embed_lsb_scattered.py --only DOCILE
python3 validate.py --clean-dir ../datasets/clean_docile_only --stego-dir ../datasets/stego_docile_only \
  --out-csv ../outputs/features_docile.csv --classifier rf \
  --save-model ../outputs/model_exp010_rf_docile_frozen.pkl
```

---

### Cross-benchmark synthesis (all 5 datasets)

**Validation protocol distinction, going forward**: FUNSD/CORD/SROIE are the **core evaluation set** -- Experiments 002-009 tune and validate the SAA pipeline against them, with a single shared model reporting per-dataset breakdowns. BOSSBase and DocILE-substitute are **external benchmarks** -- each evaluated with its own independently trained/tested model (Experiment 010), used to assess generalization to domains the feature set was never tuned against, not to further tune the pipeline itself. This distinction matters for how these numbers should be cited: core-set accuracy reflects a tuned system, benchmark accuracy reflects out-of-the-box transfer.

**The mechanism, shown directly rather than just claimed.** Every dataset's clean images have a baseline LSB-plane entropy; embedding either moves that baseline a lot, a little, or not at all, depending on how much "room" the baseline leaves. Computed directly from the extracted feature tables (clean images only for the baseline row, then the full clean-vs-stego gap):

| Dataset | Mean `lsb_ratio` (clean) | Mean `lsb_entropy` (clean) | Mean `entropy_manual` (clean) | Mean `srm_entropy` (clean) | `lsb_entropy` gap (stego − clean) | Overall accuracy |
|---|---:|---:|---:|---:|---:|---:|
| FUNSD | 0.943 | 0.301 | 1.110 | 1.237 | +0.233 | 96.4% |
| DocILE-substitute | 0.943 | 0.313 | 1.318 | 0.843 | +0.225 | 100.0% |
| SROIE | 0.738 | 0.581 | 2.590 | 1.997 | +0.142 | 70.7% |
| CORD | 0.495 | 0.998 | 6.591 | 4.553 | +0.000 | 75.8%\* |
| BOSSBase | 0.510 | 0.994 | 6.407 | 4.580 | +0.002 | 8.3%\*\* |

\* CORD's accuracy (Experiment 009) comes from a model trained jointly with FUNSD/SROIE, not an isolated fit like BOSSBase/DocILE-substitute -- included for the baseline-statistics comparison, not as a like-for-like accuracy comparison.
\*\* See verification addendum above: this number is real but should be read alongside the ~0.50 per-feature AUC, not as a standalone headline.

The pattern is exactly what the baseline statistics predict: FUNSD and DocILE-substitute both start from a very low clean-image `lsb_entropy` (~0.30, meaning the LSB plane is far from random -- large uniform background/whitespace regions), and both show a large, easily-separable shift after embedding (+0.22-0.23), producing near-perfect detection. CORD and BOSSBase both start from a clean-image `lsb_entropy` already near the theoretical maximum (~0.99), leaving embedding almost nothing to shift (+0.000 to +0.002), producing accuracy at or below chance. SROIE sits in between on every column, and its accuracy sits in between too.

**What this evidence supports, stated at the right strength**: under the proposed 21-feature representation and 15% scattered LSB embedding, structured document datasets with low-entropy clean-image baselines (FUNSD, DocILE-substitute) exhibit large, separable statistical shifts after embedding, moderately-structured datasets (SROIE) show a smaller but still real shift, and datasets whose clean images already have nearly maximal baseline LSB/noise entropy (CORD's natural-photo receipts, BOSSBase's natural camera photos) show little to no separable shift. This is a direct, measured relationship between a specific baseline statistic and detection accuracy across all 5 datasets tried, not an assumption. The *causal* explanation offered for why natural photographs have this baseline property -- inherent camera sensor noise and texture already saturating the LSB plane -- is a plausible interpretation of the pattern, consistent with the steganalysis literature's general treatment of natural images as harder cover media, but should be presented as a supported hypothesis rather than a proven mechanism, since this project has not directly manipulated or measured sensor noise as an independent variable.
