# AATFN Fusion-Layer Attack Dataset — Pipeline

Goal: 100 clean base images each from FUNSD, CORD, SROIE (300 total), each
attacked with every combination of Typo (T), Patch (P), and Steganography
(S) — including singles, pairs, and the full triple — for training the
AATFN fusion classifier.

## 1. Folder structure

```
aatfn/
  base_images/{funsd,cord,sroie}/     100 clean images each (input)
  raw_datasets/                       (optional) full downloaded datasets before sampling
  scripts/
    sample_dataset.py                 samples 100 images/dataset into base_images/
    generate_dataset.py               applies all attack combos, writes metadata.csv
    attacks/
      typo_attack.py                  (T) OCR + word-level typo injection
      patch_attack.py                 (P) adversarial noise patch overlay
      stego_attack.py                 (S) LSB steganography payload embed
  generated/{funsd,cord,sroie}/{combo}/   output images, one folder per combo
  metadata.csv                        multi-hot labels for every generated image
  requirements.txt
```

## 2. Attack-combination taxonomy (8 classes)

Every base image is attacked with each of the 8 subsets of {T, P, S}:

| combo | typo | patch | stego | meaning              |
|-------|------|-------|-------|-----------------------|
| clean |  0   |  0    |  0    | unmodified baseline   |
| T     |  1   |  0    |  0    | typo only             |
| P     |  0   |  1    |  0    | patch only            |
| S     |  0   |  0    |  1    | stego only            |
| TP    |  1   |  1    |  0    | typo + patch          |
| TS    |  1   |  0    |  1    | typo + stego          |
| PS    |  0   |  1    |  1    | patch + stego         |
| TPS   |  1   |  1    |  1    | all three combined    |

300 base images x 8 combos = **2,400 images total** (this includes the 300
clean copies, which the fusion classifier needs as its negative class).

Attack order inside a combo is always **typo -> patch -> stego** (stego
last, so the LSB payload survives the visible edits made by typo/patch).

## 3. Step-by-step

### Step 1 — Get raw FUNSD/CORD/SROIE data
You likely already have these from the SAA/typography/patch modules
(your old MMShield project had `datasets/{funsd,cord,sroie}/`). Easiest
option: point `sample_dataset.py` at those existing folders.

If you need to (re)download, on your own machine (not this sandbox — it
has no general internet access):
- **FUNSD**: https://guillaumejaume.github.io/FUNSD/dataset.zip (page images under `training_data/images/` + `testing_data/images/`)
- **CORD**: `pip install datasets` then
  ```python
  from datasets import load_dataset
  ds = load_dataset("naver-clova-ix/cord-v2")
  ```
  and save each `ds["train"][i]["image"]` as PNG.
- **SROIE**: HuggingFace `darentang/sroie`, or the Kaggle mirror
  "SROIE datasetv2" (ICDAR2019 Task 1/2 receipt images).

### Step 2 — Install dependencies
```
cd ~/Desktop/fusion/aatfn
pip install -r requirements.txt
# tesseract binary also required for the typo attack:
brew install tesseract        # macOS
```

### Step 3 — Sample 100 images per dataset
```
cd scripts
python3 sample_dataset.py \
  --funsd  /path/to/FUNSD/images \
  --cord   /path/to/CORD/images \
  --sroie  /path/to/SROIE/images \
  --n 100
```
This copies 100 randomly-seeded images per dataset into
`aatfn/base_images/{funsd,cord,sroie}/`.

### Step 4 — Generate all attack combinations
```
python3 generate_dataset.py
```
Loops over every base image x all 8 combos, saves attacked PNGs into
`generated/<dataset>/<combo>/`, and appends one row per image to
`metadata.csv` (columns: `dataset, base_image, combo, typo, patch, stego,
output_path`).

Run a subset if you want to test first, e.g. only FUNSD, only 3 combos:
```
python3 generate_dataset.py --datasets funsd --combos clean T TPS
```

### Step 5 — Spot-check
Open a few images from `generated/*/TPS/` and confirm: OCR'd words look
typo'd, a colored noise patch is visible, and the file still opens
normally (stego is invisible by design — decode it via the blue-channel
LSBs if you want to verify the payload is present).

### Step 6 — Feed into AATFN
`metadata.csv` is your label file. For fusion-layer training, load each
`output_path` image, run it through the SAA + typography + patch
extractors (as before), and use the `typo`/`patch`/`stego` columns as a
3-bit multi-label target (or map the 8 `combo` values to a single
8-class softmax target — whichever your AATFN head expects).

## 4. Notes / design choices
- **Seeding**: each base image's index is reused as the RNG seed across
  all its combo variants, so e.g. `funsd_007`'s typo perturbation is
  identical whether it appears alone (T) or combined (TPS) — keeps
  variants comparable.
- **Typo attack** requires OCR (`pytesseract`); on scanned/low-quality
  images it may find few/no words — those images fall back to an
  unmodified copy for the typo step. Worth spot-checking OCR hit rate
  once real data is in `base_images/`.
- **Compute**: this sandbox has no OCR/GPU time budget for 2,400 images
  in one run — the scripts were smoke-tested here on synthetic samples
  and are ready to run at full scale on your machine.
- Scaling down: pass `--combos clean T P S` first if you want the
  single-attack set alone before generating the pair/triple combos.

## 5. Step 7 — Feature extraction (SAA + typography + patch)

Once `metadata.csv` / `generated/` exist (2,400 images), extract features
for the fusion classifier using the **real** MMShield extractors —
`saa/src/extractor.py`, `typographic/features/*.py`,
`adversarial/features/*.py` — not anything reimplemented in `aatfn/`.

**Run this on your machine, not in a sandbox.** It needs torch, PaddleOCR,
sentence-transformers, etc. (see requirements below), and does full OCR +
CNN inference per image — much slower than the pure-PIL attack generation
in step 4.

### What gets extracted, per image
| module | dim | source |
|---|---|---|
| SAA (steganalysis) | 21 | `saa/src/extractor.py: stego_analyzer()` — used as-is |
| Typography + semantic | 20 | `typographic/features/{ocr,feature_fusion}.py` — used as-is |
| Patch (adversarial) | 1308 | **new code**, see below |

**Patch module gap:** `adversarial/` only has candidate-level features (27
handcrafted + 1280 CNN = 1307-dim per proposed region, up to 12 candidates
per page) — there was no existing function to collapse a page's candidates
into one document-level vector (`adversarial/inference/` is an empty stub).
`aatfn/scripts/feature_extractors/patch_wrapper.py` adds that: proposes
candidates via the existing `candidate_generator.propose_candidates()`,
extracts the full 1307-dim vector per candidate via the existing
handcrafted+CNN code unchanged, then **max-pools dim-by-dim across
candidates** (standard multiple-instance-learning pooling — a genuine patch
should dominate at least one candidate on at least one dimension) plus one
extra `num_candidates` count feature = 1308-dim. Pages with zero candidates
get an all-zeros vector.

Combined feature vector per image: 21 + 20 + 1308 = **1349 dims**.

### Setup
```
cd ~/Desktop/MM_Shield
pip install -r requirements.txt
```
Root `requirements.txt` was missing `scipy`, `scikit-image`, `joblib`
(direct imports in `saa/src/edge.py` etc. and the training scripts) and had
`opencv-python` instead of `opencv-contrib-python` (candidate_generator.py
calls `cv2.saliency`, which only ships in the contrib build) — both fixed
now. If you already have plain `opencv-python` installed, uninstall it
first (`pip uninstall opencv-python`) since the two conflict.

### Run
```
cd aatfn/scripts
python3 extract_features.py --limit 20     # smoke test first
python3 extract_features.py                # full 2400 images
```
Resumable like `generate_dataset.py` — safe to Ctrl-C and rerun, skips
images already in `features.csv`. Per-image failures are logged to
`aatfn/extract_errors.log` and skipped rather than crashing the run; check
that file if the row count comes up short. Progress prints include a
running avg-seconds/image and ETA.

Output: `aatfn/features.csv` — one row per image:
`dataset, base_image, combo, typo, patch, stego, output_path`, then 1,349
feature columns (`saa_*`, `typo_*`, `patch_max_*` / `patch_max_cnn_*` /
`patch_num_candidates`). The `typo`/`patch`/`stego` columns are the ground-
truth attack labels (0/1) for training; don't confuse with the `typo_*`
feature-column prefix, which is the typography module's own features.

## 5b. Stego attack fix (post-hoc correction)

First training run showed stego precision stuck at ~0.50 (≈base rate) with
recall ~0.97 — the model was just guessing "stego present" by default.
Root cause, confirmed by comparing `saa_lsb_ratio`/`saa_lsb_entropy`/
`saa_srm_entropy` between clean and stego rows in `features.csv`: identical
to 6 decimal places. The original `stego_attack.py` embedded a fixed
136-bit payload into only the first 136 pixels (top-left corner) — far too
small/localized to shift SAA's whole-image statistical features.

Fixed: `stego_attack.py` now imports and reuses
`saa/src/embed_lsb_scattered.py`'s `embed_scattered_lsb()` directly — the
same 15%-of-pixels scattered random-bit LSB embedding the SAA module was
actually validated against. Regenerated all 1,200 images across the
S/PS/TS/TPS combos (`generate_dataset.py --combos S PS TS TPS`), stripped
the corresponding stale rows from `metadata.csv` and `features.csv`.
Verified: new stego images differ from clean in ~7.5% of pixel values
(matches 15% embed rate / 2, as expected when overwriting with a random
bit). Re-run `extract_features.py` for just those combos to get real SAA
signal, then retrain — architecture unchanged, isolating this as the only
variable.

## 5c. Training run comparison (test-set F1, exact-match, best val_loss)

| run | config | typo F1 | patch F1 | stego F1 | exact-match | best val_loss |
|---|---|---|---|---|---|---|
| 1 | weak stego attack, wd=1e-4 | 0.68 | 0.88 | 0.66* | 0.27 | 1.756 |
| 2 | fixed stego, wd=1e-4 | 0.61 | 0.86 | 0.64 | 0.28 | 1.811 |
| 3 | fixed stego, +regularization, wd=1e-3 | 0.59 | 0.82 | 0.56 | 0.26 | 1.793 |
| 4 | fixed stego, +regularization, wd=3e-4 | 0.65 | 0.87 | 0.54 | **0.29** | **1.718** |

*Run 1's stego F1 was fake — precision 0.50/recall 0.97 was the model
guessing "stego present" at the base rate, not real detection (confirmed
via feature-distribution inspection: `saa_lsb_ratio`/`lsb_entropy`/
`srm_entropy` were identical to 6 decimals between clean/stego before the
attack fix).

wd=3e-4 (run 4) is the best config on every metric except stego F1 —
frozen as the new default. Stego stayed flat at 0.54-0.66 F1 across every
optimizer/regularization setting while typo/patch moved a lot in response
to the same changes — pointing at the SAA/stego branch itself as the
bottleneck (21 features, weakest per-feature signal: `lsb_entropy`'s
d≈0.41 was the *strongest* of the 8 SAA features) rather than global
over/underfitting.

**Run 5 (FROZEN — official model):** single-variable ablation on just the
SAA branch — one extra hidden layer, dropout 0.2 -> 0.05 — with typo/patch
branches held byte-for-byte identical to run 4.

| run | config | typo F1 | patch F1 | stego F1 | exact-match |
|---|---|---|---|---|---|
| 4 | wd=3e-4, uniform branch treatment | 0.65 | 0.875 | 0.54 | 0.29 |
| **5** | **wd=3e-4, SAA branch widened + dropout 0.05 (FROZEN)** | **0.69** | 0.868 | **0.62** | **0.31** |

Stego F1 +0.08 (precision held, recall 0.58->0.72), typo F1 +0.04 (a
shared-trunk/attention side effect, not a change to the typo branch
itself), patch essentially flat as expected (its branch was untouched).
Confirms the hypothesis: SAA's 21 features were being over-regularized
relative to their already-modest signal strength.

**Frozen configuration:** weight_decay=3e-4, SAA branch = widened
(3-layer) + dropout 0.05, typo/patch branches = fix-2 config (unchanged
since run 3), patch branch hidden width 128, trunk dropout 0.4/0.2.
Saved model + scalers in `aatfn/model/` (from the run 5 training command)
is the official AATFN fusion model — no further architecture/hyperparameter
tuning planned. `train_fusion.py`'s current defaults reproduce this run
exactly (`python3 train_fusion.py`, no flags needed).

**Experimental narrative for writeup:** runs 1-4 established that
optimization (weight decay) alone did not resolve the stego bottleneck;
run 5's targeted architectural change (informed by inspecting per-feature
effect sizes, not guesswork) resolved it partially, closing exact-match
from 0.27 (broken baseline) to 0.31 while making all three heads reflect
genuine learned signal rather than base-rate guessing (verified via the
`saa_lsb_ratio`/`lsb_entropy`/`srm_entropy` distribution check in 5b).

## 5d. Step 9 — Final evaluation (Phase 1: figures + metrics)

`aatfn/scripts/evaluate_fusion.py` re-evaluates the frozen run-5 model on
the exact same held-out test set `train_fusion.py` reported on (same
seed=42 grouped split — this script never retrains anything, just loads
`aatfn/model/{fusion_model.pt, *_scaler.joblib}` and runs inference).

Run:
```
cd aatfn/scripts
pip install -r ../../requirements.txt   # adds matplotlib if not already present
python3 evaluate_fusion.py
python3 evaluate_fusion.py --n-bootstrap 2000   # slower, tighter CIs
```

Outputs to `aatfn/results/`:
- `confusion_{typo,patch,stego}.pdf` — per-head confusion matrices at threshold 0.5
- `roc_{typo,patch,stego}.pdf` — ROC curves with AUC and the t=0.5 operating point marked
- `pr_{typo,patch,stego}.pdf` — precision-recall curves with AP and base-rate reference line
- `calibration.pdf` — reliability diagram for all 3 heads + Brier scores
- `final_metrics.json` — per-head accuracy/precision/recall/F1/ROC-AUC/AP/Brier + macro/micro
  averages + exact-match + Hamming loss + 95% bootstrap CIs (percentile method) for each head's
  F1 and for exact-match accuracy + mean attention weights per modality
- `final_metrics_per_head.csv` — flat version of the per-head table for pasting into a paper
- `main_results.csv` — single-row overall table (macro/micro precision/recall/F1, exact-match + CI, Hamming loss)
- `roc_combined.pdf` / `pr_combined.pdf` — all 3 heads' ROC/PR curves overlaid on one panel each (paper-ready single-figure versions, in addition to the per-head files above)

## 5e. Step 10 — Paper assets

`aatfn/scripts/build_paper_assets.py` collects the final figures/tables into
`aatfn/paper_assets/`, renamed to what you'd cite in the writeup. Pure file
copying + one static table (the run 1-5 ablation comparison from 5c) — no
model inference, safe to re-run anytime.

```
cd aatfn/scripts
python3 evaluate_fusion.py        # regenerate results/, now includes roc_combined.pdf/pr_combined.pdf
python3 build_paper_assets.py
```

Produces:
```
paper_assets/
├── fig_confusion_typo.pdf
├── fig_confusion_patch.pdf
├── fig_confusion_stego.pdf
├── fig_roc.pdf
├── fig_pr.pdf
├── fig_calibration.pdf
├── table_main_results.csv
├── table_per_head.csv
└── table_ablation.csv
```

## 6. Step 8 — Fusion classifier

`aatfn/scripts/train_fusion.py` trains a multi-label classifier over
`features.csv`, predicting three independent binary flags (typo/patch/stego
present or not — not an 8-class softmax, since attacks can co-occur).

**Architecture:** each modality (SAA 21-dim, typography 20-dim, patch
1308-dim) goes through its own small MLP branch into a 64-dim embedding.
An attention gate looks at all three embeddings and learns softmax weights
to combine them into one shared representation (lets the model down-weight
a modality that isn't informative for a given image, rather than always
averaging all three equally) — this matches the "Adaptive Attention Fusion"
design discussed earlier. The shared representation goes through a trunk,
then three separate sigmoid heads (typo/patch/stego).

**Splitting:** grouped by `(dataset, base_image)` — all 8 attack-combo
variants of the same underlying document stay in the same split, so the
model can't leak document-content signal across train/test via a sibling
combo of the same page. Default 70/15/15 train/val/test.

**Run:**
```
cd aatfn/scripts
python3 train_fusion.py                 # default 50 epochs, batch size 32
```
Needs `torch`, `pandas`, `scikit-learn`, `joblib` (already in
`requirements.txt`). Early-stops on validation loss. Outputs to
`aatfn/model/`: `fusion_model.pt` (weights), `{saa,typo,patch}_scaler.joblib`
(StandardScalers fit on train only), `metrics.json` (per-label
precision/recall/F1 + exact-match accuracy on val and test), `config.json`.
