"""
Step 3: extract SAA + typography + patch features for every image in
metadata.csv (the 2,400-image attack-combo dataset from generate_dataset.py)
and write aatfn/features.csv -- one row per image, with:

  dataset, base_image, combo, typo, patch, stego, output_path,
  saa_0..saa_20 (21 dims),
  typo_0..typo_19 (20 dims),
  patch_max_<name> / patch_max_cnn_<i> / patch_num_candidates (1308 dims)

typo/patch/stego here are the ATTACK labels from metadata.csv (0/1 ground
truth for the fusion classifier to predict) -- not to be confused with the
"typo_*" feature-column prefix, which is the TYPOGRAPHY MODULE's features.

Run this on your machine (needs torch/paddleocr/sentence-transformers/etc --
see aatfn/requirements_extraction.txt), NOT in a constrained sandbox: OCR +
CNN embeddings over 2,400 images is slow and this is not chunked for a
45-second budget the way generate_dataset.py was.

Safe to interrupt and resume: skips any output_path already present in
features.csv, flushes each row immediately.

Usage:
    cd aatfn/scripts
    python3 extract_features.py                      # all 2400 images
    python3 extract_features.py --limit 20            # smoke-test on 20 images
    python3 extract_features.py --datasets funsd       # subset
"""
import argparse
import csv
import time
import traceback
from pathlib import Path

from feature_extractors.paths import AATFN_DIR
from feature_extractors.patch_wrapper import PATCH_FEATURE_NAMES, extract_patch_features
from feature_extractors.saa_wrapper import SAA_FEATURE_NAMES, extract_saa_features
from feature_extractors.typography_wrapper import TYPOGRAPHY_FEATURE_NAMES, extract_typography_features

METADATA_PATH = AATFN_DIR / "metadata.csv"
FEATURES_PATH = AATFN_DIR / "features.csv"
ERRORS_PATH = AATFN_DIR / "extract_errors.log"

SAA_COLS = [f"saa_{n}" for n in SAA_FEATURE_NAMES]
TYPO_COLS = [f"typo_{n}" for n in TYPOGRAPHY_FEATURE_NAMES]
PATCH_COLS = PATCH_FEATURE_NAMES
FEATURE_COLS = SAA_COLS + TYPO_COLS + PATCH_COLS
META_COLS = ["dataset", "base_image", "combo", "typo", "patch", "stego", "output_path"]
ALL_COLS = META_COLS + FEATURE_COLS


def _load_done_set():
    done = set()
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["output_path"])
    return done


def extract_one(image_path: Path):
    saa_vec = extract_saa_features(str(image_path))
    typo_vec = extract_typography_features(str(image_path))
    patch_vec = extract_patch_features(str(image_path))
    return list(saa_vec) + list(typo_vec) + list(patch_vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None, help="filter to these datasets, default = all")
    ap.add_argument("--combos", nargs="+", default=None, help="filter to these combos, default = all")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N matching rows (smoke test)")
    args = ap.parse_args()

    with open(METADATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.datasets:
        rows = [r for r in rows if r["dataset"] in args.datasets]
    if args.combos:
        rows = [r for r in rows if r["combo"] in args.combos]
    if args.limit:
        rows = rows[: args.limit]

    done = _load_done_set()
    todo = [r for r in rows if r["output_path"] not in done]
    print(f"{len(rows)} rows match filters, {len(done)} already done, {len(todo)} to process")

    write_header = not FEATURES_PATH.exists()
    feat_f = open(FEATURES_PATH, "a", newline="")
    writer = csv.DictWriter(feat_f, fieldnames=ALL_COLS)
    if write_header:
        writer.writeheader()
        feat_f.flush()

    err_f = open(ERRORS_PATH, "a")

    t_start = time.time()
    for i, row in enumerate(todo, start=1):
        image_path = AATFN_DIR / row["output_path"]
        try:
            t0 = time.time()
            features = extract_one(image_path)
            elapsed = time.time() - t0

            out_row = {k: row[k] for k in META_COLS}
            out_row.update(dict(zip(FEATURE_COLS, features)))
            writer.writerow(out_row)
            feat_f.flush()

            if i % 10 == 0 or i == len(todo):
                avg = (time.time() - t_start) / i
                remaining = (len(todo) - i) * avg
                print(f"[{i}/{len(todo)}] {row['output_path']} ({elapsed:.2f}s) "
                      f"-- avg {avg:.2f}s/img, ~{remaining/60:.1f} min remaining")
        except Exception:
            err_f.write(f"{row['output_path']}\n{traceback.format_exc()}\n---\n")
            err_f.flush()
            print(f"[{i}/{len(todo)}] FAILED {row['output_path']} -- see {ERRORS_PATH}")

    feat_f.close()
    err_f.close()
    print(f"\nDone. features.csv -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
