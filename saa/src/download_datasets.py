"""
Pull clean document/benchmark images into SAA/datasets/clean/<dataset_name>/img_###.png.

Dataset sources (verified available on the HF Hub, parquet-native -- no
legacy "dataset script" loading required, which recent `datasets` versions
(v4+) dropped support for):
  FUNSD    -> nielsr/funsd-layoutlmv3   (subset "funsd"; `image` column, PIL-decoded)
  CORD     -> naver-clova-ix/cord-v2    (`image` column, PIL-decoded; donut-format
              `ground_truth` column ignored)
  SROIE    -> Voxel51/scanned_receipts  (`image` column, PIL-decoded; imagefolder
              mirror of ICDAR-SROIE -- see SROIE spec comment below for why the
              original darentang/sroie doesn't work)
  BOSSBASE -> italoaa/Cropped_BOSSBase_1.01_WOW_S-UNIWARD, cover/ subfolder only
              (256x256 grayscale natural-photo patches -- the classic image
              steganalysis benchmark, included as an out-of-domain generalization
              check since FUNSD/CORD/SROIE are all document images). Only the
              `cover/` (clean) images are pulled -- the dataset's own pre-made
              WOW/S-UNIWARD stego versions are deliberately NOT used, since mixing
              in a different embedding algorithm would confound "does the SAA
              pipeline generalize to a new image domain" with "does it detect a
              different steganography algorithm." Our own embed_lsb_scattered.py
              (same 15% scattered-LSB scheme used everywhere else in this project)
              is applied on top instead, for methodological consistency.
  DOCILE   -> Voxel51/high-quality-invoice-images-for-ocr ("image" column,
              imagefolder). Substitute for the DocILE benchmark: DocILE itself
              (Simsa et al., 2023) has no plain-image mirror on the HF Hub as of
              this writing (its real distribution is a multi-gigabyte annotated
              corpus with its own custom loader, not a drop-in imagefolder
              dataset) -- documented explicitly here per project convention
              (compare the LightGBM-for-XGBoost substitution in EXPERIMENTS.md
              Experiment 007). This is a different, but structurally similar,
              invoice/financial-document image set, serving the same role: an
              out-of-domain financial-document generalization check.

If any of these dataset ids move/rename upstream, update DATASET_SPECS below
-- everything downstream (embed_lsb_scattered.py, validate.py) only cares
about the resulting datasets/clean/<name>/*.png files, not the HF ids.

Usage:
    python download_datasets.py                          # 50 images per dataset (default), FUNSD/CORD/SROIE
    python download_datasets.py --n 10                    # smaller smoke-test pull
    python download_datasets.py --only BOSSBASE,DOCILE --n 100 --random --seed 42
"""
import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets" / "clean"

DATASET_SPECS = {
    "FUNSD": {
        "hf_id": "nielsr/funsd-layoutlmv3",
        "candidate_configs": ["funsd", None],
        "candidate_splits": ["train", "test"],
        "image_col": "image",
    },
    "CORD": {
        "hf_id": "naver-clova-ix/cord-v2",
        "candidate_configs": [None],
        "candidate_splits": ["train", "validation", "test"],
        "image_col": "image",
    },
    "SROIE": {
        # darentang/sroie ships only a legacy loading script (datasets>=4.0
        # refuses to run it), and its auto-converted parquet mirror embeds an
        # `image_path` string pointing at HF's own build server cache dir --
        # not a path that exists on any other machine, so that's a dead end
        # too. Voxel51/scanned_receipts is a genuine imagefolder-format
        # mirror of the same ICDAR-SROIE scanned receipts (713 of the 973
        # rows, full-page images) with a real `image` column.
        "hf_id": "Voxel51/scanned_receipts",
        "candidate_configs": [None],
        "candidate_splits": ["train"],
        "image_col": "image",
    },
    "BOSSBASE": {
        # cover/ only -- see module docstring. data_files restricts the
        # imagefolder loader to just that subfolder so we never accidentally
        # pull a stego (WOW/S-UNIWARD) patch as if it were clean.
        # NOTE: the dataset card's directory diagram (cropdataset/cover/...,
        # .png) does not match the actual repo layout -- verified via the
        # Hub tree API: files are flat under cover/ (no cropdataset/ prefix)
        # and are .pgm (portable graymap), not .png.
        "hf_id": "italoaa/Cropped_BOSSBase_1.01_WOW_S-UNIWARD",
        "candidate_configs": [None],
        "candidate_splits": ["train"],
        "image_col": "image",
        "data_files": "cover/*.pgm",
    },
    "DOCILE": {
        # substitute for DocILE -- see module docstring.
        "hf_id": "Voxel51/high-quality-invoice-images-for-ocr",
        "candidate_configs": [None],
        "candidate_splits": ["train"],
        "image_col": "image",
    },
}


def _first_available_split(hf_id: str, candidate_configs, candidate_splits, candidate_revisions=(None,),
                            data_files=None, streaming=True):
    """
    streaming=True (default): uses datasets' IterableDataset mode, which
    fetches shards/files lazily as they're iterated instead of downloading
    the entire matching file set up front. This matters a lot for datasets
    where we only want a small sample -- e.g. BOSSBASE's data_files glob
    matches all 20,000 cover/*.pgm files, and a non-streaming load_dataset()
    call downloads every one of them before you can take even a single row.
    Streaming mode only pulls as many files/row-groups as needed to satisfy
    however many rows the caller actually iterates.
    """
    last_err = None
    for revision in candidate_revisions:
        for config in candidate_configs:
            for split in candidate_splits:
                try:
                    kwargs = {"split": split, "streaming": streaming}
                    if revision is not None:
                        kwargs["revision"] = revision
                    if data_files is not None:
                        kwargs["data_files"] = data_files
                    if config is None:
                        ds = load_dataset(hf_id, **kwargs)
                    else:
                        ds = load_dataset(hf_id, config, **kwargs)
                    return ds, split
                except Exception as e:  # noqa: BLE001 - try the next config/split/revision
                    last_err = e
                    continue
    raise RuntimeError(
        f"Could not load any (revision, config, split) combo for {hf_id} "
        f"(tried revisions={candidate_revisions}, configs={candidate_configs}, "
        f"splits={candidate_splits}, data_files={data_files}): {last_err}"
    )


def _extract_image(row: dict, image_col: str):
    """
    Handle both auto-decoded Image-feature columns (PIL.Image objects) and
    plain string path columns (e.g. SROIE's `image_path`), which point to a
    file on disk once the dataset has been downloaded and extracted.
    """
    val = row.get(image_col)
    if val is None:
        return None
    if isinstance(val, str):
        return Image.open(val)
    return val  # already a PIL.Image (decoded Image feature)


def download_one(name: str, spec: dict, n: int, skip: int = 0, start_index: int = None,
                  random_sample: bool = False, seed: int = 42, buffer_size: int = 1000) -> int:
    """
    Save `n` images to out_dir/img_###.png. Always streams (see
    _first_available_split) so only as much of the source dataset is
    downloaded as is actually needed for this call.

    Default (random_sample=False): `skip` valid (non-None) images are
    consumed from the front of the dataset's natural stream order before
    saving begins -- this is how a second pull can fetch a *new*,
    non-overlapping batch (e.g. skip=50, n=50 to get images 50-99) instead
    of re-saving the same first N images every time. This is the mode used
    for FUNSD/CORD/SROIE, kept exactly as-is for reproducibility of
    Experiments 002-009 (streaming preserves the same row order as the old
    non-streaming ds[i] indexing did, so switching to streaming doesn't
    change which images "img_000.png" etc. refer to).

    random_sample=True: applies datasets' IterableDataset.shuffle(seed=...,
    buffer_size=...) before iterating -- a reservoir-style shuffle that only
    needs to materialize `buffer_size` rows at a time (not the whole
    dataset) to draw a reproducible, non-first-N sample. Used for new
    benchmarks (BOSSBASE, DOCILE) to avoid accidental ordering bias (e.g. a
    dataset sorted by camera model, submission date, or vendor) without
    forcing a full-dataset download just to sample 100 rows out of tens of
    thousands. `skip` still lets you draw a second, disjoint batch from the
    same shuffled stream later if needed.

    `start_index` controls the output filename numbering (defaults to `skip`).
    """
    out_dir = DATASETS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if start_index is None:
        start_index = skip

    print(f"[{name}] loading {spec['hf_id']} (streaming) ...")
    ds, split = _first_available_split(
        spec["hf_id"],
        spec["candidate_configs"],
        spec["candidate_splits"],
        spec.get("candidate_revisions", (None,)),
        data_files=spec.get("data_files"),
        streaming=True,
    )
    print(f"[{name}] using split '{split}'")

    if random_sample:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
        print(f"[{name}] random_sample=True, seed={seed}, buffer_size={buffer_size} -- shuffled stream")

    skipped = 0
    saved = 0
    for row in ds:
        if saved >= n:
            break
        img = _extract_image(row, spec["image_col"])
        if img is None:
            continue
        if skipped < skip:
            skipped += 1
            continue
        img = img.convert("RGB")
        out_path = out_dir / f"img_{start_index + saved:03d}.png"
        img.save(out_path)
        saved += 1
        if saved % 20 == 0:
            print(f"[{name}] ...{saved}/{n} saved")

    print(f"[{name}] saved {saved}/{n} images (skip={skip}, random_sample={random_sample}) to {out_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="images per dataset")
    parser.add_argument("--skip", type=int, default=0, help="valid images to skip before saving (for a second, non-overlapping pull)")
    parser.add_argument("--only", type=str, default=None, help="comma-separated subset of FUNSD,CORD,SROIE,BOSSBASE,DOCILE")
    parser.add_argument("--random", action="store_true", help="draw a fixed-seed random sample instead of a first/next-N-in-order slice")
    parser.add_argument("--seed", type=int, default=42, help="seed for --random")
    parser.add_argument("--buffer-size", type=int, default=1000,
                         help="shuffle buffer size for --random (streaming reservoir shuffle -- doesn't require a full-dataset download)")
    args = parser.parse_args()

    names = list(DATASET_SPECS.keys())
    if args.only:
        names = [n.strip().upper() for n in args.only.split(",")]

    totals = {}
    for name in names:
        spec = DATASET_SPECS[name]
        totals[name] = download_one(name, spec, args.n, skip=args.skip, random_sample=args.random,
                                     seed=args.seed, buffer_size=args.buffer_size)

    print("\nSummary:")
    for name, count in totals.items():
        print(f"  {name}: {count} images")
    print(f"  TOTAL: {sum(totals.values())} images")


if __name__ == "__main__":
    main()
