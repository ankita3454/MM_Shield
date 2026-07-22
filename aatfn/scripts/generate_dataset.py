"""
Step 2: generate the full AATFN fusion-layer attack dataset.

For every base image in aatfn/base_images/<dataset>/, this produces all
2^3 = 8 attack-combination variants:

    clean, T, P, S, TP, TS, PS, TPS

(T=typo, P=patch, S=steganography). Each variant is saved under
aatfn/generated/<dataset>/<combo>/ and a row is appended to
aatfn/metadata.csv with multi-hot labels (typo, patch, stego) so the
fusion classifier can be trained directly off this CSV.

Usage:
    python3 generate_dataset.py            # all 3 datasets
    python3 generate_dataset.py --datasets funsd cord   # subset
    python3 generate_dataset.py --datasets funsd --combos T --start 0 --end 50
        # process only images[0:50] -- useful for chunking long runs

Safe to re-run / resume: skips any (dataset, combo, image) already present
in metadata.csv, and flushes each row immediately so a partial/interrupted
run never loses already-completed work.
"""
import argparse
import csv
import itertools
from pathlib import Path

from PIL import Image

from attacks.typo_attack import apply_typo_attack
from attacks.patch_attack import apply_patch_attack
from attacks.stego_attack import apply_stego_attack

HERE = Path(__file__).resolve().parent.parent  # aatfn/
BASE_DIR = HERE / "base_images"
OUT_DIR = HERE / "generated"
META_PATH = HERE / "metadata.csv"

# combo code -> (typo, patch, stego) flags
COMBOS = {
    "clean": (0, 0, 0),
    "T": (1, 0, 0),
    "P": (0, 1, 0),
    "S": (0, 0, 1),
    "TP": (1, 1, 0),
    "TS": (1, 0, 1),
    "PS": (0, 1, 1),
    "TPS": (1, 1, 1),
}


def apply_combo(img: Image.Image, seed: int, typo: int, patch: int, stego: int) -> Image.Image:
    out = img
    # fixed order: typo -> patch -> stego (stego last so LSBs survive)
    if typo:
        out = apply_typo_attack(out, seed)
    if patch:
        out = apply_patch_attack(out, seed)
    if stego:
        out = apply_stego_attack(out, seed)
    return out


def _load_done_set():
    """(dataset, combo, base_image) triples already recorded in metadata.csv."""
    done = set()
    if META_PATH.exists():
        with open(META_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["combo"], row["base_image"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["funsd", "cord", "sroie"])
    ap.add_argument("--combos", nargs="+", default=list(COMBOS.keys()),
                     help="subset of combo codes to generate, default = all 8")
    ap.add_argument("--start", type=int, default=0, help="start index into the sorted image list")
    ap.add_argument("--end", type=int, default=None, help="end index (exclusive), default = all")
    args = ap.parse_args()

    done = _load_done_set()
    write_header = not META_PATH.exists()
    meta_f = open(META_PATH, "a", newline="")
    writer = csv.DictWriter(meta_f, fieldnames=[
        "dataset", "base_image", "combo", "typo", "patch", "stego", "output_path"
    ])
    if write_header:
        writer.writeheader()
        meta_f.flush()

    total_written = 0
    for dataset in args.datasets:
        src_dir = BASE_DIR / dataset
        images = sorted(src_dir.glob("*"))
        if not images:
            print(f"[{dataset}] no base images found in {src_dir} -- run sample_dataset.py first")
            continue
        images = images[args.start:args.end]

        for combo in args.combos:
            typo, patch, stego = COMBOS[combo]
            out_dir = OUT_DIR / dataset / combo
            out_dir.mkdir(parents=True, exist_ok=True)

            n_done = 0
            for idx, img_path in enumerate(images, start=args.start):
                if (dataset, combo, img_path.name) in done:
                    continue  # already generated in a previous chunk/run

                seed = idx  # deterministic per base image, reused across combos
                img = Image.open(img_path).convert("RGB")
                attacked = apply_combo(img, seed, typo, patch, stego)

                out_name = f"{img_path.stem}_{combo}.png"
                out_path = out_dir / out_name
                attacked.save(out_path)

                writer.writerow({
                    "dataset": dataset,
                    "base_image": img_path.name,
                    "combo": combo,
                    "typo": typo,
                    "patch": patch,
                    "stego": stego,
                    "output_path": str(out_path.relative_to(HERE)),
                })
                meta_f.flush()
                n_done += 1
                total_written += 1

            print(f"[{dataset}][{combo}] generated {n_done} new images (of {len(images)} in range) -> {out_dir}")

    meta_f.close()
    print(f"\nWrote {total_written} new rows -> {META_PATH}")


if __name__ == "__main__":
    main()
