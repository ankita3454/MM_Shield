"""
Generate the "malicious" half of the SAA dataset by embedding a scattered
15% LSB payload into clean images.

Design rationale (per project notes): a 15% scattered payload density
matches the density used in the steganalysis literature for payloads that
are sparse enough to be visually invisible but still leave a statistically
detectable trace -- this is deliberately NOT a dense, easy-to-detect
payload; it's meant to stress-test the SAA features against a realistic
adversary.

"Scattered" = a uniformly random 15% of pixel positions (across the full
image, raster order irrelevant) have their LSB overwritten with a random
bit, rather than embedding into a contiguous block. This matches how real
LSB steganography tools spread payload bits pseudo-randomly across the
carrier using a shared seed/key.

Usage:
    python embed_lsb_scattered.py --clean-dir ../datasets/clean --out-dir ../datasets/stego --ratio 0.15
    python embed_lsb_scattered.py --only SROIE      # just one dataset subfolder
    python embed_lsb_scattered.py --force            # re-embed even if output already exists

Resumable by default: if an output file already exists, it's skipped. This
matters because large scanned images (e.g. full-page SROIE receipts) can
make a full 150-image run slow enough to hit a wall-clock timeout partway
through -- re-running the same command just picks up where it left off
instead of redoing already-finished datasets.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def embed_scattered_lsb(image: Image.Image, ratio: float = 0.15, seed: int = None) -> Image.Image:
    """
    Overwrite the LSB of `ratio` fraction of pixels (chosen uniformly at
    random, without replacement, independently per color channel) with a
    random bit. Operates on RGB images; returns a new PIL Image.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    flat = arr.reshape(-1)
    n = flat.size

    n_embed = int(n * ratio)
    idx = rng.choice(n, size=n_embed, replace=False)
    payload_bits = rng.integers(0, 2, size=n_embed).astype(np.uint8)

    flat = flat.copy()
    flat[idx] = (flat[idx] & ~np.uint8(1)) | payload_bits
    stego_arr = flat.reshape(arr.shape)
    return Image.fromarray(stego_arr, mode="RGB")


def process_directory(clean_dir: str, out_dir: str, ratio: float, seed_base: int = 1000,
                       only: list = None, force: bool = False) -> int:
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    count = 0

    for dataset_subdir in sorted(clean_dir.iterdir()):
        if not dataset_subdir.is_dir():
            continue
        if only and dataset_subdir.name.upper() not in only:
            continue
        out_subdir = out_dir / dataset_subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in dataset_subdir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

        done = 0
        skipped = 0
        for i, img_path in enumerate(image_paths):
            out_path = out_subdir / img_path.name
            if not force and out_path.exists():
                skipped += 1
                count += 1
                continue
            img = Image.open(img_path)
            stego = embed_scattered_lsb(img, ratio=ratio, seed=seed_base + i)
            stego.save(out_path)
            done += 1
            count += 1

        print(f"[{dataset_subdir.name}] embedded {done} new, skipped {skipped} already-done "
              f"({len(image_paths)} total) -> {out_subdir}")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=str, default="../datasets/clean")
    parser.add_argument("--out-dir", type=str, default="../datasets/stego")
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--only", type=str, default=None, help="comma-separated subset of dataset subfolder names")
    parser.add_argument("--force", action="store_true", help="re-embed even if output file already exists")
    args = parser.parse_args()

    only = [x.strip().upper() for x in args.only.split(",")] if args.only else None
    total = process_directory(args.clean_dir, args.out_dir, args.ratio, only=only, force=args.force)
    print(f"\nTotal stego images (new + already-done): {total}")


if __name__ == "__main__":
    main()
