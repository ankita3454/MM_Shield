"""
Step 1: sample 100 clean base images per dataset (FUNSD / CORD / SROIE)
into aatfn/base_images/<dataset>/.

This does NOT download the datasets -- point RAW_DIRS at wherever you've
already downloaded/extracted FUNSD, CORD and SROIE on your machine (e.g.
your existing MMShield/SAA/datasets/ folders). See pipeline.md step 1 for
download links if you don't have them yet.

Usage:
    python3 sample_dataset.py \
        --funsd /path/to/FUNSD/images \
        --cord /path/to/CORD/images \
        --sroie /path/to/SROIE/images \
        --n 100
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
HERE = Path(__file__).resolve().parent.parent  # aatfn/


def sample_one(name: str, src_dir: Path, n: int, seed: int = 42):
    dst_dir = HERE / "base_images" / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
    )
    if len(images) < n:
        print(f"[{name}] WARNING: only found {len(images)} images in {src_dir}, "
              f"wanted {n}. Using all of them.")
    rng = random.Random(seed)
    rng.shuffle(images)
    chosen = images[:n]

    for i, src in enumerate(chosen):
        dst = dst_dir / f"{name}_{i:03d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    print(f"[{name}] copied {len(chosen)} images -> {dst_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funsd", type=Path, required=True, help="folder containing FUNSD page images")
    ap.add_argument("--cord", type=Path, required=True, help="folder containing CORD receipt images")
    ap.add_argument("--sroie", type=Path, required=True, help="folder containing SROIE receipt images")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    sample_one("funsd", args.funsd, args.n)
    sample_one("cord", args.cord, args.n)
    sample_one("sroie", args.sroie, args.n)


if __name__ == "__main__":
    main()
