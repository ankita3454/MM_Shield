"""Phase 1-2: download FUNSD, CORD, SROIE from Hugging Face and organize them into
datasets/<name>/{images/, annotations/, metadata.json}.

Dataset-specific parsing (FUNSD's 0-1000 normalized bboxes, CORD's nested Donut
ground_truth JSON, SROIE's quad-polygon bboxes) lives ONLY in this file. Every
module downstream of this one must consume exclusively the unified per-word
record format:

    {"text": str, "bbox": [xmin, ymin, xmax, ymax], "page_width": int,
     "page_height": int, "dataset": str}

and must not branch on dataset name.
"""

import json

from datasets import load_dataset

from typographic.config import BENCHMARK_DATASET, DATASETS_DIR, HF_DATASET_SOURCES, RANDOM_SEED, TRAINING_DATASETS
from typographic.features import ocr


def _funsd_records(example, dataset_name):
    # nielsr/funsd bboxes are normalized to a 0-1000 scale (standard LayoutLM
    # preprocessing), independent of the actual image dimensions - verified
    # empirically (max_x/max_y exceed actual pixel width/height on every example).
    width, height = example["image"].size
    records = []
    for word, bbox in zip(example["words"], example["bboxes"]):
        text = word.strip()
        if not text:
            continue
        x0, y0, x1, y1 = bbox
        records.append({
            "text": text,
            "bbox": [x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height],
            "page_width": width,
            "page_height": height,
            "dataset": dataset_name,
        })
    return records


def _cord_records(example, dataset_name):
    # naver-clova-ix/cord-v2 stores annotations as a nested Donut-format JSON
    # string under "ground_truth"; word-level quads are in valid_line[].words[].quad,
    # already in actual pixel coordinates.
    width, height = example["image"].size
    ground_truth = json.loads(example["ground_truth"])
    records = []
    for line in ground_truth.get("valid_line", []):
        for word in line.get("words", []):
            text = word.get("text", "").strip()
            if not text:
                continue
            quad = word["quad"]
            xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
            ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
            records.append({
                "text": text,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "page_width": width,
                "page_height": height,
                "dataset": dataset_name,
            })
    return records


def _sroie_records(example, dataset_name):
    # rth/sroie-2019-v2 stores each word's bbox as a quad polygon
    # [[x1,x2,x3,x4], [y1,y2,y3,y4]] in actual pixel coordinates.
    width, height = example["image"].size
    objects = example["objects"]
    records = []
    for (xs, ys), text in zip(objects["bbox"], objects["text"]):
        text = text.strip()
        if not text:
            continue
        records.append({
            "text": text,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "page_width": width,
            "page_height": height,
            "dataset": dataset_name,
        })
    return records


_NORMALIZERS = {
    "FUNSD": _funsd_records,
    "CORD": _cord_records,
    "SROIE": _sroie_records,
}


def download_dataset(name: str, force: bool = False) -> dict:
    """Download one dataset from Hugging Face, normalize its annotations once,
    and write images/, annotations/, and metadata.json under datasets/<name>/.

    Returns the metadata dict that was written (or already present, if force=False).
    """
    if name not in HF_DATASET_SOURCES:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {list(HF_DATASET_SOURCES)}.")

    out_dir = DATASETS_DIR / name
    images_dir = out_dir / "images"
    annotations_dir = out_dir / "annotations"
    metadata_path = out_dir / "metadata.json"

    if metadata_path.exists() and not force:
        print(f"[{name}] already downloaded at {out_dir} (pass force=True to redownload)")
        return json.loads(metadata_path.read_text())

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    repo_id = HF_DATASET_SOURCES[name]
    normalize = _NORMALIZERS[name]

    print(f"[{name}] downloading {repo_id} from Hugging Face...")
    dataset_dict = load_dataset(repo_id)

    entries = []
    counter = 0
    for split_name, split in dataset_dict.items():
        for example in split:
            records = normalize(example, name)
            if not records:
                continue  # skip pages with no usable text regions

            image_id = f"{name}_{counter:04d}"
            counter += 1

            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_filename = f"{image_id}.png"
            image.save(images_dir / image_filename)

            annotation_filename = f"{image_id}.json"
            (annotations_dir / annotation_filename).write_text(json.dumps(records, indent=2))

            entries.append({
                "image_id": image_id,
                "image_file": f"images/{image_filename}",
                "annotation_file": f"annotations/{annotation_filename}",
                "original_split": split_name,
                "page_width": records[0]["page_width"],
                "page_height": records[0]["page_height"],
                "num_words": len(records),
            })

    metadata = {"dataset": name, "source": repo_id, "num_images": len(entries), "images": entries}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[{name}] done: {len(entries)} images -> {out_dir}")
    return metadata


def download_all(force: bool = False) -> None:
    for name in TRAINING_DATASETS:
        download_dataset(name, force=force)


def download_figstep(force: bool = False) -> dict:
    """Download the FigStep external zero-shot benchmark (500 typographic
    jailbreak images, no clean counterpart class - see
    typographic/training/benchmark.py). Unlike the training datasets, FigStep
    has no per-word ground-truth annotations to normalize; only images plus
    benchmark-specific metadata (category, question, instruction) are saved."""
    out_dir = DATASETS_DIR / BENCHMARK_DATASET
    images_dir = out_dir / "images"
    metadata_path = out_dir / "metadata.json"

    if metadata_path.exists() and not force:
        print(f"[{BENCHMARK_DATASET}] already downloaded at {out_dir} (pass force=True to redownload)")
        return json.loads(metadata_path.read_text())

    images_dir.mkdir(parents=True, exist_ok=True)
    repo_id = HF_DATASET_SOURCES[BENCHMARK_DATASET]

    print(f"[{BENCHMARK_DATASET}] downloading {repo_id} from Hugging Face...")
    dataset = load_dataset(repo_id, split="test")

    entries = []
    for i, example in enumerate(dataset):
        image_id = f"{BENCHMARK_DATASET}_{i:04d}"
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_filename = f"{image_id}.png"
        image.save(images_dir / image_filename)

        entries.append({
            "image_id": image_id,
            "image_file": f"images/{image_filename}",
            "category_name": example["category_name"],
            "question": example["question"],
            "instruction": example["instruction"],
        })

    metadata = {"dataset": BENCHMARK_DATASET, "source": repo_id, "num_images": len(entries), "images": entries}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[{BENCHMARK_DATASET}] done: {len(entries)} images -> {out_dir}")
    return metadata


def download_external_sample(name: str, split: str, n: int, seed: int = RANDOM_SEED, force: bool = False) -> dict:
    """Stream-sample n pages from an external Hugging Face dataset too large to
    fully download (e.g. DocLayNet, ~32GB), and save images + OCR-derived
    annotations under datasets/<name>/ - same folder shape as download_dataset().

    Unlike the training datasets, annotations here come from our own OCR
    pipeline (features/ocr.py) rather than dataset-specific ground-truth
    parsing: it's the same source of truth used everywhere else in the
    pipeline, and avoids a one-off coordinate-system conversion (e.g.
    DocLayNet's pdf_cells are in PDF point-space, not image-pixel-space, with
    a different aspect ratio and possibly a flipped y-axis - not worth the
    risk when OCR already does this job reliably). This annotation is only
    ever used by attack_generator.py for placement decisions, exactly like
    the ground-truth annotations for FUNSD/CORD/SROIE.

    Generic by name/repo/split/n so it can be reused for future external
    benchmarks beyond DocLayNet, not just this one dataset.
    """
    out_dir = DATASETS_DIR / name
    images_dir = out_dir / "images"
    annotations_dir = out_dir / "annotations"
    metadata_path = out_dir / "metadata.json"

    if metadata_path.exists() and not force:
        print(f"[{name}] already sampled at {out_dir} (pass force=True to resample)")
        return json.loads(metadata_path.read_text())

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    repo_id = HF_DATASET_SOURCES[name]
    print(f"[{name}] stream-sampling {n} pages from {repo_id} ({split} split)...")
    stream = load_dataset(repo_id, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=max(n * 5, 1000))

    entries = []
    for i, example in enumerate(stream.take(n)):
        image_id = f"{name}_{i:04d}"

        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_filename = f"{image_id}.png"
        image.save(images_dir / image_filename)

        regions = ocr.extract_regions(str(images_dir / image_filename))
        annotation_filename = f"{image_id}.json"
        (annotations_dir / annotation_filename).write_text(json.dumps(regions, indent=2))

        doc_category = example.get("metadata", {}).get("doc_category") if isinstance(example.get("metadata"), dict) else None

        entries.append({
            "image_id": image_id,
            "dataset": name,
            "image_file": f"images/{image_filename}",
            "annotation_file": f"annotations/{annotation_filename}",
            "doc_category": doc_category,
            "page_width": image.width,
            "page_height": image.height,
            "num_regions": len(regions),
        })

        if (i + 1) % 25 == 0:
            print(f"[{name}] sampled {i + 1}/{n}")

    metadata = {"dataset": name, "source": repo_id, "split": split, "seed": seed, "num_images": len(entries), "images": entries}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[{name}] done: {len(entries)} images -> {out_dir}")
    return metadata


if __name__ == "__main__":
    download_all()
