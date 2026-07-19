"""Phase 3: sample a fixed set of clean images from each training dataset and
freeze the selection to datasets/sampled_images.json.

This file, once generated, must never change - it defines the exact 150 clean
images (50 each from FUNSD/CORD/SROIE) that attack_generator.py will produce
malicious counterparts of. Re-running this script is a no-op unless force=True
is passed explicitly.
"""

import json
import random

from typographic.config import (
    DATASETS_DIR,
    DOCLAYNET_SAMPLED_PATH,
    RANDOM_SEED,
    SAMPLED_IMAGES_PATH,
    SAMPLES_PER_DATASET,
    TRAINING_DATASETS,
)


def sample_images(force: bool = False) -> dict:
    if SAMPLED_IMAGES_PATH.exists() and not force:
        print(f"sampled_images.json already exists at {SAMPLED_IMAGES_PATH} - frozen, not regenerating "
              f"(pass force=True to override deliberately)")
        return json.loads(SAMPLED_IMAGES_PATH.read_text())

    rng = random.Random(RANDOM_SEED)
    sampled = []

    for dataset_name in TRAINING_DATASETS:
        metadata_path = DATASETS_DIR / dataset_name / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"{metadata_path} not found - run download.download_dataset('{dataset_name}') first"
            )
        metadata = json.loads(metadata_path.read_text())
        available = metadata["images"]
        if len(available) < SAMPLES_PER_DATASET:
            raise ValueError(
                f"{dataset_name} has only {len(available)} images, need {SAMPLES_PER_DATASET}"
            )

        chosen = rng.sample(available, SAMPLES_PER_DATASET)
        for entry in chosen:
            sampled.append({
                "image_id": entry["image_id"],
                "dataset": dataset_name,
                "image_file": f"{dataset_name}/{entry['image_file']}",
                "annotation_file": f"{dataset_name}/{entry['annotation_file']}",
            })

    result = {
        "seed": RANDOM_SEED,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "total": len(sampled),
        "images": sampled,
    }
    SAMPLED_IMAGES_PATH.write_text(json.dumps(result, indent=2))
    print(f"sampled {len(sampled)} images -> {SAMPLED_IMAGES_PATH}")
    return result


def sample_doclaynet(force: bool = False) -> dict:
    """Reshape datasets/DocLayNet/metadata.json (already exactly N sampled
    pages, produced by download.download_external_sample()) into the same
    frozen sampled_images.json-style format attack_generator.py expects."""
    if DOCLAYNET_SAMPLED_PATH.exists() and not force:
        print(f"{DOCLAYNET_SAMPLED_PATH} already exists - frozen, not regenerating "
              f"(pass force=True to override deliberately)")
        return json.loads(DOCLAYNET_SAMPLED_PATH.read_text())

    metadata_path = DATASETS_DIR / "DocLayNet" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found - run download.download_external_sample('DocLayNet', ...) first"
        )
    metadata = json.loads(metadata_path.read_text())

    sampled = [{
        "image_id": entry["image_id"],
        "dataset": entry["dataset"],
        "image_file": f"DocLayNet/{entry['image_file']}",
        "annotation_file": f"DocLayNet/{entry['annotation_file']}",
        "doc_category": entry.get("doc_category"),
    } for entry in metadata["images"]]

    result = {
        "seed": metadata["seed"],
        "samples_per_dataset": len(sampled),
        "total": len(sampled),
        "images": sampled,
    }
    DOCLAYNET_SAMPLED_PATH.write_text(json.dumps(result, indent=2))
    print(f"sampled {len(sampled)} DocLayNet images -> {DOCLAYNET_SAMPLED_PATH}")
    return result


if __name__ == "__main__":
    sample_images()
