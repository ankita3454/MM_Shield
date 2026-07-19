"""Phase: dataset builder.

Runs OCR -> typography/semantic feature extraction -> fusion over all 300
images (150 clean from sampled_images.json + 150 malicious from
attack_metadata.json), producing two outputs:

  - feature_dataset.csv: image_id, source_image_id, dataset, label, and the
    20 fused feature columns. This is the only file the classifier reads.
  - feature_metadata.json: everything else useful for debugging, error
    analysis, and paper figures later (attack category/phrase/font/rotation,
    OCR cache path) that has no business being in the ML-facing CSV.

OCR results are cached per image_id under outputs/ocr_cache/ since PaddleOCR
is by far the slowest stage - reruns (e.g. after a feature-formula change)
skip straight to feature computation.
"""

import csv
import json

from PIL import Image

from typographic.config import ATTACK_METADATA_PATH, DATASETS_DIR, OUTPUTS_DIR, SAMPLED_IMAGES_PATH
from typographic.features import feature_fusion, ocr

OCR_CACHE_DIR = OUTPUTS_DIR / "ocr_cache"
FEATURE_DATASET_PATH = OUTPUTS_DIR / "feature_dataset.csv"
FEATURE_METADATA_PATH = OUTPUTS_DIR / "feature_metadata.json"


def _get_regions(image_id: str, image_path) -> list[dict]:
    cache_path = OCR_CACHE_DIR / f"{image_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    regions = ocr.extract_regions(str(image_path))
    cache_path.write_text(json.dumps(regions, indent=2))
    return regions


def _iter_entries():
    """Yields (image_id, source_image_id, dataset, label, image_path, attack_info)
    for all 150 clean + 150 malicious images."""
    sampled = json.loads(SAMPLED_IMAGES_PATH.read_text())
    for entry in sampled["images"]:
        yield {
            "image_id": entry["image_id"],
            "source_image_id": entry["image_id"],
            "dataset": entry["dataset"],
            "label": "clean",
            "image_path": DATASETS_DIR / entry["image_file"],
            "attack_info": None,
        }

    attacks = json.loads(ATTACK_METADATA_PATH.read_text())
    for attack in attacks["attacks"]:
        yield {
            "image_id": attack["malicious_id"],
            "source_image_id": attack["source_image_id"],
            "dataset": attack["source_dataset"],
            "label": "malicious",
            "image_path": DATASETS_DIR / attack["image_file"],
            "attack_info": attack,
        }


def build_dataset(force: bool = False) -> None:
    if FEATURE_DATASET_PATH.exists() and FEATURE_METADATA_PATH.exists() and not force:
        print(f"{FEATURE_DATASET_PATH} and {FEATURE_METADATA_PATH} already exist - not regenerating "
              f"(pass force=True to override deliberately)")
        return

    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = feature_fusion.get_feature_names()
    csv_rows = []
    metadata_rows = []

    for i, entry in enumerate(_iter_entries()):
        image_id = entry["image_id"]
        regions = _get_regions(image_id, entry["image_path"])
        image = Image.open(entry["image_path"]).convert("RGB")
        fused = feature_fusion.fuse_features(image, regions)

        row = {
            "image_id": image_id,
            "source_image_id": entry["source_image_id"],
            "dataset": entry["dataset"],
            "label": entry["label"],
        }
        row.update(dict(zip(feature_names, fused["fused_vector"])))
        csv_rows.append(row)

        attack = entry["attack_info"]
        metadata_rows.append({
            "image_id": image_id,
            "source_image_id": entry["source_image_id"],
            "dataset": entry["dataset"],
            "label": entry["label"],
            "ocr_cache": str(OCR_CACHE_DIR / f"{image_id}.json"),
            "num_regions": fused["num_regions"],
            "attack_category": attack["category"] if attack else None,
            "attack_phrase": attack["phrase"] if attack else None,
            "font": attack["font_path"] if attack else None,
            "rotation": attack["rotation_degrees"] if attack else None,
            "color_rgb": attack["color_rgb"] if attack else None,
            "position_mode": attack["position_mode"] if attack else None,
        })

        if (i + 1) % 25 == 0:
            print(f"processed {i + 1}/300 images")

    with open(FEATURE_DATASET_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "source_image_id", "dataset", "label"] + feature_names)
        writer.writeheader()
        writer.writerows(csv_rows)

    FEATURE_METADATA_PATH.write_text(json.dumps(metadata_rows, indent=2))

    print(f"wrote {len(csv_rows)} rows -> {FEATURE_DATASET_PATH}")
    print(f"wrote {len(metadata_rows)} entries -> {FEATURE_METADATA_PATH}")


if __name__ == "__main__":
    build_dataset()
