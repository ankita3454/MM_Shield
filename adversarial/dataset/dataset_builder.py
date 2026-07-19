"""Phase 2: candidate dataset builder (two-stage design).

Runs candidate generation over all 150 clean + 150 attacked images, crops
and saves each candidate region as its own image file, and records labels/
metadata for every candidate. Deliberately does NOT compute CNN/handcrafted
features here - those are computed later (Phase 3) from these saved crops,
keeping candidate crops independently inspectable/reusable regardless of
feature method, unlike the Typographic module's single-pass dataset_builder.py.
"""

import csv
import json

from PIL import Image

from adversarial.config import ADVERSARIAL_DATASETS_DIR, ATTACK_METADATA_PATH, CANDIDATES_DIR
from adversarial.dataset.candidate_generator import generate_candidates_for_attacked, generate_candidates_for_clean
from typographic.config import DATASETS_DIR, SAMPLED_IMAGES_PATH

CANDIDATE_LABELS_PATH = ADVERSARIAL_DATASETS_DIR / "candidate_labels.csv"
CANDIDATE_METADATA_PATH = ADVERSARIAL_DATASETS_DIR / "candidate_metadata.json"

FIELDNAMES = ["candidate_id", "source_image_id", "dataset", "label", "source_type", "crop_file"]


def _crop_and_save(image: Image.Image, bbox, candidate_id: str, candidates_dir) -> str:
    crop = image.crop(tuple(bbox))
    filename = f"{candidate_id}.png"
    crop.save(candidates_dir / filename)
    return f"{candidates_dir.relative_to(DATASETS_DIR)}/{filename}"


def _candidate_row(candidate_id, source_image_id, dataset_name, candidate, crop_file):
    return {
        "candidate_id": candidate_id,
        "source_image_id": source_image_id,
        "dataset": dataset_name,
        "label": candidate["label"],
        "source_type": candidate["source_type"],
        "crop_file": crop_file,
    }


def build_candidate_dataset(
    sampled_path=SAMPLED_IMAGES_PATH,
    attack_metadata_path=ATTACK_METADATA_PATH,
    candidates_dir=CANDIDATES_DIR,
    labels_path=CANDIDATE_LABELS_PATH,
    metadata_path=CANDIDATE_METADATA_PATH,
    force: bool = False,
) -> None:
    if labels_path.exists() and metadata_path.exists() and not force:
        print(f"{labels_path} and {metadata_path} already exist - not regenerating "
              f"(pass force=True to override deliberately)")
        return

    candidates_dir.mkdir(parents=True, exist_ok=True)

    sampled = json.loads(sampled_path.read_text())
    attacks = json.loads(attack_metadata_path.read_text())
    attacks_by_source = {a["source_image_id"]: a for a in attacks["attacks"]}

    rows = []
    metadata_rows = []

    for i, entry in enumerate(sampled["images"]):
        image_id = entry["image_id"]
        dataset_name = entry["dataset"]

        clean_image = Image.open(DATASETS_DIR / entry["image_file"]).convert("RGB")
        for j, c in enumerate(generate_candidates_for_clean(clean_image)):
            candidate_id = f"{image_id}_cand{j}"
            crop_file = _crop_and_save(clean_image, c["bbox"], candidate_id, candidates_dir)
            row = _candidate_row(candidate_id, image_id, dataset_name, c, crop_file)
            rows.append(row)
            metadata_rows.append({**row, "bbox": c["bbox"], "score": c["score"]})

        attack = attacks_by_source.get(image_id)
        if attack is not None:
            attacked_image = Image.open(DATASETS_DIR / attack["image_file"]).convert("RGB")
            attacked_candidates = generate_candidates_for_attacked(attacked_image, attack["patch_bbox"])
            for j, c in enumerate(attacked_candidates):
                candidate_id = f"{attack['attack_id']}_cand{j}"
                crop_file = _crop_and_save(attacked_image, c["bbox"], candidate_id, candidates_dir)
                row = _candidate_row(candidate_id, image_id, dataset_name, c, crop_file)
                rows.append(row)
                metadata_rows.append({
                    **row, "bbox": c["bbox"], "score": c["score"], "patch_coverage": c.get("patch_coverage"),
                })

        if (i + 1) % 25 == 0:
            print(f"processed {i + 1}/{len(sampled['images'])} source images")

    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path.write_text(json.dumps(metadata_rows, indent=2))

    print(f"wrote {len(rows)} candidate rows -> {labels_path}")
    print(f"wrote {len(metadata_rows)} entries -> {metadata_path}")


if __name__ == "__main__":
    build_candidate_dataset()
