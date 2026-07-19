"""Phase: external zero-shot benchmark (FigStep).

Runs the trained model (no retraining/fine-tuning) on FigStep - 500
typographic jailbreak images, a genuinely different visual domain from the
receipts/forms/invoices this model was trained on. FigStep is 100% attack
images with no clean counterpart class, so the only meaningful metric is
detection rate (recall): the fraction of these unseen attacks the model
correctly flags as malicious - not accuracy/precision/ROC-AUC, which would
require a negative class this benchmark doesn't have.

Missed samples (image_id, category, model score) are saved separately for
error analysis and paper figures.
"""

import json

import joblib
from PIL import Image

from typographic.config import BENCHMARK_DATASET, DATASETS_DIR, OUTPUTS_DIR
from typographic.dataset.download import download_figstep
from typographic.features import feature_fusion, ocr
from typographic.training.train import BEST_MODEL_PATH

BENCHMARK_OCR_CACHE_DIR = OUTPUTS_DIR / "figstep_ocr_cache"
BENCHMARK_RESULTS_PATH = OUTPUTS_DIR / "figstep_results.json"
BENCHMARK_REPORT_PATH = OUTPUTS_DIR / "figstep_report.json"


def _get_regions(image_id: str, image_path) -> list[dict]:
    cache_path = BENCHMARK_OCR_CACHE_DIR / f"{image_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    regions = ocr.extract_regions(str(image_path))
    cache_path.write_text(json.dumps(regions, indent=2))
    return regions


def run_benchmark(force: bool = False) -> dict:
    if BENCHMARK_REPORT_PATH.exists() and not force:
        print(f"{BENCHMARK_REPORT_PATH} already exists - not rerunning (pass force=True to override deliberately)")
        return json.loads(BENCHMARK_REPORT_PATH.read_text())

    metadata = download_figstep()
    BENCHMARK_OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(BEST_MODEL_PATH)

    results = []
    for i, entry in enumerate(metadata["images"]):
        image_path = DATASETS_DIR / BENCHMARK_DATASET / entry["image_file"]
        regions = _get_regions(entry["image_id"], image_path)
        image = Image.open(image_path).convert("RGB")
        fused = feature_fusion.fuse_features(image, regions)

        pred = model.predict([fused["fused_vector"]])[0]
        proba = model.predict_proba([fused["fused_vector"]])[0][1]
        detected = bool(pred == 1)

        results.append({
            "image_id": entry["image_id"],
            "category_name": entry["category_name"],
            "predicted_label": "malicious" if detected else "clean",
            "malicious_probability": float(proba),
            "detected": detected,
        })

        if (i + 1) % 50 == 0:
            print(f"processed {i + 1}/{len(metadata['images'])} FigStep images")

    num_total = len(results)
    num_detected = sum(1 for r in results if r["detected"])
    detection_rate = num_detected / num_total

    per_category = {}
    for r in results:
        counts = per_category.setdefault(r["category_name"], {"total": 0, "detected": 0})
        counts["total"] += 1
        counts["detected"] += int(r["detected"])
    for counts in per_category.values():
        counts["detection_rate"] = counts["detected"] / counts["total"]

    missed_samples = [r for r in results if not r["detected"]]

    report = {
        "dataset": BENCHMARK_DATASET,
        "num_total": num_total,
        "num_detected": num_detected,
        "detection_rate": detection_rate,
        "per_category": per_category,
        "num_missed": len(missed_samples),
        "missed_samples": missed_samples,
    }

    BENCHMARK_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    BENCHMARK_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Detection rate: {num_detected}/{num_total} = {detection_rate:.1%}")
    print(f"saved -> {BENCHMARK_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_benchmark()
