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
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from typographic.config import (
    BENCHMARK_DATASET,
    DATASETS_DIR,
    DOCLAYNET_ATTACK_METADATA_PATH,
    DOCLAYNET_SAMPLE_SIZE,
    DOCLAYNET_SAMPLED_PATH,
    DOCLAYNET_SPLIT,
    OUTPUTS_DIR,
)
from typographic.dataset import dataset_builder, sampler
from typographic.dataset.attack_generator import DOCLAYNET_ATTACKS_DIR, generate_all_attacks
from typographic.dataset.download import download_external_sample, download_figstep
from typographic.features import feature_fusion, ocr
from typographic.training.train import BEST_MODEL_PATH

BENCHMARK_OCR_CACHE_DIR = OUTPUTS_DIR / "figstep_ocr_cache"
BENCHMARK_RESULTS_PATH = OUTPUTS_DIR / "figstep_results.json"
BENCHMARK_REPORT_PATH = OUTPUTS_DIR / "figstep_report.json"

DOCLAYNET_FEATURE_DATASET_PATH = OUTPUTS_DIR / "doclaynet_feature_dataset.csv"
DOCLAYNET_FEATURE_METADATA_PATH = OUTPUTS_DIR / "doclaynet_feature_metadata.json"
DOCLAYNET_RESULTS_PATH = OUTPUTS_DIR / "doclaynet_results.json"
DOCLAYNET_REPORT_PATH = OUTPUTS_DIR / "doclaynet_report.json"

FEATURE_NAMES = feature_fusion.get_feature_names()


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


def run_doclaynet_benchmark(force: bool = False) -> dict:
    """External zero-shot benchmark on DocLayNet: unlike FigStep, attacks are
    generated on it the same way as the training datasets, so it has a clean
    counterpart class - full accuracy/precision/recall/F1/ROC-AUC apply, plus
    a per-doc_category breakdown (financial_reports, manuals, patents, etc.)
    since DocLayNet provides that metadata. Reuses every existing pipeline
    stage (download/sample/attack-generate/feature-build) end to end; no
    retraining."""
    if DOCLAYNET_REPORT_PATH.exists() and not force:
        print(f"{DOCLAYNET_REPORT_PATH} already exists - not rerunning (pass force=True to override deliberately)")
        return json.loads(DOCLAYNET_REPORT_PATH.read_text())

    download_external_sample("DocLayNet", split=DOCLAYNET_SPLIT, n=DOCLAYNET_SAMPLE_SIZE)
    sampler.sample_doclaynet()
    generate_all_attacks(
        sampled_path=DOCLAYNET_SAMPLED_PATH,
        attacks_dir=DOCLAYNET_ATTACKS_DIR,
        attack_metadata_path=DOCLAYNET_ATTACK_METADATA_PATH,
    )
    dataset_builder.build_dataset(
        sampled_path=DOCLAYNET_SAMPLED_PATH,
        attack_metadata_path=DOCLAYNET_ATTACK_METADATA_PATH,
        feature_dataset_path=DOCLAYNET_FEATURE_DATASET_PATH,
        feature_metadata_path=DOCLAYNET_FEATURE_METADATA_PATH,
    )

    model = joblib.load(BEST_MODEL_PATH)
    df = pd.read_csv(DOCLAYNET_FEATURE_DATASET_PATH)
    metadata_rows = {m["image_id"]: m for m in json.loads(DOCLAYNET_FEATURE_METADATA_PATH.read_text())}

    X = df[FEATURE_NAMES].values
    y_true = (df["label"] == "malicious").astype(int).values
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = []
    for image_id, true_label, pred, proba in zip(df["image_id"], df["label"], y_pred, y_proba):
        results.append({
            "image_id": image_id,
            "doc_category": metadata_rows[image_id]["doc_category"],
            "true_label": true_label,
            "predicted_label": "malicious" if pred == 1 else "clean",
            "malicious_probability": float(proba),
            "correct": bool((pred == 1) == (true_label == "malicious")),
        })

    per_category = {}
    for r in results:
        cat = r["doc_category"] or "unknown"
        stats = per_category.setdefault(cat, {"total": 0, "correct": 0, "malicious_total": 0, "malicious_detected": 0})
        stats["total"] += 1
        stats["correct"] += int(r["correct"])
        if r["true_label"] == "malicious":
            stats["malicious_total"] += 1
            stats["malicious_detected"] += int(r["predicted_label"] == "malicious")
    for stats in per_category.values():
        stats["accuracy"] = stats["correct"] / stats["total"]
        stats["detection_rate"] = (stats["malicious_detected"] / stats["malicious_total"]) if stats["malicious_total"] else None

    report = {
        "dataset": "DocLayNet",
        "num_total": len(results),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_doc_category": per_category,
    }

    DOCLAYNET_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    DOCLAYNET_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "per_doc_category"}, indent=2))
    print(f"saved -> {DOCLAYNET_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_benchmark()
    run_doclaynet_benchmark()
