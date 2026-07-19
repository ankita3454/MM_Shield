"""Phase 5: external benchmark on DocLayNet.

Reuses the entire pipeline end-to-end (attack generation -> candidate
generation -> candidate dataset building -> feature extraction -> fusion) on
Typographic's frozen 200-page DocLayNet sample, then runs the already-chosen,
already-trained model (see evaluate.py's CHOSEN_RUN_ID) against it - no
retraining. DocLayNet attacks are generated the same way as the FUNSD/CORD/
SROIE training data, so they have a clean counterpart class and full
accuracy/precision/recall/F1/ROC-AUC/PR-AUC apply, unlike a zero-shot
benchmark with attack-only images.

Split into three subprocesses (candidates / features / evaluate), not just
separate functions - this is a real, reproduced native-library crash, not
caution for its own sake. Bisected directly (not guessed) to: torch
entering the same process as either paddleocr or xgboost's actual
model.predict() call segfaults (exit 139) on this machine, order-dependent
(the library imported/used *second* is the one that crashes). Confirmed
paddleocr+cv2 together are fine (candidate generation already exercises
both, e.g. Phase 2's original run); confirmed torch+cv2 together are fine
(Phase 3's cnn_features.py + handcrafted_features.py + feature_fusion.py
already ran this combination successfully). The only unsafe combinations
are torch-with-paddleocr and torch-with-xgboost-predict in one process.
Setting KMP_DUPLICATE_LIB_OK=TRUE (the usual workaround for the similar,
more common Intel-OpenMP "Error #15" case) did NOT fix it, and forcing
torch off MPS onto CPU-only did NOT fix it either - so process separation,
not an env var or device flag, is the actual fix. Three stages:
  - "candidates": download/sample (paddleocr) + attack/candidate generation
    (cv2) - no torch.
  - "features": CNN (torch) + handcrafted (cv2) feature extraction + fusion
    - no paddleocr.
  - "evaluate": loads the run's saved model/scaler/pca + metadata.json and
    predicts - no torch, no paddleocr, just joblib/pandas/sklearn.
Each runs in its own subprocess; run_doclaynet_benchmark() orchestrates all
three in sequence and is safe to call from any process regardless of what
it has already imported, since none of the three ever happens in the
caller's own process.

Locked decision (2026-07-20): APRICOT was investigated as an additional
cross-domain benchmark and rejected in favor of DocLayNet-only. Its only
public distribution is a Box shared folder - confirmed (not assumed) to
return 401 Unauthorized from Box's API without an OAuth token, and its web
UI is JS-rendered with no scriptable file listing - breaking this project's
one-command reproducibility that every other dataset here (FUNSD/CORD/
SROIE/DocLayNet, all HF-sourced) maintains. It would also move the
document-tuned candidate generator (frozen since Phase 2's whole-page-blob
debugging) into a natural-image domain (street scenes) it was never
validated against, making a poor result unfalsifiable - no way to tell
detector failure from candidate-generator domain mismatch without another
full empirical tuning cycle. See MEMORY for the full reasoning.
"""

import argparse
import json
import subprocess
import sys

from adversarial.config import (
    DOCLAYNET_ATTACK_METADATA_PATH,
    DOCLAYNET_ATTACKS_DIR,
    DOCLAYNET_CANDIDATE_LABELS_PATH,
    DOCLAYNET_CANDIDATE_METADATA_PATH,
    DOCLAYNET_CANDIDATES_DIR,
    OUTPUTS_DIR,
)
# Deliberately NOT `from adversarial.training.train import RUNS_DIR` / `from
# adversarial.training.evaluate import CHOSEN_RUN_ID` - both of those modules
# import torch (via cnn_features.py) AND xgboost at their own top level, and
# Python runs a module's top-level imports before __main__ dispatch ever
# sees --stage, so importing either here would silently drag torch and
# xgboost into every subprocess regardless of stage, undoing the isolation
# this file exists to provide. These two constants are simple enough to
# duplicate directly instead.
RUNS_DIR = OUTPUTS_DIR / "runs"
CHOSEN_RUN_ID = "fused__xgboost__pcanone__oversample"

DOCLAYNET_CNN_FEATURES_PATH = OUTPUTS_DIR / "doclaynet_cnn_features.csv"
DOCLAYNET_HANDCRAFTED_FEATURES_PATH = OUTPUTS_DIR / "doclaynet_handcrafted_features.csv"
DOCLAYNET_FUSED_FEATURES_PATH = OUTPUTS_DIR / "doclaynet_fused_features.csv"
DOCLAYNET_BENCHMARK_RESULTS_PATH = OUTPUTS_DIR / "doclaynet_benchmark_results.json"
DOCLAYNET_BENCHMARK_REPORT_PATH = OUTPUTS_DIR / "doclaynet_benchmark_report.json"


def _run_candidates_stage(force: bool = False) -> None:
    """paddleocr (via typographic.dataset.download) + cv2 (attack/candidate
    generation) - confirmed safe together, but never call this in the same
    process as anything that imports torch."""
    from adversarial.dataset.attack_generator import generate_all_attacks
    from adversarial.dataset.dataset_builder import build_candidate_dataset
    from typographic.config import DOCLAYNET_SAMPLE_SIZE, DOCLAYNET_SAMPLED_PATH, DOCLAYNET_SPLIT
    from typographic.dataset.download import download_external_sample
    from typographic.dataset.sampler import sample_doclaynet

    download_external_sample("DocLayNet", split=DOCLAYNET_SPLIT, n=DOCLAYNET_SAMPLE_SIZE)
    sample_doclaynet()
    generate_all_attacks(
        sampled_path=DOCLAYNET_SAMPLED_PATH,
        attacks_dir=DOCLAYNET_ATTACKS_DIR,
        attack_metadata_path=DOCLAYNET_ATTACK_METADATA_PATH,
        force=force,
    )
    build_candidate_dataset(
        sampled_path=DOCLAYNET_SAMPLED_PATH,
        attack_metadata_path=DOCLAYNET_ATTACK_METADATA_PATH,
        candidates_dir=DOCLAYNET_CANDIDATES_DIR,
        labels_path=DOCLAYNET_CANDIDATE_LABELS_PATH,
        metadata_path=DOCLAYNET_CANDIDATE_METADATA_PATH,
        force=force,
    )


def _run_features_stage(force: bool = False) -> None:
    """torch (CNN) + cv2 (handcrafted) - confirmed safe together (same
    combination Phase 3 already ran), but never call this in the same
    process as paddleocr or as xgboost's .predict()."""
    from adversarial.features.cnn_features import extract_cnn_features
    from adversarial.features.feature_fusion import fuse_features
    from adversarial.features.handcrafted_features import extract_handcrafted_features

    extract_cnn_features(labels_path=DOCLAYNET_CANDIDATE_LABELS_PATH, output_path=DOCLAYNET_CNN_FEATURES_PATH, force=force)
    extract_handcrafted_features(
        labels_path=DOCLAYNET_CANDIDATE_LABELS_PATH, output_path=DOCLAYNET_HANDCRAFTED_FEATURES_PATH, force=force,
    )
    fuse_features(
        labels_path=DOCLAYNET_CANDIDATE_LABELS_PATH,
        cnn_path=DOCLAYNET_CNN_FEATURES_PATH,
        handcrafted_path=DOCLAYNET_HANDCRAFTED_FEATURES_PATH,
        output_path=DOCLAYNET_FUSED_FEATURES_PATH,
        force=force,
    )


def _run_evaluate_stage(run_id: str = CHOSEN_RUN_ID, force: bool = False) -> dict:
    """Never imports torch/cnn_features/feature_fusion - reads the CSV and
    metadata.json that the build stage already wrote to disk."""
    import joblib
    import pandas as pd
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if DOCLAYNET_BENCHMARK_REPORT_PATH.exists() and not force:
        print(f"{DOCLAYNET_BENCHMARK_REPORT_PATH} already exists - not rerunning "
              f"(pass force=True to override deliberately)")
        return json.loads(DOCLAYNET_BENCHMARK_REPORT_PATH.read_text())

    run_dir = RUNS_DIR / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text())
    model = joblib.load(run_dir / "model.pkl")
    scaler = joblib.load(run_dir / "scaler.pkl")
    pca_path = run_dir / "pca.pkl"
    pca_model = joblib.load(pca_path) if pca_path.exists() else None

    feature_names = metadata["feature_names"]
    fused_df = pd.read_csv(DOCLAYNET_FUSED_FEATURES_PATH)
    X = fused_df[feature_names].values
    y_true = fused_df["label"].values.astype(int)

    X_s = scaler.transform(X)
    X_p = pca_model.transform(X_s) if pca_model is not None else X_s

    threshold = metadata.get("threshold", 0.5)
    y_proba = model.predict_proba(X_p)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = [
        {
            "candidate_id": cid,
            "source_image_id": sid,
            "source_type": stype,
            "true_label": int(t),
            "predicted_label": int(p),
            "malicious_probability": float(proba),
        }
        for cid, sid, stype, t, p, proba in zip(
            fused_df["candidate_id"], fused_df["source_image_id"], fused_df["source_type"], y_true, y_pred, y_proba,
        )
    ]

    report = {
        "run_id": run_id,
        "dataset": "DocLayNet",
        "threshold": threshold,
        "num_total": len(y_true),
        "num_positive": int((y_true == 1).sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    DOCLAYNET_BENCHMARK_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    DOCLAYNET_BENCHMARK_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"saved -> {DOCLAYNET_BENCHMARK_REPORT_PATH}")
    return report


def run_doclaynet_benchmark(run_id: str = CHOSEN_RUN_ID, force: bool = False) -> dict:
    if DOCLAYNET_BENCHMARK_REPORT_PATH.exists() and not force:
        print(f"{DOCLAYNET_BENCHMARK_REPORT_PATH} already exists - not rerunning "
              f"(pass force=True to override deliberately)")
        return json.loads(DOCLAYNET_BENCHMARK_REPORT_PATH.read_text())

    candidates_cmd = [sys.executable, "-m", "adversarial.training.benchmark", "--stage", "candidates"]
    features_cmd = [sys.executable, "-m", "adversarial.training.benchmark", "--stage", "features"]
    eval_cmd = [sys.executable, "-m", "adversarial.training.benchmark", "--stage", "evaluate", "--run-id", run_id]
    if force:
        candidates_cmd.append("--force")
        features_cmd.append("--force")
        eval_cmd.append("--force")

    subprocess.run(candidates_cmd, check=True)
    subprocess.run(features_cmd, check=True)
    subprocess.run(eval_cmd, check=True)

    return json.loads(DOCLAYNET_BENCHMARK_REPORT_PATH.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 DocLayNet benchmark")
    parser.add_argument("--stage", choices=["candidates", "features", "evaluate", "all"], default="all")
    parser.add_argument("--run-id", default=CHOSEN_RUN_ID)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage == "candidates":
        _run_candidates_stage(force=args.force)
    elif args.stage == "features":
        _run_features_stage(force=args.force)
    elif args.stage == "evaluate":
        _run_evaluate_stage(run_id=args.run_id, force=args.force)
    else:
        run_doclaynet_benchmark(run_id=args.run_id, force=args.force)
