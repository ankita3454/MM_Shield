"""Phase 4: final held-out test evaluation.

Runs exactly one chosen run (already trained and validated by train.py's
ablation sweep) against test.csv, exactly once. This file exists precisely
so test.csv is only ever touched here - never inside the sweep in train.py -
per the locked design decision that comparing many feature/classifier/PCA/
imbalance configs against test.csv would let config choices overfit to test
by repeated exposure.

Chosen config (2026-07-20, after reviewing the full ablation sweep in
experiment_log.csv): features=fused, classifier=xgboost, pca=none,
imbalance=oversample. Runner-up was cnn-only (higher validation F1: 0.933
vs 0.875), but fused was chosen deliberately over the raw validation-F1
leader - the fused representation (CNN + handcrafted, not CNN alone) was
this module's stated design premise from the start, and the validation set
is only 8 positives, too small for a 0.933-vs-0.875 gap to be a reliable
tiebreaker on its own.
"""

import json

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

from adversarial.dataset.splitter import TEST_PATH
from adversarial.training.train import FEATURE_CONFIGS, RUNS_DIR

CHOSEN_RUN_ID = "fused__xgboost__pcanone__oversample"
TEST_METRICS_PATH = RUNS_DIR / CHOSEN_RUN_ID / "test_metrics.json"


def evaluate_on_test(run_id: str = CHOSEN_RUN_ID, force: bool = False) -> dict:
    if TEST_METRICS_PATH.exists() and not force:
        print(f"{TEST_METRICS_PATH} already exists - not re-evaluating "
              f"(pass force=True to override deliberately - test.csv should be touched sparingly)")
        return json.loads(TEST_METRICS_PATH.read_text())

    run_dir = RUNS_DIR / run_id
    metadata = json.loads((run_dir / "metadata.json").read_text())
    model = joblib.load(run_dir / "model.pkl")
    scaler = joblib.load(run_dir / "scaler.pkl")
    pca_path = run_dir / "pca.pkl"
    pca_model = joblib.load(pca_path) if pca_path.exists() else None

    features_key = metadata["features"]
    feature_names = FEATURE_CONFIGS[features_key][2]

    test_df = pd.read_csv(TEST_PATH)[["candidate_id", "label"]]
    features_df = FEATURE_CONFIGS[features_key][1]()  # resume-safe loader, reads existing CSV
    merged = test_df.merge(features_df, on="candidate_id", how="inner", suffixes=("", "_feat"))
    if len(merged) != len(test_df):
        raise ValueError(
            f"{len(test_df) - len(merged)} candidate_id(s) from {TEST_PATH} missing in "
            f"{features_key} features"
        )

    X_test = merged[feature_names].values
    y_test = merged["label"].values.astype(int)

    X_test_s = scaler.transform(X_test)
    X_test_p = pca_model.transform(X_test_s) if pca_model is not None else X_test_s

    threshold = metadata.get("threshold", 0.5)
    test_proba = model.predict_proba(X_test_p)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "run_id": run_id,
        "threshold": threshold,
        "n_test": len(y_test), "n_test_positive": int((y_test == 1).sum()),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "test_pr_auc": float(average_precision_score(y_test, test_proba)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }
    TEST_METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"FINAL TEST EVALUATION ({run_id}):")
    print(f"  n_test={metrics['n_test']} (n_test_positive={metrics['n_test_positive']})")
    print(f"  precision={metrics['test_precision']:.3f} recall={metrics['test_recall']:.3f} "
          f"f1={metrics['test_f1']:.3f}")
    print(f"  roc_auc={metrics['test_roc_auc']:.3f} pr_auc={metrics['test_pr_auc']:.3f}")
    print(f"  confusion_matrix={metrics['confusion_matrix']}")
    print(f"saved -> {TEST_METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate_on_test()
