"""Phase: internal evaluation.

Loads best_model.pkl and evaluates it ONCE on test.csv - the held-out split
that train.py never touches. FinInject (external, zero-shot) benchmarking is
a separate, later module.
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from typographic.config import OUTPUTS_DIR
from typographic.dataset.splitter import TEST_PATH
from typographic.features import feature_fusion
from typographic.training.train import BEST_MODEL_PATH

EVALUATION_REPORT_PATH = OUTPUTS_DIR / "evaluation_report.json"

FEATURE_NAMES = feature_fusion.get_feature_names()


def evaluate_on_test() -> dict:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"{BEST_MODEL_PATH} not found - run train.train_and_select() first")

    model = joblib.load(BEST_MODEL_PATH)

    df = pd.read_csv(TEST_PATH)
    X_test = df[FEATURE_NAMES].values
    y_test = (df["label"] == "malicious").astype(int).values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    report = {
        "num_test_samples": len(df),
        "accuracy": float((y_pred == y_test).mean()),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }

    EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "pr_curve"}, indent=2))
    print(f"saved -> {EVALUATION_REPORT_PATH}")
    return report


if __name__ == "__main__":
    evaluate_on_test()
