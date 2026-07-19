"""Phase: model training.

Trains Random Forest, XGBoost, and SVM on train.csv, compares them on
validation.csv (never test.csv - that stays untouched until evaluate.py),
picks the best by validation F1, retrains it on train+validation combined,
and saves it as best_model.pkl.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from typographic.config import OUTPUTS_DIR, RANDOM_SEED
from typographic.dataset.splitter import TRAIN_PATH, VALIDATION_PATH
from typographic.features import feature_fusion

BEST_MODEL_PATH = OUTPUTS_DIR / "best_model.pkl"
TRAINING_REPORT_PATH = OUTPUTS_DIR / "training_report.json"

FEATURE_NAMES = feature_fusion.get_feature_names()


def _load_xy(path):
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES].values
    y = (df["label"] == "malicious").astype(int).values
    return X, y


def _candidate_models() -> dict:
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=200, random_state=RANDOM_SEED, eval_metric="logloss")),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_SEED)),
        ]),
    }


def train_and_select(force: bool = False) -> dict:
    if BEST_MODEL_PATH.exists() and not force:
        print(f"{BEST_MODEL_PATH} already exists - not retraining (pass force=True to override deliberately)")
        return json.loads(TRAINING_REPORT_PATH.read_text())

    X_train, y_train = _load_xy(TRAIN_PATH)
    X_val, y_val = _load_xy(VALIDATION_PATH)

    results = {}
    for name, model in _candidate_models().items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        results[name] = {
            "val_f1": float(f1_score(y_val, val_pred)),
            "val_roc_auc": float(roc_auc_score(y_val, val_proba)),
        }
        print(f"{name}: val_f1={results[name]['val_f1']:.3f} val_roc_auc={results[name]['val_roc_auc']:.3f}")

    best_name = max(results, key=lambda n: results[n]["val_f1"])
    print(f"best model: {best_name}")

    # Retrain the chosen model on train+validation combined; test.csv is never touched here.
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_model = _candidate_models()[best_name]
    best_model.fit(X_train_val, y_train_val)

    joblib.dump(best_model, BEST_MODEL_PATH)

    report = {"model_comparison": results, "best_model": best_name, "feature_names": FEATURE_NAMES}
    TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"saved -> {BEST_MODEL_PATH}")
    return report


if __name__ == "__main__":
    train_and_select()
