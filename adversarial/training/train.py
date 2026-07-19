"""Phase 4: training as an experiment framework, not a single script.

The research question is "can these fused features distinguish adversarial
patches from normal regions?" - answering it well requires comparing feature
configurations (cnn-only / handcrafted-only / fused), classifiers, and
imbalance-handling strategies against each other, not committing to one
combination up front. Every axis is a CLI flag:

    python -m adversarial.training.train --features fused --classifier xgboost --pca 64 --imbalance class_weight
    python -m adversarial.training.train --features handcrafted --classifier rf --pca none

--classifier all (the default) trains rf/xgboost/svm in one invocation and
prints a comparison table.

test.csv is never touched here (locked design decision, 2026-07-19): every
run reports validation-set metrics only. Comparing many feature/classifier/
PCA/imbalance combinations against test.csv would let config choices overfit
to test by repeated exposure. Once a config is chosen from the validation
comparison table, evaluate.py runs it once, and only once, against test.csv.

PCA (when requested) is fit only on the training split's scaled features and
applied to validation - never fit on validation, per the same fit-on-train-
only principle used for the scaler and for feature_fusion.py's decision to
keep dimensionality reduction out of dataset-building entirely.

Each (features, classifier, pca, imbalance) combination gets its own run
directory under adversarial/outputs/runs/ with the trained model, scaler,
PCA transform (if any), config/data metadata, and metrics - resume-safe like
every other module in this project, skipped on repeat invocations unless
--force is passed. Every run additionally appends one row to
experiment_log.csv so comparison tables across many invocations don't
require re-parsing every run directory.
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from adversarial.config import OUTPUTS_DIR, RANDOM_SEED
from adversarial.dataset.splitter import TRAIN_PATH, VALIDATION_PATH, split_dataset
from adversarial.features.cnn_features import CNN_EMBEDDING_DIM, CNN_FEATURES_PATH, extract_cnn_features
from adversarial.features.feature_fusion import FUSED_FEATURES_PATH, fuse_features
from adversarial.features.handcrafted_features import (
    HANDCRAFTED_FEATURES_PATH,
    RAW_FEATURE_NAMES,
    extract_handcrafted_features,
)

RUNS_DIR = OUTPUTS_DIR / "runs"
EXPERIMENT_LOG_PATH = OUTPUTS_DIR / "experiment_log.csv"

FEATURE_CONFIGS = {
    "cnn": (CNN_FEATURES_PATH, extract_cnn_features, [f"cnn_{i}" for i in range(CNN_EMBEDDING_DIM)]),
    "handcrafted": (HANDCRAFTED_FEATURES_PATH, extract_handcrafted_features, list(RAW_FEATURE_NAMES)),
    "fused": (
        FUSED_FEATURES_PATH,
        fuse_features,
        [f"cnn_{i}" for i in range(CNN_EMBEDDING_DIM)] + list(RAW_FEATURE_NAMES),
    ),
}

CLASSIFIER_NAMES = ["rf", "xgboost", "svm"]

_EXPERIMENT_LOG_FIELDS = [
    "run_id", "features", "classifier", "pca", "imbalance", "threshold",
    "n_train", "n_train_positive", "n_val", "n_val_positive",
    "val_precision", "val_recall", "val_f1", "val_roc_auc", "val_pr_auc",
    "timestamp",
]


def _load_xy(split_path, features_key: str):
    _, loader_fn, feature_names = FEATURE_CONFIGS[features_key]
    features_df = loader_fn()
    split_df = pd.read_csv(split_path)[["candidate_id", "label"]]

    merged = split_df.merge(features_df, on="candidate_id", how="inner", suffixes=("", "_feat"))
    if len(merged) != len(split_df):
        raise ValueError(
            f"{len(split_df) - len(merged)} candidate_id(s) from {split_path} missing in "
            f"{features_key} features - re-run the Phase 3 feature extraction for this split"
        )

    X = merged[feature_names].values
    y = merged["label"].values.astype(int)
    return X, y


def _random_oversample(X, y, seed: int):
    """Duplicate positive-class rows (with replacement) until class counts match.
    Fit-data-only operation - never applied to validation/test."""
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_pos) >= len(idx_neg):
        return X, y
    extra = rng.choice(idx_pos, size=len(idx_neg) - len(idx_pos), replace=True)
    all_idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def _select_threshold(y_true, proba) -> float:
    """Classification threshold maximizing F1 on the given (validation-only)
    labels/probabilities. Sweeps every observed probability as a candidate
    boundary (not a coarse grid) so the true optimum for this prediction set
    is found exactly. Locked design (2026-07-20): every run's .predict() call
    was implicitly using the raw 0.5 default, never actually tuned - this is
    the fix, and it is fit on validation only, exactly once per run, never
    on test.csv or DocLayNet (mirrors the same fit-on-train/validation-only
    principle already used for the scaler and PCA)."""
    best_threshold, best_f1 = 0.5, -1.0
    for t in np.unique(proba):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)
    return best_threshold


def _build_classifier(name: str, imbalance: str, scale_pos_weight: float):
    use_class_weight = imbalance == "class_weight"
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
            class_weight="balanced" if use_class_weight else None,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=200, random_state=RANDOM_SEED, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
        )
    if name == "svm":
        return SVC(
            probability=True, random_state=RANDOM_SEED,
            class_weight="balanced" if use_class_weight else None,
        )
    raise ValueError(f"unknown classifier: {name}")


def _run_id(features: str, classifier: str, pca: str, imbalance: str) -> str:
    return f"{features}__{classifier}__pca{pca}__{imbalance}"


def _append_experiment_log(row: dict) -> None:
    write_header = not EXPERIMENT_LOG_PATH.exists()
    row_df = pd.DataFrame([{k: row.get(k) for k in _EXPERIMENT_LOG_FIELDS}])
    row_df.to_csv(EXPERIMENT_LOG_PATH, mode="a", index=False, header=write_header)


def run_experiment(features: str, classifier: str, pca: str, imbalance: str, force: bool = False) -> dict:
    pca_label = "none" if pca in (None, "none") else str(pca)
    run_id = _run_id(features, classifier, pca_label, imbalance)
    run_dir = RUNS_DIR / run_id
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists() and not force:
        print(f"{run_id}: already trained - not retraining (pass force=True to override deliberately)")
        return json.loads(metrics_path.read_text())

    X_train, y_train = _load_xy(TRAIN_PATH, features)
    X_val, y_val = _load_xy(VALIDATION_PATH, features)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    pca_model = None
    if pca_label != "none":
        n_components = int(pca_label)
        pca_model = PCA(n_components=n_components, random_state=RANDOM_SEED).fit(X_train_s)
        X_train_p = pca_model.transform(X_train_s)
        X_val_p = pca_model.transform(X_val_s)
    else:
        X_train_p, X_val_p = X_train_s, X_val_s

    if imbalance == "oversample":
        X_fit, y_fit = _random_oversample(X_train_p, y_train, RANDOM_SEED)
    else:
        X_fit, y_fit = X_train_p, y_train

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = _build_classifier(classifier, imbalance, scale_pos_weight)
    model.fit(X_fit, y_fit)

    val_proba = model.predict_proba(X_val_p)[:, 1]
    threshold = _select_threshold(y_val, val_proba)
    val_pred = (val_proba >= threshold).astype(int)

    metrics = {
        "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val, val_proba)),
        "val_pr_auc": float(average_precision_score(y_val, val_proba)),
        "confusion_matrix": confusion_matrix(y_val, val_pred).tolist(),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.pkl")
    joblib.dump(scaler, run_dir / "scaler.pkl")
    if pca_model is not None:
        joblib.dump(pca_model, run_dir / "pca.pkl")

    metadata = {
        "features": features,
        "classifier": classifier,
        "pca": pca_label,
        "imbalance": imbalance,
        "threshold": threshold,
        "feature_names": FEATURE_CONFIGS[features][2],
        "raw_feature_dim": X_train.shape[1],
        "n_train": len(y_train), "n_train_positive": n_pos,
        "n_val": len(y_val), "n_val_positive": int((y_val == 1).sum()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))

    _append_experiment_log({**metadata, "run_id": run_id, **metrics})

    print(f"{run_id}: val_f1={metrics['val_f1']:.3f} val_roc_auc={metrics['val_roc_auc']:.3f} "
          f"val_pr_auc={metrics['val_pr_auc']:.3f} -> {run_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 4 training experiment runner")
    parser.add_argument("--features", choices=list(FEATURE_CONFIGS), default="fused")
    parser.add_argument("--classifier", choices=CLASSIFIER_NAMES + ["all"], default="all")
    parser.add_argument("--pca", default="none", help="'none' or an integer component count, e.g. 64")
    parser.add_argument("--imbalance", choices=["class_weight", "oversample"], default="class_weight")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_dataset()  # resume-safe: no-op if train/validation/test.csv already exist

    classifiers = CLASSIFIER_NAMES if args.classifier == "all" else [args.classifier]
    results = {}
    for name in classifiers:
        results[name] = run_experiment(args.features, name, args.pca, args.imbalance, force=args.force)

    if len(results) > 1:
        print(f"\ncomparison (--features {args.features} --pca {args.pca} --imbalance {args.imbalance}):")
        for name, m in sorted(results.items(), key=lambda kv: -kv[1]["val_f1"]):
            print(f"  {name}: precision={m['val_precision']:.3f} recall={m['val_recall']:.3f} "
                  f"f1={m['val_f1']:.3f} roc_auc={m['val_roc_auc']:.3f} pr_auc={m['val_pr_auc']:.3f}")


if __name__ == "__main__":
    main()
