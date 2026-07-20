"""
Baseline SAA validation: extract the frozen 21-feature vector for every
clean/stego image, fit StandardScaler + a classifier, and report accuracy
overall and broken out per source dataset (FUNSD/CORD/SROIE), matching the
per-dataset reporting used in the prior (deleted) codebase's results table.

Mandatory per the design contract: StandardScaler always precedes the
classifier, since feature magnitudes span many orders of magnitude
(e.g. total_frequency_energy ~1e14 vs lsb_ratio ~0.3).

Usage:
    python validate.py --clean-dir ../datasets/clean --stego-dir ../datasets/stego
"""
import argparse
import os
from pathlib import Path

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

from extractor import stego_analyzer, FEATURE_NAMES


def build_dataset(clean_dir: str, stego_dir: str, out_csv: str = None, flush_every: int = 10,
                   resume: bool = True) -> pd.DataFrame:
    """
    Walk clean_dir/<dataset_name>/*.png and stego_dir/<dataset_name>/*.png,
    extract the 21-feature vector for each image, and return a tidy
    DataFrame with columns: [*FEATURE_NAMES, label, dataset, path].
    label: 0 = clean, 1 = stego

    Resumable: if `out_csv` already exists, any image `path` already present
    there is skipped (feature extraction re-run only on new/missing images),
    and progress is flushed to `out_csv` every `flush_every` newly-extracted
    images. This matters because extracting all 21 features (FFT, SRM
    convolutions, sliding-window variance, chi-square) over ~300 full-size
    document images is slow enough to hit a wall-clock timeout partway
    through -- re-running the same command just picks up where it left off.
    """
    clean_dir = Path(clean_dir)
    stego_dir = Path(stego_dir)

    existing_rows = []
    done_paths = set()
    if resume and out_csv and os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        existing_rows = existing_df.to_dict("records")
        done_paths = set(existing_df["path"])
        print(f"  resuming from {out_csv}: {len(done_paths)} images already extracted")

    rows = list(existing_rows)
    new_count = 0

    for label, root in ((0, clean_dir), (1, stego_dir)):
        if not root.exists():
            continue
        for dataset_subdir in sorted(root.iterdir()):
            if not dataset_subdir.is_dir():
                continue
            for img_path in sorted(dataset_subdir.iterdir()):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                path_str = str(img_path)
                if path_str in done_paths:
                    continue
                try:
                    vec = stego_analyzer(path_str)
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: failed on {img_path}: {e}")
                    continue
                row = dict(zip(FEATURE_NAMES, vec))
                row["label"] = label
                row["dataset"] = dataset_subdir.name
                row["path"] = path_str
                rows.append(row)
                done_paths.add(path_str)
                new_count += 1

                if out_csv and new_count % flush_every == 0:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                    print(f"  ...extracted {len(rows)} total ({new_count} new this run), checkpointed -> {out_csv}")

    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def _build_classifier(name: str, seed: int, max_features="sqrt"):
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    if name == "rf":
        # class_weight="balanced" matters here: mixing FUNSD (huge, easy
        # signal) with CORD/SROIE (weak signal) in one training set means an
        # unweighted tree ensemble can end up dominated by whichever
        # dataset's split of examples happens to be easiest to separate.
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features=max_features,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lgbm":
        # Experiment 007: boosting instead of bagging. min_child_samples
        # lowered from LightGBM's default (20) since our training set is
        # only 210 rows -- the default would forbid most useful splits.
        if not _HAS_LGBM:
            raise ImportError("lightgbm is not installed -- pip install lightgbm")
        return LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            min_child_samples=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown classifier '{name}', expected 'svm', 'rf', or 'lgbm'")


def run_validation(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42, classifier: str = "rf",
                    max_features="sqrt") -> dict:
    X = df[FEATURE_NAMES].values
    y = df["label"].values
    groups = df["dataset"].values

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = _build_classifier(classifier, seed, max_features=max_features)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    overall_acc = accuracy_score(y_test, y_pred)

    per_dataset_acc = {}
    for ds_name in np.unique(groups_test):
        mask = groups_test == ds_name
        per_dataset_acc[ds_name] = accuracy_score(y_test[mask], y_pred[mask])

    report = classification_report(y_test, y_pred, target_names=["clean", "stego"])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # rows=true, cols=pred; [[TN,FP],[FN,TP]]
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])

    feature_importance = None
    if classifier in ("rf", "lgbm"):
        # cast to plain float -- LightGBM returns int32 split counts, which
        # json.dump chokes on downstream in save_frozen_model
        feature_importance = sorted(
            zip(FEATURE_NAMES, (float(v) for v in clf.feature_importances_)), key=lambda x: -x[1]
        )

    return {
        "overall_accuracy": overall_acc,
        "per_dataset_accuracy": per_dataset_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "precision": {"clean": float(precision[0]), "stego": float(precision[1])},
        "recall": {"clean": float(recall[0]), "stego": float(recall[1])},
        "f1": {"clean": float(f1[0]), "stego": float(f1[1])},
        "feature_importance": feature_importance,
        "scaler": scaler,
        "clf": clf,
        "classifier_name": classifier,
        "test_size": test_size,
        "seed": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def save_frozen_model(results: dict, out_path: str) -> None:
    """
    Persist scaler + classifier + all metrics as a single pickle, so a
    "frozen baseline" is an actual reproducible artifact, not just numbers
    in a markdown table that silently go stale the next time someone edits
    extractor.py or validate.py.
    """
    payload = {
        "scaler": results["scaler"],
        "clf": results["clf"],
        "classifier_name": results["classifier_name"],
        "feature_names": FEATURE_NAMES,
        "test_size": results["test_size"],
        "seed": results["seed"],
        "overall_accuracy": results["overall_accuracy"],
        "per_dataset_accuracy": results["per_dataset_accuracy"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "feature_importance": results["feature_importance"],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    # also drop a human-readable JSON sidecar with everything except the
    # actual model objects, for quick diffing between frozen versions
    json_path = os.path.splitext(out_path)[0] + ".json"
    json_payload = {k: v for k, v in payload.items() if k not in ("scaler", "clf")}
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=str, default="../datasets/clean")
    parser.add_argument("--stego-dir", type=str, default="../datasets/stego")
    parser.add_argument("--out-csv", type=str, default="../outputs/features.csv")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier", type=str, default="rf", choices=["svm", "rf", "lgbm"])
    parser.add_argument("--max-features", type=str, default="sqrt",
                         help="RF only: 'sqrt', 'None', or an int (e.g. '5')")
    parser.add_argument("--force-reextract", action="store_true",
                         help="ignore any existing --out-csv and recompute every feature from scratch")
    parser.add_argument("--save-model", type=str, default=None,
                         help="path to freeze scaler+classifier+metrics as a pickle (+ .json sidecar)")
    args = parser.parse_args()

    max_features = args.max_features
    if max_features == "None":
        max_features = None
    elif max_features.isdigit():
        max_features = int(max_features)

    print("Extracting features for all clean + stego images...")
    df = build_dataset(args.clean_dir, args.stego_dir, out_csv=args.out_csv, resume=not args.force_reextract)
    print(f"Total images processed: {len(df)} "
          f"(clean={sum(df['label'] == 0)}, stego={sum(df['label'] == 1)})")
    print(f"Saved feature table -> {args.out_csv}")

    print(f"\nTraining StandardScaler + {args.classifier.upper()} ...")
    results = run_validation(df, test_size=args.test_size, seed=args.seed, classifier=args.classifier,
                              max_features=max_features)

    print(f"\nTrain/test split: {results['n_train']} train / {results['n_test']} test (seed={results['seed']})")
    print(f"\nOverall accuracy: {results['overall_accuracy']:.4f}")
    print("\nPer-dataset accuracy:")
    for name, acc in results["per_dataset_accuracy"].items():
        print(f"  {name}: {acc:.4f}")
    print("\nClassification report:")
    print(results["classification_report"])

    cm = results["confusion_matrix"]
    print("Confusion matrix (rows=true, cols=predicted; order=[clean, stego]):")
    print(f"                 pred_clean  pred_stego")
    print(f"  true_clean     {cm[0][0]:>10d}  {cm[0][1]:>10d}")
    print(f"  true_stego     {cm[1][0]:>10d}  {cm[1][1]:>10d}")

    if results["feature_importance"]:
        print(f"\nTop 10 feature importances ({args.classifier.upper()}):")
        for name, imp in results["feature_importance"][:10]:
            print(f"  {name:28s} {imp:.4f}")

    if args.save_model:
        save_frozen_model(results, args.save_model)
        print(f"\nFroze model + metrics -> {args.save_model} (+ {os.path.splitext(args.save_model)[0]}.json)")


if __name__ == "__main__":
    main()
