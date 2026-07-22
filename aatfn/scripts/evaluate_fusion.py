"""
Phase 1: final evaluation of the frozen AATFN fusion model (run 5) on the
held-out test set. Produces every figure/table needed for the results
section: per-head confusion matrices, ROC curves, PR curves, a combined
calibration/reliability plot with Brier scores, a final metrics table
(accuracy/precision/recall/F1/ROC-AUC per head + micro/macro averages +
exact-match + Hamming loss), and bootstrap 95% CIs for the headline
numbers.

Reconstructs the exact same train/val/test split used by train_fusion.py
(same seed=42, same grouped-by-page split) so "test set" here is identical
to what train_fusion.py reported metrics on -- this script doesn't retrain
anything, it just re-evaluates the already-frozen, already-saved model.

Usage:
    cd aatfn/scripts
    python3 evaluate_fusion.py
    python3 evaluate_fusion.py --n-bootstrap 2000   # more precise CIs, slower
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from feature_extractors.paths import AATFN_DIR
from train_fusion import (
    FEATURES_PATH,
    LABEL_COLS,
    MODEL_DIR,
    PATCH_COLS,
    SAA_COLS,
    TYPO_COLS,
    FusionModel,
    _grouped_split,
    _to_tensors,
)

RESULTS_DIR = AATFN_DIR / "results"
HEAD_DISPLAY_NAMES = {"typo": "Typography", "patch": "Patch", "stego": "Steganography"}
THRESHOLD = 0.5


def load_frozen_model_and_test_split(seed: int = 42):
    df = pd.read_csv(FEATURES_PATH)
    _, _, test_df = _grouped_split(df, seed=seed)

    import joblib

    saa_scaler = joblib.load(MODEL_DIR / "saa_scaler.joblib")
    typo_scaler = joblib.load(MODEL_DIR / "typo_scaler.joblib")
    patch_scaler = joblib.load(MODEL_DIR / "patch_scaler.joblib")

    saa_te, typo_te, patch_te, y_te = _to_tensors(test_df, saa_scaler, typo_scaler, patch_scaler)

    model = FusionModel(len(SAA_COLS), len(TYPO_COLS), len(PATCH_COLS))
    model.load_state_dict(torch.load(MODEL_DIR / "fusion_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        typo_logit, patch_logit, stego_logit, attn_weights = model(saa_te, typo_te, patch_te)
        probs = torch.stack([
            torch.sigmoid(typo_logit), torch.sigmoid(patch_logit), torch.sigmoid(stego_logit)
        ], dim=1).numpy()

    y_true = y_te.numpy().astype(int)
    return test_df, y_true, probs, attn_weights.numpy()


def plot_confusion_matrices(y_true, probs):
    for i, name in enumerate(LABEL_COLS):
        pred = (probs[:, i] >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_true[:, i], pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No attack", "Attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No attack", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{HEAD_DISPLAY_NAMES[name]} — Confusion Matrix")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_{name}.pdf")
        plt.close(fig)


def plot_roc_curves(y_true, probs):
    aucs = {}
    for i, name in enumerate(LABEL_COLS):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], probs[:, i])
        auc_val = roc_auc_score(y_true[:, i], probs[:, i])
        aucs[name] = float(auc_val)

        op_idx = int(np.argmin(np.abs(thresholds - THRESHOLD)))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
        ax.plot(fpr[op_idx], tpr[op_idx], "o", color="red",
                label=f"Operating point (t={THRESHOLD})")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{HEAD_DISPLAY_NAMES[name]} — ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"roc_{name}.pdf")
        plt.close(fig)
    return aucs


def plot_pr_curves(y_true, probs):
    ap_scores = {}
    for i, name in enumerate(LABEL_COLS):
        precision, recall, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        ap = average_precision_score(y_true[:, i], probs[:, i])
        ap_scores[name] = float(ap)
        base_rate = float(y_true[:, i].mean())

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
        ax.axhline(base_rate, linestyle="--", color="gray", label=f"Base rate ({base_rate:.2f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{HEAD_DISPLAY_NAMES[name]} — Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"pr_{name}.pdf")
        plt.close(fig)
    return ap_scores


def plot_roc_combined(y_true, probs):
    """All 3 heads' ROC curves on one figure -- paper-ready single-panel version."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance (AUC = 0.500)")
    for i, name in enumerate(LABEL_COLS):
        fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
        auc_val = roc_auc_score(y_true[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{HEAD_DISPLAY_NAMES[name]} (AUC = {auc_val:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Heads")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_combined.pdf")
    plt.close(fig)


def plot_pr_combined(y_true, probs):
    """All 3 heads' PR curves on one figure -- paper-ready single-panel version."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, name in enumerate(LABEL_COLS):
        precision, recall, _ = precision_recall_curve(y_true[:, i], probs[:, i])
        ap = average_precision_score(y_true[:, i], probs[:, i])
        ax.plot(recall, precision, label=f"{HEAD_DISPLAY_NAMES[name]} (AP = {ap:.3f})")
    ax.axhline(0.5, linestyle="--", color="gray", label="Base rate (0.50, all heads)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Heads")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_combined.pdf")
    plt.close(fig)


def plot_calibration(y_true, probs):
    brier_scores = {}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    for i, name in enumerate(LABEL_COLS):
        brier = brier_score_loss(y_true[:, i], probs[:, i])
        brier_scores[name] = float(brier)
        prob_true, prob_pred = calibration_curve(y_true[:, i], probs[:, i], n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o",
                label=f"{HEAD_DISPLAY_NAMES[name]} (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram (all heads)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "calibration.pdf")
    plt.close(fig)
    return brier_scores


def compute_final_metrics_table(y_true, probs, roc_aucs, ap_scores, brier_scores):
    pred = (probs >= THRESHOLD).astype(int)

    per_head = {}
    for i, name in enumerate(LABEL_COLS):
        per_head[name] = {
            "accuracy": float(accuracy_score(y_true[:, i], pred[:, i])),
            "precision": float(precision_score(y_true[:, i], pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], pred[:, i], zero_division=0)),
            "roc_auc": roc_aucs[name],
            "average_precision": ap_scores[name],
            "brier_score": brier_scores[name],
            "base_rate": float(y_true[:, i].mean()),
        }

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, pred, average="micro", zero_division=0)

    overall = {
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f1),
        "exact_match_accuracy": float(np.mean(np.all(pred == y_true, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, pred)),
    }
    return {"per_head": per_head, "overall": overall}


def bootstrap_cis(y_true, probs, n_bootstrap: int, seed: int = 42, alpha: float = 0.05):
    """95% bootstrap CIs (percentile method) for per-head F1 and exact-match accuracy."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pred = (probs >= THRESHOLD).astype(int)

    boot_f1 = {name: [] for name in LABEL_COLS}
    boot_exact = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], pred[idx]
        for i, name in enumerate(LABEL_COLS):
            boot_f1[name].append(f1_score(yt[:, i], yp[:, i], zero_division=0))
        boot_exact.append(np.mean(np.all(yp == yt, axis=1)))

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    cis = {}
    for name in LABEL_COLS:
        arr = np.array(boot_f1[name])
        cis[f"{name}_f1"] = {
            "point": float(f1_score(y_true[:, LABEL_COLS.index(name)], pred[:, LABEL_COLS.index(name)], zero_division=0)),
            "ci_lo": float(np.percentile(arr, lo)), "ci_hi": float(np.percentile(arr, hi)),
        }
    arr = np.array(boot_exact)
    cis["exact_match_accuracy"] = {
        "point": float(np.mean(np.all(pred == y_true, axis=1))),
        "ci_lo": float(np.percentile(arr, lo)), "ci_hi": float(np.percentile(arr, hi)),
    }
    return cis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if not (MODEL_DIR / "fusion_model.pt").exists():
        raise SystemExit(f"{MODEL_DIR / 'fusion_model.pt'} not found -- run train_fusion.py first")

    test_df, y_true, probs, attn_weights = load_frozen_model_and_test_split(seed=args.seed)
    print(f"Evaluating frozen model on {len(test_df)} held-out test images")
    print(f"Mean attention weights (saa, typo, patch): {attn_weights.mean(axis=0).round(3).tolist()}")

    plot_confusion_matrices(y_true, probs)
    roc_aucs = plot_roc_curves(y_true, probs)
    plot_roc_combined(y_true, probs)
    plot_pr_combined(y_true, probs)
    ap_scores = plot_pr_curves(y_true, probs)
    brier_scores = plot_calibration(y_true, probs)
    print(f"Saved 10 figures -> {RESULTS_DIR}")

    metrics_table = compute_final_metrics_table(y_true, probs, roc_aucs, ap_scores, brier_scores)
    print("\n=== Final metrics table ===")
    print(json.dumps(metrics_table, indent=2))

    print(f"\nRunning {args.n_bootstrap}-sample bootstrap for 95% CIs...")
    cis = bootstrap_cis(y_true, probs, n_bootstrap=args.n_bootstrap, seed=args.seed)
    print("\n=== Bootstrap 95% CIs ===")
    for k, v in cis.items():
        print(f"  {k}: {v['point']:.3f}  (95% CI [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}])")

    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump({"metrics": metrics_table, "bootstrap_ci": cis,
                    "mean_attention_weights": {"saa": float(attn_weights.mean(axis=0)[0]),
                                                "typo": float(attn_weights.mean(axis=0)[1]),
                                                "patch": float(attn_weights.mean(axis=0)[2])}}, f, indent=2)

    # flat CSV version of the per-head table, handy for pasting into a paper
    rows = []
    for name in LABEL_COLS:
        row = {"head": HEAD_DISPLAY_NAMES[name], **metrics_table["per_head"][name]}
        row["f1_ci_lo"] = cis[f"{name}_f1"]["ci_lo"]
        row["f1_ci_hi"] = cis[f"{name}_f1"]["ci_hi"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "final_metrics_per_head.csv", index=False)

    # single-row "overall" table -- macro/micro averages, exact-match, hamming loss
    overall_row = {**metrics_table["overall"]}
    overall_row["exact_match_ci_lo"] = cis["exact_match_accuracy"]["ci_lo"]
    overall_row["exact_match_ci_hi"] = cis["exact_match_accuracy"]["ci_hi"]
    pd.DataFrame([overall_row]).to_csv(RESULTS_DIR / "main_results.csv", index=False)

    print(f"\nSaved final_metrics.json + final_metrics_per_head.csv + main_results.csv -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
