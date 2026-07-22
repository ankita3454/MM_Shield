"""
Step 8: train the AATFN fusion classifier on aatfn/features.csv.

Architecture (matches the agreed design: 3 independent binary flags, not an
8-class softmax -- typo/patch/stego attacks are not mutually exclusive):

    SAA branch (21)   -> MLP -> 64-dim embedding  --\
    Typo branch (20)  -> MLP -> 64-dim embedding  ---+-> attention-weighted
    Patch branch(1308)-> MLP -> 64-dim embedding  --/     fusion -> shared
                                                            trunk -> 3 sigmoid
                                                            heads (typo/
                                                            patch/stego)

Attention fusion: a small gating MLP looks at the concatenation of the three
branch embeddings and outputs 3 softmax weights, which weight-sum the
embeddings into one 64-dim shared representation. This lets the model
down-weight a modality that isn't informative for a given image rather than
always averaging all three equally.

Splits are grouped by (dataset, base_image) -- all 8 attack-combo variants
of the same underlying document stay in the same split, so the model can't
"cheat" by memorizing document content that leaked across train/test via a
sibling combo of the same page.

Usage:
    cd aatfn/scripts
    python3 train_fusion.py                  # default 70/15/15 split, 50 epochs
    python3 train_fusion.py --epochs 100 --batch-size 64
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from feature_extractors.paths import AATFN_DIR
from feature_extractors.patch_wrapper import PATCH_FEATURE_NAMES
from feature_extractors.saa_wrapper import SAA_FEATURE_NAMES
from feature_extractors.typography_wrapper import TYPOGRAPHY_FEATURE_NAMES

FEATURES_PATH = AATFN_DIR / "features.csv"
MODEL_DIR = AATFN_DIR / "model"

SAA_COLS = [f"saa_{n}" for n in SAA_FEATURE_NAMES]
TYPO_COLS = [f"typo_{n}" for n in TYPOGRAPHY_FEATURE_NAMES]
PATCH_COLS = PATCH_FEATURE_NAMES
LABEL_COLS = ["typo", "patch", "stego"]  # ground-truth attack flags

EMBED_DIM = 64


class FusionModel(nn.Module):
    """
    Regularization pass (fix 2, applied only after the stego dataset fix
    (fix 1) was verified to add real signal but not by itself close the
    train/val loss gap -- train_loss kept dropping post-fix while val_loss
    still plateaued/climbed after ~epoch 2-6, so the overfitting is a
    genuinely separate problem from the stego label-noise one):

      - patch branch hidden width 256 -> 128 (it was the single largest
        source of parameters by far, at 1308 input dims vs 21/20 for the
        other two branches -- most likely to memorize on 1680 train rows)
      - dropout added after EVERY branch's projection to embed_dim (was
        only inside the patch branch before), so no branch's embedding
        reaches the attention gate un-regularized
      - trunk dropout raised 0.3 -> 0.4, plus a second dropout added
        before the output heads (was only one dropout, mid-trunk)
      - weight_decay raised via the --weight-decay CLI default (see
        argparse below): 1e-4 -> 1e-3

    Fix 2b (single-variable ablation, applied after wd=3e-4 was frozen as
    the best config across all three heads jointly -- typo/patch F1 both
    recovered close to the original run, val_loss and exact-match both
    best of any run so far, but stego F1 stayed flat at 0.54-0.66 across
    every optimizer/regularization setting tried, while typo/patch moved
    a lot -- meaning the bottleneck is the stego/SAA branch specifically,
    not global optimization): the SAA branch was getting the exact same
    dropout (0.2) and shallow 2-layer shape as typo_branch despite having
    only 21 input features vs typo's 20 -- fine on paper, but SAA's
    per-feature signal is also far weaker (lsb_entropy effect size
    d=0.41 was the STRONGEST of the 8 SAA features after the stego
    fix) -- dropping even a small fraction of 21 already-weak features
    plausibly removes a meaningful share of the branch's total
    information. saa_branch gets one more hidden layer (more capacity to
    model interactions among the 21 features before the shared 64-dim
    projection) and near-zero dropout (0.05 instead of 0.2). typo_branch
    and patch_branch are UNCHANGED from the fix-2 config, on purpose --
    isolating this as the only variable in this run.
    """
    def __init__(self, saa_dim, typo_dim, patch_dim, embed_dim=EMBED_DIM):
        super().__init__()
        self.saa_branch = nn.Sequential(
            nn.Linear(saa_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.Dropout(0.05),
        )
        self.typo_branch = nn.Sequential(
            nn.Linear(typo_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim), nn.Dropout(0.2)
        )
        self.patch_branch = nn.Sequential(
            nn.Linear(patch_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, embed_dim), nn.Dropout(0.3),
        )

        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 3, 32), nn.ReLU(), nn.Linear(32, 3)
        )

        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.typo_head = nn.Linear(32, 1)
        self.patch_head = nn.Linear(32, 1)
        self.stego_head = nn.Linear(32, 1)

    def forward(self, saa_x, typo_x, patch_x):
        e_saa = self.saa_branch(saa_x)
        e_typo = self.typo_branch(typo_x)
        e_patch = self.patch_branch(patch_x)

        stacked = torch.stack([e_saa, e_typo, e_patch], dim=1)  # (B, 3, embed_dim)
        gate_input = torch.cat([e_saa, e_typo, e_patch], dim=1)  # (B, 3*embed_dim)
        weights = torch.softmax(self.attention(gate_input), dim=1)  # (B, 3)
        shared = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, embed_dim)

        trunk_out = self.trunk(shared)
        return (
            self.typo_head(trunk_out).squeeze(-1),
            self.patch_head(trunk_out).squeeze(-1),
            self.stego_head(trunk_out).squeeze(-1),
            weights,
        )


def _grouped_split(df, test_frac=0.15, val_frac=0.15, seed=42):
    groups = df["dataset"] + "_" + df["base_image"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx]
    train_val_groups = groups.iloc[train_val_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    train_idx_rel, val_idx_rel = next(gss2.split(train_val_df, groups=train_val_groups))

    train_df = train_val_df.iloc[train_idx_rel]
    val_df = train_val_df.iloc[val_idx_rel]
    test_df = df.iloc[test_idx]
    return train_df, val_df, test_df


def _to_tensors(df, saa_scaler, typo_scaler, patch_scaler, fit=False):
    saa_x = df[SAA_COLS].values.astype(np.float32)
    typo_x = df[TYPO_COLS].values.astype(np.float32)
    patch_x = df[PATCH_COLS].values.astype(np.float32)

    if fit:
        saa_x = saa_scaler.fit_transform(saa_x)
        typo_x = typo_scaler.fit_transform(typo_x)
        patch_x = patch_scaler.fit_transform(patch_x)
    else:
        saa_x = saa_scaler.transform(saa_x)
        typo_x = typo_scaler.transform(typo_x)
        patch_x = patch_scaler.transform(patch_x)

    y = df[LABEL_COLS].values.astype(np.float32)
    return (
        torch.tensor(saa_x), torch.tensor(typo_x), torch.tensor(patch_x), torch.tensor(y)
    )


def evaluate(model, saa_x, typo_x, patch_x, y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        typo_logit, patch_logit, stego_logit, _ = model(saa_x, typo_x, patch_x)
        preds = torch.stack([
            torch.sigmoid(typo_logit), torch.sigmoid(patch_logit), torch.sigmoid(stego_logit)
        ], dim=1).numpy()
    pred_labels = (preds >= threshold).astype(int)
    true_labels = y.numpy().astype(int)

    metrics = {}
    for i, name in enumerate(LABEL_COLS):
        metrics[name] = {
            "precision": float(precision_score(true_labels[:, i], pred_labels[:, i], zero_division=0)),
            "recall": float(recall_score(true_labels[:, i], pred_labels[:, i], zero_division=0)),
            "f1": float(f1_score(true_labels[:, i], pred_labels[:, i], zero_division=0)),
        }
    exact_match = float(np.mean(np.all(pred_labels == true_labels, axis=1)))
    metrics["exact_match_accuracy"] = exact_match
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=3e-4,
                     help="frozen at 3e-4 -- best val_loss/exact-match across a 1e-4/3e-4/1e-3 sweep, "
                          "see pipeline.md for the comparison table")
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience on val loss")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not FEATURES_PATH.exists():
        raise SystemExit(f"{FEATURES_PATH} not found -- run extract_features.py first")

    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} rows from {FEATURES_PATH}")

    train_df, val_df, test_df = _grouped_split(df, seed=args.seed)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
          f"(grouped by dataset+base_image, no leakage across combos of the same page)")

    saa_scaler, typo_scaler, patch_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    saa_tr, typo_tr, patch_tr, y_tr = _to_tensors(train_df, saa_scaler, typo_scaler, patch_scaler, fit=True)
    saa_val, typo_val, patch_val, y_val = _to_tensors(val_df, saa_scaler, typo_scaler, patch_scaler)
    saa_te, typo_te, patch_te, y_te = _to_tensors(test_df, saa_scaler, typo_scaler, patch_scaler)

    model = FusionModel(len(SAA_COLS), len(TYPO_COLS), len(PATCH_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    n = len(train_df)
    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            typo_logit, patch_logit, stego_logit, _ = model(saa_tr[idx], typo_tr[idx], patch_tr[idx])
            loss = (
                criterion(typo_logit, y_tr[idx, 0])
                + criterion(patch_logit, y_tr[idx, 1])
                + criterion(stego_logit, y_tr[idx, 2])
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            typo_logit, patch_logit, stego_logit, _ = model(saa_val, typo_val, patch_val)
            val_loss = (
                criterion(typo_logit, y_val[:, 0])
                + criterion(patch_logit, y_val[:, 1])
                + criterion(stego_logit, y_val[:, 2])
            ).item()

        print(f"epoch {epoch:3d}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    val_metrics = evaluate(model, saa_val, typo_val, patch_val, y_val)
    test_metrics = evaluate(model, saa_te, typo_te, patch_te, y_te)

    print("\n=== Validation metrics ===")
    print(json.dumps(val_metrics, indent=2))
    print("\n=== Test metrics ===")
    print(json.dumps(test_metrics, indent=2))

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "fusion_model.pt")
    import joblib
    joblib.dump(saa_scaler, MODEL_DIR / "saa_scaler.joblib")
    joblib.dump(typo_scaler, MODEL_DIR / "typo_scaler.joblib")
    joblib.dump(patch_scaler, MODEL_DIR / "patch_scaler.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump({
            "saa_dim": len(SAA_COLS), "typo_dim": len(TYPO_COLS), "patch_dim": len(PATCH_COLS),
            "embed_dim": EMBED_DIM, "label_cols": LABEL_COLS,
        }, f, indent=2)

    print(f"\nSaved model + scalers + metrics -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
