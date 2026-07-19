"""Phase 3: feature fusion.

Joins cnn_features.csv (1280-dim EfficientNet-B0 embedding per candidate)
with handcrafted_features.csv (27 patch-local descriptors per candidate) on
candidate_id, into one raw fused_features.csv.

Deliberately no PCA or any other learned transform here (locked design -
see MEMORY): dimensionality reduction is a training-time concern. Fitting
PCA inside dataset-building would make the dataset depend on a specific
learned transform, so regenerating a different train/val/test split would
require rebuilding the whole fused dataset. Keeping this CSV raw and full-
dimensional keeps it reusable across any future split or model choice; PCA
(or feature selection, or nothing) belongs in train.py, fit only on the
training split.
"""

import pandas as pd

from adversarial.config import OUTPUTS_DIR
from adversarial.features.cnn_features import CNN_FEATURES_PATH, extract_cnn_features
from adversarial.features.handcrafted_features import HANDCRAFTED_FEATURES_PATH, extract_handcrafted_features

FUSED_FEATURES_PATH = OUTPUTS_DIR / "fused_features.csv"

_METADATA_COLUMNS = ["candidate_id", "source_image_id", "dataset", "label", "source_type"]


def fuse_features(
    labels_path=None,
    cnn_path=CNN_FEATURES_PATH,
    handcrafted_path=HANDCRAFTED_FEATURES_PATH,
    output_path=FUSED_FEATURES_PATH,
    force: bool = False,
) -> pd.DataFrame:
    if output_path.exists() and not force:
        print(f"{output_path} already exists - not regenerating (pass force=True to override deliberately)")
        return pd.read_csv(output_path)

    extract_cnn_kwargs = {"output_path": cnn_path}
    extract_handcrafted_kwargs = {"output_path": handcrafted_path}
    if labels_path is not None:
        extract_cnn_kwargs["labels_path"] = labels_path
        extract_handcrafted_kwargs["labels_path"] = labels_path

    if not cnn_path.exists():
        extract_cnn_features(**extract_cnn_kwargs)
    if not handcrafted_path.exists():
        extract_handcrafted_features(**extract_handcrafted_kwargs)

    cnn_df = pd.read_csv(cnn_path)
    handcrafted_df = pd.read_csv(handcrafted_path)

    if cnn_df["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate_id in cnn_features.csv")
    if handcrafted_df["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate_id in handcrafted_features.csv")

    handcrafted_only = handcrafted_df.drop(columns=[c for c in _METADATA_COLUMNS if c != "candidate_id"])
    fused_df = cnn_df.merge(handcrafted_only, on="candidate_id", how="inner", validate="one_to_one")

    if len(fused_df) != len(cnn_df) or len(fused_df) != len(handcrafted_df):
        raise ValueError(
            f"row count mismatch after join: cnn={len(cnn_df)}, handcrafted={len(handcrafted_df)}, "
            f"fused={len(fused_df)} - candidate_id sets differ between feature files"
        )

    if fused_df.isna().any().any():
        raise ValueError("NaNs present in fused_features.csv after join")

    fused_df = fused_df.sort_values("candidate_id").reset_index(drop=True)
    fused_df.to_csv(output_path, index=False)
    print(f"wrote {len(fused_df)} rows, {fused_df.shape[1]} columns -> {output_path}")
    return fused_df


if __name__ == "__main__":
    fuse_features()
