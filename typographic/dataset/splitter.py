"""Phase: train/validation/test splitter.

Runs AFTER feature extraction (deliberately reordered from the original spec)
so split membership never has to be threaded through per-image feature files.

Groups feature_dataset.csv rows by source_image_id - a clean image and its
malicious counterpart always land in the same split, preventing the
classifier from partially memorizing a document rather than learning the
injection pattern. Groups are further stratified by source dataset
(FUNSD/CORD/SROIE) so each split gets a proportional share of all three,
not an accidental skew.
"""

import random

import pandas as pd

from typographic.config import (
    OUTPUTS_DIR,
    RANDOM_SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from typographic.dataset.dataset_builder import FEATURE_DATASET_PATH

TRAIN_PATH = OUTPUTS_DIR / "train.csv"
VALIDATION_PATH = OUTPUTS_DIR / "validation.csv"
TEST_PATH = OUTPUTS_DIR / "test.csv"


def _split_groups(group_ids: list[str], rng: random.Random) -> tuple[set, set, set]:
    shuffled = group_ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * TRAIN_FRACTION)
    n_val = round(n * VALIDATION_FRACTION)
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])
    test_ids = set(shuffled[n_train + n_val:])
    return train_ids, val_ids, test_ids


def split_dataset(force: bool = False) -> None:
    if TRAIN_PATH.exists() and VALIDATION_PATH.exists() and TEST_PATH.exists() and not force:
        print(f"train/validation/test.csv already exist under {OUTPUTS_DIR} - not regenerating "
              f"(pass force=True to override deliberately)")
        return

    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(f"{FEATURE_DATASET_PATH} not found - run dataset_builder.build_dataset() first")

    df = pd.read_csv(FEATURE_DATASET_PATH)
    rng = random.Random(RANDOM_SEED)

    train_ids, val_ids, test_ids = set(), set(), set()
    # Stratify by dataset: split each source dataset's groups independently,
    # then union, so FUNSD/CORD/SROIE are proportionally represented in every split.
    for dataset_name, dataset_df in df.groupby("dataset"):
        group_ids = sorted(dataset_df["source_image_id"].unique())
        t_ids, v_ids, te_ids = _split_groups(group_ids, rng)
        train_ids |= t_ids
        val_ids |= v_ids
        test_ids |= te_ids

    train_df = df[df["source_image_id"].isin(train_ids)]
    val_df = df[df["source_image_id"].isin(val_ids)]
    test_df = df[df["source_image_id"].isin(test_ids)]

    assert not (set(train_df["source_image_id"]) & set(val_df["source_image_id"]))
    assert not (set(train_df["source_image_id"]) & set(test_df["source_image_id"]))
    assert not (set(val_df["source_image_id"]) & set(test_df["source_image_id"]))

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"train: {len(train_df)} rows ({len(train_ids)} groups) -> {TRAIN_PATH}")
    print(f"validation: {len(val_df)} rows ({len(val_ids)} groups) -> {VALIDATION_PATH}")
    print(f"test: {len(test_df)} rows ({len(test_ids)} groups) -> {TEST_PATH}")


if __name__ == "__main__":
    split_dataset()
