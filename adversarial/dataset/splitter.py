"""Phase 2: train/validation/test splitter for candidate-level rows.

Groups candidate_labels.csv rows by source_image_id (the ORIGINAL clean
image id - shared by a clean image's own candidates and its attacked
counterpart's candidates) so a document's clean and attacked candidates
always land in the same split, preventing the classifier from partially
memorizing a document rather than learning the patch-detection task.
Groups are further stratified by source dataset (FUNSD/CORD/SROIE) so each
split gets a proportional share of all three.
"""

import random

import pandas as pd

from adversarial.config import OUTPUTS_DIR
from adversarial.dataset.dataset_builder import CANDIDATE_LABELS_PATH
from typographic.config import RANDOM_SEED, TEST_FRACTION, TRAIN_FRACTION, VALIDATION_FRACTION

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

    if not CANDIDATE_LABELS_PATH.exists():
        raise FileNotFoundError(f"{CANDIDATE_LABELS_PATH} not found - run dataset_builder.build_candidate_dataset() first")

    df = pd.read_csv(CANDIDATE_LABELS_PATH)
    rng = random.Random(RANDOM_SEED)

    train_ids, val_ids, test_ids = set(), set(), set()
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
