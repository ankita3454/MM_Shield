"""Phase 3: CNN feature extraction for candidate crops.

Extracts EfficientNet-B0 (ImageNet-pretrained, classifier head replaced with
Identity) embeddings for every candidate crop from Phase 2's
candidate_labels.csv. Deterministic (eval mode - no dropout/augmentation),
batched for speed, resume-safe (skips entirely if cnn_features.csv already
exists, matching the idempotent pattern used everywhere else in this
project - inference over ~3000 crops is fast enough that whole-file resume
is sufficient, no need for finer-grained per-row checkpointing).

There is no existing EfficientNet-B0 loading code in this repo to reuse -
the Typographic module has no CNN/torch dependency at all (PaddleOCR +
MiniLM only). This follows the same lazy-singleton-model conventions
established there (see typographic/features/semantic_features.py).
"""

import pandas as pd
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from adversarial.config import OUTPUTS_DIR
from adversarial.dataset.dataset_builder import CANDIDATE_LABELS_PATH
from typographic.config import DATASETS_DIR

CNN_FEATURES_PATH = OUTPUTS_DIR / "cnn_features.csv"
CNN_EMBEDDING_DIM = 1280
BATCH_SIZE = 16

_model = None
_transform = None
_device = None


def _get_model():
    global _model, _transform, _device
    if _model is None:
        _device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        model.to(_device)
        _model = model
        _transform = weights.transforms()
    return _model, _transform, _device


def embed_crops(image_paths: list) -> list[list[float]]:
    """Raw 1280-dim EfficientNet-B0 embedding per image path, batched."""
    model, transform, device = _get_model()
    embeddings = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            batch_embeddings = model(batch)
        embeddings.extend(batch_embeddings.cpu().numpy().tolist())
    return embeddings


def extract_cnn_features(
    labels_path=CANDIDATE_LABELS_PATH,
    output_path=CNN_FEATURES_PATH,
    force: bool = False,
) -> pd.DataFrame:
    if output_path.exists() and not force:
        print(f"{output_path} already exists - not regenerating (pass force=True to override deliberately)")
        return pd.read_csv(output_path)

    df = pd.read_csv(labels_path)
    feature_names = [f"cnn_{i}" for i in range(CNN_EMBEDDING_DIM)]

    rows = []
    total = len(df)
    for i in range(0, total, BATCH_SIZE):
        batch_df = df.iloc[i:i + BATCH_SIZE]
        paths = [DATASETS_DIR / p for p in batch_df["crop_file"]]
        embeddings = embed_crops(paths)
        for (_, candidate), embedding in zip(batch_df.iterrows(), embeddings):
            row = {
                "candidate_id": candidate["candidate_id"],
                "source_image_id": candidate["source_image_id"],
                "dataset": candidate["dataset"],
                "label": candidate["label"],
                "source_type": candidate["source_type"],
            }
            row.update(dict(zip(feature_names, embedding)))
            rows.append(row)

        done = min(i + BATCH_SIZE, total)
        if done % (BATCH_SIZE * 10) == 0 or done == total:
            print(f"processed {done}/{total} candidates")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"wrote {len(out_df)} rows -> {output_path}")
    return out_df


if __name__ == "__main__":
    extract_cnn_features()
