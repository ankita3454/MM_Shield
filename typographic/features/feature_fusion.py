"""Phase: feature fusion.

Combines the 12 document-level typography features with the 8 document-level
semantic similarity features into one fixed, deterministic 20-dim vector.
Consumes only the unified OCR region format - no dataset-specific logic.
"""

from PIL import Image

from typographic.features import semantic_features, typography_features


def get_feature_names() -> list[str]:
    return typography_features.RAW_FEATURE_NAMES + semantic_features.get_semantic_feature_names()


def fuse_features(image: Image.Image, regions: list[dict]) -> dict:
    """Extract typography + semantic document-level features and fuse them into
    one fixed-length vector, given an image and its OCR regions (unified format)."""
    typo_doc_features = typography_features.extract_typography_features(image, regions)
    semantic_doc_features = semantic_features.extract_semantic_features(regions)

    feature_names = get_feature_names()
    fused = {**typo_doc_features, **semantic_doc_features}
    fused_vector = [fused[name] for name in feature_names]

    return {
        "num_regions": len(regions),
        "typography_features": typo_doc_features,
        "semantic_features": semantic_doc_features,
        "feature_names": feature_names,
        "fused_vector": fused_vector,
    }
