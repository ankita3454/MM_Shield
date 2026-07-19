"""Phase: semantic feature extraction.

Generates raw all-MiniLM-L6-v2 sentence embeddings per OCR region (no
training - the model is used purely for inference), but per the locked
anti-overfitting design, only a derived similarity-score summary against the
attack phrase bank (attack_templates.json) is fused into the classifier's
feature vector - never the raw 384-dim embedding directly. This keeps the
semantic contribution from dominating the much lower-dimensional typography
features on a ~300-image dataset.

Consumes only the unified region format {text, bbox, ...} produced by ocr.py.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from typographic.config import ATTACK_TEMPLATES_PATH

_MODEL_NAME = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.5

_model = None
_category_centroids = None  # dict[str, np.ndarray], built once from attack_templates.json


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _get_category_centroids() -> dict:
    global _category_centroids
    if _category_centroids is None:
        templates = json.loads(ATTACK_TEMPLATES_PATH.read_text())
        model = _get_model()
        centroids = {}
        for category, phrases in templates.items():
            embeddings = model.encode(phrases, normalize_embeddings=True)
            centroids[category] = np.mean(embeddings, axis=0)
        _category_centroids = centroids
    return _category_centroids


def embed_regions(regions: list[dict]) -> np.ndarray:
    """Raw MiniLM embeddings, one 384-dim row per region. Not fused directly -
    exposed for inspection/debugging and as the basis for the similarity features."""
    if not regions:
        return np.zeros((0, 384))
    model = _get_model()
    texts = [r.get("text", "") for r in regions]
    return model.encode(texts, normalize_embeddings=True)


def _region_category_similarities(embeddings: np.ndarray) -> dict:
    """cosine similarity (dot product, since embeddings are normalized) of each
    region against each category centroid -> {category: [sim_per_region]}"""
    centroids = _get_category_centroids()
    return {category: embeddings @ centroid for category, centroid in centroids.items()}


def aggregate_document_features(regions: list[dict], embeddings: np.ndarray | None = None) -> dict:
    """8 document-level features: max similarity per category (6), overall max (1),
    count of regions above the injection-similarity threshold (1)."""
    categories = list(_get_category_centroids().keys())
    if not regions:
        doc_features = {f"{cat}_max_sim": 0.0 for cat in categories}
        doc_features["overall_max_sim"] = 0.0
        doc_features["count_above_threshold"] = 0.0
        return doc_features

    if embeddings is None:
        embeddings = embed_regions(regions)

    per_category_sims = _region_category_similarities(embeddings)

    doc_features = {f"{cat}_max_sim": float(np.max(sims)) for cat, sims in per_category_sims.items()}
    all_sims = np.stack(list(per_category_sims.values()), axis=0)  # (num_categories, num_regions)
    doc_features["overall_max_sim"] = float(np.max(all_sims))
    doc_features["count_above_threshold"] = float(np.sum(np.max(all_sims, axis=0) > _SIMILARITY_THRESHOLD))
    return doc_features


def extract_semantic_features(regions: list[dict]) -> dict:
    """Convenience entry point: raw region embeddings -> 8-dim document vector."""
    embeddings = embed_regions(regions)
    return aggregate_document_features(regions, embeddings)


SEMANTIC_FEATURE_NAMES = None  # populated lazily below, once centroids (and thus categories) are known


def get_semantic_feature_names() -> list[str]:
    global SEMANTIC_FEATURE_NAMES
    if SEMANTIC_FEATURE_NAMES is None:
        categories = list(_get_category_centroids().keys())
        SEMANTIC_FEATURE_NAMES = [f"{cat}_max_sim" for cat in categories] + ["overall_max_sim", "count_above_threshold"]
    return SEMANTIC_FEATURE_NAMES
