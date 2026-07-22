"""Thin wrapper around typographic/features/{ocr,feature_fusion}.py.

20-dim fused vector (12 typography + 8 semantic). Runs PaddleOCR (loaded
lazily, once) then the typography+semantic fusion on the resulting regions.
"""
from PIL import Image

from .paths import setup_sys_path

setup_sys_path()
from typographic.features import feature_fusion, ocr  # noqa: E402

TYPOGRAPHY_FEATURE_NAMES = feature_fusion.get_feature_names()
TYPOGRAPHY_DIM = len(TYPOGRAPHY_FEATURE_NAMES)  # 20


def extract_typography_features(image_path: str):
    """-> list[float] length 20 (fused_vector)"""
    regions = ocr.extract_regions(str(image_path))
    with Image.open(image_path) as image:
        result = feature_fusion.fuse_features(image, regions)
    return result["fused_vector"]
