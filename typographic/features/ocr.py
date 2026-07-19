"""Phase: OCR text extraction.

Wraps PaddleOCR (PP-OCRv6) and converts its output into the unified region
format consumed by every downstream feature module:

    {"text": str, "bbox": [xmin, ymin, xmax, ymax], "confidence": float,
     "rotation_degrees": float, "page_width": int, "page_height": int}

Region granularity matches whatever PaddleOCR natively detects (line/phrase
-level) - no added word/paragraph clustering. Rotation is derived from each
detected quad polygon's top edge rather than PaddleOCR's coarse upright/
upside-down textline-orientation classifier, since attacks use arbitrary
rotation angles, not just 0/180.
"""

import math

from PIL import Image
from paddleocr import PaddleOCR

_ocr_engine = None


def _get_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_engine


def _quad_rotation_degrees(poly) -> float:
    # poly is 4 corners ordered [top-left, top-right, bottom-right, bottom-left];
    # angle of the top edge relative to horizontal gives the text's rotation.
    (x0, y0), (x1, y1) = poly[0], poly[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def extract_regions(image_path: str) -> list[dict]:
    """Run OCR on an image file, returning one unified record per detected region."""
    engine = _get_engine()
    result = engine.predict(image_path)[0]

    with Image.open(image_path) as img:
        page_width, page_height = img.size

    regions = []
    for text, score, box, poly in zip(
        result["rec_texts"], result["rec_scores"], result["rec_boxes"], result["rec_polys"]
    ):
        text = text.strip()
        if not text:
            continue
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        regions.append({
            "text": text,
            "bbox": [x0, y0, x1, y1],
            "confidence": float(score),
            "rotation_degrees": _quad_rotation_degrees(poly.tolist()),
            "page_width": page_width,
            "page_height": page_height,
        })
    return regions
