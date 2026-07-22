"""
Steganography attack (S) -- "strg"
-------------------------------------
Scattered 15% LSB embedding -- reuses the exact algorithm from
saa/src/embed_lsb_scattered.py (the validated SAA pipeline's own stego
generator) rather than a bespoke implementation, so the fusion dataset's
stego attack is statistically consistent with what the SAA feature
extractor (entropy, LSB ratio/entropy, chi-square, SRM) was built and
tuned against.

FIXED (was): the original version here embedded a tiny fixed 136-bit
payload into only the first 136 pixels of the blue channel -- confirmed
via features.csv inspection to leave literally zero signal in SAA's
whole-image statistical features (lsb_ratio/lsb_entropy/srm_entropy
identical to 6 decimal places between clean and "stego" images). A 15%
scattered payload across all channels is large enough for those global
statistics to actually shift.
"""
import sys
from pathlib import Path

from PIL import Image

_SAA_SRC = Path(__file__).resolve().parents[3] / "saa" / "src"
if str(_SAA_SRC) not in sys.path:
    sys.path.insert(0, str(_SAA_SRC))

from embed_lsb_scattered import embed_scattered_lsb  # noqa: E402

STEGO_RATIO = 0.15


def apply_stego_attack(img: Image.Image, seed: int) -> Image.Image:
    """Return a copy of img with a scattered 15% LSB payload embedded
    (same algorithm as saa/src/embed_lsb_scattered.py)."""
    return embed_scattered_lsb(img, ratio=STEGO_RATIO, seed=seed)
