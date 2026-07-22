"""
Adversarial patch attack (P)
------------------------------
Stamps a structured-noise "patch" onto the document image at a
pseudo-random location. Simulates a physical/digital sticker-style
adversarial patch used to fool downstream vision models (e.g. layout or
classification models) or to occlude document content.
"""
import random
import numpy as np
from PIL import Image


def _make_patch(size: int, rng: random.Random) -> Image.Image:
    """Generate a structured-noise square patch (checkerboard + random RGB noise)."""
    noise = rng.getrandbits
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    block = max(2, size // 8)
    for by in range(0, size, block):
        for bx in range(0, size, block):
            color = (
                rng.randrange(0, 256),
                rng.randrange(0, 256),
                rng.randrange(0, 256),
            )
            arr[by:by + block, bx:bx + block] = color
    return Image.fromarray(arr, mode="RGB")


def apply_patch_attack(img: Image.Image, seed: int, patch_frac: float = 0.16) -> Image.Image:
    """Return a copy of img with a random-noise adversarial patch pasted on it."""
    rng = random.Random(seed + 1)  # offset seed so patch != stego randomness
    out = img.convert("RGB").copy()
    w, h = out.size

    size = max(24, int(min(w, h) * patch_frac))
    patch = _make_patch(size, rng)

    max_x = max(1, w - size)
    max_y = max(1, h - size)
    px = rng.randrange(0, max_x)
    py = rng.randrange(0, max_y)

    out.paste(patch, (px, py))
    return out
