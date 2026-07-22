"""
Typo attack (T)
----------------
OCR-locates words on a document image, picks a handful of them, whites out
their bounding boxes and redraws a character-perturbed ("typo") version of
the same word in roughly the same place/size. This mimics an attacker
tampering with document text (e.g. altering an amount or a name) while
keeping the page visually plausible.

Requires: pytesseract + the `tesseract` binary on PATH, Pillow.
"""
import random
import pytesseract
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _typo_word(word: str, rng: random.Random) -> str:
    """Apply one character-level perturbation: swap, delete, or duplicate."""
    if len(word) < 3:
        return word + word[-1]  # duplicate last char for very short words
    op = rng.choice(["swap", "delete", "duplicate", "substitute"])
    i = rng.randrange(1, len(word) - 1)
    if op == "swap" and i < len(word) - 1:
        chars = list(word)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "duplicate":
        return word[:i] + word[i] + word[i:]
    # substitute with a visually-close character
    lookalikes = {"o": "0", "l": "1", "s": "5", "b": "6", "e": "3", "a": "@"}
    c = word[i].lower()
    repl = lookalikes.get(c, rng.choice("abcdefghijklmnopqrstuvwxyz"))
    return word[:i] + repl + word[i + 1:]


def apply_typo_attack(img: Image.Image, seed: int, max_words: int = 4) -> Image.Image:
    """Return a copy of img with a few OCR-detected words replaced by typo'd text."""
    rng = random.Random(seed)
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    data = pytesseract.image_to_data(out, output_type=pytesseract.Output.DICT)
    candidates = [
        i for i, w in enumerate(data["text"])
        if w.strip().isalpha() and len(w.strip()) >= 3
    ]
    if not candidates:
        return out  # nothing OCR-able; return unmodified copy

    rng.shuffle(candidates)
    chosen = candidates[:max_words]

    for i in chosen:
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        word = data["text"][i]
        typo = _typo_word(word, rng)

        # white-out the original word region, then draw the perturbed word
        draw.rectangle([x - 1, y - 1, x + w + 1, y + h + 1], fill="white")
        font_size = max(8, int(h * 0.9))
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except OSError:
            font = ImageFont.load_default()
        draw.text((x, y), typo, fill="black", font=font)

    return out
