"""Phase 4: generate one malicious counterpart for each of the 150 frozen clean
images, by rendering a randomly chosen injection phrase (from attack_templates.json)
onto a copy of the image with randomized font, size, rotation, color, position,
and spacing.

Reads datasets/sampled_images.json (frozen, never regenerated here) and each
clean image's normalized annotation (word bboxes) to decide realistic placement.
Writes malicious images to datasets/attacks/ and records every randomized
parameter in datasets/attack_metadata.json for later analysis/reproducibility.
"""

import json
import random

from PIL import Image, ImageDraw, ImageFont

from typographic.config import (
    ATTACK_METADATA_PATH,
    ATTACK_TEMPLATES_PATH,
    DATASETS_DIR,
    RANDOM_SEED,
    SAMPLED_IMAGES_PATH,
)

ATTACKS_DIR = DATASETS_DIR / "attacks"
DOCLAYNET_ATTACKS_DIR = DATASETS_DIR / "doclaynet_attacks"

# A small, fixed set of macOS system fonts spanning serif/sans/monospace/display
# styles, so injected text doesn't all look the same. Falls back to Pillow's
# bundled default font if a path isn't available on the current machine.
_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
]

POSITION_MODES = ["top_margin", "bottom_margin", "random", "overlay_on_text"]


def _load_font(size: int):
    path = random.choice(_FONT_PATHS)
    try:
        return ImageFont.truetype(path, size), path
    except OSError:
        return ImageFont.load_default(size=size), "PIL_default"


def _spaced_text(text: str, extra_spaces: int) -> str:
    if extra_spaces <= 0:
        return text
    pad = " " * extra_spaces
    return pad.join(text.split(" "))


def _choose_position(mode, canvas_size, text_size, word_bboxes, rng):
    W, H = canvas_size
    tw, th = text_size

    if word_bboxes:
        top_of_content = min(b[1] for b in word_bboxes)
        bottom_of_content = max(b[3] for b in word_bboxes)
    else:
        top_of_content, bottom_of_content = H * 0.1, H * 0.9

    if mode == "top_margin":
        y_max = max(int(top_of_content - th), 0)
        x = rng.randint(0, max(W - tw, 0))
        y = rng.randint(0, y_max) if y_max > 0 else 0
    elif mode == "bottom_margin":
        y_min = min(int(bottom_of_content), max(H - th, 0))
        x = rng.randint(0, max(W - tw, 0))
        y = rng.randint(y_min, max(H - th, y_min))
    elif mode == "overlay_on_text" and word_bboxes:
        target = rng.choice(word_bboxes)
        x = int(max(min(target[0], W - tw), 0))
        y = int(max(min(target[1], H - th), 0))
    else:  # "random" fallback
        x = rng.randint(0, max(W - tw, 0))
        y = rng.randint(0, max(H - th, 0))

    return x, y


def generate_attack(clean_entry: dict, templates: dict, rng: random.Random, attacks_dir=ATTACKS_DIR) -> dict:
    image_path = DATASETS_DIR / clean_entry["image_file"]
    annotation_path = DATASETS_DIR / clean_entry["annotation_file"]

    image = Image.open(image_path).convert("RGB")
    word_records = json.loads(annotation_path.read_text())
    word_bboxes = [w["bbox"] for w in word_records]

    category = rng.choice(list(templates.keys()))
    phrase = rng.choice(templates[category])

    font_size = rng.randint(max(int(image.height * 0.02), 12), int(image.height * 0.06))
    font, font_path = _load_font(font_size)

    extra_spaces = rng.choice([0, 0, 0, 1, 2])
    rendered_text = _spaced_text(phrase, extra_spaces)

    rotation = rng.choice([0, 0, 0, rng.uniform(-25, 25), rng.uniform(-180, 180)])

    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))

    # Render text on a transparent layer first so it can be rotated cleanly,
    # then composited onto the image at the chosen position.
    scratch = Image.new("RGBA", (1, 1))
    draw_scratch = ImageDraw.Draw(scratch)
    bbox = draw_scratch.textbbox((0, 0), rendered_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    text_layer = Image.new("RGBA", (text_w + 4, text_h + 4), (0, 0, 0, 0))
    ImageDraw.Draw(text_layer).text((2 - bbox[0], 2 - bbox[1]), rendered_text, font=font, fill=color + (255,))
    if rotation:
        text_layer = text_layer.rotate(rotation, expand=True)

    position_mode = rng.choice(POSITION_MODES)
    x, y = _choose_position(position_mode, image.size, text_layer.size, word_bboxes, rng)

    composed = image.convert("RGBA")
    composed.alpha_composite(text_layer, dest=(x, y))
    composed = composed.convert("RGB")

    malicious_id = f"{clean_entry['image_id']}_malicious"
    output_filename = f"{malicious_id}.png"
    composed.save(attacks_dir / output_filename)

    return {
        "malicious_id": malicious_id,
        "source_image_id": clean_entry["image_id"],
        "source_dataset": clean_entry["dataset"],
        "image_file": f"{attacks_dir.relative_to(DATASETS_DIR)}/{output_filename}",
        "category": category,
        "phrase": phrase,
        "rendered_text": rendered_text,
        "font_path": font_path,
        "font_size": font_size,
        "rotation_degrees": rotation,
        "color_rgb": list(color),
        "position_mode": position_mode,
        "position_xy": [x, y],
        "extra_word_spacing": extra_spaces,
    }


def generate_all_attacks(
    sampled_path=SAMPLED_IMAGES_PATH,
    attacks_dir=ATTACKS_DIR,
    attack_metadata_path=ATTACK_METADATA_PATH,
    force: bool = False,
) -> dict:
    """Generate one malicious counterpart for every entry in sampled_path.
    Defaults reproduce the original 150-image FUNSD/CORD/SROIE run exactly;
    pass different paths (as done for DocLayNet) to reuse this on another
    sampled-image set without touching the existing frozen output."""
    if attack_metadata_path.exists() and not force:
        print(f"{attack_metadata_path} already exists - not regenerating "
              f"(pass force=True to override deliberately)")
        return json.loads(attack_metadata_path.read_text())

    if not sampled_path.exists():
        raise FileNotFoundError(f"{sampled_path} not found - sample the images first")

    attacks_dir.mkdir(parents=True, exist_ok=True)

    sampled = json.loads(sampled_path.read_text())
    templates = json.loads(ATTACK_TEMPLATES_PATH.read_text())
    rng = random.Random(RANDOM_SEED)

    attacks = []
    for clean_entry in sampled["images"]:
        attacks.append(generate_attack(clean_entry, templates, rng, attacks_dir=attacks_dir))

    metadata = {"seed": RANDOM_SEED, "total": len(attacks), "attacks": attacks}
    attack_metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"generated {len(attacks)} malicious images -> {attacks_dir}")
    return metadata


if __name__ == "__main__":
    generate_all_attacks()
