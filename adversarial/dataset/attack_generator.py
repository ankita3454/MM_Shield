"""Phase 2: adversarial patch attack generator.

Generates a small, fixed library of procedural patch images (checkerboard,
random noise, QR-like block pattern, warning-colored block, geometric logo)
- fully reproducible from code alone, no external asset/dataset dependency,
no licensing questions. These are explicitly synthetic patch stickers, not
real gradient-optimized adversarial perturbations (that would require
white-box access to a target model - out of scope here); framed honestly as
such, matching the old reference implementation's own honest disclosure.

For each clean image, composites a randomly chosen patch with randomized
scale/rotation/opacity/Gaussian-blur/position (Gaussian blur specifically to
soften edges the way a printed-and-rescanned or photographed patch would,
plus occasional brightness shift and JPEG-compression-artifact simulation
for added realism), and records the exact ground-truth patch bbox.
"""

import io
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from adversarial.config import ATTACK_METADATA_PATH, ATTACKS_DIR, PATCH_LIBRARY_DIR
from typographic.config import DATASETS_DIR, RANDOM_SEED, SAMPLED_IMAGES_PATH

PATCH_SIZE = 300  # base size (px) patches are generated at, before per-attack scaling
PATCH_TYPES = ["checkerboard", "noise", "qr_like", "warning_block", "geometric_logo"]


def _make_checkerboard(rng):
    img = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
    draw = ImageDraw.Draw(img)
    cell = PATCH_SIZE // 8
    color_a = tuple(rng.randint(0, 255) for _ in range(3))
    color_b = tuple(rng.randint(0, 255) for _ in range(3))
    for row in range(8):
        for col in range(8):
            color = color_a if (row + col) % 2 == 0 else color_b
            draw.rectangle([col * cell, row * cell, (col + 1) * cell, (row + 1) * cell], fill=color)
    return img.convert("RGBA")


def _make_noise(rng):
    arr = np.random.RandomState(rng.randint(0, 2**31 - 1)).randint(0, 256, (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB").convert("RGBA")


def _make_qr_like(rng):
    img = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), "white")
    draw = ImageDraw.Draw(img)
    cell = PATCH_SIZE // 12
    for row in range(12):
        for col in range(12):
            if rng.random() < 0.5:
                draw.rectangle([col * cell, row * cell, (col + 1) * cell, (row + 1) * cell], fill="black")
    for cx, cy in [(0, 0), (PATCH_SIZE - 3 * cell, 0), (0, PATCH_SIZE - 3 * cell)]:
        draw.rectangle([cx, cy, cx + 3 * cell, cy + 3 * cell], outline="black", width=max(cell // 2, 1))
    return img.convert("RGBA")


def _make_warning_block(rng):
    img = Image.new("RGBA", (PATCH_SIZE, PATCH_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base_color = rng.choice([(255, 200, 0), (255, 80, 0), (220, 0, 0)])
    draw.rectangle([0, 0, PATCH_SIZE, PATCH_SIZE], fill=base_color + (255,))
    stripe_w = PATCH_SIZE // 8
    for i in range(-PATCH_SIZE, PATCH_SIZE, stripe_w * 2):
        draw.polygon(
            [(i, 0), (i + stripe_w, 0), (i + stripe_w + PATCH_SIZE, PATCH_SIZE), (i + PATCH_SIZE, PATCH_SIZE)],
            fill=(0, 0, 0, 200),
        )
    return img


def _make_geometric_logo(rng):
    img = Image.new("RGBA", (PATCH_SIZE, PATCH_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, PATCH_SIZE, PATCH_SIZE], fill=tuple(rng.randint(0, 255) for _ in range(3)) + (255,))
    for _ in range(3):
        shape_color = tuple(rng.randint(0, 255) for _ in range(3)) + (255,)
        x0, y0 = rng.randint(0, PATCH_SIZE // 2), rng.randint(0, PATCH_SIZE // 2)
        x1, y1 = x0 + rng.randint(20, PATCH_SIZE // 2), y0 + rng.randint(20, PATCH_SIZE // 2)
        if rng.random() < 0.5:
            draw.ellipse([x0, y0, x1, y1], fill=shape_color)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=shape_color)
    return img


_PATCH_BUILDERS = {
    "checkerboard": _make_checkerboard,
    "noise": _make_noise,
    "qr_like": _make_qr_like,
    "warning_block": _make_warning_block,
    "geometric_logo": _make_geometric_logo,
}


def build_patch_library(force: bool = False) -> dict:
    """Generate (once) and cache the fixed 5-patch library under
    PATCH_LIBRARY_DIR. Uses its own fixed sub-seed, independent of the
    per-attack rng, so the library itself never changes between runs."""
    PATCH_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    library = {}
    for name in PATCH_TYPES:
        path = PATCH_LIBRARY_DIR / f"{name}.png"
        if path.exists() and not force:
            library[name] = Image.open(path).convert("RGBA")
            continue
        patch = _PATCH_BUILDERS[name](rng)
        patch.save(path)
        library[name] = patch
    return library


def generate_attack(clean_entry: dict, library: dict, rng: random.Random, attacks_dir=ATTACKS_DIR) -> dict:
    image_path = DATASETS_DIR / clean_entry["image_file"]
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    patch_type = rng.choice(PATCH_TYPES)
    patch = library[patch_type].copy()

    scale_fraction = rng.uniform(0.08, 0.35)
    target_size = max(int(scale_fraction * min(W, H)), 16)
    patch = patch.resize((target_size, target_size), Image.LANCZOS)

    # Mostly mild rotation (a roughly-upright sticker), occasionally more extreme.
    rotation = rng.choice([rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-180, 180)])
    patch = patch.rotate(rotation, expand=True)

    # Softens edges the way a printed-and-rescanned or photographed patch would.
    blur_radius = rng.uniform(0.5, 2.0)
    patch = patch.filter(ImageFilter.GaussianBlur(blur_radius))

    brightness_factor = None
    if rng.random() < 0.5:
        brightness_factor = rng.uniform(0.8, 1.2)
        patch = ImageEnhance.Brightness(patch).enhance(brightness_factor)

    opacity = rng.uniform(0.6, 1.0)
    alpha = patch.getchannel("A").point(lambda a: int(a * opacity))
    patch.putalpha(alpha)

    patch_w, patch_h = patch.size
    x = rng.randint(0, max(W - patch_w, 0))
    y = rng.randint(0, max(H - patch_h, 0))

    composed = image.convert("RGBA")
    composed.alpha_composite(patch, dest=(x, y))
    composed = composed.convert("RGB")

    jpeg_quality = None
    if rng.random() < 0.7:
        jpeg_quality = rng.randint(50, 90)
        buffer = io.BytesIO()
        composed.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        composed = Image.open(buffer).convert("RGB")

    attack_id = f"{clean_entry['image_id']}_patch"
    output_filename = f"{attack_id}.png"
    composed.save(attacks_dir / output_filename)

    return {
        "attack_id": attack_id,
        "source_image_id": clean_entry["image_id"],
        "source_dataset": clean_entry["dataset"],
        "image_file": f"{attacks_dir.relative_to(DATASETS_DIR)}/{output_filename}",
        "patch_type": patch_type,
        "patch_bbox": [x, y, x + patch_w, y + patch_h],
        "page_width": W,
        "page_height": H,
        "scale_fraction": scale_fraction,
        "rotation_degrees": rotation,
        "opacity": opacity,
        "blur_radius": blur_radius,
        "brightness_factor": brightness_factor,
        "jpeg_quality": jpeg_quality,
    }


def generate_all_attacks(
    sampled_path=SAMPLED_IMAGES_PATH,
    attacks_dir=ATTACKS_DIR,
    attack_metadata_path=ATTACK_METADATA_PATH,
    force: bool = False,
) -> dict:
    if attack_metadata_path.exists() and not force:
        print(f"{attack_metadata_path} already exists - not regenerating "
              f"(pass force=True to override deliberately)")
        return json.loads(attack_metadata_path.read_text())

    if not sampled_path.exists():
        raise FileNotFoundError(f"{sampled_path} not found - sample the images first")

    attacks_dir.mkdir(parents=True, exist_ok=True)

    library = build_patch_library()
    sampled = json.loads(sampled_path.read_text())
    rng = random.Random(RANDOM_SEED)

    attacks = []
    for clean_entry in sampled["images"]:
        attacks.append(generate_attack(clean_entry, library, rng, attacks_dir=attacks_dir))

    metadata = {"seed": RANDOM_SEED, "total": len(attacks), "attacks": attacks}
    attack_metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"generated {len(attacks)} patched images -> {attacks_dir}")
    return metadata


if __name__ == "__main__":
    generate_all_attacks()
