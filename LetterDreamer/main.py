"""
Synthetic dataset generator that creates letter-like shapes using fonts and transformations.
Produces 32x32 grayscale BMP images that can be fed into LetterLearner.

Generates letters A-Z using:
- Various built-in and system fonts
- Random transformations (rotation, scale, position)
- Noise and blur augmentation
- Optional stroke variations
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

SIZE = 32  # Output image size.
LETTERS = [chr(ord('A') + i) for i in range(26)]

# Common fonts on Windows that work well for handwriting-like appearance
FONT_CANDIDATES = [
    # Sans-serif fonts (clean, readable)
    "arial.ttf",
    "verdana.ttf",
    "tahoma.ttf",
    "calibri.ttf",
    "segoeui.ttf",
    # Serif fonts (more traditional)
    "times.ttf",
    "georgia.ttf",
    "garamond.ttf",
    # Monospace fonts
    "consola.ttf",
    "cour.ttf",
    # Handwriting-style fonts (if available)
    "comic.ttf",
    "segoesc.ttf",
    # Bold variants for variety
    "arialbd.ttf",
    "verdanab.ttf",
    "timesbd.ttf",
    "calibrib.ttf",
    "consolab.ttf",
]


def find_available_fonts() -> List[Path]:
    """Find fonts that are actually available on the system."""
    fonts_dirs = [
        Path("C:/Windows/Fonts"),
        Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
    ]
    
    available = []
    for fonts_dir in fonts_dirs:
        if not fonts_dir.exists():
            continue
        for font_name in FONT_CANDIDATES:
            font_path = fonts_dir / font_name
            if font_path.exists():
                available.append(font_path)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for path in available:
        if path.name.lower() not in seen:
            seen.add(path.name.lower())
            unique.append(path)
    
    return unique


def load_fonts(font_paths: List[Path], base_size: int = 24) -> List[ImageFont.FreeTypeFont]:
    """Load fonts at specified size."""
    fonts = []
    for path in font_paths:
        try:
            font = ImageFont.truetype(str(path), base_size)
            fonts.append(font)
        except Exception:
            continue
    
    # Fallback to default font if no fonts found
    if not fonts:
        try:
            fonts.append(ImageFont.load_default())
        except:
            pass
    
    return fonts


def draw_letter_with_font(
    letter: str,
    font: ImageFont.FreeTypeFont,
    intensity: int,
    noise_sigma: float,
    rotation: float,
    scale: float,
    offset_x: int,
    offset_y: int,
    thickness_mode: str,
) -> Image.Image:
    """Draw a single letter with specified parameters."""
    # Create larger canvas for transformations
    canvas_size = int(SIZE * 2.5)
    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)
    
    # Get text bounding box
    try:
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width, text_height = SIZE // 2, SIZE // 2
    
    # Center the letter
    x = (canvas_size - text_width) // 2
    y = (canvas_size - text_height) // 2
    
    # Draw the letter
    if thickness_mode == "normal":
        draw.text((x, y), letter, fill=intensity, font=font)
    elif thickness_mode == "bold":
        # Simulate bold by drawing multiple times with small offsets
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text((x + dx, y + dy), letter, fill=intensity, font=font)
    elif thickness_mode == "thin":
        # Draw normally, will be handled by erosion later
        draw.text((x, y), letter, fill=intensity, font=font)
    
    # Apply rotation
    if abs(rotation) > 0.5:
        image = image.rotate(rotation, resample=Image.BILINEAR, fillcolor=0, center=(canvas_size//2, canvas_size//2))
    
    # Apply scale and crop to final size
    if abs(scale - 1.0) > 0.01:
        new_size = int(canvas_size * scale)
        if new_size > 10:
            image = image.resize((new_size, new_size), Image.LANCZOS)
            # Re-center after scaling
            if new_size > canvas_size:
                left = (new_size - canvas_size) // 2
                image = image.crop((left, left, left + canvas_size, left + canvas_size))
            else:
                new_image = Image.new("L", (canvas_size, canvas_size), 0)
                paste_pos = (canvas_size - new_size) // 2
                new_image.paste(image, (paste_pos, paste_pos))
                image = new_image
    
    # Crop to final size with offset
    left = (canvas_size - SIZE) // 2 + offset_x
    top = (canvas_size - SIZE) // 2 + offset_y
    left = max(0, min(left, canvas_size - SIZE))
    top = max(0, min(top, canvas_size - SIZE))
    image = image.crop((left, top, left + SIZE, top + SIZE))
    
    # Apply thin effect (erosion)
    if thickness_mode == "thin":
        image = image.filter(ImageFilter.MinFilter(3))
    
    return image


def add_noise(image: Image.Image, noise_sigma: float) -> Image.Image:
    """Add Gaussian noise to the image (keeps black background)."""
    array = np.asarray(image, dtype=np.float32)
    noise = np.random.normal(0.0, noise_sigma, size=array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    # No background lift - keep black background
    return Image.fromarray(noisy, mode="L")


def generate_letter_image(
    letter: str,
    fonts: List[ImageFont.FreeTypeFont],
    noise_sigma: float,
    blur_max: float,
    use_lowercase: bool,
) -> Image.Image:
    """Generate a single letter image with random augmentations."""
    # Random parameters
    font = random.choice(fonts) if fonts else None
    # Higher minimum intensity (200-255) to ensure letters are clearly visible
    intensity = random.randint(200, 255)
    rotation = random.uniform(-15, 15)
    scale = random.uniform(0.7, 1.3)
    offset_x = random.randint(-4, 4)
    offset_y = random.randint(-4, 4)
    thickness_mode = random.choices(
        ["normal", "bold", "thin"],
        weights=[0.6, 0.25, 0.15]
    )[0]
    
    # Randomly use lowercase
    display_letter = letter.lower() if use_lowercase and random.random() < 0.3 else letter
    
    # Draw the letter
    if font:
        image = draw_letter_with_font(
            display_letter, font, intensity, noise_sigma,
            rotation, scale, offset_x, offset_y, thickness_mode
        )
    else:
        # Fallback: simple text on canvas
        image = Image.new("L", (SIZE, SIZE), color=0)
        draw = ImageDraw.Draw(image)
        draw.text((SIZE//4, SIZE//4), display_letter, fill=intensity)
    
    # Add noise
    image = add_noise(image, noise_sigma)
    
    # Optional blur
    if blur_max > 0.01:
        radius = random.uniform(0.0, blur_max)
        if radius > 0.02:
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # No invert - always black background with white letters
    
    return image


def ensure_dirs(base: Path, labels: Iterable[str]) -> None:
    """Create directories for each label."""
    for label in labels:
        (base / label).mkdir(parents=True, exist_ok=True)


def next_index_for(target_dir: Path) -> int:
    """Find the next available index for BMP files in target_dir."""
    max_index = 0
    for path in target_dir.glob("*.bmp"):
        try:
            file_number = int(path.stem)
            max_index = max(max_index, file_number)
        except ValueError:
            continue
    return max_index + 1


def generate_dataset(
    base_dir: Path,
    samples_per_class: int,
    noise_sigma: float,
    split_mode: bool,
    val_split: float,
    test_split: float,
    blur_max: float,
    max_retry: int,
    use_lowercase: bool,
    fonts: List[ImageFont.FreeTypeFont],
) -> Tuple[int, Counter]:
    """Generate the complete dataset."""
    
    # Validate samples_per_class limit
    if samples_per_class > 9999:
        raise ValueError(
            f"samples_per_class ({samples_per_class}) exceeds maximum of 9999"
        )
    
    if split_mode:
        val_ratio = max(0.0, min(1.0, val_split))
        test_ratio = max(0.0, min(1.0, test_split))
        if val_ratio + test_ratio >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")
        train_ratio = 1.0 - val_ratio - test_ratio
        splits = [
            ("train", train_ratio),
            ("val", val_ratio),
            ("test", test_ratio),
        ]
        for split_name, _ in splits:
            ensure_dirs(base_dir / split_name, LETTERS)
    else:
        ensure_dirs(base_dir, LETTERS)
        splits = [(None, 1.0)]
    
    seen_hashes: set = set()
    created_total = 0
    created_by_tag: Counter = Counter()
    
    def generate_unique(letter: str) -> Image.Image:
        """Generate a unique image for the letter."""
        last_image = None
        for _ in range(max(1, max_retry)):
            img = generate_letter_image(
                letter, fonts, noise_sigma, blur_max, use_lowercase
            )
            digest = hashlib.md5(img.tobytes()).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                return img
            last_image = img
        # Accept duplicate as fallback
        if last_image:
            seen_hashes.add(hashlib.md5(last_image.tobytes()).hexdigest())
        return last_image
    
    for letter in LETTERS:
        if split_mode:
            allocations = {}
            remaining = samples_per_class
            for split_name, ratio in splits[:-1]:
                count = int(round(samples_per_class * ratio))
                allocations[split_name] = max(0, count)
                remaining -= count
            allocations[splits[-1][0]] = max(0, remaining)
        else:
            allocations = {None: samples_per_class}
        
        for split_name, _ in splits:
            count = allocations.get(split_name if split_mode else None, 0)
            target_root = base_dir / split_name if split_name else base_dir
            ensure_dirs(target_root, [letter])
            target_dir = target_root / letter
            tag = split_name or "all"
            start_index = next_index_for(target_dir)
            
            # Check limit
            final_index = start_index + count - 1
            if final_index > 9999:
                raise ValueError(
                    f"Cannot generate {count} samples for class {letter} in {target_dir}: "
                    f"would exceed 9999 limit"
                )
            
            for idx in range(count):
                image = generate_unique(letter)
                if image:
                    filename = f"{start_index + idx:04d}.bmp"
                    image.save(target_dir / filename)
                    created_total += 1
                    created_by_tag[tag] += 1
    
    return created_total, created_by_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic letter images (A-Z) for LetterLearner."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("..") / "shared" / "data" / "synthetic",
        help="Directory where class subfolders will be created.",
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of images per class (default: 500)."
    )
    parser.add_argument(
        "--noise", type=float, default=18.0,
        help="Gaussian noise sigma (default: 18.0)."
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing files before generation."
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Create train/val/test subdirectories."
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Validation proportion (default: 0.2)."
    )
    parser.add_argument(
        "--test_split", type=float, default=0.1,
        help="Test proportion (default: 0.1)."
    )
    parser.add_argument(
        "--blur", type=float, default=0.8,
        help="Maximum Gaussian blur radius (default: 0.8)."
    )
    parser.add_argument(
        "--max_retry", type=int, default=5,
        help="Attempts to generate unique images."
    )
    parser.add_argument(
        "--lowercase", action="store_true",
        help="Include lowercase variants (30% chance per image)."
    )
    return parser.parse_args()


def prepare_output_dir(base_dir: Path, clean: bool) -> None:
    """Prepare the output directory."""
    if base_dir.exists() and clean:
        for item in base_dir.rglob("*"):
            if item.is_file():
                item.unlink()
        for sub in sorted((p for p in base_dir.rglob("*") if p.is_dir()), reverse=True):
            if not any(sub.iterdir()):
                sub.rmdir()
    base_dir.mkdir(parents=True, exist_ok=True)


def has_existing_data(base_dir: Path) -> bool:
    """Check if directory contains BMP files."""
    if not base_dir.exists():
        return False
    return next(base_dir.rglob("*.bmp"), None) is not None


def resolve_existing_output(base_dir: Path, clean_requested: bool) -> str:
    """Handle existing data interactively."""
    if clean_requested:
        return "replace"
    if not has_existing_data(base_dir):
        return "append"
    
    prompt = (
        f"Ve složce '{base_dir}' již existují vygenerovaná data.\n"
        "Zvolte akci: [P]řepsat / [Z]achovat / [D]oplnit: "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice in ("p", "prepsat", "overwrite", "replace", "y"):
            return "replace"
        if choice in ("z", "zachovat", "skip", "n", "ne"):
            return "skip"
        if choice in ("d", "doplnit", "append", "a"):
            return "append"
        print("Neplatná volba. Zadejte P, Z nebo D.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    
    print("=" * 60)
    print("🔤 LetterDreamer - Synthetic Letter Generator")
    print("=" * 60)
    
    action = resolve_existing_output(output_dir, args.clean)
    if action == "skip":
        print(f"Data ve složce {output_dir} zůstala zachována. Generování neproběhlo.")
        return
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Find and load fonts
    print("\nHledám dostupné fonty...")
    font_paths = find_available_fonts()
    print(f"Nalezeno {len(font_paths)} fontů")
    
    # Load fonts at various sizes for variety
    fonts = []
    for size in [18, 20, 22, 24, 26, 28]:
        fonts.extend(load_fonts(font_paths, base_size=size))
    print(f"Načteno {len(fonts)} variant fontů")
    
    if not fonts:
        print("⚠️  Varování: Nebyly nalezeny žádné fonty. Použiji fallback.")
    
    clean_flag = action == "replace"
    prepare_output_dir(output_dir, clean=clean_flag)
    
    print(f"\nGeneruji {args.samples} vzorků pro každé z 26 písmen...")
    print(f"Výstupní složka: {output_dir}")
    if args.split:
        print(f"Split: train={1-args.val_split-args.test_split:.0%}, val={args.val_split:.0%}, test={args.test_split:.0%}")
    
    generated_total, generated_breakdown = generate_dataset(
        base_dir=output_dir,
        samples_per_class=args.samples,
        noise_sigma=args.noise,
        split_mode=args.split,
        val_split=args.val_split,
        test_split=args.test_split,
        blur_max=max(0.0, args.blur),
        max_retry=max(1, args.max_retry),
        use_lowercase=args.lowercase,
        fonts=fonts,
    )
    
    print("\n" + "=" * 60)
    if generated_total:
        print(f"✓ Hotovo: vygenerováno {generated_total} obrázků")
        print(f"  Složka: {output_dir}")
        if generated_breakdown:
            print("\nSouhrn podle složek:")
            for tag, count in sorted(generated_breakdown.items()):
                label = "celkem" if tag == "all" else tag
                print(f"  {label}: {count}")
    else:
        print(f"Nebyly vytvořeny žádné nové soubory v {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
