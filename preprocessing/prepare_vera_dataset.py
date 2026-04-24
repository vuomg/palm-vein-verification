"""
prepare_vera_dataset.py

Xử lý VERA Palm Vein Dataset ROI:
1. Flatten cấu trúc {ID}-{Gender}/{Session}/ → class {ID}_{Hand}/
2. Treat L và R tay của cùng 1 người là 2 identity riêng biệt
3. Apply enhancement (CLAHE pipeline) từ palm_vein_enhancement.py
4. Output: VERA_enhanced/<class_id>/<img>.png

VERA Dataset Structure (input roi/):
    {ID}-{Gender}/
        01/   → {ID}_L_1_1.png ... {ID}_L_1_5.png, {ID}_R_1_1.png ... {ID}_R_1_5.png
        02/   → {ID}_L_2_1.png ... {ID}_L_2_5.png, {ID}_R_2_1.png ... {ID}_R_2_5.png

Output Structure:
    VERA_enhanced/
        001_L/   → 10 images (session1 × 5 + session2 × 5, tay trái)
        001_R/   → 10 images (tay phải)
        ...
        110_L/
        110_R/
    Total: 220 classes × 10 images = 2200 images

Usage:
    python prepare_vera_dataset.py
    python prepare_vera_dataset.py --input VERA-Dataset/VERA/VERA-Palmvein/roi --output VERA_enhanced
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from palm_vein_enhancement import enhance_palm_veins


def prepare_vera_dataset(
    input_dir: str = "VERA-Dataset/VERA/VERA-Palmvein/roi",
    output_dir: str = "VERA_enhanced",
    target_size: int = 128,
):
    """
    Flatten + enhance VERA roi images.

    Class naming: {subject_id}_{hand}  (e.g., "001_L", "001_R")
    Each class gets 10 images: 2 sessions × 5 captures per hand.

    Args:
        input_dir: Path to VERA roi/ directory
        output_dir: Output directory for enhanced images
        target_size: Resize target before CLAHE (default: 128)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_path.resolve()}")
        return

    # Subject folders: e.g. 001-M, 025-F
    subject_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} subject folders in {input_path.resolve()}")

    total_classes = 0
    total_images = 0
    failed_images = 0

    for subject_dir in tqdm(subject_dirs, desc="Subjects"):
        # Parse subject ID from folder name (e.g. "001-M" → "001")
        subject_id = subject_dir.name.split("-")[0]  # "001"

        # Collect all images from both sessions, grouped by hand
        hand_images: dict[str, list[Path]] = {"L": [], "R": []}

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for img_path in sorted(session_dir.glob("*.png")):
                # Filename pattern: {ID}_{Hand}_{Session}_{Capture}.png
                parts = img_path.stem.split("_")
                if len(parts) < 2:
                    continue
                hand = parts[1].upper()  # "L" or "R"
                if hand in hand_images:
                    hand_images[hand].append(img_path)

        # Write one output folder per hand
        for hand, img_paths in hand_images.items():
            if not img_paths:
                continue

            class_name = f"{subject_id}_{hand}"  # e.g. "001_L"
            class_dir = output_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for idx, img_path in enumerate(img_paths):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"\nWARN: Cannot read {img_path}")
                        failed_images += 1
                        continue

                    # Resize to target_size if needed
                    if img.shape[0] != target_size or img.shape[1] != target_size:
                        img = cv2.resize(img, (target_size, target_size),
                                         interpolation=cv2.INTER_AREA)

                    enhanced = enhance_palm_veins(img)

                    out_name = f"img_{idx + 1:03d}.png"
                    cv2.imwrite(str(class_dir / out_name), enhanced)
                    total_images += 1

                except Exception as e:
                    print(f"\nERROR processing {img_path}: {e}")
                    failed_images += 1

            total_classes += 1

    print("\n" + "=" * 60)
    print("VERA Dataset Enhancement Summary")
    print("=" * 60)
    print(f"   Classes (subject × hand): {total_classes}")
    print(f"   Total images:             {total_images}")
    print(f"   Failed:                   {failed_images}")
    print(f"   Output:                   {output_path.resolve()}")
    print("=" * 60)
    print("\nNext step:")
    print(f'   python split_vera_openset.py --input {output_dir} --output VERA_dataset_openset')

    return total_classes, total_images


def main():
    parser = argparse.ArgumentParser(description="Prepare VERA Palm Vein Dataset")
    parser.add_argument("--input", type=str,
                        default="VERA-Dataset/VERA/VERA-Palmvein/roi",
                        help="Path to VERA roi/ directory")
    parser.add_argument("--output", type=str, default="VERA_enhanced",
                        help="Output directory (default: VERA_enhanced)")
    parser.add_argument("--target-size", type=int, default=128,
                        help="Resize target before CLAHE (default: 128)")
    args = parser.parse_args()

    prepare_vera_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
