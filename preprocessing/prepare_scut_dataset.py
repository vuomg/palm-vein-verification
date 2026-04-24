"""
prepare_scut_dataset.py

Xử lý SCUT Palm Vein Dataset ROI:
1. Flatten + enhance SCUT roi images
2. Class naming: {class_id}  (e.g., "001", "002", ..., "1100")
3. Apply enhancement (CLAHE pipeline) từ palm_vein_enhancement.py
4. Output: SCUT_enhanced/<class_id>/<img>.png

SCUT Dataset Structure (input):
    scale1.0/
        1/     → 1_1.jpg, 1_2.jpg, ..., 1_10.jpg
        2/     → 2_1.jpg, 2_2.jpg, ..., 2_10.jpg
        ...
        1100/  → 1100_1.jpg, ..., 1100_10.jpg

Output Structure:
    SCUT_enhanced/
        001/   → 10 images (img_001.png, img_002.png, ..., img_010.png)
        002/   → 10 images
        ...
        1100/  → 10 images
    Total: 1100 classes × ~10 images = ~11k images

Usage:
    python prepare_scut_dataset.py
    python prepare_scut_dataset.py --input "C:\\Users\\DELL\\Downloads\\SCUT_PV_V1\\ROI\\ROIscale\\scale1.0" --output SCUT_enhanced
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from palm_vein_enhancement import enhance_palm_veins


def prepare_scut_dataset(
    input_dir: str = "C:\\Users\\DELL\\Downloads\\SCUT_PV_V1\\ROI\\ROIscale\\scale1.0",
    output_dir: str = "SCUT_enhanced",
    target_size: int = 128,
):
    """
    Flatten + enhance SCUT roi images.

    Class naming: {class_id}  (e.g., "001", "002", ..., "1100")
    Each class gets ~10 images from source folder.

    Args:
        input_dir: Path to SCUT scale1.0/ directory
        output_dir: Output directory for enhanced images
        target_size: Resize target before CLAHE (default: 128)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_path.resolve()}")
        return

    # Class folders: e.g. 1, 2, 3, ..., 1100
    class_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()],
                       key=lambda x: int(x.name))
    print(f"Found {len(class_dirs)} class folders in {input_path.resolve()}")

    total_classes = 0
    total_images = 0
    failed_images = 0

    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_id = class_dir.name  # "1", "2", ..., "1100"
        class_id_padded = str(class_id).zfill(4)  # "0001", "0002", ..., "1100"

        # Collect all images in this class folder
        img_paths = sorted(class_dir.glob("*.jpg"))
        if not img_paths:
            img_paths = sorted(class_dir.glob("*.png"))
        if not img_paths:
            img_paths = sorted(class_dir.glob("*.bmp"))

        if not img_paths:
            continue

        # Create output class directory
        out_class_dir = output_path / class_id_padded
        out_class_dir.mkdir(parents=True, exist_ok=True)

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
                cv2.imwrite(str(out_class_dir / out_name), enhanced)
                total_images += 1

            except Exception as e:
                print(f"\nERROR processing {img_path}: {e}")
                failed_images += 1

        total_classes += 1

    print("\n" + "=" * 60)
    print("SCUT Dataset Enhancement Summary")
    print("=" * 60)
    print(f"   Classes:             {total_classes}")
    print(f"   Total images:        {total_images}")
    print(f"   Failed:              {failed_images}")
    print(f"   Output:              {output_path.resolve()}")
    print("=" * 60)
    print("\nNext step:")
    print(f'   python split_scut_openset.py --input {output_dir} --output SCUT_dataset_openset')
    print("\nOr train directly:")
    print(f'   python train.py --model sca_mobilenet --dataset {output_dir} --database SCUT --epochs 100')

    return total_classes, total_images


def main():
    parser = argparse.ArgumentParser(description="Prepare SCUT Palm Vein Dataset")
    parser.add_argument("--input", type=str,
                        default="C:\\Users\\DELL\\Downloads\\SCUT_PV_V1\\ROI\\ROIscale\\scale1.0",
                        help="Path to SCUT scale1.0/ directory")
    parser.add_argument("--output", type=str, default="SCUT_enhanced",
                        help="Output directory (default: SCUT_enhanced)")
    parser.add_argument("--target-size", type=int, default=128,
                        help="Resize target before CLAHE (default: 128)")
    args = parser.parse_args()

    prepare_scut_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
