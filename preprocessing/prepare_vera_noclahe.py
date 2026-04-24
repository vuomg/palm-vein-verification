"""
prepare_vera_noclahe.py

Flatten VERA roi structure WITHOUT CLAHE enhancement.
Chỉ resize 128x128, không apply bất kỳ enhancement nào.
Dùng để ablation: so sánh với/không có CLAHE.

Output: VERA_noclahe/
"""
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def prepare_vera_noclahe(
    input_dir: str = "VERA-Dataset/VERA/VERA-Palmvein/roi",
    output_dir: str = "VERA_noclahe",
    target_size: int = 128,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: {input_path.resolve()} not found")
        return

    subject_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} subjects in {input_path.resolve()}")

    total_classes, total_images, failed = 0, 0, 0

    for subject_dir in tqdm(subject_dirs, desc="Subjects"):
        subject_id = subject_dir.name.split("-")[0]
        hand_images = {"L": [], "R": []}

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for img_path in sorted(session_dir.glob("*.png")):
                parts = img_path.stem.split("_")
                if len(parts) < 2:
                    continue
                hand = parts[1].upper()
                if hand in hand_images:
                    hand_images[hand].append(img_path)

        for hand, img_paths in hand_images.items():
            if not img_paths:
                continue
            class_dir = output_path / f"{subject_id}_{hand}"
            class_dir.mkdir(parents=True, exist_ok=True)

            for idx, img_path in enumerate(img_paths):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        failed += 1
                        continue
                    # Only resize — NO CLAHE, NO enhancement
                    if img.shape[0] != target_size or img.shape[1] != target_size:
                        img = cv2.resize(img, (target_size, target_size),
                                         interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(class_dir / f"img_{idx+1:03d}.png"), img)
                    total_images += 1
                except Exception as e:
                    print(f"\nERROR {img_path}: {e}")
                    failed += 1
            total_classes += 1

    print(f"\nDone: {total_classes} classes, {total_images} images, {failed} failed")
    print(f"Output: {output_path.resolve()}")
    print(f"\nNext: python split_vera_openset.py --input {output_dir} --output VERA_noclahe_openset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="VERA-Dataset/VERA/VERA-Palmvein/roi")
    parser.add_argument("--output", default="VERA_noclahe")
    parser.add_argument("--target-size", type=int, default=128)
    args = parser.parse_args()
    prepare_vera_noclahe(args.input, args.output, args.target_size)
