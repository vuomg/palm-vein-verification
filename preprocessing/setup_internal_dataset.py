"""
setup_internal_dataset.py

Setup internal palm vein dataset from raw .raw files:
  Phase 1: ROI extraction with multiprocessing (raw → 128×128 PNG with CLAHE)
  Phase 2: Open-set split (70/30 identity-level disjoint)

Source: C:\project\auto (1549 identities × 10 images)
Pipeline: palm_vein_processing.py (GrabCut → Distance Transform → ROI k=2.6 → CLAHE → 128×128)

Usage:
    python -u setup_internal_dataset.py
    python -u setup_internal_dataset.py --input "C:/project/auto" --output dataset_internal_openset --seed 42 --workers 4
    python -u setup_internal_dataset.py --skip-extraction --roi-dir dataset_internal_roi
"""

import os
import sys
import json
import shutil
import random
import argparse
import time
import glob as glob_module
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial


def _read_raw_image(img_path):
    """Read a .raw NIR image → grayscale numpy array (640×480)."""
    with open(img_path, 'rb') as f:
        raw_data = f.read()
    pixels = np.frombuffer(raw_data, dtype=np.uint8)
    image = pixels[:320 * 960].reshape(320, 960)
    left = image[:, 0:480]
    return cv2.resize(left, (480, 640), interpolation=cv2.INTER_CUBIC)


def _crop_roi_from_coords(img, roi_txt_path, target_size=(128, 128)):
    """Crop ROI using 4-point coordinates from roi_*.txt via perspective transform."""
    roi_text = open(roi_txt_path, 'r').read().strip()
    coords = list(map(int, roi_text.split(',')))
    src_pts = np.array(coords, dtype=np.float32).reshape(4, 2)
    dst_pts = np.array([
        [target_size[1] - 1, 0],
        [0, 0],
        [0, target_size[0] - 1],
        [target_size[1] - 1, target_size[0] - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (target_size[1], target_size[0]))


def process_single_image(args_tuple):
    """Process a single .raw image → ROI PNG. Designed for multiprocessing."""
    img_path, input_dir, output_dir, no_clahe, use_roi_coords = args_tuple

    rel_path = os.path.relpath(img_path, input_dir)
    rel_dir = os.path.dirname(rel_path)

    if rel_dir:
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
    else:
        output_subdir = output_dir

    base_name = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    output_path = os.path.join(output_subdir, base_name)

    if os.path.exists(output_path):
        return (True, rel_path, "skipped (exists)")

    try:
        if img_path.lower().endswith('.raw'):
            img = _read_raw_image(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return (False, rel_path, "Could not read image")

        if use_roi_coords:
            # Use device-provided ROI coordinates from roi_*.txt
            fname = os.path.basename(img_path)
            num = fname.replace('img_', '').replace('.raw', '')
            roi_txt = os.path.join(os.path.dirname(img_path), f'roi_{num}.txt')
            if not os.path.exists(roi_txt):
                return (False, rel_path, f"ROI file not found: roi_{num}.txt")
            roi = _crop_roi_from_coords(img, roi_txt, target_size=(128, 128))
        else:
            # GrabCut + Distance Transform pipeline
            from palm_vein_preprocessing import (
                preprocess_image, segment_hand,
                compute_palm_center_and_reference_length, extract_roi_scale_normalized,
            )
            preprocessed = preprocess_image(img)
            mask, contour = segment_hand(preprocessed)
            palm_center, L = compute_palm_center_and_reference_length(mask, method='percentile')
            roi, _ = extract_roi_scale_normalized(
                preprocessed, mask, palm_center, L,
                k=2.6, target_size=(128, 128), debug=False
            )

        if not no_clahe:
            from palm_vein_enhancement import enhance_palm_veins
            roi = enhance_palm_veins(roi)

        cv2.imwrite(output_path, roi)
        return (True, rel_path, "ok")

    except Exception as e:
        return (False, rel_path, str(e))


def extract_rois_parallel(input_dir: str, roi_dir: str, n_workers: int = 4, no_clahe: bool = False, use_roi_coords: bool = False):
    """Phase 1: Extract ROI from .raw files using multiprocessing."""
    os.makedirs(roi_dir, exist_ok=True)

    image_files = []
    for ext in ['.raw']:
        image_files.extend(glob_module.glob(os.path.join(input_dir, '**', f"*{ext}"), recursive=True))
    if not image_files:
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend(glob_module.glob(os.path.join(input_dir, '**', f"*{ext}"), recursive=True))

    if not image_files:
        raise ValueError(f"No images found in: {input_dir}")

    print("=" * 60)
    print("PHASE 1: ROI EXTRACTION (PARALLEL)")
    print("=" * 60)
    print(f"Input:   {input_dir}")
    print(f"Output:  {roi_dir}")
    print(f"Images:  {len(image_files)}")
    print(f"Workers: {n_workers}")
    if use_roi_coords:
        pipeline = f"ROI coords -> {'NO CLAHE' if no_clahe else 'CLAHE'} -> 128x128"
    else:
        pipeline = f"GrabCut -> DT -> ROI(k=2.6) -> {'NO CLAHE' if no_clahe else 'CLAHE'} -> 128x128"
    print(f"Pipeline: {pipeline}")
    print("=" * 60, flush=True)

    tasks = [(f, input_dir, roi_dir, no_clahe, use_roi_coords) for f in image_files]

    success_count = 0
    error_count = 0
    skip_count = 0
    errors_list = []
    start_time = time.time()

    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_image, tasks), 1):
            ok, rel_path, msg = result
            if msg == "skipped (exists)":
                skip_count += 1
            elif ok:
                success_count += 1
            else:
                error_count += 1
                errors_list.append((rel_path, msg))

            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i) / rate / 60 if rate > 0 else 0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {i}/{len(tasks)} ({i*100/len(tasks):.1f}%) | "
                      f"{rate:.1f} img/s | ETA: {eta:.0f}min | "
                      f"OK:{success_count} Skip:{skip_count} Err:{error_count}", flush=True)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Phase 1 complete: {success_count} success, {skip_count} skipped, {error_count} errors")
    print(f"Time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    if errors_list:
        print(f"\nFirst 20 errors:")
        for path, msg in errors_list[:20]:
            print(f"  {path}: {msg}")
    print("=" * 60, flush=True)

    return success_count, error_count, errors_list


def split_openset(roi_dir: str, output_dir: str, train_ratio: float = 0.7, seed: int = 42):
    """Phase 2: Split ROI dataset into train/test (open-set protocol)."""
    roi_path = Path(roi_dir)
    output_path = Path(output_dir)

    if not roi_path.exists():
        print(f"ERROR: ROI directory not found: {roi_path.resolve()}")
        return None

    identity_dirs = sorted(
        [d for d in roi_path.iterdir() if d.is_dir() and any(d.glob("*.png"))],
        key=lambda x: x.name
    )
    total_ids = len(identity_dirs)

    print("\n" + "=" * 60)
    print("PHASE 2: OPEN-SET SPLIT")
    print("=" * 60)
    print(f"Total identities with images: {total_ids}")

    if total_ids == 0:
        print("ERROR: No identity directories with .png files found!")
        return None

    random.seed(seed)
    names = [d.name for d in identity_dirs]
    random.shuffle(names)

    n_train = int(total_ids * train_ratio)
    train_ids = sorted(names[:n_train])
    test_ids = sorted(names[n_train:])

    print(f"Train: {len(train_ids)} identities ({len(train_ids)/total_ids*100:.1f}%)")
    print(f"Test:  {len(test_ids)} identities ({len(test_ids)/total_ids*100:.1f}%)")

    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, f"Identity overlap detected: {overlap}"

    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_images = 0
    test_images = 0
    errors = []

    for i, name in enumerate(train_ids, 1):
        src = roi_path / name
        dst = train_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        try:
            shutil.copytree(src, dst)
            train_images += len(list(dst.glob("*.png")))
        except Exception as e:
            errors.append(f"Train copy {name}: {e}")
        if i % 200 == 0:
            print(f"  Train: {i}/{len(train_ids)} copied...", flush=True)

    for i, name in enumerate(test_ids, 1):
        src = roi_path / name
        dst = test_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        try:
            shutil.copytree(src, dst)
            test_images += len(list(dst.glob("*.png")))
        except Exception as e:
            errors.append(f"Test copy {name}: {e}")
        if i % 200 == 0:
            print(f"  Test: {i}/{len(test_ids)} copied...", flush=True)

    split_info = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "open-set verification (internal dataset)",
        "source_raw_dir": str(Path(roi_dir).resolve()),
        "output_dir": str(output_path.resolve()),
        "train_user_ratio": train_ratio,
        "test_user_ratio": round(1 - train_ratio, 4),
        "random_seed": seed,
        "total_users": total_ids,
        "train": {
            "identities": len(train_ids),
            "total_images": train_images,
            "avg_per_identity": round(train_images / len(train_ids), 1) if train_ids else 0,
        },
        "test": {
            "identities": len(test_ids),
            "total_images": test_images,
            "avg_per_identity": round(test_images / len(test_ids), 1) if test_ids else 0,
        },
        "user_overlap": 0,
        "train_user_names": train_ids,
        "test_user_names": test_ids,
    }

    if errors:
        split_info["errors"] = errors

    json_path = output_path / "split_info_openset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("INTERNAL DATASET OPEN-SET SPLIT COMPLETE")
    print("=" * 60)
    print(f"Total identities: {total_ids}")
    print(f"Train: {len(train_ids)} identities / {train_images} images")
    print(f"Test:  {len(test_ids)} identities / {test_images} images")
    print(f"Overlap: 0 (open-set confirmed)")
    print(f"Seed: {seed}")
    print(f"Output: {output_path.resolve()}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  - {e}")
    print("=" * 60)
    print(f"\nNext step:")
    print(f"  python train.py --model sca_mobilenet --dataset {output_dir} --database default --epochs 100")

    return split_info


def main():
    parser = argparse.ArgumentParser(
        description="Setup internal palm vein dataset (ROI extraction + open-set split)"
    )
    parser.add_argument("--input", default="C:/project/auto",
                        help="Input directory with .raw files (default: C:/project/auto)")
    parser.add_argument("--roi-dir", default="dataset_internal_roi",
                        help="Intermediate ROI output directory (default: dataset_internal_roi)")
    parser.add_argument("--output", default="dataset_internal_openset",
                        help="Final output directory with train/test split")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Train ratio at identity level (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help=f"Number of worker processes (default: {max(1, cpu_count()-2)})")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip ROI extraction, only do split (roi-dir must exist)")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Skip CLAHE enhancement (save raw ROI without enhancement)")
    parser.add_argument("--use-roi-coords", action="store_true",
                        help="Use ROI coordinates from roi_*.txt instead of GrabCut segmentation")

    args = parser.parse_args()

    start = time.time()

    if not args.skip_extraction:
        extract_rois_parallel(args.input, args.roi_dir, args.workers, args.no_clahe, args.use_roi_coords)
    else:
        print(f"Skipping ROI extraction, using existing: {args.roi_dir}")

    split_openset(args.roi_dir, args.output, args.train_ratio, args.seed)

    total = time.time() - start
    print(f"\nTotal time: {total/60:.1f} minutes")


if __name__ == "__main__":
    main()
