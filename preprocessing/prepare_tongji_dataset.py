"""
prepare_tongji_dataset.py

Xử lý TONGJI Dataset ROI:
1. Nhóm 10 ảnh liên tiếp thành 1 user (00001-00010 = user 1, ...)
2. Apply enhancement (CLAHE pipeline) từ palm_vein_enhancement.py
3. Output: TONGJI_enhanced/<user_id>/<img>.bmp

TONGJI Dataset Structure:
- ROI/session1/00001.bmp ... 06000.bmp (600 users × 10 images)
- ROI/session2/00001.bmp ... 06000.bmp (600 users × 10 images)
- Total: 1200 identities, 12000 images

Usage:
    python prepare_tongji_dataset.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import enhancement functions
from palm_vein_enhancement import enhance_palm_veins


def prepare_tongji_dataset(
    input_dir: str = "TONGJI_Dataset/ROI",
    output_dir: str = "TONGJI_enhanced",
    images_per_user: int = 10,
    target_size: int = 128
):
    """
    Process TONGJI ROI images: group by user, enhance, and save.
    
    Args:
        input_dir: Path to TONGJI ROI directory (contains session1, session2)
        output_dir: Output directory for enhanced images
        images_per_user: Number of images per user (default: 10)
        target_size: Target image size (default: 128)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    sessions = ["session1", "session2"]
    total_users = 0
    total_images = 0
    failed_images = 0
    
    for session in sessions:
        session_dir = input_path / session
        if not session_dir.exists():
            print(f"⚠️  Session directory not found: {session_dir}")
            continue
        
        # Get all BMP files sorted numerically
        bmp_files = sorted(session_dir.glob("*.bmp"), key=lambda p: int(p.stem))
        print(f"\n📁 {session}: Found {len(bmp_files)} images")
        
        if len(bmp_files) == 0:
            continue
        
        # Calculate number of users
        num_users = len(bmp_files) // images_per_user
        remainder = len(bmp_files) % images_per_user
        if remainder > 0:
            print(f"⚠️  {remainder} remaining images (not a full user group) - skipping")
        
        print(f"   → {num_users} users × {images_per_user} images")
        
        # Session prefix: s1 or s2
        session_prefix = session.replace("session", "s")
        
        for user_idx in tqdm(range(num_users), desc=f"Processing {session}"):
            user_id = f"tongji_{session_prefix}_{user_idx + 1:03d}"
            user_dir = output_path / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            start_idx = user_idx * images_per_user
            end_idx = start_idx + images_per_user
            user_files = bmp_files[start_idx:end_idx]
            
            for img_idx, img_path in enumerate(user_files):
                try:
                    # Read grayscale image
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"\n⚠️  Failed to read: {img_path}")
                        failed_images += 1
                        continue
                    
                    # Resize to target size if needed
                    if img.shape[0] != target_size or img.shape[1] != target_size:
                        img = cv2.resize(img, (target_size, target_size), 
                                        interpolation=cv2.INTER_AREA)
                    
                    # Apply enhancement
                    enhanced = enhance_palm_veins(img)
                    
                    # Save enhanced image
                    out_filename = f"img_{img_idx + 1:03d}.bmp"
                    out_path = user_dir / out_filename
                    cv2.imwrite(str(out_path), enhanced)
                    total_images += 1
                    
                except Exception as e:
                    print(f"\n⚠️  Error processing {img_path}: {e}")
                    failed_images += 1
            
            total_users += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TONGJI Dataset Enhancement Summary")
    print("=" * 60)
    print(f"   Total users:  {total_users}")
    print(f"   Total images: {total_images}")
    print(f"   Failed:       {failed_images}")
    print(f"   Output:       {output_path.resolve()}")
    print("=" * 60)
    
    return total_users, total_images


if __name__ == "__main__":
    prepare_tongji_dataset()
