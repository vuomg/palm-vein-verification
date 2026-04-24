"""
split_tongji_openset.py

Split TONGJI enhanced dataset into train/test for open-set verification.
Follows the same protocol as final_dataset_openset (70/30 split at user level).

Usage:
    python split_tongji_openset.py
"""

import os
import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def split_tongji_openset(
    input_dir: str = "TONGJI_enhanced",
    output_dir: str = "TONGJI_dataset_openset",
    train_ratio: float = 0.7,
    seed: int = 42
):
    """
    Split TONGJI enhanced dataset into train/test (open-set protocol).
    
    Args:
        input_dir: Path to enhanced TONGJI dataset
        output_dir: Output directory for split dataset
        train_ratio: Ratio of users for training (default: 0.7)
        seed: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all user directories
    user_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    total_users = len(user_dirs)
    
    print(f"📊 Total users found: {total_users}")
    
    if total_users == 0:
        print("❌ No user directories found!")
        return
    
    # Random split at user level
    random.seed(seed)
    user_names = [d.name for d in user_dirs]
    random.shuffle(user_names)
    
    num_train = int(total_users * train_ratio)
    train_users = sorted(user_names[:num_train])
    test_users = sorted(user_names[num_train:])
    
    print(f"   Train: {len(train_users)} users ({len(train_users)/total_users*100:.1f}%)")
    print(f"   Test:  {len(test_users)} users ({len(test_users)/total_users*100:.1f}%)")
    
    # Verify no overlap
    overlap = set(train_users) & set(test_users)
    assert len(overlap) == 0, f"User overlap detected: {overlap}"
    
    # Create output directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    train_images = 0
    test_images = 0
    
    for user_name in tqdm(train_users, desc="Copying train"):
        src = input_path / user_name
        dst = train_dir / user_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        train_images += len(list(dst.glob("*")))
    
    for user_name in tqdm(test_users, desc="Copying test"):
        src = input_path / user_name
        dst = test_dir / user_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        test_images += len(list(dst.glob("*")))
    
    # Save split info (same format as final_dataset_openset)
    split_info = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "open-set verification (TONGJI dataset)",
        "input_dir": str(input_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "train_user_ratio": train_ratio,
        "test_user_ratio": round(1 - train_ratio, 4),
        "random_seed": seed,
        "total_users": total_users,
        "train": {
            "users": len(train_users),
            "total_images": train_images,
            "avg_per_user": round(train_images / len(train_users), 1) if train_users else 0,
            "user_percentage": len(train_users) / total_users * 100
        },
        "test": {
            "users": len(test_users),
            "total_images": test_images,
            "avg_per_user": round(test_images / len(test_users), 1) if test_users else 0,
            "user_percentage": len(test_users) / total_users * 100
        },
        "user_overlap": len(overlap),
        "train_user_names": train_users,
        "test_user_names": test_users
    }
    
    json_path = output_path / "split_info_openset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TONGJI Open-Set Split Summary")
    print("=" * 60)
    print(f"   Total users:    {total_users}")
    print(f"   Train users:    {len(train_users)} ({train_images} images)")
    print(f"   Test users:     {len(test_users)} ({test_images} images)")
    print(f"   User overlap:   {len(overlap)}")
    print(f"   Output:         {output_path.resolve()}")
    print(f"   Split info:     {json_path}")
    print("=" * 60)
    
    return split_info


if __name__ == "__main__":
    split_tongji_openset()
