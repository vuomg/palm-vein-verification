#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Splitter for Verification Task
Chia dataset theo PER-USER: mỗi user có cả train và test images
Phù hợp cho VERIFICATION (1:1 matching) - CLOSED-SET

Paper: RSNet (verification task)
Split: 70% train, 30% test PER USER (7 train + 3 test images per user)

Usage:
    python dataset_splitter.py --input roi_dataset --output final_dataset
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime
import random
import json


def split_dataset_openset(
    input_dir, 
    output_dir, 
    train_ratio=0.7,
    max_users=None, 
    skip_confirmation=False, 
    random_seed=42
):
    """
    Chia dataset cho OPEN-SET PROTOCOL (như RSNet paper).
    
    Train và test có USERS HOÀN TOÀN KHÁC NHAU (no overlap).
    Chia USERS: 70% train users, 30% test users
    Mỗi user: TẤT CẢ images đi vào set tương ứng
    
    Args:
        input_dir: Thư mục chứa ROI images (all users)
        output_dir: Thư mục output (train/ and test/)
        train_ratio: Tỷ lệ train USERS (default: 0.7 = 70% users for train)
        max_users: Giới hạn số user (None = tất cả)
        skip_confirmation: Bỏ qua xác nhận
        random_seed: Random seed cho reproducibility
    
    Example:
        Input:  roi_dataset/
                ├── user_0001/ (10 images)
                ├── user_0002/ (10 images)
                ├── ...
                └── user_1549/ (10 images)
        
        Output: final_dataset/
                ├── train/
                │   ├── user_0001/ (10 images)  ← 70% of USERS (1084 users)
                │   ├── user_0002/ (10 images)     ALL their images
                │   └── ...
                └── test/
                    ├── user_1085/ (10 images)  ← 30% of USERS (465 users)
                    ├── user_1086/ (10 images)     DIFFERENT users!
                    └── ...                          ALL their images
    
    For OPEN-SET VERIFICATION (Paper Protocol):
        - Training: Learn features from train users
        - Testing:  Evaluate on UNSEEN users (generalization)
                   Genuine pairs (same test user, diff images)
                   Impostor pairs (different test users)
    """
    
    print("="*70)
    print("DATASET SPLITTER - OPEN-SET PROTOCOL (RSNet Paper)")
    print("="*70)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Train ratio: {train_ratio*100:.0f}% of USERS")
    print(f"Test ratio:  {(1-train_ratio)*100:.0f}% of USERS")
    print(f"Random seed: {random_seed}")
    print(f"Strategy:    User-level split (DIFFERENT users in train & test)")
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    print("ℹ️  OPEN-SET PROTOCOL (RSNet Paper):")
    print("   → Train and test have DIFFERENT users (no overlap)")
    print("   → Train: 70% of users (all their images)")
    print("   → Test:  30% of users (all their images) - UNSEEN identities")
    print("   → Tests model's generalization to new users")
    print("="*70)
    
    if not skip_confirmation:
        confirm = input("\nContinue? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Seed for reproducibility
    random.seed(random_seed)
    
    # Get all user directories
    input_path = Path(input_dir)
    user_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    if not user_dirs:
        print("\n❌ ERROR: No user directories found!")
        return
    
    print(f"\n✓ Found {len(user_dirs)} users")
    
    # Limit users if specified
    if max_users:
        user_dirs = user_dirs[:max_users]
        print(f"⚠️  Limited to {max_users} users")
    
    total_users = len(user_dirs)
    
    # Shuffle users for random split
    random.shuffle(user_dirs)
    
    # Split USERS (not images!)
    n_train_users = max(1, int(total_users * train_ratio))
    train_users = user_dirs[:n_train_users]
    test_users = user_dirs[n_train_users:]
    
    print(f"\n📊 User Split:")
    print(f"  Train users: {len(train_users)} ({len(train_users)/total_users*100:.1f}%)")
    print(f"  Test users:  {len(test_users)} ({len(test_users)/total_users*100:.1f}%)")
    print(f"  Overlap:     0 users (open-set confirmed)")
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    total_train_images = 0
    total_test_images = 0
    users_with_warnings = []
    
    print(f"\nCopying train users (all their images)...")
    
    # Process train users - copy ALL their images
    for i, user_dir in enumerate(train_users, 1):
        user_name = user_dir.name
        
        # Get all images
        images = (
            list(user_dir.glob('*.png')) +
            list(user_dir.glob('*.jpg')) +
            list(user_dir.glob('*.bmp'))
        )
        
        if len(images) == 0:
            print(f"⚠️  Warning: {user_name} has no images, skipping")
            users_with_warnings.append(user_name)
            continue
        
        # Create user directory in train
        train_user_dir = train_dir / user_name
        train_user_dir.mkdir(exist_ok=True)
        
        # Copy ALL images
        for img in images:
            shutil.copy2(img, train_user_dir / img.name)
        
        total_train_images += len(images)
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(train_users)} train users...")
    
    print(f"\nCopying test users (all their images)...")
    
    # Process test users - copy ALL their images
    for i, user_dir in enumerate(test_users, 1):
        user_name = user_dir.name
        
        # Get all images
        images = (
            list(user_dir.glob('*.png')) +
            list(user_dir.glob('*.jpg')) +
            list(user_dir.glob('*.bmp'))
        )
        
        if len(images) == 0:
            print(f"⚠️  Warning: {user_name} has no images, skipping")
            users_with_warnings.append(user_name)
            continue
        
        # Create user directory in test
        test_user_dir = test_dir / user_name
        test_user_dir.mkdir(exist_ok=True)
        
        # Copy ALL images
        for img in images:
            shutil.copy2(img, test_user_dir / img.name)
        
        total_test_images += len(images)
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(test_users)} test users...")
    
    # Calculate statistics
    avg_train = total_train_images / len(train_users) if len(train_users) > 0 else 0
    avg_test = total_test_images / len(test_users) if len(test_users) > 0 else 0
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY - OPEN-SET SPLIT (RSNet Paper Protocol)")
    print("="*70)
    print(f"Total users:           {total_users}")
    print(f"Users with warnings:   {len(users_with_warnings)}")
    print()
    print(f"TRAIN SET:")
    print(f"  Users:               {len(train_users)} ({len(train_users)/total_users*100:.1f}%)")
    print(f"  Total images:        {total_train_images}")
    print(f"  Avg per user:        {avg_train:.1f} images")
    print()
    print(f"TEST SET:")
    print(f"  Users:               {len(test_users)} ({len(test_users)/total_users*100:.1f}%)")
    print(f"  Total images:        {total_test_images}")
    print(f"  Avg per user:        {avg_test:.1f} images")
    print()
    print(f"✓ User overlap:        0 (OPEN-SET confirmed)")
    print(f"Total images:          {total_train_images + total_test_images}")
    print()
    print(f"Output directory:      {output_dir}")
    print(f"Completed:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Save split info
    split_info = {
        'timestamp': datetime.now().isoformat(),
        'task_type': 'open-set verification (RSNet paper protocol)',
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'train_user_ratio': train_ratio,
        'test_user_ratio': 1 - train_ratio,
        'random_seed': random_seed,
        'total_users': total_users,
        'train': {
            'users': len(train_users),
            'total_images': total_train_images,
            'avg_per_user': avg_train,
            'user_percentage': len(train_users) / total_users * 100
        },
        'test': {
            'users': len(test_users),  # DIFFERENT users!
            'total_images': total_test_images,
            'avg_per_user': avg_test,
            'user_percentage': len(test_users) / total_users * 100
        },
        'user_overlap': 0,  # Open-set!
        'users_with_warnings': users_with_warnings,
        'train_user_names': [u.name for u in train_users],
        'test_user_names': [u.name for u in test_users]
    }
    
    info_path = output_path / 'split_info_openset.json'
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Split info saved: {info_path}")
    print()
    print("✅ Dataset ready for OPEN-SET verification training!")
    print()
    print("Next steps:")
    print(f'  1. python train.py --dataset "{output_dir}" --epochs 100')
    print(f'     → Training will detect open-set automatically')
    print(f'  2. Check rsnet_results/rejection_analysis/ for user-level metrics')


def main():
    parser = argparse.ArgumentParser(
        description='Dataset Splitter for OPEN-SET Verification Task (User-level Split)'
    )
    
    parser.add_argument('--input', type=str, default='roi_dataset',
                       help='Input directory with all users (default: roi_dataset)')
    parser.add_argument('--output', type=str, default='final_dataset',
                       help='Output directory (default: final_dataset)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Train ratio per user (default: 0.7 = 70%% train, 30%% test)')
    parser.add_argument('--max-users', type=int, default=None,
                       help='Maximum number of users (default: all)')
    parser.add_argument('--skip-confirmation', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ ERROR: Input directory not found: {args.input}")
        return
    
    split_dataset_openset(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        max_users=args.max_users,
        skip_confirmation=args.skip_confirmation,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
