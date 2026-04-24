"""
split_scut_openset.py

Split SCUT enhanced dataset into train/test for open-set verification.
Follows the same protocol as VERA split (70/30 split at identity level).

Identity = class id (e.g. "0001", "0002", ..., "1100").

Usage:
    python split_scut_openset.py
    python split_scut_openset.py --input SCUT_enhanced --output SCUT_dataset_openset --seed 42
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def split_scut_openset(
    input_dir: str = "SCUT_enhanced",
    output_dir: str = "SCUT_dataset_openset",
    train_ratio: float = 0.7,
    seed: int = 42,
):
    """
    Split SCUT enhanced dataset into train/test (open-set protocol).

    Identity-level split — train and test identities are completely disjoint.
    70% identities → train, 30% identities → test.

    Args:
        input_dir: Path to SCUT_enhanced directory
        output_dir: Output directory for split dataset
        train_ratio: Ratio of identities for training (default: 0.7)
        seed: Random seed for reproducibility (default: 42)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_path.resolve()}")
        print(f"Run prepare_scut_dataset.py first.")
        return

    # Each subdirectory is an identity class (e.g. "0001", "0002", ...)
    identity_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()],
                          key=lambda x: int(x.name))
    total_ids = len(identity_dirs)

    print(f"Total identities found: {total_ids}")

    if total_ids == 0:
        print("ERROR: No identity directories found!")
        return

    # Random identity-level split
    random.seed(seed)
    names = [d.name for d in identity_dirs]
    random.shuffle(names)

    n_train = int(total_ids * train_ratio)
    train_ids = sorted(names[:n_train])
    test_ids = sorted(names[n_train:])

    print(f"   Train: {len(train_ids)} identities ({len(train_ids)/total_ids*100:.1f}%)")
    print(f"   Test:  {len(test_ids)} identities ({len(test_ids)/total_ids*100:.1f}%)")

    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, f"Identity overlap detected: {overlap}"

    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_images = 0
    test_images = 0

    for name in tqdm(train_ids, desc="Copying train"):
        src = input_path / name
        dst = train_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        train_images += len(list(dst.glob("*")))

    for name in tqdm(test_ids, desc="Copying test"):
        src = input_path / name
        dst = test_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        test_images += len(list(dst.glob("*")))

    split_info = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "open-set verification (SCUT dataset)",
        "input_dir": str(input_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "train_identity_ratio": train_ratio,
        "test_identity_ratio": round(1 - train_ratio, 4),
        "random_seed": seed,
        "total_identities": total_ids,
        "train": {
            "identities": len(train_ids),
            "total_images": train_images,
            "avg_per_identity": round(train_images / len(train_ids), 1) if train_ids else 0,
            "identity_percentage": len(train_ids) / total_ids * 100,
        },
        "test": {
            "identities": len(test_ids),
            "total_images": test_images,
            "avg_per_identity": round(test_images / len(test_ids), 1) if test_ids else 0,
            "identity_percentage": len(test_ids) / total_ids * 100,
        },
        "identity_overlap": len(overlap),
        "train_identity_names": train_ids,
        "test_identity_names": test_ids,
    }

    json_path = output_path / "split_info_openset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SCUT Open-Set Split Summary")
    print("=" * 60)
    print(f"   Total identities: {total_ids}")
    print(f"   Train:            {len(train_ids)} identities / {train_images} images")
    print(f"   Test:             {len(test_ids)} identities / {test_images} images")
    print(f"   Overlap:          0 (open-set confirmed)")
    print(f"   Output:           {output_path.resolve()}")
    print(f"   Split info:       {json_path}")
    print("=" * 60)
    print("\nNext step:")
    print(f'   python train.py --model sca_mobilenet --dataset {output_dir} --database SCUT --epochs 100')

    return split_info


def main():
    parser = argparse.ArgumentParser(description="Split SCUT dataset for open-set verification")
    parser.add_argument("--input", type=str, default="SCUT_enhanced",
                        help="Input directory (output of prepare_scut_dataset.py)")
    parser.add_argument("--output", type=str, default="SCUT_dataset_openset",
                        help="Output directory (default: SCUT_dataset_openset)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Train/test ratio at identity level (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    split_scut_openset(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
