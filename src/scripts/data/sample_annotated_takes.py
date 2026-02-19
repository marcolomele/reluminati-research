"""
Sample takes that have relation annotations from relations_*.json files.

Extracts UIDs (ann_id keys) from each annotation, groups by split, samples the given
percentages, and saves sampled UIDs to JSON.

Usage:
    python sample_annotated_takes.py --root /path/to/annotations --train_pct 0.1 --val_pct 0.2 --test_pct 0.2 --output sampled.json
"""

import argparse
import json
from pathlib import Path
import random


def load_annotated_takes(root: Path) -> dict:
    """Load relations files and extract UIDs (ann_id keys) from annotations."""
    splits = {"train": [], "val": [], "test": []}
    
    for split_name in ["train", "val", "test"]:
        file_path = root / f"relations_{split_name}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract UIDs (ann_id keys) from annotations
        uids = []
        for ann_id, annotation in data.get('annotations', {}).items():
            uids.append(ann_id)
        
        splits[split_name] = sorted(uids)
        print(f"Loaded {len(splits[split_name])} annotated takes from {split_name}")
    
    return splits


def sample_takes(takes: list, ratio: float, seed: int) -> list:
    """Randomly sample a fraction of UIDs."""
    if ratio <= 0:
        return []
    if ratio >= 1.0:
        return takes
    random.seed(seed)
    n = max(1, int(len(takes) * ratio))
    sampled = random.sample(takes, n)
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Sample annotated takes from relations_*.json files"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Directory containing relations_*.json files"
    )
    parser.add_argument(
        "--train_pct",
        type=float,
        default=0.1,
        help="Fraction of train UIDs to sample (default: 0.1)",
    )
    parser.add_argument(
        "--val_pct",
        type=float,
        default=0.2,
        help="Fraction of val UIDs to sample (default: 0.2)",
    )
    parser.add_argument(
        "--test_pct",
        type=float,
        default=0.2,
        help="Fraction of test UIDs to sample (default: 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for sampled UIDs JSON",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    # Load all annotated UIDs
    all_splits = load_annotated_takes(root)
    
    # Sample from each split
    sampled = {
        "train": sample_takes(all_splits["train"], args.train_pct, args.seed),
        "val": sample_takes(all_splits["val"], args.val_pct, args.seed + 1),
        "test": sample_takes(all_splits["test"], args.test_pct, args.seed + 2),
    }

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"\nSaved to {out_path}")
    print(f"  Train: {len(sampled['train'])} / {len(all_splits['train'])}")
    print(f"  Val:   {len(sampled['val'])} / {len(all_splits['val'])}")
    print(f"  Test:  {len(sampled['test'])} / {len(all_splits['test'])}")


if __name__ == "__main__":
    main()
