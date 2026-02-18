"""
Sample a fraction of takes from an existing splits.json.

Reads splits.json from --root (train/val/test UIDs), randomly samples the given
percentages of each split, and saves the sampled UIDs to a JSON file in the same directory.

Usage:
    python sample_takes.py --root ../output_dir_cooking --train_pct 0.1 --val_pct 0.2 --test_pct 0.2 --seed 42

NOTE: This script assumes that splits.json exists in --root directory.
"""

import argparse
import json
from pathlib import Path
import random


def load_split(root: Path) -> dict:
    """Load split.json from root. Returns dict with 'train', 'val', 'test' keys."""
    path = root / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"splits.json not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def sample_uids(uids: list, ratio: float, seed: int) -> list:
    """Randomly sample a fraction of UIDs. Returns list."""
    if ratio <= 0:
        return []
    if ratio >= 1.0:
        return uids
    random.seed(seed)
    n = max(1, int(len(uids) * ratio))
    sampled = random.sample(uids, n)
    return sampled


def sample_splits(split_dict: dict, train_pct: float, val_pct: float, test_pct: float, seed: int) -> dict:
    """Sample from each split by the given ratios (0.0–1.0)."""
    return {
        "train": sample_uids(split_dict["train"], train_pct, seed),
        "val": sample_uids(split_dict["val"], val_pct, seed + 1),
        "test": sample_uids(split_dict["test"], test_pct, seed + 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sample takes from split.json and save sampled UIDs to JSON"
    )
    parser.add_argument("--root", type=str, required=True, help="Directory containing split.json")
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
        default="sampled_split.json",
        help="Output path (absolute or relative to --root) (default: sampled_split.json)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    split_dict = load_split(root)
    sampled = sample_splits(
        split_dict,
        args.train_pct,
        args.val_pct,
        args.test_pct,
        args.seed,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"Saved to {out_path}")
    print(f"  Train: {len(sampled['train'])} / {len(split_dict['train'])}")
    print(f"  Val:   {len(sampled['val'])} / {len(split_dict['val'])}")
    print(f"  Test:  {len(sampled['test'])} / {len(split_dict['test'])}")


if __name__ == "__main__":
    main()
