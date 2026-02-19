"""
Convert sampled take_names to UIDs using relations files.

Given the output of sample_annotated_takes.py (with take_names),
finds the corresponding UIDs from relations_*.json and outputs a new JSON with UIDs.

Usage:
    python convert_names_to_uids.py --sampled sampled_annotated_takes.json --relations-root /path/to/annotations --output sampled_uids.json
"""

import argparse
import json
from pathlib import Path


def convert_to_uids(sampled_path: Path, relations_root: Path) -> dict:
    """Convert sampled take_names to UIDs."""
    
    # Load sampled take_names
    with open(sampled_path, 'r') as f:
        sampled = json.load(f)
    
    # Build take_name to UID mapping from relations files
    take_name_to_uid = {}
    
    for split_name in ["train", "val", "test"]:
        file_path = relations_root / f"relations_{split_name}.json"
        if not file_path.exists():
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for ann_id, annotation in data.get('annotations', {}).items():
            take_name = annotation.get('take_name')
            
            if take_name:
                take_name_to_uid[take_name] = ann_id
    
    # Convert sampled take_names to UIDs
    result = {}
    missing = []
    
    for split_name, take_names in sampled.items():
        result[split_name] = []
        for take_name in take_names:
            if take_name in take_name_to_uid:
                result[split_name].append(take_name_to_uid[take_name])
            else:
                missing.append(take_name)
                print(f"Warning: No UID found for take_name: {take_name}")
    
    if missing:
        print(f"\nTotal missing: {len(missing)}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert sampled take_names to UIDs"
    )
    parser.add_argument(
        "--sampled",
        type=str,
        required=True,
        help="Path to sampled_annotated_takes.json (with take_names)"
    )
    parser.add_argument(
        "--relations-root",
        type=str,
        required=True,
        help="Directory containing relations_*.json files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for UIDs JSON"
    )
    args = parser.parse_args()
    
    result = convert_to_uids(
        Path(args.sampled),
        Path(args.relations_root)
    )
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved to {out_path}")
    for split_name, uids in result.items():
        print(f"  {split_name}: {len(uids)} UIDs")


if __name__ == "__main__":
    main()
