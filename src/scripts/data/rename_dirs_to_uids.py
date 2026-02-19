"""
Rename take_name directories to their corresponding UUIDs.

Reads relations_*.json to build take_name -> UUID mapping,
then renames directories in the output directory.

Usage:
    python rename_dirs_to_uids.py --data-dir /path/to/output_dir_all --relations-root /path/to/annotations [--dry-run]
"""

import argparse
import json
import shutil
from pathlib import Path


def build_take_name_to_uid_mapping(relations_root: Path) -> dict:
    """Build take_name to UID mapping from relations files."""
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
    
    return take_name_to_uid


def rename_directories(data_dir: Path, take_name_to_uid: dict, dry_run: bool = False):
    """Rename directories from take_name to UUID."""
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
    
    renamed = 0
    skipped = 0
    
    # Get all subdirectories
    for take_dir in data_dir.iterdir():
        if not take_dir.is_dir():
            continue
        
        take_name = take_dir.name
        
        # Skip if already looks like a UUID (contains dashes and hex chars)
        if '-' in take_name and len(take_name) > 30:
            print(f"Skipping (already UUID?): {take_name}")
            skipped += 1
            continue
        
        # Get corresponding UUID
        if take_name not in take_name_to_uid:
            print(f"Warning: No UUID found for: {take_name}")
            skipped += 1
            continue
        
        uid = take_name_to_uid[take_name]
        new_path = data_dir / uid
        
        if new_path.exists():
            print(f"Warning: Target already exists: {uid}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"[DRY RUN] Would rename: {take_name} -> {uid}")
        else:
            take_dir.rename(new_path)
            print(f"Renamed: {take_name} -> {uid}")
        
        renamed += 1
    
    print(f"\nSummary:")
    print(f"  Renamed: {renamed}")
    print(f"  Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename take_name directories to UUIDs"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing take directories (e.g., output_dir_all)"
    )
    parser.add_argument(
        "--relations-root",
        type=str,
        required=True,
        help="Directory containing relations_*.json files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually renaming"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    relations_root = Path(args.relations_root)
    
    print("Building take_name -> UUID mapping...")
    mapping = build_take_name_to_uid_mapping(relations_root)
    print(f"Found {len(mapping)} take_name -> UUID mappings\n")
    
    if args.dry_run:
        print("=== DRY RUN MODE ===\n")
    
    rename_directories(data_dir, mapping, args.dry_run)


if __name__ == "__main__":
    main()
