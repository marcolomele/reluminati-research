import glob, os

import numpy as np
import json

from collections import defaultdict
from pathlib import Path
import argparse

def make_pairs(data_dir, split='train', setting='egoexo'):
    
    with open(f'{data_dir}/split.json', 'r') as fp:
        splits = json.load(fp)

    split_takes = splits[split]

    pairs = []
    skipped = 0
    
    for take in split_takes:
        # Check if take directory exists
        take_dir = Path(data_dir) / take
        annotation_file = take_dir / 'annotation.json'
        
        if not take_dir.exists():
            print(f"Warning: Directory not found for take {take}, skipping...")
            skipped += 1
            continue
        
        if not annotation_file.exists():
            print(f"Warning: annotation.json not found for take {take}, skipping...")
            skipped += 1
            continue
        
        try:
            with open(annotation_file, 'r') as fp:
                annotation = json.load(fp)
        except Exception as e:
            print(f"Warning: Failed to read annotation for take {take}: {e}, skipping...")
            skipped += 1
            continue

        for obj_name in annotation['masks']:

            ego_cam = None
            for cam in annotation['masks'][obj_name]:
                if 'aria' in cam:
                    ego_cam = cam

            if ego_cam is None:
                continue

            for cam in annotation['masks'][obj_name]:
                
                if 'aria' in cam:
                    continue

                if setting == 'egoexo':
                    dcam = ego_cam
                elif setting == 'exoego':
                    dcam = cam
                else:
                    raise Exception(f"Setting {setting} not recognized.")

                for idx in annotation['masks'][obj_name][dcam]:
                    cam_rgb_path = f'{data_dir}//{take}//{cam}//{obj_name}//rgb//{idx}'
                    cam_mask_path = f'{data_dir}//{take}//{cam}//{obj_name}//mask//{idx}'

                    aria_rgb_path = f'{data_dir}//{take}//{ego_cam}//{obj_name}//rgb//{idx}'
                    aria_mask_path = f'{data_dir}//{take}//{ego_cam}//{obj_name}//mask//{idx}'

                    pairs.append( (aria_rgb_path, aria_mask_path, cam_rgb_path, cam_mask_path) )

    print(f'{split} - pairs: {len(pairs)}, skipped: {skipped}')
    with open(f'{data_dir}/{split}_{setting}_pairs.json', 'w') as fp:
        json.dump(pairs, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    make_pairs(args.data_dir, 'train', 'egoexo')
    make_pairs(args.data_dir, 'val', 'egoexo')

    make_pairs(args.data_dir, 'train', 'exoego')
    make_pairs(args.data_dir, 'val', 'exoego')