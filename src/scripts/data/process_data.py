"""
This script reads raw takes from a source directory (e.g. on a research cluster),
extracts annotated frames, and organizes them for baseline model training.

Source directory structure:
    source-dir/
        takes/<take_name>/frame_aligned_videos/...
        annotations/splits.json                    # {uid: split_name, ...}
        annotations/relations_train.json
        annotations/relations_val.json
        annotations/relations_test.json

Usage:
    python process_data.py --source-dir /path/to/egoexo/root
    python process_data.py --scenario cooking --source-dir /path/to/egoexo/root
    python process_data.py --keyword basketball --source-dir /path/to/egoexo/root
    python process_data.py --keyword cooking --limit 10 --source-dir /path/to/egoexo/root
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import cv2
from decord import VideoReader, cpu
from lzstring import LZString


def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string."""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def decode_mask(width: int, height: int, encoded_mask: str):
    """Decode LZString compressed mask to COCO RLE format."""
    try:
        decomp_string = LZString.decompressFromEncodedURIComponent(encoded_mask)
    except:
        return None
    
    decomp_encoded = decomp_string.encode()
    rle_obj = {
        "size": [height, width],
        "counts": decomp_encoded,
    }
    rle_obj['counts'] = rle_obj['counts'].decode('ascii')
    return rle_obj


def process_masks(object_masks: Dict[str, Any]):
    """
    Process and decode all masks in object_masks annotation in format: {object_id}/{cam_id}/{frame_id} -> COCO RLE
    """
    processed_masks = {}
    
    for object_id, obj_data in object_masks.items():
        processed_masks[object_id] = {}
        
        for cam_id, cam_data in obj_data.items():
            processed_masks[object_id][cam_id] = {}
            
            annotations = cam_data.get("annotation", {})
            for frame_id, frame_data in annotations.items():
                width = frame_data.get("width")
                height = frame_data.get("height")
                encoded_mask = frame_data.get("encodedMask")
                
                if width and height and encoded_mask:
                    coco_mask = decode_mask(width, height, encoded_mask)
                    if coco_mask:
                        processed_masks[object_id][cam_id][frame_id] = coco_mask
    
    return processed_masks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Locate and process EgoExo4D data for correspondence task"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario category to filter (e.g. 'cooking', 'basketball'). Case-insensitive substring match on scenario field."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Root directory containing takes/ and annotations/ folders"
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Filter takes by keyword contained in the take folder name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of takes to process (randomly sampled from keyword-matched takes)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed data (default: ../output_dir_{scenario})"
    )
    return parser.parse_args()


def load_split_uids(source_dir: Path) -> Dict[str, List[str]]:
    """Load UIDs from source_dir/annotations/splits.json (uid -> split mapping)."""
    split_file = source_dir / "annotations" / "splits.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        data = json.load(f)
    uid_to_split = data.get("take_uid_to_split", data)

    splits: Dict[str, List[str]] = {}
    for uid, split_name in uid_to_split.items():
        splits.setdefault(split_name, []).append(uid)

    print(f"Loaded splits from {split_file}:")
    for split_name, uids in splits.items():
        print(f"  {split_name}: {len(uids)} UIDs")
    return splits


def load_annotations(source_dir: Path) -> Dict[str, Tuple[Dict[str, Any], str]]:
    """Load all annotations from source_dir/annotations/relations_{split}.json."""
    all_annotations = {}
    annotation_dir = source_dir / "annotations"
    for split_name in ("train", "val", "test"):
        file_path = annotation_dir / f"relations_{split_name}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for uid, ann in data.get('annotations', {}).items():
                all_annotations[uid] = (ann, split_name)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    return all_annotations


def get_all_cameras_from_annotation(annotation: Dict[str, Any]) -> Dict[str, List[int]]:
    """Extract all cameras and their annotated frames from annotation."""
    camera_frames = {}
    
    for obj_data in annotation.get('object_masks', {}).values():
        for cam_name, cam_data in obj_data.items():
            if cam_name not in camera_frames:
                camera_frames[cam_name] = set()
            camera_frames[cam_name].update(cam_data.get('annotated_frames', []))
    
    return {cam: sorted(list(frames)) for cam, frames in camera_frames.items()}


def extract_frames_from_video(
    video_path: Path,
    frame_indices: List[int],
    output_dir: Path,
    is_aria: bool = False
) -> bool:
    """Extract specific frames from video and save as JPEGs."""
    if not video_path.exists():
        print(f"Video not found: {video_path.name}")
        return False
    
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        valid_indices = [idx for idx in frame_indices if idx < len(vr)]
        
        if not valid_indices:
            print(f"No valid frame indices for {video_path.name}")
            return False
        
        if len(valid_indices) < len(frame_indices):
            print(f"Warning: {len(frame_indices) - len(valid_indices)} frame indices out of bounds for {video_path}")
        
        frames = vr.get_batch(valid_indices).asnumpy()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for frame, idx in zip(frames, valid_indices):
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{idx}.jpg"), frame_bgr)
        
        print(f"Extracted {len(valid_indices)} frames")
        return True
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return False

def process_take(
    uid: str,
    annotation: Dict[str, Any],
    source_dir: Path,
    output_dir: Path
) -> bool:
    """Process a single take: locate videos in source_dir, extract frames, create annotation.json."""
    take_start = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"Processing: {uid}")
    print(f"{'='*80}")
    
    take_name = annotation.get('take_name')
    if not take_name:
        print(f"Error: No take_name found in annotation for {uid}")
        return False
    
    print(f"Take name: {take_name}")
    
    # Locate videos in source directory
    takes_dir = source_dir / "takes" / take_name / "frame_aligned_videos" / "downscaled" / "448"
    if not takes_dir.exists():
        takes_dir = source_dir / "takes" / take_name / "frame_aligned_videos"
        if not takes_dir.exists():
            takes_dir = source_dir / "takes" / take_name
            if not takes_dir.exists():
                print(f"Error: Take not found at {source_dir / 'takes' / take_name}")
                return False
    
    # Get cameras and frames
    camera_frames = get_all_cameras_from_annotation(annotation)
    if not camera_frames:
        print(f"Warning: No cameras found in annotation for {uid}")
        return False
    
    print(f"Found {len(camera_frames)} cameras to process")
    
    # Create take output directory
    take_output_dir = output_dir / uid
    take_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process cameras
    success_count = 0
    for cam_name, frame_indices in camera_frames.items():
        print(f"\nProcessing camera: {cam_name}")
        print(f"  Annotated frames: {len(frame_indices)}")
        
        # Find video file
        video_path = takes_dir / f"{cam_name}.mp4"
        is_aria = 'aria' in cam_name.lower()
        
        # Extract frames
        cam_output_dir = take_output_dir / cam_name
        if extract_frames_from_video(video_path, frame_indices, cam_output_dir, is_aria):
            success_count += 1
    
    if success_count == 0:
        print(f"Error: No frames extracted for {uid}")
        return False
    
    print(f"\nSuccessfully processed {success_count}/{len(camera_frames)} cameras")
    
    # Process masks - decode from LZString to COCO RLE format
    print("\nProcessing and decoding masks...")
    object_masks = annotation.get("object_masks", {})
    decoded_masks = process_masks(object_masks)
    print(f"Decoded masks for {len(decoded_masks)} objects")
    
    # Get all extracted frame indices (union of all cameras)
    all_frame_indices = set()
    for frames in camera_frames.values():
        all_frame_indices.update(frames)
    subsample_idx = sorted(list(all_frame_indices))
    
    # Create annotation.json
    annotation_output = {
        "scenario": annotation.get("scenario", ""),
        "take_name": take_name,
        "object_masks": object_masks,
        "masks": decoded_masks,
        "subsample_idx": subsample_idx
    }
    
    annotation_path = take_output_dir / "annotation.json"
    with open(annotation_path, 'w') as f:
        json.dump(annotation_output, f, indent=2)
    
    print(f"Created annotation file: {annotation_path}")
    
    duration = (datetime.now() - take_start).total_seconds()
    print(f"\nTake processed in {format_duration(duration)}")
    
    return True


def main():
    """Main execution function."""
    script_start = datetime.now()
    args = parse_args()
    
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    scenario_label = args.scenario if args.scenario else "all"
    print(f"\n{'='*80}")
    print(f"EgoExo4D Data Locate and Process")
    print(f"Scenario: {scenario_label}")
    print(f"Source dir: {source_dir}")
    if args.keyword:
        print(f"Keyword filter: {args.keyword}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Started: {script_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Load splits (informational only -- not used for UID selection)
    try:
        splits = load_split_uids(source_dir)
    except Exception as e:
        print(f"Warning: Could not load splits.json: {e}")
        splits = {}
    
    annotations = load_annotations(source_dir)
    if not annotations:
        print("Error: No annotations loaded. Check that relations_*.json files exist.")
        return 1
    
    # Build UID list from annotations (only UIDs with actual annotation data)
    all_uids = list(annotations.keys())
    print(f"\nLoaded {len(all_uids)} annotated UIDs")
    
    # Filter by scenario (case-insensitive substring match)
    if args.scenario:
        kw = args.scenario.lower()
        all_uids = [uid for uid in all_uids if kw in annotations[uid][0].get('scenario', '').lower()]
        print(f"Scenario filter '{args.scenario}': {len(all_uids)} matching UIDs")
    
    # Filter by keyword on take_name (case-insensitive substring match)
    if args.keyword:
        kw = args.keyword.lower()
        matching_uids = [uid for uid in all_uids if kw in annotations[uid][0].get('take_name', '').lower()]
        print(f"Keyword '{args.keyword}': {len(matching_uids)} matching UIDs")
        if args.limit and len(matching_uids) > args.limit:
            by_split: Dict[str, List[str]] = defaultdict(list)
            for uid in matching_uids:
                by_split[annotations[uid][1]].append(uid)
            total = len(matching_uids)
            target_per_split = {s: max(0, round(args.limit * len(uids) / total)) for s, uids in by_split.items()}
            n = sum(target_per_split.values())
            while n < args.limit and any(target_per_split[s] < len(by_split[s]) for s in target_per_split):
                for s in sorted(target_per_split.keys(), key=lambda x: -len(by_split[x])):
                    if target_per_split[s] < len(by_split[s]) and n < args.limit:
                        target_per_split[s] += 1
                        n += 1
                        break
            matching_uids = []
            for s, uids in by_split.items():
                matching_uids.extend(random.sample(uids, min(target_per_split[s], len(uids))))
            random.shuffle(matching_uids)
            print(f"Sampled {len(matching_uids)} takes (split-proportional)")
        all_uids = matching_uids
    
    print(f"\nTotal UIDs to process: {len(all_uids)}\n")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"../output_dir_{scenario_label}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Process UIDs
    success_count = 0
    failed_uids = []
    uid_times = []
    
    for i, uid in enumerate(all_uids, 1):
        uid_start = datetime.now()
        
        print(f"\n{'*'*80}")
        print(f"Progress: {i}/{len(all_uids)} | Elapsed: {format_duration((uid_start - script_start).total_seconds())}")
        print(f"{'*'*80}")
        
        if uid not in annotations:
            print(f"Annotation not found for {uid}")
            failed_uids.append((uid, "annotation_not_found"))
            continue
        
        annotation, split_name = annotations[uid]
        print(f"Found in {split_name} split")
        
        # Process
        if process_take(uid, annotation, source_dir, output_dir):
            success_count += 1
            uid_duration = (datetime.now() - uid_start).total_seconds()
            uid_times.append(uid_duration)
            
            if len(uid_times) > 0:
                avg_time = sum(uid_times) / len(uid_times)
                remaining = (len(all_uids) - i) * avg_time
                print(f"\nEstimated time remaining: {format_duration(remaining)}")
        else:
            failed_uids.append((uid, "processing_failed"))
    
    # Summary
    total_duration = (datetime.now() - script_start).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {format_duration(total_duration)}")
    print(f"Total UIDs: {len(all_uids)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_uids)}")
    
    if uid_times:
        avg_time = sum(uid_times) / len(uid_times)
        print(f"Average time per UID: {format_duration(avg_time)}")
    
    if failed_uids:
        print(f"\nFailed UIDs:")
        for uid, reason in failed_uids:
            print(f"  {uid}: {reason}")
    
    return 0 if len(failed_uids) == 0 else 1


if __name__ == "__main__":
    exit(main())