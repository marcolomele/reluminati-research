"""
Compute object displacement and size-change metrics for the baseline set.

For each (scenario, take_uid, tracked_object, camera):
  - Decode masks at every annotated frame to get centroid and area.
  - Sum frame-to-frame Euclidean centroid displacement -> total_displacement.
  - Sum frame-to-frame absolute area change -> total_area_change.
  - Mean area across all frames -> average_area.

Outputs (CSV):
  - baseline_object_displacement.csv   (scenario, take_uid, tracked_object, camera, total_displacement)
  - baseline_object_size_change.csv    (scenario, take_uid, tracked_object, camera, total_area_change, average_area)
  - baseline_object_dynamics_joint.csv (all columns combined)

Usage:
    python src/scripts/data/analyze_baseline_object_dynamics.py
    python src/scripts/data/analyze_baseline_object_dynamics.py --root /path/to/data --output-dir /path/to/output
"""

import argparse
import csv
import math
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _baseline_common import (
    load_baseline_uids,
    load_all_annotations,
    build_uid_scenario_map,
    decode_mask_full,
    mask_centroid,
)


def compute_track_dynamics(frame_annotations: dict) -> dict | None:
    """Compute displacement and area-change metrics for one object-camera track."""
    sorted_frames = sorted(frame_annotations.keys(), key=lambda k: int(k))
    if not sorted_frames:
        return None

    centroids = []
    areas = []
    decode_errors = 0

    for frame_str in sorted_frames:
        try:
            mask = decode_mask_full(frame_annotations[frame_str])
            area = int(mask.sum())
            cx, cy = mask_centroid(mask)
            centroids.append((cx, cy))
            areas.append(area)
        except Exception:
            decode_errors += 1

    if len(centroids) < 1:
        return None

    total_displacement = 0.0
    for j in range(1, len(centroids)):
        dx = centroids[j][0] - centroids[j - 1][0]
        dy = centroids[j][1] - centroids[j - 1][1]
        total_displacement += math.sqrt(dx * dx + dy * dy)

    total_area_change = 0.0
    for j in range(1, len(areas)):
        total_area_change += abs(areas[j] - areas[j - 1])

    average_area = float(np.mean(areas)) if areas else 0.0

    return {
        "total_displacement": round(total_displacement, 2),
        "total_area_change": round(total_area_change, 2),
        "average_area": round(average_area, 2),
        "n_frames": len(centroids),
        "decode_errors": decode_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline object dynamics analysis")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.output_dir or root / "baseline_analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root: {root}")
    print(f"Output: {out_dir}")

    baseline_uids = load_baseline_uids(root)
    annotations = load_all_annotations(root)
    uid_scenario = build_uid_scenario_map(annotations)

    print(f"Baseline UIDs: {len(baseline_uids)}")

    joint_rows = []
    total_decode_errors = 0

    for i, uid in enumerate(sorted(baseline_uids)):
        ann = annotations.get(uid)
        if not ann:
            print(f"  WARNING: UID {uid} not found in annotations")
            continue

        scenario = uid_scenario.get(uid, "unknown")
        object_masks = ann.get("object_masks", {})

        for obj_name, cameras in object_masks.items():
            for cam_name, cam_data in cameras.items():
                frame_annotations = cam_data.get("annotation", {})
                result = compute_track_dynamics(frame_annotations)
                if result is None:
                    continue

                total_decode_errors += result["decode_errors"]
                joint_rows.append({
                    "scenario": scenario,
                    "take_uid": uid,
                    "tracked_object": obj_name,
                    "camera": cam_name,
                    "total_displacement": result["total_displacement"],
                    "total_area_change": result["total_area_change"],
                    "average_area": result["average_area"],
                })

        if (i + 1) % 5 == 0 or i == len(baseline_uids) - 1:
            print(f"  Processed {i + 1}/{len(baseline_uids)} UIDs, {len(joint_rows)} tracks so far")

    if total_decode_errors:
        print(f"  WARNING: {total_decode_errors} total frame decode errors")

    joint_rows.sort(key=lambda r: (r["scenario"], r["take_uid"], r["tracked_object"], r["camera"]))

    # Displacement CSV
    path_disp = out_dir / "baseline_object_displacement.csv"
    disp_fields = ["scenario", "take_uid", "tracked_object", "camera", "total_displacement"]
    with open(path_disp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=disp_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(joint_rows)
    print(f"Wrote {path_disp} ({len(joint_rows)} rows)")

    # Size change CSV
    path_size = out_dir / "baseline_object_size_change.csv"
    size_fields = ["scenario", "take_uid", "tracked_object", "camera", "total_area_change", "average_area"]
    with open(path_size, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=size_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(joint_rows)
    print(f"Wrote {path_size} ({len(joint_rows)} rows)")

    # Joint CSV
    path_joint = out_dir / "baseline_object_dynamics_joint.csv"
    joint_fields = ["scenario", "take_uid", "tracked_object", "camera",
                    "total_displacement", "total_area_change", "average_area"]
    with open(path_joint, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=joint_fields)
        w.writeheader()
        w.writerows(joint_rows)
    print(f"Wrote {path_joint} ({len(joint_rows)} rows)")


if __name__ == "__main__":
    main()
