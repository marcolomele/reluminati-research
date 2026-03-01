"""
Compute additional cross-view correspondence statistics for the baseline set.

Metrics:
  - Camera coverage per take (number of ego/exo cameras, cameras per object)
  - Track temporal stats (annotated frame count, frame span, temporal density)
  - Object diversity per take and per scenario

Outputs (CSV):
  - baseline_camera_coverage.csv       (scenario, take_uid, tracked_object, n_cameras, n_ego_cameras, n_exo_cameras)
  - baseline_track_temporal_stats.csv   (scenario, take_uid, tracked_object, camera, n_annotated_frames, frame_span, temporal_density)
  - baseline_take_object_diversity.csv  (scenario, take_uid, n_tracked_objects)

Usage:
    python src/scripts/data/analyze_baseline_extra_stats.py
    python src/scripts/data/analyze_baseline_extra_stats.py --root /path/to/data --output-dir /path/to/output
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _baseline_common import (
    load_baseline_uids,
    load_all_annotations,
    build_uid_scenario_map,
)


def main():
    parser = argparse.ArgumentParser(description="Baseline extra correspondence stats")
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

    camera_rows = []
    temporal_rows = []
    diversity_rows = []

    for uid in sorted(baseline_uids):
        ann = annotations.get(uid)
        if not ann:
            print(f"  WARNING: UID {uid} not found in annotations")
            continue

        scenario = uid_scenario.get(uid, "unknown")
        object_masks = ann.get("object_masks", {})

        n_tracked_objects = len(object_masks)

        for obj_name, cameras in object_masks.items():
            cam_names = list(cameras.keys())
            ego_cams = [c for c in cam_names if c.startswith("aria")]
            exo_cams = [c for c in cam_names if not c.startswith("aria")]

            camera_rows.append({
                "scenario": scenario,
                "take_uid": uid,
                "tracked_object": obj_name,
                "n_cameras": len(cam_names),
                "n_ego_cameras": len(ego_cams),
                "n_exo_cameras": len(exo_cams),
            })

            for cam_name, cam_data in cameras.items():
                annotated_frames = cam_data.get("annotated_frames", [])
                n_frames = len(annotated_frames)

                if n_frames > 0:
                    frame_nums = sorted(int(f) for f in annotated_frames)
                    frame_span = frame_nums[-1] - frame_nums[0]
                    # temporal_density: fraction of the span that is annotated
                    # (1.0 = every possible frame annotated, lower = sparser)
                    expected_frames = (frame_span // 30 + 1) if frame_span > 0 else 1
                    temporal_density = round(n_frames / expected_frames, 4) if expected_frames > 0 else 1.0
                else:
                    frame_span = 0
                    temporal_density = 0.0

                temporal_rows.append({
                    "scenario": scenario,
                    "take_uid": uid,
                    "tracked_object": obj_name,
                    "camera": cam_name,
                    "n_annotated_frames": n_frames,
                    "frame_span": frame_span,
                    "temporal_density": temporal_density,
                })

        diversity_rows.append({
            "scenario": scenario,
            "take_uid": uid,
            "n_tracked_objects": n_tracked_objects,
        })

    # Write camera coverage
    path_cam = out_dir / "baseline_camera_coverage.csv"
    cam_fields = ["scenario", "take_uid", "tracked_object", "n_cameras", "n_ego_cameras", "n_exo_cameras"]
    with open(path_cam, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cam_fields)
        w.writeheader()
        for row in sorted(camera_rows, key=lambda r: (r["scenario"], r["take_uid"], r["tracked_object"])):
            w.writerow(row)
    print(f"Wrote {path_cam} ({len(camera_rows)} rows)")

    # Write track temporal stats
    path_temp = out_dir / "baseline_track_temporal_stats.csv"
    temp_fields = ["scenario", "take_uid", "tracked_object", "camera",
                   "n_annotated_frames", "frame_span", "temporal_density"]
    with open(path_temp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=temp_fields)
        w.writeheader()
        for row in sorted(temporal_rows, key=lambda r: (r["scenario"], r["take_uid"], r["tracked_object"], r["camera"])):
            w.writerow(row)
    print(f"Wrote {path_temp} ({len(temporal_rows)} rows)")

    # Write object diversity
    path_div = out_dir / "baseline_take_object_diversity.csv"
    div_fields = ["scenario", "take_uid", "n_tracked_objects"]
    with open(path_div, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=div_fields)
        w.writeheader()
        for row in sorted(diversity_rows, key=lambda r: (r["scenario"], r["take_uid"])):
            w.writerow(row)
    print(f"Wrote {path_div} ({len(diversity_rows)} rows)")


if __name__ == "__main__":
    main()
