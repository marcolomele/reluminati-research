"""
Compute per-frame mask area for all tracked objects in the baseline set.

Uses mask_utils.area() on RLE for speed (no full mask decode).

Outputs (CSV):
  - baseline_mask_area_all.csv                (scenario, take_uid, tracked_object, camera, frame, mask_area)
  - baseline_mask_area_summary_by_scenario.csv (scenario, count, mean, median, p05, p25, p75, p95)

Usage:
    python src/scripts/data/analyze_baseline_mask_sizes.py
    python src/scripts/data/analyze_baseline_mask_sizes.py --root /path/to/data --output-dir /path/to/output
"""

import argparse
import csv
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _baseline_common import (
    load_baseline_uids,
    load_all_annotations,
    build_uid_scenario_map,
    mask_area_from_annotation,
)


def main():
    parser = argparse.ArgumentParser(description="Baseline mask area analysis")
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

    rows = []
    decode_errors = 0

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
                for frame_str, frame_ann in frame_annotations.items():
                    try:
                        area = mask_area_from_annotation(frame_ann)
                        rows.append({
                            "scenario": scenario,
                            "take_uid": uid,
                            "tracked_object": obj_name,
                            "camera": cam_name,
                            "frame": int(frame_str),
                            "mask_area": area,
                        })
                    except Exception:
                        decode_errors += 1

        if (i + 1) % 10 == 0 or i == len(baseline_uids) - 1:
            print(f"  Processed {i + 1}/{len(baseline_uids)} UIDs, {len(rows)} rows so far")

    if decode_errors:
        print(f"  WARNING: {decode_errors} frames failed to decode")

    # Write per-frame CSV
    path_all = out_dir / "baseline_mask_area_all.csv"
    fieldnames = ["scenario", "take_uid", "tracked_object", "camera", "frame", "mask_area"]
    with open(path_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["scenario"], r["take_uid"], r["tracked_object"], r["camera"], r["frame"])):
            w.writerow(row)
    print(f"Wrote {path_all} ({len(rows)} rows)")

    # Compute summary by scenario
    areas_by_scenario: dict[str, list[int]] = {}
    for row in rows:
        areas_by_scenario.setdefault(row["scenario"], []).append(row["mask_area"])

    path_summary = out_dir / "baseline_mask_area_summary_by_scenario.csv"
    summary_fields = ["scenario", "count", "mean", "median", "p05", "p25", "p75", "p95"]
    with open(path_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for s in sorted(areas_by_scenario):
            a = np.array(areas_by_scenario[s])
            w.writerow({
                "scenario": s,
                "count": len(a),
                "mean": f"{a.mean():.1f}",
                "median": f"{np.median(a):.1f}",
                "p05": f"{np.percentile(a, 5):.1f}",
                "p25": f"{np.percentile(a, 25):.1f}",
                "p75": f"{np.percentile(a, 75):.1f}",
                "p95": f"{np.percentile(a, 95):.1f}",
            })
    print(f"Wrote {path_summary} ({len(areas_by_scenario)} rows)")


if __name__ == "__main__":
    main()
