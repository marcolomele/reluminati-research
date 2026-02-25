"""
Create a multi-frame baseline CSV for video-window experiments.

For each (take_uid, object_name, src_camera, dest_camera) stream in the
existing single-frame baseline CSV, this script expands the selection to
`window_size` consecutive annotated frames centred on the original baseline
frame.  Only frames where BOTH src and dst cameras have a valid mask are
included.

Output CSV has the same schema as the input:
    take_uid, object_name, src_camera, dest_camera, frame

Rows are sorted by (take_uid, object_name, src_camera, dest_camera, frame)
so that the video-window cache in experiment.py counts frames correctly.

Usage:
    python create_multiframe_csv.py \
        --baseline /data/video_datasets/3164542/lm_eec_pairs_from_vlm460850.csv \
        --root     /data/video_datasets/3321908/output_dir_all/ \
        --output   /data/video_datasets/3164542/lm_eec_pairs_multiframe_w10.csv \
        --window   10
"""

import argparse
import json
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="Path to single-frame baseline CSV")
    p.add_argument("--root",     required=True, help="Root data directory")
    p.add_argument("--output",   required=True, help="Output CSV path")
    p.add_argument("--window",   type=int, default=10, help="Frames per stream (default 10)")
    return p.parse_args()


def main():
    args = parse_args()
    baseline = pd.read_csv(args.baseline)
    print(f"Loaded {len(baseline)} baseline pairs from {args.baseline}")

    # Deduplicate to unique streams; for each stream use the median baseline frame as anchor
    stream_cols = ["take_uid", "object_name", "src_camera", "dest_camera"]
    streams = (
        baseline.groupby(stream_cols)["frame"]
        .apply(lambda frames: sorted(frames.astype(str).tolist(), key=int)[len(frames) // 2])
        .reset_index()
        .rename(columns={"frame": "anchor_frame"})
    )
    print(f"Unique streams: {len(streams)}")

    ann_cache = {}
    rows = []
    skipped = 0

    for _, row in streams.iterrows():
        uid      = str(row["take_uid"])
        obj      = str(row["object_name"])
        src_cam  = str(row["src_camera"])
        dst_cam  = str(row["dest_camera"])
        base_frm = str(row["anchor_frame"])

        # Load annotation (cached per take)
        if uid not in ann_cache:
            ann_path = os.path.join(args.root, uid, "annotation.json")
            with open(ann_path) as f:
                ann_cache[uid] = json.load(f)
        ann = ann_cache[uid]

        # Frames valid for BOTH src and dst cameras
        try:
            src_frames = set(ann["masks"][obj][src_cam].keys())
            dst_frames = set(ann["masks"][obj][dst_cam].keys())
            valid = sorted(src_frames & dst_frames, key=int)
        except KeyError:
            print(f"  SKIP (no mask): {uid}/{obj}/{src_cam}/{dst_cam}")
            skipped += 1
            continue

        if not valid:
            skipped += 1
            continue

        # Find position of anchor frame (or nearest available)
        if base_frm in set(valid):
            idx = valid.index(base_frm)
        else:
            idx = min(range(len(valid)), key=lambda i: abs(int(valid[i]) - int(base_frm)))

        # Centred window of `window` frames, clamped to list bounds
        half  = args.window // 2
        start = max(0, idx - half)
        end   = min(len(valid), start + args.window)
        start = max(0, end - args.window)   # re-adjust if end hit the boundary
        window_frames = valid[start:end]

        for frm in window_frames:
            rows.append({
                "take_uid":    uid,
                "object_name": obj,
                "src_camera":  src_cam,
                "dest_camera": dst_cam,
                "frame":       frm,
            })

    df = pd.DataFrame(rows)
    # Sort by stream then frame — required for video-window cache to work correctly
    df = df.sort_values(["take_uid", "object_name", "src_camera", "dest_camera", "frame"]) \
           .reset_index(drop=True)

    df.to_csv(args.output, index=False)

    n_streams = len(baseline) - skipped
    print(f"\nDone — {len(df)} rows across {n_streams} streams "
          f"(avg {len(df)/n_streams:.1f} frames/stream, window={args.window})")
    print(f"Skipped: {skipped}")
    print(f"Output: {args.output}")

    # Quick stats
    sizes = df.groupby(["take_uid", "object_name", "src_camera", "dest_camera"]).size()
    print(f"\nFrames per stream — min:{sizes.min()}  median:{int(sizes.median())}  max:{sizes.max()}")


if __name__ == "__main__":
    main()
