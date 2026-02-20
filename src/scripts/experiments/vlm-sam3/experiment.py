"""
Experiment pipeline for evaluating VLM + SAM3 object correspondence.

Runs three ablation experiments per (source, destination) pair:
  EXP-A:  source image with mask overlay only
  EXP-B:  clean source image + source image with mask overlay
  EXP-C:  clean source image + source image with mask overlay + destination image

Usage:
    python experiment.py --config config.json

TODO: 
* Make mask decoding more efficient by pre-extracting masks to disk in the
root folder using the ego-exo correspondence structure (see load_relation_masks.py
and process_data.py). Current approach decodes COCO RLE from annotation.json at
runtime for each pair.
* List of API keys for limits + sleep between requests.
* add congif arguments to output dataframe columns
* every X amount of frames, save to local the predicted masks for each of the three experiments to be able to compare qualitatively the model
"""

import argparse
import base64
import io
import json
import logging
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from sklearn.metrics import balanced_accuracy_score
from skimage import segmentation
from ollama import Client
from huggingface_hub import login
from transformers import Sam3Processor, Sam3Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENTS = ["EXP-A", "EXP-B", "EXP-C"]
SAVE_INTERVAL = 50


# ─── Configuration ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="VLM + SAM3 experiment pipeline")
    parser.add_argument("--config", required=True, help="Path to config.json")
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return json.load(f)


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_pairs(root_dir, split, direction):
    """Load pair tuples from {split}_{direction}_pairs.json in root directory."""
    pairs_path = os.path.join(root_dir, f"{split}_{direction}_pairs.json")
    logger.info("Loading pairs: %s", pairs_path)
    with open(pairs_path) as f:
        pairs = json.load(f)
    logger.info("Loaded %d pairs", len(pairs))
    return pairs


def subsample_pairs(pairs, pct, seed):
    """Randomly subsample pairs by percentage (0-1). Returns all if pct >= 1."""
    if pct >= 1.0:
        return pairs
    random.seed(seed)
    n = max(1, int(len(pairs) * pct))
    sampled = random.sample(pairs, n)
    logger.info("Subsampled %d / %d pairs (%.0f%%, seed=%d)", n, len(pairs), pct * 100, seed)
    return sampled


def parse_pair(pair, root_dir):
    """
    Extract metadata from a 4-tuple pair.

    Pair: [aria_rgb, aria_mask, cam_rgb, cam_mask]
    Path structure: root_dir / take_uid / camera / object / {rgb|mask} / frame
    """
    aria_rgb, _, cam_rgb, _ = pair

    parts_aria = Path(os.path.relpath(aria_rgb, root_dir)).parts
    parts_cam = Path(os.path.relpath(cam_rgb, root_dir)).parts

    return {
        "take_uid": parts_aria[0],
        "aria_camera": parts_aria[1],
        "object_name": parts_aria[2],
        "frame": parts_aria[4],
        "cam_camera": parts_cam[1],
        "aria_rgb_path": aria_rgb,
        "cam_rgb_path": cam_rgb,
    }


def resolve_img(path):
    """Resolve image path, trying common extensions if the bare path is missing."""
    if os.path.isfile(path):
        return path
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = path + ext
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Image not found: {path}")


# ─── Mask Handling ───────────────────────────────────────────────────────────


def decode_rle(rle):
    """Decode a COCO-RLE dict into a boolean numpy mask."""
    if isinstance(rle, dict) and "counts" in rle:
        return mask_utils.decode(rle).astype(bool)
    return None


def load_annotation(root_dir, take_uid):
    """Load annotation.json for a take (contains pre-decoded COCO RLE masks)."""
    path = os.path.join(root_dir, take_uid, "annotation.json")
    with open(path) as f:
        return json.load(f)


def get_mask(ann, obj, cam, frame):
    """Look up and decode a single mask from the annotation dict."""
    try:
        return decode_rle(ann["masks"][obj][cam][str(frame)])
    except KeyError:
        return None


# ─── Image Helpers ───────────────────────────────────────────────────────────


def create_overlay(img_path, mask_np, alpha=0.5):
    """Return a PIL Image with a red overlay on the masked region."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    if mask_np.shape[0] != h or mask_np.shape[1] != w:
        mask_np = cv2.resize(
            mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    arr = np.array(img)
    red = np.zeros_like(arr)
    red[:] = [255, 0, 0]
    m = mask_np == 1
    arr[m] = (arr[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_b64(pil_img, fmt="JPEG"):
    """Base64-encode a PIL Image."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path):
    """Base64-encode an image file on disk."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─── VLM ─────────────────────────────────────────────────────────────────────


def init_vlm(cfg):
    """Initialize the Ollama VLM client."""
    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {cfg['vlm-api-key']}"},
    )


def vlm_caption(client, model, images_b64, prompt, retries=3):
    """Query the VLM with images + prompt. Returns the generated text."""
    msgs = [{"role": "user", "content": prompt, "images": images_b64}]

    for attempt in range(retries):
        try:
            text = ""
            for part in client.chat(model, messages=msgs, stream=True):
                text += part.message.content
            return text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 30 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
            else:
                logger.error("VLM error: %s", e)
                return "ERROR"
    return "ERROR"


# ─── SAM3 ────────────────────────────────────────────────────────────────────


def init_sam3(cfg):
    """Load the SAM3 model and processor onto the best available device."""
    login(token=cfg["huggingface-api-key"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg["huggingface-model"]
    model = Sam3Model.from_pretrained(name).to(device)
    proc = Sam3Processor.from_pretrained(name)
    logger.info("SAM3 ready on %s (%s)", device, name)
    return model, proc, device


def sam3_segment(model, proc, img_path, text_prompt, device):
    """Run text-prompted SAM3 segmentation and return post-processed results."""
    img = Image.open(img_path).convert("RGB")
    inputs = proc(images=img, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)

    return proc.post_process_instance_segmentation(
        out,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]


# ─── Metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(gt, sam_out):
    """Compute IoU, balanced accuracy, contour accuracy, and label error."""
    masks = sam_out["masks"]
    if len(masks) == 0:
        return {"iou": 0.0, "ba": 0.0, "ca": 0.0, "le": 1.0}

    best_iou, best_pred = 0.0, None
    for t in masks:
        p = t.cpu().numpy().astype(bool)
        inter = np.logical_and(p, gt).sum()
        union = np.logical_or(p, gt).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_pred = iou, p

    if best_pred is None:
        return {"iou": 0.0, "ba": 0.0, "ca": 0.0, "le": 1.0}

    ba = balanced_accuracy_score(gt.flatten(), best_pred.flatten())
    le = float(np.mean(best_pred != gt))

    gt_b = segmentation.find_boundaries(gt, mode="inner")
    pr_b = segmentation.find_boundaries(best_pred, mode="inner")
    ca = (
        float(np.logical_and(gt_b, pr_b).sum() / np.logical_or(gt_b, pr_b).sum())
        if np.any(gt_b | pr_b)
        else 0.0
    )
    return {"iou": best_iou, "ba": ba, "ca": ca, "le": le}


def spatial_covariates(mask_np):
    """Compute size category, relative area, centricity from a GT mask."""
    H, W = mask_np.shape
    rel_area = float(np.sum(mask_np)) / (H * W)

    coords = np.argwhere(mask_np)
    if len(coords) == 0:
        return {
            "obj_rel_area": 0.0,
            "obj_dist_center": 1.0,
            "obj_size_cat": "none",
            "obj_is_peripheral": True,
        }

    cy, cx = coords.mean(axis=0)
    dist = np.hypot(cy - H / 2, cx - W / 2) / np.hypot(H / 2, W / 2)

    if rel_area < 0.005:
        cat = "small"
    elif rel_area < 0.05:
        cat = "medium"
    else:
        cat = "large"

    return {
        "obj_rel_area": rel_area,
        "obj_dist_center": float(dist),
        "obj_size_cat": cat,
        "obj_is_peripheral": dist > 0.5,
    }


# ─── Per-Pair Pipeline ──────────────────────────────────────────────────────


def process_pair(meta, cfg, vlm, sam_m, sam_p, dev, ann):
    """
    Run EXP-A / EXP-B / EXP-C for one pair and return a list of result dicts.
    """
    direction = cfg["direction"].replace("-", "")

    if direction == "egoexo":
        src_cam, dst_cam = meta["aria_camera"], meta["cam_camera"]
        src_rgb, dst_rgb = meta["aria_rgb_path"], meta["cam_rgb_path"]
    else:
        src_cam, dst_cam = meta["cam_camera"], meta["aria_camera"]
        src_rgb, dst_rgb = meta["cam_rgb_path"], meta["aria_rgb_path"]

    src_rgb = resolve_img(src_rgb)
    dst_rgb = resolve_img(dst_rgb)
    obj, frame = meta["object_name"], meta["frame"]

    src_mask = get_mask(ann, obj, src_cam, frame)
    dst_gt = get_mask(ann, obj, dst_cam, frame)
    if src_mask is None or dst_gt is None:
        logger.warning("Mask missing: %s / %s / %s – skipped", meta["take_uid"], obj, frame)
        return []

    spatial = spatial_covariates(dst_gt)

    overlay_img = create_overlay(src_rgb, src_mask)
    overlay_b64 = pil_to_b64(overlay_img)
    src_b64 = file_to_b64(src_rgb)
    dst_b64 = file_to_b64(dst_rgb)

    exp_spec = {
        "EXP-A": {"images": [overlay_b64], "prompt": cfg["prompt-exp-a"]},
        "EXP-B": {"images": [src_b64, overlay_b64], "prompt": cfg["prompt-exp-b"]},
        "EXP-C": {"images": [src_b64, overlay_b64, dst_b64], "prompt": cfg["prompt-exp-c"]},
    }

    model_name = cfg["vlm-model"]
    rows = []

    for exp_id in EXPERIMENTS:
        spec = exp_spec[exp_id]
        caption = vlm_caption(vlm, model_name, spec["images"], spec["prompt"])
        sam_out = sam3_segment(sam_m, sam_p, dst_rgb, caption, dev)

        h_s, w_s = ( #height and width of the predicted mask 
            sam_out["masks"][0].shape[-2:] if len(sam_out["masks"]) > 0 else dst_gt.shape
        )
        if dst_gt.shape != (h_s, w_s):
            gt_resized = cv2.resize(
                dst_gt.astype(np.uint8), (w_s, h_s), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            gt_resized = dst_gt

        metrics = compute_metrics(gt_resized, sam_out)

        rows.append({
            "take_uid": meta["take_uid"],
            "object_name": obj,
            "src_camera": src_cam,
            "dest_camera": dst_cam,
            "frame": frame,
            "experiment": exp_id,
            "vlm_model": model_name,
            "vlm_output": caption,
            **metrics,
            **spatial,
        })

    return rows


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    cfg = load_config(args.config)

    root = cfg["root-data-directory"]
    split = cfg["split"]
    direction = cfg["direction"]

    pairs = load_pairs(root, split, direction)
    pairs = subsample_pairs(pairs, cfg["subset-run-percentage"], cfg["subset-seed"])

    vlm = init_vlm(cfg)
    sam_m, sam_p, dev = init_sam3(cfg)

    ann_cache = {}
    all_rows = []
    n = len(pairs)
    t0 = time.time()

    for i, pair in enumerate(pairs):
        meta = parse_pair(pair, root)
        uid = meta["take_uid"]

        logger.info(
            "[%d/%d] %s | %s | frame %s",
            i + 1, n, uid, meta["object_name"], meta["frame"],
        )

        if uid not in ann_cache:
            ann_cache[uid] = load_annotation(root, uid)

        try:
            all_rows.extend(process_pair(meta, cfg, vlm, sam_m, sam_p, dev, ann_cache[uid]))
        except Exception as e:
            logger.error("Pair %d failed: %s", i + 1, e)

        if (i + 1) % SAVE_INTERVAL == 0 and all_rows:
            pd.DataFrame(all_rows).to_csv("results_partial.csv", index=False)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            logger.info("Checkpoint saved (%d rows). ETA %.0fs", len(all_rows), eta)

    if not all_rows:
        logger.warning("No results collected.")
        return

    df = pd.DataFrame(all_rows)
    tag = f"{split}_{direction}_{cfg['vlm-model']}"

    raw_path = f"results_{tag}.csv"
    df.to_csv(raw_path, index=False)
    logger.info("Raw results → %s", raw_path)

    summary = df.groupby("experiment")[["iou", "ba", "ca", "le"]].mean()
    summary_path = f"summary_{tag}.csv"
    summary.to_csv(summary_path)
    logger.info("Summary → %s", summary_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Pairs: {n}  |  Result rows: {len(df)}  |  Time: {elapsed:.0f}s")
    print(f"\nMean metrics by experiment:\n{summary.to_string()}")
    print(f"\nMean IoU by object size:\n{df.groupby('obj_size_cat')[['iou']].mean().to_string()}")


if __name__ == "__main__":
    main()
