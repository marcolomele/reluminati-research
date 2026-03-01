"""
Experiment pipeline for evaluating VLM + SAM3 object correspondence.

Experiments are fully config-driven via config.json. Each entry in
cfg["experiments"] specifies an id, image tokens, prompt key, model,
num_predict, and optional num_frames for multi-frame inputs.

Built-in image tokens:
  src_overlay  – source frame with red mask overlay
  src_clean    – source frame with true colours
  src_bbox     – source frame with red bounding box around object
  src_crop     – source frame cropped to object bounding box
  dst          – destination frame (single frame only)

Usage:
    python experiment.py --config config.json
"""

import argparse
import base64
import glob
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

SAVE_INTERVAL = 50
DEFAULT_VLM_REPROMPT_GROWTH_THRESHOLD = 0.20
DEFAULT_VLM_REPROMPT_MOVEMENT_THRESHOLD = 0.20

# Use SLURM job ID as the output filename so each run has its own file.
# Falls back to "local" when running outside SLURM.
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


# ─── Configuration ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="VLM + SAM3 experiment pipeline")
    parser.add_argument("--config", required=True, help="Path to config.json")
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return json.load(f)


def get_api_keys(cfg, list_key, single_key, provider_name):
    """
    Read API keys from config.

    Preferred format is a list (e.g. `vlm-api-keys`), with fallback to the
    legacy single-key field for backward compatibility.
    """
    keys = cfg.get(list_key)
    if keys is None and single_key in cfg:
        keys = [cfg[single_key]]

    if not isinstance(keys, list) or not keys:
        raise ValueError(
            f"Missing {provider_name} API keys. Provide `{list_key}` (list) in config."
        )

    cleaned = [k.strip() for k in keys if isinstance(k, str) and k.strip()]
    if not cleaned:
        raise ValueError(
            f"No valid {provider_name} API keys found in `{list_key}`."
        )
    return cleaned


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_pairs_from_json(cfg):
    """
    Load pairs from {split}_{direction}_pairs.json and return meta dicts.

    Resolves src/dst according to cfg["direction"] so callers get unified keys:
    take_uid, object_name, frame, src_camera, dst_camera, src_rgb_path, dst_rgb_path
    """
    root_dir = cfg["root-data-directory"]
    split = cfg["split"]
    direction = cfg["direction"]
    pairs_path = os.path.join(root_dir, f"{split}_{direction}_pairs.json")
    logger.info("Loading pairs: %s", pairs_path)
    with open(pairs_path) as f:
        raw_pairs = json.load(f)
    logger.info("Loaded %d pairs", len(raw_pairs))

    direction_norm = direction.replace("-", "")
    result = []
    for pair in raw_pairs:
        aria_rgb, _, cam_rgb, _ = pair

        parts_aria = Path(os.path.relpath(aria_rgb, root_dir)).parts
        parts_cam = Path(os.path.relpath(cam_rgb, root_dir)).parts

        take_uid = parts_aria[0]
        aria_camera = parts_aria[1]
        object_name = parts_aria[2]
        frame = parts_aria[4]
        cam_camera = parts_cam[1]

        if direction_norm == "egoexo":
            src_camera, dst_camera = aria_camera, cam_camera
        else:
            src_camera, dst_camera = cam_camera, aria_camera

        # Images live at root/take/camera/frame.jpg (flat — no object/rgb subdirs)
        src_rgb_path = os.path.join(root_dir, take_uid, src_camera, frame)
        dst_rgb_path = os.path.join(root_dir, take_uid, dst_camera, frame)

        result.append({
            "take_uid": take_uid,
            "object_name": object_name,
            "frame": frame,
            "src_camera": src_camera,
            "dst_camera": dst_camera,
            "src_rgb_path": src_rgb_path,
            "dst_rgb_path": dst_rgb_path,
        })
    return result


def load_pairs_from_csv(csv_path, root_dir):
    """
    Load pairs from CSV baseline file and return meta dicts.

    CSV columns: take_uid, object_name, src_camera, dest_camera, frame
    Image paths reconstructed as: root_dir/take_uid/camera/object_name/rgb/frame
    """
    logger.info("Loading pairs from CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d pairs from CSV", len(df))
    result = []
    for _, row in df.iterrows():
        take_uid = str(row["take_uid"])
        obj = str(row["object_name"])
        src_cam = str(row["src_camera"])
        dst_cam = str(row["dest_camera"])
        frame = str(row["frame"])
        src_rgb_path = os.path.join(root_dir, take_uid, src_cam, frame)
        dst_rgb_path = os.path.join(root_dir, take_uid, dst_cam, frame)
        result.append({
            "take_uid": take_uid,
            "object_name": obj,
            "frame": frame,
            "src_camera": src_cam,
            "dst_camera": dst_cam,
            "src_rgb_path": src_rgb_path,
            "dst_rgb_path": dst_rgb_path,
        })
    return result


def load_pairs(cfg):
    """Dispatcher: load pairs from CSV baseline or JSON depending on config."""
    if "pairs-csv" in cfg:
        return load_pairs_from_csv(cfg["pairs-csv"], cfg["root-data-directory"])
    return load_pairs_from_json(cfg)


def subsample_pairs(pairs, pct, seed):
    """Randomly subsample pairs by percentage (0-1). Returns all if pct >= 1."""
    if pct >= 1.0:
        return pairs
    random.seed(seed)
    n = max(1, int(len(pairs) * pct))
    sampled = random.sample(pairs, n)
    logger.info("Subsampled %d / %d pairs (%.0f%%, seed=%d)", n, len(pairs), pct * 100, seed)
    return sampled


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


def _align_mask(mask_np, img_w, img_h):
    """Resize mask to match (img_w, img_h) if needed. Returns bool array."""
    if mask_np.shape[0] != img_h or mask_np.shape[1] != img_w:
        return cv2.resize(
            mask_np.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return mask_np


def mask_bbox(mask_np):
    """Return (rmin, rmax, cmin, cmax) tight bounding box, or None if mask is empty."""
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any():
        return None
    rmin, rmax = int(np.where(rows)[0][[0, -1]][0]), int(np.where(rows)[0][[0, -1]][1])
    cmin, cmax = int(np.where(cols)[0][[0, -1]][0]), int(np.where(cols)[0][[0, -1]][1])
    return rmin, rmax, cmin, cmax


def create_overlay(img_path, mask_np, alpha=0.5):
    """Return a PIL Image with a red overlay on the masked region."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    aligned = _align_mask(mask_np, w, h)
    arr = np.array(img)
    red = np.zeros_like(arr)
    red[:] = [255, 0, 0]
    m = aligned == 1
    arr[m] = (arr[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return Image.fromarray(arr)


def create_bbox_image(img_path, mask_np, color=(255, 0, 0), thickness=3):
    """Return a PIL Image with a colored rectangle drawn around the masked region."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    aligned = _align_mask(mask_np, w, h)
    bbox = mask_bbox(aligned)
    if bbox is None:
        return img
    rmin, rmax, cmin, cmax = bbox
    arr = np.array(img)
    cv2.rectangle(arr, (cmin, rmin), (cmax, rmax), color, thickness)
    return Image.fromarray(arr)


def create_crop_image(img_path, mask_np, padding=10):
    """Return a PIL Image cropped to the bounding box of the mask (+ padding)."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    aligned = _align_mask(mask_np, w, h)
    bbox = mask_bbox(aligned)
    if bbox is None:
        return img
    rmin, rmax, cmin, cmax = bbox
    left = max(0, cmin - padding)
    upper = max(0, rmin - padding)
    right = min(w, cmax + padding)
    lower = min(h, rmax + padding)
    return img.crop((left, upper, right, lower))


def pil_to_b64(pil_img, fmt="JPEG"):
    """Base64-encode a PIL Image."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def file_to_b64(path):
    """Base64-encode an image file on disk."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def sample_source_frames(ann, obj, cam, n_frames, take_uid, root_dir, current_frame=None):
    """
    Return n_frames (img_path, mask_np) tuples for a source stream.

    When current_frame is provided, selects the n_frames annotated frames
    immediately PRECEDING (and including) current_frame, giving the VLM
    genuine temporal context for the current query.

    Falls back to evenly-spaced sampling across the whole video only when
    current_frame is None — this is intentionally avoided for multi-frame
    experiments because random video-wide sampling produces semantically
    unrelated frames that confuse the model.
    """
    try:
        all_frame_ids = sorted(ann["masks"][obj][cam].keys(), key=lambda x: int(x))
    except KeyError:
        return []
    if not all_frame_ids:
        return []

    if current_frame is not None:
        current_int = int(current_frame)
        # Keep only frames up to and including the current one
        preceding = [f for f in all_frame_ids if int(f) <= current_int]
        if not preceding:
            preceding = all_frame_ids[:n_frames]  # edge case: current before all annotations
        frame_ids_to_use = preceding[-n_frames:]   # take the last n (most recent)
    else:
        # Legacy fallback — evenly spaced across the full video
        indices = np.linspace(0, len(all_frame_ids) - 1, n_frames, dtype=int)
        frame_ids_to_use = [all_frame_ids[i] for i in indices]

    result = []
    for fid in frame_ids_to_use:
        img_path = os.path.join(root_dir, take_uid, cam, fid)
        try:
            img_path = resolve_img(img_path)
        except FileNotFoundError:
            continue
        mask_np = decode_rle(ann["masks"][obj][cam][fid])
        if mask_np is not None:
            result.append((img_path, mask_np))
    return result


def build_images_for_token(token, src_rgb, src_mask, dst_rgb, ann, take_uid, obj, src_cam, root_dir, num_frames, current_frame=None):
    """
    Dispatch on token and return a list of base64-encoded image strings.

    Multi-frame tokens (src_overlay, src_clean) return num_frames images when
    num_frames > 1; single-frame tokens (src_bbox, src_crop, dst) always return
    exactly one image.

    current_frame must be passed for multi-frame tokens so that
    sample_source_frames can select frames preceding the current query,
    not random frames from across the whole video.
    """
    if token == "src_overlay":
        if num_frames > 1:
            frames = sample_source_frames(ann, obj, src_cam, num_frames, take_uid, root_dir, current_frame=current_frame)
            return [pil_to_b64(create_overlay(fp, m)) for fp, m in frames]
        return [pil_to_b64(create_overlay(src_rgb, src_mask))]
    elif token == "src_clean":
        if num_frames > 1:
            frames = sample_source_frames(ann, obj, src_cam, num_frames, take_uid, root_dir, current_frame=current_frame)
            return [file_to_b64(fp) for fp, _ in frames]
        return [file_to_b64(src_rgb)]
    elif token == "src_bbox":
        return [pil_to_b64(create_bbox_image(src_rgb, src_mask))]
    elif token == "src_crop":
        return [pil_to_b64(create_crop_image(src_rgb, src_mask))]
    elif token == "dst":
        return [file_to_b64(dst_rgb)]
    else:
        raise ValueError(f"Unknown image token: {token!r}")


# ─── VLM ─────────────────────────────────────────────────────────────────────


def init_vlm(cfg):
    """Initialize the Ollama cloud VLM client."""
    keys = get_api_keys(cfg, "vlm-api-keys", "vlm-api-key", "Ollama")
    logger.info("Loaded %d Ollama API key(s).", len(keys))
    return {
        "keys": keys,
        "idx": 0,
        "client": Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {keys[0]}"},
        ),
    }


def init_local_vlm(base_url):
    """Initialize a local Ollama client (no auth required)."""
    logger.info("Initializing local Ollama client at %s", base_url)
    return {
        "keys": ["local"],
        "idx": 0,
        "client": Client(host=base_url),
    }


def _ollama_error_requires_key_rotation(err):
    txt = str(err).lower()
    return any(
        x in txt
        for x in (
            "401",
            "403",
            "429",
            "quota",
            "rate limit",
            "too many requests",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "invalid token",
        )
    )


def _rotate_vlm_key(vlm_state):
    """Switch to the next Ollama key and rebuild the client. Returns False if exhausted."""
    next_idx = vlm_state["idx"] + 1
    if next_idx >= len(vlm_state["keys"]):
        return False
    vlm_state["idx"] = next_idx
    new_key = vlm_state["keys"][next_idx]
    vlm_state["client"] = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {new_key}"},
    )
    return True


def vlm_caption(vlm_state, model, images_b64, prompt, num_predict=32):
    """Query the VLM with images + prompt. Returns the generated text."""
    msgs = [{"role": "user", "content": prompt, "images": images_b64}]

    while True:
        try:
            text = ""
            for part in vlm_state["client"].chat(
                model,
                messages=msgs,
                stream=True,
                options={
                    "temperature": 0,
                    "seed": 777,
                    "num_predict": num_predict,
                },
            ):
                text += part.message.content
            return text.strip()
        except Exception as e:
            if _ollama_error_requires_key_rotation(e):
                switched = _rotate_vlm_key(vlm_state)
                if switched:
                    logger.warning(
                        "Ollama request failed; switched to API key #%d/%d.",
                        vlm_state["idx"] + 1,
                        len(vlm_state["keys"]),
                    )
                    continue
                logger.error("Ollama request failed and no API keys remain: %s", e)
                return "ERROR"
            else:
                logger.error("VLM error: %s", e)
                return "ERROR"


# ─── SAM3 ────────────────────────────────────────────────────────────────────


def init_sam3(cfg):
    """Load the SAM3 model and processor onto the best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg["huggingface-model"]
    keys = get_api_keys(cfg, "huggingface-api-keys", "huggingface-api-key", "Hugging Face")
    logger.info("Loaded %d Hugging Face API key(s).", len(keys))

    last_error = None
    for idx, key in enumerate(keys):
        try:
            login(token=key)
            model = Sam3Model.from_pretrained(name).to(device)
            proc = Sam3Processor.from_pretrained(name)
            logger.info("SAM3 ready on %s (%s) using HF key #%d/%d", device, name, idx + 1, len(keys))
            return model, proc, device
        except Exception as e:
            last_error = e
            if idx < len(keys) - 1:
                logger.warning(
                    "Hugging Face init failed with key #%d/%d; trying next key. Error: %s",
                    idx + 1,
                    len(keys),
                    e,
                )
                continue
            break

    raise RuntimeError(
        "Failed to initialize SAM3 with all Hugging Face API keys."
    ) from last_error


_SAM3_MAX_TOKENS = 28  # SAM3 text encoder hard limit is 32 (incl. special tokens)

def _prepare_sam3_text(text: str, proc) -> str:
    """
    Extract the core description from a (possibly long CoT) VLM output and
    truncate to SAM3's token limit.

    Strategy:
      1. If the text contains a recognisable "final answer" marker, extract
         just that phrase.
      2. Otherwise take the last short (≤ 10 words) non-empty line.
      3. Tokenise with the SAM3 processor and hard-truncate to _SAM3_MAX_TOKENS.
    """
    import re as _re

    text = (text or "").strip()

    # 1. Look for explicit "final answer" / "answer:" marker
    m = _re.search(
        r"(?:final\s+answer|answer)[:\s]+([^\n]{3,120})",
        text, _re.IGNORECASE
    )
    if m:
        text = m.group(1).strip().strip('"').strip("'")
    else:
        # 2. Take the last non-empty line that looks like a short description
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            # Prefer a short final line (≤ 12 words), otherwise fall back to full text
            candidate = lines[-1]
            # Strip leading numbering like "4." or "(4)"
            candidate = _re.sub(r"^\s*[\d]+[.)]\s*", "", candidate).strip()
            if len(candidate.split()) <= 12:
                text = candidate

    # 3. Tokenise and hard-truncate
    token_ids = proc.tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) > _SAM3_MAX_TOKENS:
        text = proc.tokenizer.decode(token_ids[:_SAM3_MAX_TOKENS], skip_special_tokens=True)

    return text


def sam3_segment(model, proc, img_path, text_prompt, device):
    """Run text-prompted SAM3 segmentation and return post-processed results."""
    text_prompt = _prepare_sam3_text(text_prompt, proc)
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
    """Compute IoU, balanced accuracy, contour accuracy, and location error."""
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


def mask_centroid(mask_np):
    """Return (cy, cx) centroid in pixel space, or None for empty masks."""
    coords = np.argwhere(mask_np)
    if len(coords) == 0:
        return None
    cy, cx = coords.mean(axis=0)
    return float(cy), float(cx)


def _should_reprompt(
    src_mask_area,
    src_centroid,
    src_shape,
    cache_entry,
    growth_threshold,
    movement_threshold,
):
    """
    Decide whether to refresh VLM captions based on growth and movement.

    We reprompt on first sighting, or when mask area grows by at least
    `growth_threshold` relative to the best area seen so far, or when the
    object centroid shifts by at least `movement_threshold` in x or y.
    """
    if cache_entry is None:
        return True, "cold_start", 0.0, 0.0, 0.0

    best_area = cache_entry["best_src_mask_area"]
    growth = 0.0
    if best_area <= 0:
        growth_trigger = src_mask_area > 0
    else:
        growth = (src_mask_area - best_area) / best_area
        growth_trigger = growth >= growth_threshold

    dx_ratio, dy_ratio = 0.0, 0.0
    movement_trigger = False
    ref_centroid = cache_entry.get("ref_src_centroid")
    if src_centroid is not None and ref_centroid is not None:
        h, w = src_shape
        dy_ratio = abs(src_centroid[0] - ref_centroid[0]) / max(h, 1)
        dx_ratio = abs(src_centroid[1] - ref_centroid[1]) / max(w, 1)
        movement_trigger = dx_ratio >= movement_threshold or dy_ratio >= movement_threshold

    reprompt = growth_trigger or movement_trigger
    if growth_trigger and movement_trigger:
        reason = "growth+movement"
    elif growth_trigger:
        reason = "growth"
    elif movement_trigger:
        reason = "movement"
    else:
        reason = "reuse"
    return reprompt, reason, growth, dx_ratio, dy_ratio


def process_pair(meta, cfg, vlm, sam_m, sam_p, dev, ann, caption_cache, vlm_clients=None):
    """
    Run all config-defined experiments for one pair and return a list of result dicts.
    """
    src_cam = meta["src_camera"]
    dst_cam = meta["dst_camera"]
    src_rgb = resolve_img(meta["src_rgb_path"])
    dst_rgb = resolve_img(meta["dst_rgb_path"])
    obj = meta["object_name"]
    frame = meta["frame"]
    take_uid = meta["take_uid"]
    root_dir = cfg["root-data-directory"]

    src_mask = get_mask(ann, obj, src_cam, frame)
    dst_gt = get_mask(ann, obj, dst_cam, frame)
    if src_mask is None or dst_gt is None:
        logger.warning("Mask missing: %s / %s / %s – skipped", take_uid, obj, frame)
        return []

    spatial = spatial_covariates(dst_gt)
    src_mask_area = int(np.sum(src_mask))
    src_centroid = mask_centroid(src_mask)
    growth_threshold = float(
        cfg.get("vlm-reprompt-growth-threshold", DEFAULT_VLM_REPROMPT_GROWTH_THRESHOLD)
    )
    movement_threshold = float(
        cfg.get("vlm-reprompt-movement-threshold", DEFAULT_VLM_REPROMPT_MOVEMENT_THRESHOLD)
    )

    rows = []
    for exp_cfg in cfg["experiments"]:
        exp_id = exp_cfg["id"]
        exp_model = exp_cfg.get("vlm_model", cfg.get("vlm-model", ""))
        exp_predict = int(exp_cfg.get("num_predict", 32))
        exp_frames = int(exp_cfg.get("num_frames", 1))
        exp_tokens = exp_cfg["images"]
        exp_prompt_key = exp_cfg["prompt"]
        exp_prompt = cfg["prompts"].get(exp_prompt_key, exp_prompt_key)

        # Per-experiment cache key so each experiment's caption is tracked independently.
        cache_key = (take_uid, obj, src_cam, dst_cam, exp_id)
        cache_entry = caption_cache.get(cache_key)
        video_window = int(exp_cfg.get("video_window", 0))

        if video_window > 0:
            # Window-based reuse: call VLM once every `video_window` consecutive frames.
            # Pairs must be sorted by stream (take/object/cameras/frame) for this to be
            # meaningful — main() enforces this automatically when video_window > 0.
            frame_count = 0 if cache_entry is None else cache_entry.get("window_frame_count", video_window)
            reprompt = (cache_entry is None) or (frame_count >= video_window)
            reprompt_reason = "cold_start" if cache_entry is None else ("window_start" if reprompt else "window_reuse")
            growth, dx_ratio, dy_ratio = 0.0, 0.0, 0.0
        else:
            reprompt, reprompt_reason, growth, dx_ratio, dy_ratio = _should_reprompt(
                src_mask_area,
                src_centroid,
                src_mask.shape,
                cache_entry,
                growth_threshold,
                movement_threshold,
            )

        if reprompt:
            images_b64 = [
                img
                for token in exp_tokens
                for img in build_images_for_token(
                    token, src_rgb, src_mask, dst_rgb, ann, take_uid, obj, src_cam, root_dir, exp_frames, current_frame=frame
                )
            ]
            exp_base_url = exp_cfg.get("vlm_base_url")
            active_vlm = (
                vlm_clients[exp_base_url]
                if exp_base_url and vlm_clients and exp_base_url in vlm_clients
                else vlm
            )
            caption = vlm_caption(active_vlm, exp_model, images_b64, exp_prompt, num_predict=exp_predict)
            prev_best_area = 0 if cache_entry is None else cache_entry["best_src_mask_area"]
            caption_cache[cache_key] = {
                "best_src_mask_area": max(prev_best_area, src_mask_area),
                "ref_src_centroid": src_centroid,
                "caption": caption,
                **({"window_frame_count": 1} if video_window > 0 else {}),
            }
            logger.info(
                "VLM reprompted (%s) [%s]: %s | %s | frame %s | src_area=%d | growth=%.3f | dx=%.3f | dy=%.3f",
                reprompt_reason, exp_id, take_uid, obj, frame, src_mask_area, growth, dx_ratio, dy_ratio,
            )
        else:
            caption = cache_entry["caption"]
            if video_window > 0:
                cache_entry["window_frame_count"] = frame_count + 1

        sam_out = sam3_segment(sam_m, sam_p, dst_rgb, caption, dev)

        h_s, w_s = (
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
            "take_uid": take_uid,
            "object_name": obj,
            "src_camera": src_cam,
            "dest_camera": dst_cam,
            "frame": frame,
            "experiment": exp_id,
            "vlm_model": exp_model,
            "vlm_output": caption,
            "vlm_reprompted": reprompt,
            "vlm_reprompt_reason": reprompt_reason,
            "vlm_num_frames": exp_frames,
            "vlm_video_window": video_window,
            "src_mask_area": src_mask_area,
            "vlm_growth_threshold": growth_threshold,
            "vlm_movement_threshold": movement_threshold,
            "src_move_x_ratio": dx_ratio,
            "src_move_y_ratio": dy_ratio,
            **metrics,
            **spatial,
        })

    return rows


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    cfg = load_config(args.config)

    using_csv = "pairs-csv" in cfg
    exp_ids = [e["id"] for e in cfg["experiments"]]
    logger.info("Experiments: %s", exp_ids)

    pairs = load_pairs(cfg)
    if not using_csv:
        pairs = subsample_pairs(pairs, cfg["subset-run-percentage"], cfg["subset-seed"])

    if any(int(e.get("video_window", 0)) > 0 for e in cfg["experiments"]):
        pairs.sort(key=lambda m: (
            m["take_uid"], m["object_name"], m["src_camera"], m["dst_camera"], int(m["frame"])
        ))
        logger.info("Pairs sorted by stream for video-window experiments.")

    vlm = init_vlm(cfg)
    # Build per-base-url clients for any experiments using a local Ollama server.
    vlm_clients = {}
    for exp_cfg in cfg["experiments"]:
        base_url = exp_cfg.get("vlm_base_url")
        if base_url and base_url not in vlm_clients:
            vlm_clients[base_url] = init_local_vlm(base_url)
    sam_m, sam_p, dev = init_sam3(cfg)

    ann_cache = {}
    caption_cache = {}
    all_rows = []

    # Resume: scan ALL results_*.csv files in the working directory to find
    # pairs already completed across any previous or current job runs.
    done_pairs = set()
    checkpoint_path = f"results_{JOB_ID}.csv"
    existing_csvs = sorted(glob.glob("results_*.csv"))
    if existing_csvs:
        try:
            frames = []
            for p in existing_csvs:
                try:
                    frames.append(pd.read_csv(p, dtype=str))
                except Exception as e:
                    logger.warning("Skipping unreadable checkpoint %s: %s", p, e)
            if frames:
                done_df = pd.concat(frames, ignore_index=True).drop_duplicates()
                key_cols = ["take_uid", "object_name", "src_camera", "dest_camera", "frame"]
                if all(c in done_df.columns for c in key_cols + ["experiment"]):
                    done_pairs_exps = done_df.groupby(key_cols)["experiment"].apply(set).to_dict()
                    all_exp_ids = set(exp_ids)
                    for key, exps in done_pairs_exps.items():
                        if all_exp_ids <= exps:
                            done_pairs.add(key)
                    all_rows = done_df.to_dict("records")
                    logger.info(
                        "Resuming from %d file(s): %d pairs already done (%d rows loaded).",
                        len(existing_csvs), len(done_pairs), len(all_rows),
                    )
        except Exception as e:
            logger.warning("Could not load checkpoints for resume: %s", e)

    # dst_camera in meta corresponds to dest_camera in results CSV
    pairs = [
        m for m in pairs
        if (m["take_uid"], m["object_name"], m["src_camera"], m["dst_camera"], m["frame"])
        not in done_pairs
    ]
    logger.info("%d pairs remaining to process.", len(pairs))

    n = len(pairs)
    t0 = time.time()

    for i, meta in enumerate(pairs):
        uid = meta["take_uid"]

        logger.info(
            "[%d/%d] %s | %s | frame %s",
            i + 1, n, uid, meta["object_name"], meta["frame"],
        )

        if uid not in ann_cache:
            ann_cache[uid] = load_annotation(cfg["root-data-directory"], uid)

        try:
            all_rows.extend(
                process_pair(
                    meta,
                    cfg,
                    vlm,
                    sam_m,
                    sam_p,
                    dev,
                    ann_cache[uid],
                    caption_cache,
                    vlm_clients=vlm_clients,
                )
            )
        except Exception as e:
            logger.error("Pair %d failed: %s", i + 1, e)

        if (i + 1) % SAVE_INTERVAL == 0 and all_rows:
            pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            logger.info("Checkpoint saved → %s (%d rows). ETA %.0fs", checkpoint_path, len(all_rows), eta)

    if not all_rows:
        logger.warning("No results collected.")
        return

    df = pd.DataFrame(all_rows)

    # Final write — same file used for checkpointing (results_{JOB_ID}.csv)
    df.to_csv(checkpoint_path, index=False)
    logger.info("Final results → %s", checkpoint_path)

    summary = df.groupby("experiment")[["iou", "ba", "ca", "le"]].mean()
    summary_path = f"summary_{JOB_ID}.csv"
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
