"""
Shared utilities for baseline analysis scripts.

Provides scenario classification, baseline UID extraction, annotation loading,
and mask decoding helpers used across all analyze_baseline_*.py scripts.
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

SCENARIO_KEYWORDS = {
    "basketball": [
        "basketball", "drills", "layup", "mikan", "jump shooting",
        "mid-range", "reverse layup",
    ],
    "bike repair": [
        "wheel", "tire", "flat", "tube", "chain", "lubricate",
        "derailleur", "derailueur", "bike",
    ],
    "cooking": [
        "cooking", "making", "omelet", "scrambled", "eggs", "noodles",
        "pasta", "salad", "greek", "cucumber", "tomato", "sesame",
        "ginger", "milk tea", "chai tea", "chai", "coffee", "latte",
    ],
    "music": [
        "piano", "violin", "guitar", "suzuki", "playing", "scales",
        "arpeggios", "freeplaying",
    ],
    "health": [
        "first aid", "cpr", "covid", "antigen", "rapid", "test",
    ],
    "dance": ["dance", "dancing"],
    "soccer": ["soccer", "football"],
    "rock climbing": ["rock climbing", "climbing"],
}

RELATIONS_FILES = [
    "relations_test.json",
    "relations_train.json",
    "relations_val.json",
]


def classify_scenario(title: str) -> str | None:
    """Map a scenario title to one of the 8 canonical categories, or None."""
    title_lower = title.lower()
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                return scenario
    return None


def load_baseline_uids(root: Path) -> set[str]:
    """Extract unique UIDs from baseline_egoexo_pairs.json."""
    pairs = json.loads((root / "baseline_egoexo_pairs.json").read_text())
    return {p[0].split("//")[1] for p in pairs}


def load_baseline_pairs(root: Path) -> list[list[str]]:
    """Load all pairs from baseline_egoexo_pairs.json."""
    return json.loads((root / "baseline_egoexo_pairs.json").read_text())


def load_all_annotations(root: Path) -> dict:
    """Load and merge annotations from all relations_*.json files."""
    merged = {}
    for fname in RELATIONS_FILES:
        path = root / fname
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        data = json.loads(path.read_text())
        for uid, ann in data.get("annotations", {}).items():
            merged[uid] = ann
    return merged


def build_uid_scenario_map(annotations: dict) -> dict[str, str]:
    """Map each UID to its canonical scenario category."""
    uid_scenario = {}
    for uid, ann in annotations.items():
        cat = classify_scenario(ann.get("scenario", ""))
        if cat:
            uid_scenario[uid] = cat
    return uid_scenario


# ---------------------------------------------------------------------------
# Mask decoding helpers (require lzstring + pycocotools on the remote machine)
# ---------------------------------------------------------------------------

_MASK_LIBS_AVAILABLE = False
_LZString = None
_mask_utils = None


def _ensure_mask_libs():
    global _MASK_LIBS_AVAILABLE, _LZString, _mask_utils
    if _MASK_LIBS_AVAILABLE:
        return
    try:
        from lzstring import LZString as _LZ
        from pycocotools import mask as _mu
        _LZString = _LZ
        _mask_utils = _mu
        _MASK_LIBS_AVAILABLE = True
    except ImportError:
        print("ERROR: lzstring and pycocotools are required for mask decoding.")
        print("Install with: pip install lzstring pycocotools")
        sys.exit(1)


def decode_rle(annotation_obj: dict) -> dict:
    """Decompress encodedMask into a COCO RLE dict (without full decode)."""
    _ensure_mask_libs()
    encoded_mask = annotation_obj["encodedMask"]
    decomp = _LZString.decompressFromEncodedURIComponent(encoded_mask)
    return {
        "size": [annotation_obj["height"], annotation_obj["width"]],
        "counts": decomp.encode().decode("ascii"),
    }


def mask_area_from_annotation(annotation_obj: dict) -> int:
    """Compute mask area directly from RLE (fast, no full decode)."""
    _ensure_mask_libs()
    rle = decode_rle(annotation_obj)
    return int(_mask_utils.area(rle))


def decode_mask_full(annotation_obj: dict) -> np.ndarray:
    """Decode to full binary mask array (height, width)."""
    _ensure_mask_libs()
    rle = decode_rle(annotation_obj)
    return _mask_utils.decode(rle)


def mask_centroid(binary_mask: np.ndarray) -> tuple[float, float]:
    """Compute centroid (cx, cy) of a binary mask. Returns image-center if empty."""
    ys, xs = np.where(binary_mask)
    if len(ys) > 0:
        return float(xs.mean()), float(ys.mean())
    h, w = binary_mask.shape
    return w / 2.0, h / 2.0
