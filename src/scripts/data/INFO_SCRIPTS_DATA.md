## Purpose

Miscellaneous scripts for data processing, feature extraction, and pipeline utilities.

## Scripts

### Data Processing

- **`process_data.py`** – Loads EgoExo4D videos, extracts annotated frames, and organizes them for training. Supports all scenarios and train/val/test splits.

- **`create_pairs.py`** – Builds egoexo and exoego pair lists from `split.json` and per-take `annotation.json`. Writes `{split}_{egoexo|exoego}_pairs.json` (e.g. `train_egoexo_pairs.json`). Requires `--data_dir`.

- **`sample_takes.py`** – Samples a fraction of takes from an existing `splits.json` at `--root`. Returns a JSON with sampled UIDs per split. Use `--train_pct`, `--val_pct`, `--test_pct`, `--seed`.

- **`sample_annotated_takes.py`** – Samples takes that have relation annotations from `relations_*.json`. Extracts UIDs from annotations, groups by split, samples by percentage, and saves to JSON.

### Baseline

- **`generate_baseline_pairs.py`** – Samples 10 takes per scenario from `{test,train,val}_egoexo_pairs.json`, maps UIDs to scenarios via `relations_*.json` and keyword matching, and writes `baseline_egoexo_pairs.json` and `baseline_exoego_pairs.json` (mirror). Run from repo root.

- **`verify_baseline_pairs.py`** – Verifies baseline pair files: entry counts, mirror property, unique UIDs, takes/pairs per scenario, and spot-checks. Run from repo root.

### Feature Extraction

- **`load_relation_masks.py`** – Helper to load and decode relation masks from EgoExo4D annotations. Demonstrates loading relation JSON, navigating the structure, and decoding encoded masks to binary numpy arrays. Usage: `python load_relation_masks.py <annotation_path> [take_uid] [object_name] [camera_name]`.

### Utilities

TBD
