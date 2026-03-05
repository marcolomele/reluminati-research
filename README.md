# reluminati-research

Research on **object correspondence across ego/exo viewpoints** using VLM + SAM3.

**Task**: Given a source frame (egocentric or exocentric) with an annotated object, find and segment the same object in a destination frame from the opposite viewpoint.

---

## Pipeline

```
Source frame(s) ──► VLM (qwen3-vl) ──► description or bbox ──► SAM3 ──► mask ──► IoU
```

Two modes:
- **Text mode** (default): VLM describes the object → SAM3 text-grounded segmentation
- **BBox mode** (EXP-C-box): VLM localizes object as `x1,y1,x2,y2` → SAM3 box prompt (bypasses SAM3's 28-token text limit)

---

## Setup

```bash
conda activate vlm-sam3
cd src/scripts/experiments/vlm-sam3
```

Config files are local-only (skip-worktree). Copy from a template and fill in API keys:
```bash
# Edit API keys in config.json
cp config.json config.local.json
```

---

## Running Experiments

```bash
# Submit to SLURM
sbatch smoke.sbatch               # main experiments (config.json)
sbatch smoke_box.sbatch           # box localization experiments (config.box.json)
sbatch smoke_multiframe.sbatch    # multiframe ablation (config.multiframe.json)

# Monitor
squeue -u $USER
tail -f smoke_<JOBID>.out
```

Results are saved to `/data/video_datasets/3164542/results_{JOB_ID}.csv`.

---

## Experiment Results Summary

| Experiment | Model | Success Rate | Notes |
|-----------|-------|-------------|-------|
| EXP-C | qwen3-vl:235b-cloud | 30.9% | 25.5% missing VLM outputs |
| EXP-C-cot | qwen3-vl:235b-instruct | **44.4%** | 0% missing — **BEST** |
| EXP-C-ctx | qwen3-vl:235b-cloud | 17.6% | Neighbor context hurts (SAM3 token truncation) |
| EXP-C-ctx-cot | qwen3-vl:235b-instruct | 24.1% | Worse than plain cot |
| EXP-C-box | qwen3-vl:235b-cloud | TBD | VLM→bbox→SAM3 box prompt |
| EXP-C-box-cot | qwen3-vl:235b-instruct | TBD | CoT variant |
| EXP-F-mf1/3/5/10 | qwen3-vl:235b-instruct | TBD | Multiframe temporal ablation |

**LM-EEC baseline**: ego→exo IoU=0.569, exo→ego IoU=0.660

**Key insight**: When VLM correctly describes the object, SAM3 achieves IoU=0.718 mean (83.4% have IoU>0.5) — **better than LM-EEC baseline**. The bottleneck is VLM localization accuracy.

---

## Key Config Fields

```json
{
  "experiments": [{
    "id": "EXP-X",
    "images": ["src_clean", "src_overlay", "dst"],
    "prompt": "clean-overlay-dst",
    "vlm_model": "qwen3-vl:235b-cloud",
    "num_predict": 64,
    "num_frames": 1,
    "vlm_output_type": "text"
  }]
}
```

Image tokens: `src_overlay`, `src_clean`, `src_bbox`, `src_crop`, `dst`

`vlm_output_type`: `"text"` (default) or `"bbox"` (box localization mode)

---

## Data

- **Dataset**: EgoExo4D subset, `/data/video_datasets/3321908/output_dir_all/` (180 takes, 11GB)
- **Baseline CSV**: `/data/video_datasets/3164542/baseline_multiframe_w10.csv` (1910 rows: 50 takes × ~38 frames)
- **Image quality**: 448×448 (egocentric aria), 796×448 (exo) — downsampled from original 1404×1404

---

## Known Issues / Gotchas

1. **SAM3 28-token limit**: text encoder has `max_position_embeddings=32`. Long descriptions crash. Use `_prepare_sam3_text()` which truncates to 28 tokens.
2. **Resume dtype bug**: loading old CSVs mixes `str` and `float` types. Always apply `pd.to_numeric()` before `.mean()`.
3. **VLM missing outputs**: `num_predict=32` → 20-25% empty responses. Use 64+ for cloud, 512 for instruct.
4. **Config files are local-only**: use `git update-index --skip-worktree` to exclude from git tracking.
5. **Neighbor context prompts**: longer prompts → SAM3 truncation kills description quality → worse performance.

---

## Notebooks

| Notebook | Description |
|---------|-------------|
| `notebooks/results_analysis_463119.ipynb` | Job 463119 analysis (2×2 factorial: cloud/instruct × plain/context) |
| `notebooks/experiment-analysis.ipynb` | Marco's multi-experiment analysis |
| `notebooks/baseline-overview.ipynb` | Dataset statistics |

---

## Team

- Gio (Giovanni Mantovani): `gio/config-driven-experiments` branch
- Marco: main branch integration, analysis notebooks
- Filippo: local model experiments, multiframe bug catch
