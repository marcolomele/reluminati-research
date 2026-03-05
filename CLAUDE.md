# CLAUDE.md — Developer Context for Claude Code

This file provides persistent context for Claude Code sessions. Keep it updated after every session.

---

## Project Overview

VLM + SAM3 object correspondence pipeline. Given a source ego/exo frame with an annotated object, find and segment the same object in a destination frame from a different viewpoint (ego↔exo).

**Pipeline**: Source frame(s) → VLM (qwen3-vl) → text description or bbox → SAM3 → segmentation mask → IoU vs ground truth

---

## Repository Structure

```
reluminati-research/
├── README.md
├── CLAUDE.md                          ← this file
├── notebooks/
│   ├── baseline-overview.ipynb        ← dataset stats (Marco)
│   ├── experiment-analysis.ipynb      ← Marco's analysis notebook
│   ├── results_analysis.ipynb         ← generic/old analysis
│   ├── results_analysis_463119.ipynb  ← job 463119 analysis (gio)
│   └── VLM_SAM3_Experiments.ipynb     ← old experiments
└── src/scripts/
    ├── data/                          ← dataset processing scripts
    └── experiments/vlm-sam3/
        ├── experiment.py              ← MAIN PIPELINE (~1050 lines)
        ├── config.json                ← active config (skip-worktree, local)
        ├── config.box.json            ← box-only smoke config (skip-worktree)
        ├── config.multiframe.json     ← multiframe ablation config (skip-worktree)
        ├── config.cloud_only.json     ← cloud-only config (skip-worktree)
        ├── smoke.sbatch               ← main SLURM script (skip-worktree)
        ├── smoke_box.sbatch           ← box experiments SLURM script (skip-worktree)
        ├── smoke_multiframe.sbatch    ← multiframe SLURM script (skip-worktree)
        ├── create_multiframe_csv.py   ← generates baseline_multiframe_w10.csv
        └── export_to_gsheets.py       ← export results to Google Sheets
```

**Note**: All `config*.json` and `smoke*.sbatch` files are git skip-worktree (local-only, not pushed). They contain API keys.

---

## Data Paths

| Path | Description |
|------|-------------|
| `/data/video_datasets/3321908/output_dir_all/` | Dataset root: 180 takes, 11GB |
| `/data/video_datasets/3164542/baseline_multiframe_w10.csv` | Current baseline: 191 streams × 10 frames = 1910 rows |
| `/data/video_datasets/3164542/lm_eec_pairs_from_vlm460850.csv` | Old baseline (253 pairs) |
| `/data/video_datasets/3164542/results_partial.csv` | Partial results accumulation |

**Data layout**: `root/{take_uid}/{camera}/{object_name}/rgb/{frame}` (no extension — use `resolve_img()` which tries `.jpg`)

**Image sizes**: 448×448 (aria/egocentric), 796×448 (exo) — heavily downsampled from original EgoExo4D 1404×1404

**Baseline**: 50 balanced takes, 10 per scenario (basketball, bike repair, cooking, health, music)

---

## Baseline Comparison (LM-EEC)

| Direction | IoU |
|-----------|-----|
| ego→exo | 0.569 |
| exo→ego | 0.660 |

---

## HPC / SLURM

- **Partition**: `long_gpu` (14h limit) — use this, NOT `gpu`
- **Resources**: `--gres=gpu:1 --cpus-per-task=4 --mem=16G`
- **Submit**: `cd src/scripts/experiments/vlm-sam3 && sbatch smoke.sbatch`
- **Monitor**: `squeue -u 3164542`, `sacct -u 3164542 --format=JobID,JobName,State,Elapsed,ExitCode`
- **Logs**: `smoke_{JOBID}.out`, `smoke_{JOBID}.err` (in vlm-sam3 directory)
- **Output CSV**: `/data/video_datasets/3164542/results_{JOB_ID}.csv`

---

## Experiment Design

### Completed Jobs

| Job ID | Config | Status | Notes |
|--------|--------|--------|-------|
| 460850 | old/manual | COMPLETED | 253 pairs, old code |
| 462563 | config.json (10 exps) | COMPLETED | 1100 pairs, 11000 rows |
| 463119 | config.json (4 exps: C/cot/ctx/ctx-cot) | FAILED at 75% | Crash at .mean() due to resume dtype bug |
| 465254 | config.json (4 exps resume) | PENDING | Finishes remaining 477 pairs |
| 465270 | config.box.json (2 exps) | PENDING | EXP-C-box smoke test (19 pairs) |
| 465278 | config.multiframe.json (4 exps) | PENDING | EXP-F-mf1/3/5/10 smoke test (19 pairs) |

### Current Experiment IDs (config.json)

| ID | Model | Images | Prompt | Notes |
|----|-------|--------|--------|-------|
| EXP-C | 235b-cloud | src_clean, src_overlay, dst | clean-overlay-dst | 30.9% success, 25.5% missing |
| EXP-C-cot | 235b-instruct | src_clean, src_overlay, dst | clean-overlay-dst | **44.4% success**, 0% missing — BEST |
| EXP-C-ctx | 235b-cloud | src_clean, src_overlay, dst | neighbor-context | 17.6% — WORSE (SAM3 token truncation) |
| EXP-C-ctx-cot | 235b-instruct | src_clean, src_overlay, dst | neighbor-context-cot | 24.1% — worse than cot |
| EXP-C-box | 235b-cloud | src_clean, src_overlay, dst | localize-bbox | PENDING (new!) |
| EXP-C-box-cot | 235b-instruct | src_clean, src_overlay, dst | localize-bbox-cot | PENDING (new!) |

### Multiframe Ablation (config.multiframe.json)

| ID | num_frames | Model | Notes |
|----|-----------|-------|-------|
| EXP-F-mf1 | 1 | 235b-instruct | baseline |
| EXP-F-mf3 | 3 | 235b-instruct | |
| EXP-F-mf5 | 5 | 235b-instruct | fix applied |
| EXP-F-mf10 | 10 | 235b-instruct | |

---

## Key Findings So Far

1. **EXP-C-cot is best**: 44.4% success, 0% missing. Instruct model (512 tokens) beats cloud (64 tokens).
2. **When VLM works, SAM3 works**: IoU=0.718 mean on success cases, 83.4% IoU>0.5 — **beats LM-EEC baseline**!
3. **Main failure mode**: 56% of pairs fail to localize correctly (VLM outputs bad/empty description → SAM3 fails)
4. **num_predict matters**: 32 tokens → 20-25% missing VLM outputs; 512 tokens → 0% missing
5. **Neighbor context hurts**: -13pp vs EXP-C (more tokens → SAM3 truncation kills description quality)
6. **SAM3 hard 28-token limit**: max_position_embeddings=32, ~28 usable tokens. Long CoT outputs get truncated.
7. **EXP-C-box hypothesis**: VLM outputs x1,y1,x2,y2 → SAM3 box prompt (bypasses token limit entirely)
8. **Temporal fix**: multiframe sample_source_frames used linspace across whole video (wrong) → now picks N preceding frames before current_frame

---

## Critical Bugs Found and Fixed

### 1. Resume dtype contamination (job 463119 crash)
**Error**: `TypeError: Could not convert string '0.009...' to numeric` at `.mean()`
**Cause**: Old CSV loaded as `dtype=str`, new rows as float. Pandas can't `.mean()` mixed string/float.
**Fix**: Add before summary:
```python
for _c in ["iou", "ba", "ca", "le"]:
    if _c in df.columns:
        df[_c] = pd.to_numeric(df[_c], errors="coerce")
```

### 2. SAM3 token overflow (sequence length error)
**Error**: `Sequence length N exceeds model maximum (32)`
**Cause**: CoT outputs 100-200 tokens fed directly to SAM3 text encoder
**Fix**: `_prepare_sam3_text()` — extract final answer from CoT ("final answer:" marker), truncate to 28 tokens

### 3. sample_source_frames temporal bug
**Error**: linspace selected random frames across full video
**Cause**: `np.linspace(0, len(all_frame_ids)-1, n_frames, dtype=int)` ignores current_frame position
**Fix**: `current_frame` parameter — select N most recent annotated frames before current_frame

### 4. DOS line endings in sbatch
**Error**: `sbatch: error: Batch script contains DOS line breaks (\r\n)`
**Fix**: `sed -i 's/\r//' smoke_box.sbatch`

### 5. num_predict=32 missing outputs
**Error**: 20-25% of VLM outputs are empty/truncated (cloud model)
**Cause**: num_predict=32 too short for "color object" answers
**Fix**: num_predict=64 for cloud, 512 for instruct. But 64 still gives 25.5% missing! Use 128-256 for cloud.

---

## Key Architecture Decisions

### VLM as Localizer (EXP-C-box) — NEW
Instead of VLM → text description → SAM3 text prompt, use:
**VLM → bbox coordinates (x1,y1,x2,y2) → SAM3 box prompt**

This bypasses SAM3's 28-token limit entirely.

```python
# SAM3 with box prompt (no text encoder)
inputs = proc(
    images=img,
    input_boxes=[[[x1, y1, x2, y2]]],  # batch × n_boxes × 4
    return_tensors="pt",
)
```

Key functions: `parse_vlm_bbox()`, `sam3_segment_box()`
Routing: `exp_cfg.get("vlm_output_type", "text")` — set `"bbox"` for box experiments.

### Config-driven Experiments
All experiments defined in `config.json`. Add new experiment by adding entry to `"experiments": [...]`.
Key fields: `id`, `images`, `prompt`, `vlm_model`, `num_predict`, `num_frames` (optional), `vlm_output_type` (optional, "text"|"bbox")

### Resume Logic
Script saves to `results_{JOB_ID}.csv` every 50 pairs. On restart, loads previous CSV and skips already-processed (pair_id, experiment) tuples. **Warning**: old CSV may have different dtype — always apply `pd.to_numeric()` after loading.

---

## API Keys

API keys live **only** in local `config.json` (skip-worktree, never committed).
- Ollama keys: 3 keys, rotated via `vlm-api-keys` list in config
- HuggingFace key: `huggingface-api-keys` list in config

---

## Git Setup

- Branch: `gio/config-driven-experiments`
- Remote: `https://github.com/marcolomele/reluminati-research.git`
- Push requires PAT: `git push https://GioviManto:<PAT>@github.com/marcolomele/reluminati-research.git gio/config-driven-experiments`
- Config/sbatch files: `git update-index --skip-worktree <file>` (local-only, not pushed)

---

## TODO / Next Steps

- [ ] Analyze EXP-C-box results (job 465270) — does bbox approach beat 44.4%?
- [ ] Analyze EXP-F multiframe results (job 465278) — sweet spot for frame count?
- [ ] After smoke tests OK, run full experiment (set `subset-run-percentage: 1.0`)
- [ ] Fix EXP-C num_predict: 64→128 to reduce 25.5% missing
- [ ] Download high-res EgoExo4D images for local model testing
- [ ] Add results to Google Sheet (export_to_gsheets.py)
- [ ] Remove Filippo's API keys from any committed files
