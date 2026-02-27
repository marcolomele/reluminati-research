#!/bin/bash
#SBATCH -J vlm_sam3_smoke
#SBATCH -p long_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 14:00:00
#SBATCH -o smoke_%j.out
#SBATCH -e smoke_%j.err

set -eo pipefail

source ~/.bashrc
conda activate vlm-sam3
export HUGGINGFACE_HUB_TOKEN=HUGGINGFACE_TOKEN_HERE

echo "===== ENV READY ====="
which python
python --version

# ── Local Ollama server (for 8b / 32b experiments) ──────────────────────────
export OLLAMA_MODELS=/data/video_datasets/3164542/ollama_models
export OLLAMA_HOST=http://localhost:11435
OLLAMA_BIN="$HOME/.local/bin/ollama"
if [ -x "$OLLAMA_BIN" ]; then
    echo "===== STARTING LOCAL OLLAMA (port 11435) ====="
    "$OLLAMA_BIN" serve &
    OLLAMA_PID=$!
    sleep 8   # wait for server to be ready
    echo "Ollama PID: $OLLAMA_PID"
else
    echo "WARNING: Ollama not found at $OLLAMA_BIN – local model experiments will fail."
    echo "Run setup_ollama_local.sh from the login node first."
    OLLAMA_PID=""
fi

echo "===== STARTING SCRIPT ====="

export PYTHONUNBUFFERED=1
python -u experiment.py --config config.json

echo "===== SCRIPT FINISHED ====="

# ── Shut down local Ollama ───────────────────────────────────────────────────
if [ -n "$OLLAMA_PID" ]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
    echo "Ollama server stopped."
fi
