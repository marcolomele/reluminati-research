#!/bin/bash
#SBATCH -J vlm_sam3_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 05:00:00
#SBATCH -o smoke_%j.out
#SBATCH -e smoke_%j.err

set -eo pipefail

source ~/.bashrc
conda activate vlm-sam3
export HUGGINGFACE_HUB_TOKEN=HUGGINGFACE_TOKEN_HERE

echo "===== ENV READY ====="
which python
python --version

echo "===== STARTING SCRIPT ====="

export PYTHONUNBUFFERED=1
python -u experiment.py --config config.json

echo "===== SCRIPT FINISHED ====="
