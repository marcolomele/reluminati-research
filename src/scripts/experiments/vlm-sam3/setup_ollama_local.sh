#!/bin/bash
# Run this script ONCE from the LOGIN NODE to install Ollama and pull the
# local VLM models needed for the size-ablation experiments (8b, 32b).
#
# Models are stored in OLLAMA_MODELS so they persist across jobs.
# The binary is installed to ~/.local/bin/ollama (no root required).
#
# Usage: bash setup_ollama_local.sh

set -eo pipefail

OLLAMA_MODELS_DIR="/data/video_datasets/3164542/ollama_models"
OLLAMA_BIN="$HOME/.local/bin/ollama"

# ── 1. Install Ollama binary ─────────────────────────────────────────────────
if [ -x "$OLLAMA_BIN" ]; then
    echo "[OK] Ollama already installed at $OLLAMA_BIN"
else
    echo "Downloading Ollama binary ..."
    mkdir -p "$HOME/.local/bin"
    OLLAMA_VERSION=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
    TARBALL_URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
    TMP_TAR=$(mktemp /tmp/ollama-XXXX.tar.zst)
    curl -fL "$TARBALL_URL" -o "$TMP_TAR"
    # Extract just the ollama binary (it lives at bin/ollama inside the archive)
    tar --zstd -xf "$TMP_TAR" -C "$HOME/.local" bin/ollama
    rm -f "$TMP_TAR"
    echo "[OK] Ollama ${OLLAMA_VERSION} installed at $OLLAMA_BIN"
fi

# ── 2. Create model storage dir ──────────────────────────────────────────────
mkdir -p "$OLLAMA_MODELS_DIR"
echo "Model storage: $OLLAMA_MODELS_DIR"

# ── 3. Start a temporary Ollama server on port 11435 to avoid conflict with
#       any shared server already running on the default port 11434 ───────────
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="http://localhost:11435"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
sleep 5
echo "Ollama server running on :11435 (PID $OLLAMA_PID)"

# ── 4. Pull models (OLLAMA_HOST directs pull to OUR server on :11435) ────────
echo "Pulling qwen3-vl:8b  (~6 GB) ..."
OLLAMA_HOST="http://localhost:11435" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" "$OLLAMA_BIN" pull qwen3-vl:8b

echo "Pulling qwen3-vl:32b (~21 GB) ..."
OLLAMA_HOST="http://localhost:11435" OLLAMA_MODELS="$OLLAMA_MODELS_DIR" "$OLLAMA_BIN" pull qwen3-vl:32b

# ── 5. Stop temporary server ─────────────────────────────────────────────────
kill "$OLLAMA_PID" 2>/dev/null || true
unset OLLAMA_HOST
echo ""
echo "===== Setup complete ====="
echo "Models stored in: $OLLAMA_MODELS_DIR"
echo "You can now submit the SLURM job: sbatch smoke.sbatch"
echo ""
echo "NOTE: The 32b model needs ~21 GB VRAM. If your GPU has only 16 GB, the"
echo "      EXP-A-32b and EXP-C-32b experiments will fail with OOM. In that"
echo "      case, request a larger GPU partition (e.g. gpunew) or remove those"
echo "      experiments from config.json."
