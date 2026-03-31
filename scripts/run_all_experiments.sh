#!/bin/bash
# Master experiment script for cloud GPU
# Runs all eviction gradient + NIAH experiments
#
# Usage: ./run_all_experiments.sh <model_path> <wiki_path>
#
# Setup on cloud GPU:
#   1. Clone this repo
#   2. Clone llama.cpp and apply patch:
#      git clone https://github.com/ggml-org/llama.cpp
#      cd llama.cpp
#      git apply ../llama_cpp/patches/eviction-completion.patch
#      cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)
#   3. Download model + wiki data
#   4. Run this script

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> <wiki.test.raw>}"
WIKI="${2:?Usage: $0 <model.gguf> <wiki.test.raw>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Super KV Compress — Full Experiment Suite"
echo "=========================================="
echo ""

# 1. PPL Eviction Gradients at multiple context sizes
for CTX in 4096 16384 32768 65536 131072; do
    echo ""
    echo "=== PPL Gradient: ctx=$CTX ==="
    bash "$SCRIPT_DIR/run_eviction_gradient.sh" "$MODEL" "$WIKI" "$CTX" || {
        echo "  FAILED or OOM at ctx=$CTX, skipping"
        continue
    }
done

# 2. NIAH at multiple context sizes + eviction ratios
for CTX in 4096 16384 32768; do
    echo ""
    echo "=== NIAH: ctx=$CTX ==="
    bash "$SCRIPT_DIR/run_niah_eviction.sh" "$MODEL" "$CTX" || {
        echo "  FAILED at ctx=$CTX, skipping"
        continue
    }
done

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "  Results in: results/"
echo "=========================================="
