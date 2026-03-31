#!/bin/bash
# Run LongBench v2 across all paper configurations.
# Usage: bash scripts/run_longbench_all.sh [model_path] [binary_path]

MODEL="${1:-/root/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
BINARY="${2:-/root/llama.cpp/build/bin/llama-cli}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== LongBench v2 — All Configs ==="
echo "Model: $MODEL"
echo "Binary: $BINARY"
echo "Start: $(date)"
echo

# 1. q4_0 baseline (no eviction)
echo ">>> Config 1/4: q4_0 baseline"
python3 "$SCRIPT_DIR/run_longbench.py" \
    --model "$MODEL" --binary "$BINARY" \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --limit 20

echo
echo ">>> Config 2/4: q4_0 + 50% eviction"
python3 "$SCRIPT_DIR/run_longbench.py" \
    --model "$MODEL" --binary "$BINARY" \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --evict-pct 50 \
    --limit 20

echo
echo ">>> Config 3/4: q4_0 + 70% eviction"
python3 "$SCRIPT_DIR/run_longbench.py" \
    --model "$MODEL" --binary "$BINARY" \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --evict-pct 70 \
    --limit 20

echo
echo ">>> Config 4/4: q4_0 + 85% eviction"
python3 "$SCRIPT_DIR/run_longbench.py" \
    --model "$MODEL" --binary "$BINARY" \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --evict-pct 85 \
    --limit 20

echo
echo "=== All configs done: $(date) ==="
echo "Results in benchmarks/longbench/results/"
