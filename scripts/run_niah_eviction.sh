#!/bin/bash
# NIAH + Eviction Experiment
# Tests needle retrieval at various eviction ratios
#
# Usage: ./run_niah_eviction.sh <model_path> [ctx_size]
#
# Prerequisites:
#   - llama-completion with eviction support (apply patches/eviction-completion.patch)
#   - niah_test.py in same directory or scripts/

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [ctx_size]}"
CTX="${2:-16384}"

NGL=99
SINK=128
OUTDIR="results/niah/ctx${CTX}"
mkdir -p "$OUTDIR"

echo "=== NIAH + Eviction Experiment ==="
echo "Model: $MODEL"
echo "Ctx:   $CTX"
echo ""

# Eviction ratios to test
RATIOS="0.00 0.30 0.50 0.70 0.85"

# Find niah_test.py
NIAH_SCRIPT="$(dirname "$0")/niah_test.py"
if [ ! -f "$NIAH_SCRIPT" ]; then
    NIAH_SCRIPT="scripts/niah_test.py"
fi

for RATIO in $RATIOS; do
    echo "--- NIAH at eviction ratio: $RATIO ---"
    OUTFILE="$OUTDIR/niah_evict_${RATIO}.txt"

    if [ -f "$OUTFILE" ] && grep -q "Accuracy" "$OUTFILE"; then
        echo "  Already done, skipping"
        cat "$OUTFILE" | grep -E "Accuracy|FOUND|MISS"
        continue
    fi

    CMD="python3 $NIAH_SCRIPT --model $MODEL --ctx $CTX --cache-types f16 --ngl $NGL --positions 0.1,0.25,0.5,0.75,0.9"

    if [ "$RATIO" != "0.00" ]; then
        CMD="$CMD --evict-mode 1 --evict-ratio $RATIO --evict-sink $SINK"
    fi

    echo "  Running: $CMD"
    eval "$CMD" 2>&1 | tee "$OUTFILE"
    echo ""
done

# Summary
echo ""
echo "=== NIAH Summary ==="
for RATIO in $RATIOS; do
    OUTFILE="$OUTDIR/niah_evict_${RATIO}.txt"
    if [ -f "$OUTFILE" ]; then
        ACCURACY=$(grep "Accuracy" "$OUTFILE" 2>/dev/null | tail -1 || echo "N/A")
        echo "Eviction $RATIO: $ACCURACY"
    fi
done
