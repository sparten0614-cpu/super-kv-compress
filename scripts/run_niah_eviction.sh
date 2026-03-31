#!/bin/bash
# NIAH + Eviction Experiment (StreamingLLM + H2O comparison)
# Runs full Pareto curve: eviction ratio vs NIAH accuracy
#
# Usage: ./run_niah_eviction.sh <model_path> [ctx_size]
#
# Prerequisites:
#   - llama-completion with eviction + H2O support
#     (apply patches/h2o-eviction-phase1.patch which includes eviction-completion.patch)
#   - niah_test.py in same directory or scripts/

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [ctx_size]}"
CTX="${2:-16384}"

NGL=99
SINK=128
OUTDIR="results/niah/ctx${CTX}"
mkdir -p "$OUTDIR"

echo "=== NIAH + Eviction Pareto Curve ==="
echo "Model: $MODEL"
echo "Ctx:   $CTX"
echo ""

# Find niah_test.py
NIAH_SCRIPT="$(dirname "$0")/niah_test.py"
if [ ! -f "$NIAH_SCRIPT" ]; then
    NIAH_SCRIPT="scripts/niah_test.py"
fi

# Eviction ratios for full gradient
RATIOS="0.00 0.30 0.50 0.60 0.70 0.80 0.85 0.90"

# ==========================================
# Part 1: StreamingLLM (position-based) NIAH
# ==========================================
echo "========== StreamingLLM (position-based) =========="
for RATIO in $RATIOS; do
    OUTFILE="$OUTDIR/niah_streamingllm_${RATIO}.txt"

    if [ -f "$OUTFILE" ] && grep -q "Accuracy" "$OUTFILE"; then
        echo "StreamingLLM $RATIO: already done, skipping"
        grep "Accuracy" "$OUTFILE" | tail -1
        continue
    fi

    echo "--- StreamingLLM eviction=$RATIO ---"
    CMD="python3 $NIAH_SCRIPT --model $MODEL --ctx $CTX --cache-types f16 --ngl $NGL --positions 0.1,0.25,0.5,0.75,0.9"

    if [ "$RATIO" != "0.00" ]; then
        CMD="$CMD --evict-mode 1 --evict-ratio $RATIO --evict-sink $SINK"
    fi

    eval "$CMD" 2>&1 | tee "$OUTFILE"
    echo ""
done

# ==========================================
# Part 2: H2O (attention-aware) NIAH
# ==========================================
echo ""
echo "========== H2O (attention-aware) =========="

# Skip 0.00 (baseline is same for both methods)
H2O_RATIOS="0.30 0.50 0.60 0.70 0.80 0.85 0.90"

for RATIO in $H2O_RATIOS; do
    OUTFILE="$OUTDIR/niah_h2o_${RATIO}.txt"

    if [ -f "$OUTFILE" ] && grep -q "Accuracy" "$OUTFILE"; then
        echo "H2O $RATIO: already done, skipping"
        grep "Accuracy" "$OUTFILE" | tail -1
        continue
    fi

    echo "--- H2O eviction=$RATIO ---"
    # H2O uses --h2o-eviction flag (forces non-flash attention)
    CMD="python3 $NIAH_SCRIPT --model $MODEL --ctx $CTX --cache-types f16 --ngl $NGL --positions 0.1,0.25,0.5,0.75,0.9"
    CMD="$CMD --evict-mode 1 --evict-ratio $RATIO --evict-sink $SINK --h2o-eviction"

    eval "$CMD" 2>&1 | tee "$OUTFILE"
    echo ""
done

# ==========================================
# Summary: Pareto Curve
# ==========================================
echo ""
echo "=========================================="
echo "  NIAH Pareto Curve Summary (ctx=$CTX)"
echo "=========================================="
echo ""
printf "%-10s | %-15s | %-15s\n" "Eviction" "StreamingLLM" "H2O"
printf "%-10s-+-%-15s-+-%-15s\n" "----------" "---------------" "---------------"

for RATIO in $RATIOS; do
    SL_FILE="$OUTDIR/niah_streamingllm_${RATIO}.txt"
    H2O_FILE="$OUTDIR/niah_h2o_${RATIO}.txt"

    SL_ACC="N/A"
    H2O_ACC="N/A"

    if [ -f "$SL_FILE" ]; then
        SL_ACC=$(grep -oP '\d+%' "$SL_FILE" 2>/dev/null | tail -1 || echo "N/A")
    fi
    if [ "$RATIO" = "0.00" ]; then
        H2O_ACC="(same)"
    elif [ -f "$H2O_FILE" ]; then
        H2O_ACC=$(grep -oP '\d+%' "$H2O_FILE" 2>/dev/null | tail -1 || echo "N/A")
    fi

    printf "%-10s | %-15s | %-15s\n" "$RATIO" "$SL_ACC" "$H2O_ACC"
done
