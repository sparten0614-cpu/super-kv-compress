#!/bin/bash
# KV Cache Quantization Gradient — PPL + NIAH at each quant level
#
# Usage: ./run_quant_gradient.sh <model_path> <wiki_path> [ctx_size]
#
# Tests: f16, q8_0, q5_0, q5_1, q4_0, q4_1
# Metrics: PPL (perplexity) + NIAH (retrieval accuracy)

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> <wiki.test.raw> [ctx_size]}"
WIKI="${2:?Usage: $0 <model.gguf> <wiki.test.raw> [ctx_size]}"
CTX="${3:-4096}"

NGL=99
CHUNKS=5
if [ "$CTX" -ge 16384 ]; then CHUNKS=2; fi
if [ "$CTX" -ge 32768 ]; then CHUNKS=1; fi

PERPLEXITY="${LLAMA_PERPLEXITY:-$(which llama-perplexity 2>/dev/null || echo ./build/bin/llama-perplexity)}"
NIAH_SCRIPT="$(dirname "$0")/niah_test.py"
if [ ! -f "$NIAH_SCRIPT" ]; then NIAH_SCRIPT="scripts/niah_test.py"; fi

OUTDIR="results/quant_gradient/ctx${CTX}"
mkdir -p "$OUTDIR"

QUANT_TYPES="f16 q8_0 q5_0 q5_1 q4_0 q4_1"

echo "=========================================="
echo "  KV Cache Quantization Gradient"
echo "  Model: $MODEL"
echo "  Ctx: $CTX, Chunks: $CHUNKS"
echo "  Types: $QUANT_TYPES"
echo "=========================================="

# ==========================================
# Part 1: PPL at each quantization level
# ==========================================
echo ""
echo "========== PPL Gradient =========="

for QT in $QUANT_TYPES; do
    OUTFILE="$OUTDIR/ppl_${QT}.txt"

    if [ -f "$OUTFILE" ] && grep -q "Final estimate" "$OUTFILE"; then
        PPL=$(grep "Final estimate" "$OUTFILE" | grep -oP 'PPL = [\d.]+' | awk '{print $3}')
        echo "$QT: PPL=$PPL (cached)"
        continue
    fi

    echo "--- PPL: cache-type=$QT ---"

    CMD="$PERPLEXITY -m $MODEL -ngl $NGL -c $CTX --chunks $CHUNKS -fa on -f $WIKI"

    if [ "$QT" != "f16" ]; then
        CMD="$CMD --cache-type-k $QT --cache-type-v $QT"
    fi

    eval "$CMD" 2>&1 | tee "$OUTFILE"

    PPL=$(grep "Final estimate" "$OUTFILE" 2>/dev/null | grep -oP 'PPL = [\d.]+' | awk '{print $3}' || echo "N/A")
    echo "  → PPL = $PPL"
    echo ""
done

# ==========================================
# Part 2: NIAH at each quantization level
# ==========================================
echo ""
echo "========== NIAH Gradient =========="

for QT in $QUANT_TYPES; do
    OUTFILE="$OUTDIR/niah_${QT}.txt"

    if [ -f "$OUTFILE" ] && grep -q "Accuracy" "$OUTFILE"; then
        echo "$QT NIAH: $(grep 'Accuracy' "$OUTFILE" | tail -1) (cached)"
        continue
    fi

    echo "--- NIAH: cache-type=$QT ---"
    python3 "$NIAH_SCRIPT" --model "$MODEL" --ctx "$CTX" --cache-types "$QT" \
        --ngl "$NGL" --positions "0.1,0.25,0.5,0.75,0.9" 2>&1 | tee "$OUTFILE"
    echo ""
done

# ==========================================
# Summary Table
# ==========================================
echo ""
echo "=========================================="
echo "  Quantization Gradient Summary (ctx=$CTX)"
echo "=========================================="
echo ""

# Get baseline PPL
BASELINE_PPL=""
BASELINE_FILE="$OUTDIR/ppl_f16.txt"
if [ -f "$BASELINE_FILE" ]; then
    BASELINE_PPL=$(grep "Final estimate" "$BASELINE_FILE" 2>/dev/null | grep -oP 'PPL = [\d.]+' | awk '{print $3}')
fi

printf "%-8s | %-8s | %-10s | %-12s | %-10s\n" "Type" "Bits" "PPL" "PPL Δ%" "NIAH"
printf "%-8s-+-%-8s-+-%-10s-+-%-12s-+-%-10s\n" "--------" "--------" "----------" "------------" "----------"

for QT in $QUANT_TYPES; do
    # Bits per value
    case $QT in
        f16)  BITS="16.0" ;;
        q8_0) BITS="8.5" ;;
        q5_0) BITS="5.5" ;;
        q5_1) BITS="6.0" ;;
        q4_0) BITS="4.5" ;;
        q4_1) BITS="5.0" ;;
        *)    BITS="?" ;;
    esac

    # PPL
    PPL_FILE="$OUTDIR/ppl_${QT}.txt"
    PPL="N/A"
    DELTA="N/A"
    if [ -f "$PPL_FILE" ]; then
        PPL=$(grep "Final estimate" "$PPL_FILE" 2>/dev/null | grep -oP 'PPL = [\d.]+' | awk '{print $3}' || echo "N/A")
        if [ -n "$BASELINE_PPL" ] && [ "$PPL" != "N/A" ] && [ "$BASELINE_PPL" != "N/A" ]; then
            DELTA=$(python3 -c "print(f'+{($PPL-$BASELINE_PPL)/$BASELINE_PPL*100:.2f}%')" 2>/dev/null || echo "N/A")
            if [ "$QT" = "f16" ]; then DELTA="baseline"; fi
        fi
    fi

    # NIAH
    NIAH_FILE="$OUTDIR/niah_${QT}.txt"
    NIAH="N/A"
    if [ -f "$NIAH_FILE" ]; then
        NIAH=$(grep -oP '\d+%' "$NIAH_FILE" 2>/dev/null | tail -1 || echo "N/A")
    fi

    printf "%-8s | %-8s | %-10s | %-12s | %-10s\n" "$QT" "$BITS" "$PPL" "$DELTA" "$NIAH"
done

echo ""
echo "Compression ratios: q8_0=1.88x, q5_0=2.91x, q5_1=2.67x, q4_0=3.56x, q4_1=3.20x"
