#!/bin/bash
# StreamingLLM Eviction Gradient Experiment
# Run on cloud GPU (Vast.ai / RunPod / etc.)
#
# Usage: ./run_eviction_gradient.sh <model_path> <wiki_path> [ctx_size]
#
# Prerequisites:
#   - llama.cpp built with eviction support (apply patches/eviction-completion.patch)
#   - llama-perplexity binary in PATH or ./build/bin/

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> <wiki.test.raw> [ctx_size]}"
WIKI="${2:?Usage: $0 <model.gguf> <wiki.test.raw> [ctx_size]}"
CTX="${3:-4096}"

# Auto-detect binary
PERPLEXITY="${LLAMA_PERPLEXITY:-$(which llama-perplexity 2>/dev/null || echo ./build/bin/llama-perplexity)}"

# Chunks: more for smaller ctx, fewer for larger (memory constraint)
if [ "$CTX" -le 4096 ]; then
    CHUNKS=5
elif [ "$CTX" -le 16384 ]; then
    CHUNKS=3
elif [ "$CTX" -le 32768 ]; then
    CHUNKS=2
else
    CHUNKS=1
fi

SINK=128
NGL=99  # Cloud GPU should have enough VRAM
OUTDIR="results/eviction/ctx${CTX}"
mkdir -p "$OUTDIR"

echo "=== StreamingLLM Eviction Gradient ==="
echo "Model: $MODEL"
echo "Wiki:  $WIKI"
echo "Ctx:   $CTX, Chunks: $CHUNKS, Sink: $SINK"
echo "Output: $OUTDIR"
echo ""

# Eviction ratios to test
RATIOS="0.00 0.10 0.30 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90"

for RATIO in $RATIOS; do
    echo "--- Eviction ratio: $RATIO ---"
    OUTFILE="$OUTDIR/evict_${RATIO}.txt"

    if [ -f "$OUTFILE" ] && grep -q "Final estimate" "$OUTFILE"; then
        echo "  Already done, skipping ($(grep 'Final estimate' "$OUTFILE"))"
        continue
    fi

    "$PERPLEXITY" \
        -m "$MODEL" \
        -ngl "$NGL" \
        -c "$CTX" \
        --chunks "$CHUNKS" \
        -fa on \
        -f "$WIKI" \
        --evict-mode 1 \
        --evict-sink "$SINK" \
        --evict-ratio "$RATIO" \
        2>&1 | tee "$OUTFILE"

    # Extract PPL
    PPL=$(grep "Final estimate" "$OUTFILE" | grep -oP 'PPL = [\d.]+' | awk '{print $3}')
    echo "  PPL = $PPL"
    echo ""
done

# Summary
echo ""
echo "=== Summary ==="
echo "| Eviction | PPL | File |"
echo "|----------|-----|------|"
for RATIO in $RATIOS; do
    OUTFILE="$OUTDIR/evict_${RATIO}.txt"
    if [ -f "$OUTFILE" ]; then
        PPL=$(grep "Final estimate" "$OUTFILE" 2>/dev/null | grep -oP 'PPL = [\d.]+' | awk '{print $3}' || echo "N/A")
        echo "| $RATIO | $PPL | $OUTFILE |"
    fi
done
