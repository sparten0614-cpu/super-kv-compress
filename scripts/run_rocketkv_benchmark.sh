#!/bin/bash
# RocketKV Benchmark Script
# Runs NIAH + LongBench at multiple token budgets
#
# Usage: bash run_rocketkv_benchmark.sh [rocketkv_dir]
#
# Prerequisites: run setup_rocketkv.sh first

set -euo pipefail

ROCKET_DIR="${1:-RocketKV}"
RESULTS_DIR="results/rocketkv"
mkdir -p "$RESULTS_DIR"

if [ ! -d "$ROCKET_DIR" ]; then
    echo "ERROR: RocketKV directory not found at $ROCKET_DIR"
    echo "Run setup_rocketkv.sh first"
    exit 1
fi

cd "$ROCKET_DIR"

MODEL="llama3.1-8b-instruct"
BUDGETS="256 512 1024 2048 4096"

echo "=========================================="
echo "  RocketKV Benchmark Suite"
echo "  Model: $MODEL"
echo "  Budgets: $BUDGETS"
echo "=========================================="

# 1. Baseline (full KV, no compression)
echo ""
echo "=== NIAH Baseline ==="
if [ ! -f "../$RESULTS_DIR/niah_baseline.done" ]; then
    bash scripts/paulgraham_passkey/${MODEL}.sh baseline ../$RESULTS_DIR/niah_baseline 2>&1 | tee ../$RESULTS_DIR/niah_baseline.log
    touch ../$RESULTS_DIR/niah_baseline.done
    echo "NIAH baseline done"
else
    echo "NIAH baseline already done, skipping"
fi

echo ""
echo "=== LongBench Baseline ==="
if [ ! -f "../$RESULTS_DIR/longbench_baseline.done" ]; then
    bash scripts/longbench/${MODEL}.sh baseline ../$RESULTS_DIR/longbench_baseline 2>&1 | tee ../$RESULTS_DIR/longbench_baseline.log
    touch ../$RESULTS_DIR/longbench_baseline.done
    echo "LongBench baseline done"
else
    echo "LongBench baseline already done, skipping"
fi

# 2. RocketKV at each budget
for BUDGET in $BUDGETS; do
    echo ""
    echo "=== NIAH RocketKV budget=$BUDGET ==="
    if [ ! -f "../$RESULTS_DIR/niah_rocket_${BUDGET}.done" ]; then
        bash scripts/paulgraham_passkey/${MODEL}.sh rocket ../$RESULTS_DIR/niah_rocket_${BUDGET} ${BUDGET} 2>&1 | tee ../$RESULTS_DIR/niah_rocket_${BUDGET}.log
        touch ../$RESULTS_DIR/niah_rocket_${BUDGET}.done
        echo "NIAH budget=$BUDGET done"
    else
        echo "NIAH budget=$BUDGET already done, skipping"
    fi

    echo ""
    echo "=== LongBench RocketKV budget=$BUDGET ==="
    if [ ! -f "../$RESULTS_DIR/longbench_rocket_${BUDGET}.done" ]; then
        bash scripts/longbench/${MODEL}.sh rocket ../$RESULTS_DIR/longbench_rocket_${BUDGET} ${BUDGET} 2>&1 | tee ../$RESULTS_DIR/longbench_rocket_${BUDGET}.log
        touch ../$RESULTS_DIR/longbench_rocket_${BUDGET}.done
        echo "LongBench budget=$BUDGET done"
    else
        echo "LongBench budget=$BUDGET already done, skipping"
    fi
done

echo ""
echo "=========================================="
echo "  All benchmarks complete!"
echo "  Results in: $RESULTS_DIR/"
echo "=========================================="

# Summary
echo ""
echo "=== Quick Summary ==="
for BUDGET in baseline $BUDGETS; do
    NIAH_DIR="../$RESULTS_DIR/niah_${BUDGET}"
    [ "$BUDGET" != "baseline" ] && NIAH_DIR="../$RESULTS_DIR/niah_rocket_${BUDGET}"
    if [ -d "$NIAH_DIR" ]; then
        echo "NIAH budget=$BUDGET:"
        find "$NIAH_DIR" -name "*.json" -exec python3 -c "
import json, sys, glob
files = glob.glob('${NIAH_DIR}/**/*.json', recursive=True)
if files:
    total = correct = 0
    for f in files:
        try:
            d = json.load(open(f))
            if 'accuracy' in d: total += 1; correct += (1 if d['accuracy'] >= 0.5 else 0)
        except: pass
    if total: print(f'  {correct}/{total} = {correct/total*100:.1f}%')
" \; 2>/dev/null || echo "  (no results yet)"
    fi
done
