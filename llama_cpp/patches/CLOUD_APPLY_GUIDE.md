# Cloud Patch Apply Guide

## Current Cloud State (assumed)
- eviction-completion.patch ✅
- eviction-perplexity.patch ✅
- H2O delta (manual) ✅

## What's Missing
- SnapKV eviction
- Expected Attention Phase 2
- --kv-type-map
- K-quant types (Q2_K-Q6_K)

## Recommended: Apply all-eviction-h2o.patch from scratch

**Cleanest approach:** Start from vanilla llama.cpp and apply the all-in-one patch:

```bash
cd llama.cpp
git checkout -- .                    # reset to clean state
git apply ../super-kv-compress/llama_cpp/patches/all-eviction-h2o.patch
cmake --build build -j$(nproc)
```

This single patch includes EVERYTHING:
- Perplexity eviction (StreamingLLM)
- Completion eviction (StreamingLLM)
- H2O attention-aware eviction
- SnapKV one-shot prefill eviction
- Expected Attention Phase 2 (pure K-based scoring)
- --kv-type-map per-layer precision
- K-quant types (Q2_K-Q6_K)
- Debug logging

## Test Commands After Apply

### Expected Attention + NIAH (THE KEY EXPERIMENT)
```bash
# 85% eviction — will Expected Attention beat StreamingLLM's 20% NIAH?
python3 ../super-kv-compress/scripts/niah_test.py \
    --model /path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --ctx 16384 --ngl 99 \
    --cache-types f16 \
    --positions 0.1,0.25,0.5,0.75,0.9 \
    --evict-mode 1 --evict-ratio 0.85 --evict-sink 128 \
    --expected-attn-eviction

# 50% eviction — compare with StreamingLLM's 60% NIAH
python3 ../super-kv-compress/scripts/niah_test.py \
    --model /path/to/model.gguf \
    --ctx 16384 --ngl 99 \
    --cache-types f16 \
    --positions 0.1,0.25,0.5,0.75,0.9 \
    --evict-mode 1 --evict-ratio 0.50 --evict-sink 128 \
    --expected-attn-eviction
```

### SnapKV + NIAH
```bash
python3 ../super-kv-compress/scripts/niah_test.py \
    --model /path/to/model.gguf \
    --ctx 16384 --ngl 99 \
    --cache-types f16 \
    --positions 0.1,0.25,0.5,0.75,0.9 \
    --evict-mode 1 --evict-ratio 0.85 --evict-sink 128 \
    --snapkv-eviction
```

### Full Pareto gradient (StreamingLLM vs Expected Attention)
```bash
bash ../super-kv-compress/scripts/run_niah_eviction.sh /path/to/model.gguf 16384
```

### Verify flags work
```bash
./build/bin/llama-completion --help | grep -E "h2o|snapkv|expected|kv-type-map|evict"
```

Expected output:
```
--evict-ratio N
--evict-mode N
--evict-sink N
--h2o-eviction
--snapkv-eviction
--expected-attn-eviction
--kv-type-map MAP
```

## Important Notes
- Expected Attention and SnapKV both auto-enable H2O scoring (eval callback)
- This forces non-flash attention during prefill (needed for score extraction)
- After one-shot eviction, flash attention is NOT re-enabled (future optimization)
- Expected Attention stores K vectors in memory (~8MB for 16K ctx) — acceptable
