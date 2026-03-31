# Quick Start: Find Optimal KV Compression in 5 Minutes

Get the best KV cache compression configuration for any model using our GQA-aware framework.

## Prerequisites

```bash
# llama.cpp (latest main branch, with flash attention quant support)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j

# WikiText-2 test set (for PPL evaluation)
wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/wiki.test.raw

# Our toolkit
git clone https://github.com/sparten0614-cpu/super-kv-compress
cd super-kv-compress
```

## Step 1: Identify Your Model's GQA Configuration (30 seconds)

```bash
# Check model architecture
llama.cpp/build/bin/llama-cli -m your-model.gguf --verbose 2>&1 | grep -E "n_head|n_head_kv"
```

Look for `n_head` (query heads) and `n_head_kv` (KV heads). Compute GQA ratio = n_head / n_head_kv.

| Model Family | n_head_kv | GQA Ratio | K Minimum |
|-------------|-----------|-----------|-----------|
| Llama 3.x 8B | 8 | 4:1 | q4_0 safe |
| Mistral 7B | 8 | 4:1 | q4_0 safe |
| Qwen 2.5 7B | 4 | 7:1 | **q8_0 required** |
| Llama 3.x 70B | 8 | 8:1 | q4_0 safe |

**Rule of thumb:** If n_head_kv ≤ 4, use K=q8_0. If n_head_kv ≥ 8, K=q4_0 is safe.

## Step 2: Run Quick Diagnostic (2 minutes)

Test the three most informative configs:

```bash
PPLBIN=llama.cpp/build/bin/llama-perplexity
MODEL=your-model.gguf
WIKI=wiki.test.raw

# Baseline (f16 KV cache)
$PPLBIN -m $MODEL -f $WIKI -c 4096 --chunks 5 -ngl 99 -fa on

# Conservative: K=q8_0, V=q4_0 (works on ALL models)
$PPLBIN -m $MODEL -f $WIKI -c 4096 --chunks 5 -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q4_0

# Aggressive: K=q4_0, V=q4_0 (only if GQA ratio ≤ 4)
$PPLBIN -m $MODEL -f $WIKI -c 4096 --chunks 5 -ngl 99 -fa on \
  --cache-type-k q4_0 --cache-type-v q4_0
```

## Step 3: Interpret Results (30 seconds)

| PPL Δ% | Verdict | Action |
|--------|---------|--------|
| < 0.5% | Lossless | Use this config in production |
| 0.5% - 1% | Near-lossless | Safe for most use cases |
| 1% - 5% | Moderate | OK for batch inference, not retrieval-critical |
| > 5% | Too aggressive | Back off to higher K precision |
| > 100% | Catastrophic | GQA-incompatible — increase K precision |

**If q4_0 shows > 10% degradation:**
Your model likely has high GQA ratio or outlier layers. Use K=q8_0, V=q4_0 instead.

## Step 4: NIAH Verification (2 minutes)

PPL alone is not enough. Verify retrieval works:

```bash
python3 scripts/niah_test.py --model $MODEL --ctx 4096 --ngl 99 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --positions 0.1,0.5,0.9
```

Expected output: all positions should show `FOUND`. If any show `MISS`, the config is too aggressive for retrieval tasks.

## Step 5: Deploy

Use the optimal config in your inference server:

```bash
# llama.cpp server
llama.cpp/build/bin/llama-server -m $MODEL -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q4_0

# Or for GQA ≤ 4 models (more aggressive)
llama.cpp/build/bin/llama-server -m $MODEL -ngl 99 -fa on \
  --cache-type-k q4_0 --cache-type-v q4_0
```

## Quick Reference: Recommended Configs

| Model GQA | Config | Compression | Expected PPL Δ | NIAH |
|-----------|--------|-------------|-----------------|------|
| Any | K=q8_0, V=q8_0 | 2x | ~0% | 100% |
| Any | K=q8_0, V=q4_0 | 2.5x | < 1.5% | 100% |
| GQA ≤ 4 | K=q4_0, V=q4_0 | 4x | ~3% | 100% |
| GQA ≤ 4 | TQKV_6 | 2.67x | < 0.2% | 100% |
| GQA ≥ 7 | K=q8_0, V=q4_0 | 2.5x | < 1.5% | 100% |
| GQA ≥ 7 | K=q4_0, V=q4_0 | 4x | **UNSAFE** | **0%** |

## Automated Search (Optional)

For thorough optimization, use the AutoResearch pipeline:

```bash
# Full automated search (3 phases, ~4 hours)
python3 scripts/runner.py --model $MODEL --wiki $WIKI --ctx 4096

# Quick grid search only (~1 hour)
python3 scripts/runner.py --model $MODEL --wiki $WIKI --only-phase 1

# View Pareto frontier
cat results/autoresearch/pareto_final.csv
```

## Common Pitfalls

1. **Don't use q4_0 on Qwen/high-GQA models without checking first.** PPL can go from 5 to 6000+.
2. **Don't trust PPL alone for eviction configs.** PPL can improve while retrieval breaks.
3. **Build with `-DGGML_CUDA_FA_ALL_QUANTS=ON`** for mixed K/V types. Without it, flash attention only supports matched types.
4. **Longer context = more eviction tolerance.** Test eviction at your target context length, not just 4K.
5. **V is always safe at q4_0.** Focus your precision budget on K.
