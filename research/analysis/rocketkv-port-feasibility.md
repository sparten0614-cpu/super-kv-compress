# RocketKV → llama.cpp Port Feasibility Analysis

## Overview

RocketKV (NVIDIA, ICML 2025) is a two-stage KV cache compression method:
- Stage 1: SnapKV-based permanent eviction (prefill time)
- Stage 2: Hybrid Sparse Attention / HSA (per decode step)

Claims: up to 400x token selection ratio, 3.7x decode speedup, 32.6% peak memory reduction.
Code: https://github.com/NVlabs/RocketKV (Python/PyTorch, flash-attn 2.6.3)

## Stage 1: SnapKV Eviction — Feasible (Medium Effort)

### What It Does
At end of prefill, use attention scores from an "observation window" (last 32 tokens)
to permanently evict low-importance prompt tokens from KV cache.

### Porting Assessment

**Feasibility: HIGH** — This is essentially our H2O work with a different scoring strategy.

Key differences from H2O:
1. **Prefill-only** (not per-decode-step) → simpler, no ongoing overhead
2. **Observation window** (last 32 tokens) instead of cumulative scoring
3. **Pooling** with kernel_size=63 to keep token clusters (not individual tokens)

### Implementation in llama.cpp

```
After prefill (all prompt tokens decoded):
  1. Extract attention weights for last 32 tokens (observation window)
  2. Sum attention across observation window queries → importance score per KV token
  3. Apply 1D avg pooling (kernel=63) to smooth scores
  4. Keep top-K tokens + always keep sink tokens
  5. Remove rest via llama_memory_seq_rm()
```

**Changes needed:**
- Same eval callback mechanism as H2O Phase 1 (already implemented)
- New scoring: window-based instead of cumulative
- Pooling layer (simple 1D convolution, ~20 lines)
- Trigger: after prefill complete, before first generation token

**Work estimate: 1-2 days** (leveraging existing H2O infrastructure)

## Stage 2: HSA (Hybrid Sparse Attention) — Hard (High Effort)

### What It Does
For each decode step, instead of attending to all remaining KV tokens:
1. **Page indexing:** Group KV tokens into pages, store per-page max/min along head dimension
2. **2D approximate scoring:** For query q:
   - Find k1 largest absolute values in q along head dimension
   - Use only those k1 dimensions to compute approximate page scores
   - Select top-k2 pages/tokens
3. **Sparse attention:** Exact attention only over selected tokens

### Porting Assessment

**Feasibility: LOW** — Requires fundamental changes to the attention computation path.

### Barriers

**Barrier 1: Page metadata management**
- Need to maintain per-page max(K) and min(K) tensors
- These update every time a new token enters the KV cache
- Additional storage: `2 * n_pages * k1 * sizeof(float)` per layer
- In ggml: new tensors in kv_layer, new update ops in the graph

**Barrier 2: Custom sparse attention kernel**
- Standard ggml attention: `Q @ K^T` → softmax → `@ V` (dense matmul)
- HSA needs: `Q @ K_subset^T` where K_subset is non-contiguous
- Options:
  a. Gather selected K/V rows into contiguous buffer → dense attention (works but memory copy overhead)
  b. Custom sparse matmul kernel (Metal + CUDA) — complex

**Barrier 3: Two-pass query analysis**
- First pass: find k1 largest dims in Q (top-k on d_head dimensions)
- Second pass: approximate scoring using those dims only
- In ggml graph: need 2 sequential ops before the actual attention
- This changes the attention graph structure fundamentally

**Barrier 4: Dynamic token selection**
- Each decode step selects different token subsets
- The selection depends on the current query (data-dependent control flow)
- ggml's static graph model doesn't naturally support data-dependent branching
- Would need custom ops or a multi-graph approach

**Barrier 5: Backend compatibility**
- Need Metal, CUDA, and CPU implementations of:
  - Page index construction
  - 2D approximate scoring
  - Sparse gather + attention
- ~500-1000 lines per backend

### Work estimate: 3-6 weeks (one engineer, full-time)

## Recommendation: Port Stage 1 Only

### Why
- Stage 1 (SnapKV eviction) gives the permanent memory savings
- Stage 2 (HSA) gives decode speed but NOT additional memory savings
- For our goal (memory compression), Stage 1 is sufficient
- Stage 1 leverages existing H2O infrastructure, Stage 2 requires new kernels

### Combined Strategy

```
Phase 1: TQKV quantization (already done)     → 2.67-3.56x memory
Phase 2: SnapKV Stage 1 eviction (1-2 days)   → additional 2-5x memory
Phase 3: Multi-tier decay (design done)        → additional 1.3-1.5x
Combined: 7-27x memory compression
```

### Alternative: Use RocketKV PyTorch Directly

For benchmarking and paper comparison:
- Run RocketKV's PyTorch implementation as-is on cloud GPU
- Compare against our llama.cpp quantization + eviction
- No porting needed — already have benchmark scripts pushed

## Summary

| Component | Effort | Value for Memory | Value for Speed | Recommend |
|-----------|--------|-----------------|-----------------|-----------|
| Stage 1 (SnapKV eviction) | 1-2 days | HIGH | None | ✅ Port |
| Stage 2 (HSA) | 3-6 weeks | None | HIGH | ❌ Skip |
| Full RocketKV port | 4-8 weeks | HIGH | HIGH | ❌ Too expensive |
| PyTorch benchmark | 1 day | N/A (comparison only) | N/A | ✅ Already done |
