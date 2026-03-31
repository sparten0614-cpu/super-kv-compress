# H2O Attention-Aware Eviction — Design Document

## 1. Problem Statement

StreamingLLM eviction removes KV cache entries by **position** (oldest first, keeping sink tokens).
This destroys retrieval for non-recent information:
- 85% eviction + NIAH = 20% accuracy (only 90% position found)
- High-attention "needle" tokens are evicted regardless of importance

**Goal:** Implement attention-aware eviction that keeps tokens with highest accumulated attention scores, enabling aggressive eviction (85%+) while preserving retrieval quality.

## 2. H2O Algorithm (Heavy-Hitter Oracle)

**Paper:** "H2O: Heavy-Hitter Oracle" (Zhang et al., 2023)

### Core Idea
- Track cumulative attention score for each KV cache token
- When cache is full, evict the token with the **lowest** accumulated score
- "Heavy hitters" = tokens that consistently receive high attention across layers/steps

### Algorithm
```
For each decoding step:
  1. Compute attention: attn_weights = softmax(Q @ K^T / sqrt(d))
  2. Update scores: score[i] += sum(attn_weights[:, i])  # sum across query positions
  3. If cache_size > budget:
     evict token j = argmin(score[j]) for j not in {sink_tokens}
```

### Key Properties
- Sink tokens (first few tokens) naturally accumulate high scores → protected
- Information-dense tokens (facts, names, numbers) get high attention → protected
- Filler tokens (repetitive text) get low attention → evicted first
- Adapts per-sequence (different content → different eviction pattern)

## 3. llama.cpp Architecture Analysis

### Attention Computation Paths

**Path A: Non-Flash (explicit matmul)**
```
Location: src/llama-graph.cpp:1848-1920
Flow: kq = mul_mat(k, q) → softmax → kqv = mul_mat(v, kq)
```
- Attention weights (`kq` after softmax, line 1892) are **explicitly available**
- Can extract per-token scores by summing columns of attention matrix
- Performance: slower than flash attention

**Path B: Flash Attention (fused kernel)**
```
Location: src/llama-graph.cpp:1807-1847
Flow: ggml_flash_attn_ext(q, k, v, mask, scale, ...) → output
```
- Attention weights are computed **inside the fused kernel** and never materialized
- Cannot extract scores without modifying the Metal/CUDA kernel
- Performance: faster (memory-efficient)

### Implementation Options

#### Option 1: Non-Flash Path Only (Simplest)
- Add score accumulation after softmax in the non-flash path
- Force `flash_attn=off` when H2O eviction is enabled
- **Pro:** Minimal code changes, can extract exact scores
- **Con:** Slower (non-flash attention)

#### Option 2: Modified Flash Attention Kernel
- Modify Metal/CUDA flash attention kernels to output per-column attention sums
- Add an auxiliary output tensor for score accumulation
- **Pro:** Full speed flash attention + H2O
- **Con:** Complex kernel modification, must modify multiple backends

#### Option 3: Approximate Scores (No Attention Extraction)
- Use query-key dot product magnitude as proxy for attention score
- For each cached KV token k_i: score[i] ≈ ||k_i|| or max(Q @ k_i)
- **Pro:** No attention modification needed, works with flash attn
- **Con:** Approximate, may miss softmax nonlinearity effects

#### Option 4: Hybrid — Use Recent Layer Scores
- Run one decoding step with non-flash to collect scores
- Use those scores for the next N evictions
- Switch back to flash attn for normal computation
- **Pro:** Good balance of accuracy and speed
- **Con:** Complex control flow

### Recommended Approach: Option 1 first, Option 2 later

1. Start with Option 1 (non-flash + exact scores) — proves the concept
2. If NIAH results are good, implement Option 2 for production speed
3. Option 3 as a quick fallback if kernel modification is too complex

## 4. Data Structures

### Score Tracker
```c
// Per-layer, per-sequence score accumulator
struct h2o_scores {
    float * scores;      // [n_kv_max] cumulative attention score per token
    int   * token_pos;   // [n_kv_max] original position of each cached token
    int     n_cached;    // number of tokens currently in cache
    int     budget;      // max tokens to keep
    int     n_sink;      // number of sink tokens (never evicted)
};
```

### Memory Overhead
- Per token: 4 bytes (float score) + 4 bytes (int pos) = 8 bytes
- For 128K context: 128K × 8 = 1MB — negligible vs KV cache (GBs)

## 5. Integration Points in llama.cpp

### Score Update (after each attention computation)
```
Location: src/llama-graph.cpp, after line 1892 (softmax)
Insert: h2o_update_scores(layer, kq_softmax)
```

### Eviction (before each decode batch)
```
Location: tools/completion/completion.cpp (our eviction code)
Replace: position-based eviction → score-based eviction
API: llama_memory_seq_rm(mem, seq, pos_start, pos_end)
Need: sort scores, find bottom-K tokens, evict them
```

### Challenge: Graph vs Imperative
- llama.cpp builds a **computation graph** at model load time
- Score accumulation needs to happen **during execution**, not at graph build time
- Options:
  a. Use `ggml_map_custom` to insert custom ops that read attention weights
  b. Use a callback mechanism (like the existing `cb()` callback)
  c. Post-process: after each `llama_decode()`, read the softmax output tensor

## 6. Implementation Plan

### Phase 1: Proof of Concept (non-flash, exact scores)
1. Add `--h2o-eviction` flag to llama-completion
2. Force non-flash attention when H2O is enabled
3. After each `llama_decode()`, extract attention weights from graph
4. Accumulate per-token scores
5. On eviction trigger, sort by score and evict lowest
6. Test with NIAH at 85% eviction → target 80%+ accuracy

### Phase 2: Production (flash attention compatible)
1. Modify Metal flash attention kernel to output column sums
2. Add auxiliary score output to `ggml_flash_attn_ext`
3. Benchmark speed impact
4. Verify NIAH accuracy matches Phase 1

### Phase 3: Optimization
1. Approximate methods (Option 3) for even lower overhead
2. Per-head vs aggregated scores experiment
3. Score decay (exponential moving average vs cumulative sum)
4. Interaction with TQKV quantization

## 7. Expected Results

Based on H2O paper:
- 85% eviction should maintain high retrieval quality (vs 20% with StreamingLLM)
- PPL degradation expected to be lower than StreamingLLM at same eviction ratio
- Combined: TQKV_6 (2.67x) × H2O 85% (6.67x) = 17.8x at hopefully <2% PPL

## 8. Open Questions

1. **Flash attention score extraction:** Is there a way to get partial scores from the fused kernel without full materialization?
2. **Multi-head aggregation:** Sum across heads, or per-head eviction?
3. **Score decay:** Cumulative vs EMA? Old tokens accumulate more score just from existing longer.
4. **Layer variation:** Do all layers agree on which tokens are important? Or per-layer eviction?
5. **Interaction with TQKV:** Does quantization noise affect score accuracy?
