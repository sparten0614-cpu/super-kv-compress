# Attention Prediction: Can We Solve the Causal Limitation of Eviction?

**Author:** 宁宁
**Date:** 2026-04-01
**Status:** Literature survey + feasibility analysis

---

## 1. The Problem

KV cache eviction suffers a **causal limitation**: eviction decisions are made at prefill time (past), but the tokens that matter depend on future queries (unknown). This is why H2O and StreamingLLM both fail NIAH at 50%+ eviction — the needle receives low attention during prefill but is critical for the future retrieval query.

**Core question:** Can we predict which tokens will receive high attention from future queries, using only information available at prefill time?

If yes → eviction becomes NIAH-safe.
If no → eviction is fundamentally limited, and we must use other approaches (quantization, selective recompute).

---

## 2. Taxonomy of Approaches

The literature reveals **four distinct strategies** for approximating future attention:

### Strategy A: Query-Aware (cheat — query available at eviction time)
Methods that use the actual query/instruction visible at the end of the prompt.

| Method | Mechanism | NIAH | Compression | Limitation |
|--------|-----------|------|-------------|------------|
| **SnapKV** (2024) | Attention from prompt tail as observation window | Yes | 8x mem | Requires query in prompt; fails for pre-cached contexts |
| **Quest** (ICML 2024) | Current decode query → top-K KV pages via min/max keys | Partial | 2-7x latency | Per-step decode only; coarse page granularity |
| **RetrievalAttention** (MS, 2024) | ANNS vector search over keys using decode query | Yes | 97-99% offloaded | CPU-GPU transfer overhead; decode-only |
| **HashEvict** (2024) | LSH cosine similarity to current query | Yes | 30-70% | Decode-only |

**Verdict:** These "solve" the problem by assuming it away. Not applicable when the query arrives after prefill (RAG, multi-turn, pre-caching).

### Strategy B: Learned Prediction (train a predictor)
Methods that train a model to predict future attention importance.

| Method | Mechanism | NIAH | Training Cost | Key Finding |
|--------|-----------|------|---------------|-------------|
| **LookaheadKV** (2026) | Learnable "lookahead tokens" + LoRA produce pseudo-query vectors approximating future attention | Yes (RULER) | LoRA SFT | Most direct solution; explicitly generates pseudo-future-queries at prefill |
| **ForesightKV** (2026) | Distills oracle future attention into lightweight scorer via SL + RL (GRPO) | Not tested | SL + RL | Targets reasoning models (long CoT) |
| **KVP** (2026) | Per-head RL agents learn scoring from oracle traces | Partial (RULER) | RL per head | "Recovers non-local structure" but failure cases exist |
| **DMC** (2024) | End-to-end learned merge gates during continued pre-training | Not reported | Continued pre-training | Merge > evict; 4x lossless |
| **Attention-Gate** (2024) | Fine-tuned binary keep/drop per token per head | Not reported | SFT/CPT | |
| **Write-Gate KV** (Dec 2025) | MLP gate at write time, trained via knowledge distillation | Near-lossless at 10% | KD | Uses pre- and post-RoPE keys |
| **TRIM-KV** (Dec 2025) | Per-layer retention gate with temporal decay | Not reported | KD from frozen LM | Gates "naturally recover" sink + sliding window |

**Best in class: LookaheadKV** — directly addresses the causal limitation by learning pseudo-future-queries. Up to 14.5x eviction cost reduction, RULER tested at 4K-32K.

### Strategy C: Statistical Modeling (no training, predict query distribution)
Model future queries statistically and compute expected attention.

| Method | Mechanism | NIAH | Compression | Key Insight |
|--------|-----------|------|-------------|-------------|
| **Expected Attention** (Oct 2025) | Model future Q as Gaussian → closed-form expected attention score $\hat{z}_i = \exp(\bar{\mu}_q^T k_i / \sqrt{d} + k_i^T \bar{\Sigma}_q k_i / 2d)$ | Yes (125K) | 2x@50%, 12x@decode | Gaussian assumption on hidden states → tractable |
| **OracleKV** (ICML 2025 Oral) | Inject "guidance prompt" encoding population-level query intent → attention from guidance → token scores | Yes | Plug-and-play gains | No training; query-intent prior from data statistics |

**Best in class: Expected Attention** — mathematically principled, training-free, NIAH-safe to 125K. The Gaussian query model is elegant.

### Strategy D: Information-Theoretic (query-agnostic, reconstruction-based)
Sidestep future attention entirely — retain tokens that are informationally irreplaceable.

| Method | Mechanism | NIAH | Compression |
|--------|-----------|------|-------------|
| **KVzip** (NeurIPS 2025 Oral) | Score tokens by reconstruction difficulty from remaining cache | Yes | 3-4x |

**Insight:** Any needle that can't be reconstructed from surrounding context will be retained regardless of query. Elegant but limited compression.

---

## 3. Theoretical Limits of Attention Prediction

### 3.1 TAPPA Framework (2025)

Attention predictability is governed by **query self-similarity** — how similar consecutive query vectors are:

- **High q-similarity** (factual QA, retrieval, NIAH): attention is **highly predictable** from past patterns. Re-access patterns (vertical lines in attention maps) emerge when a dominant low-frequency RoPE channel exists.
- **Low q-similarity** (multi-hop reasoning, variable tracking): attention is **fundamentally unpredictable** from local context.

**Practical rule:** q-similarity can be computed per-head and used as an allocation signal — heads with low q-similarity get larger cache budgets.

### 3.2 Information-Theoretic Bound (Jan 2026)

"On the Limits of Learned Importance Scoring" provides the most pessimistic result:

| Signal Source | Mutual Information with Future Attention |
|--------------|----------------------------------------|
| Position | 0.28-0.31 bits |
| Prefill attention | 0.28-0.31 bits |
| Key vectors alone | **0.12 bits** |
| Learned scorer (1.7M params) | No improvement over heuristics |

**Key finding:** A trained 1.7M-parameter scorer showed **no statistically significant improvement** over simple position + prefill-attention baselines at any retention level.

**Circular dependence:** Future attention → depends on future queries → depends on generation trajectory → depends on cache retention. This creates a fundamental information-theoretic obstacle.

### 3.3 Sparse Frontier (2025)

Attention predictability is **task-dependent**:
- NIAH, single-turn QA: attention is sparse, localized, tolerates 95% sparsity
- Variable tracking, multi-hop: attention scatters across distant positions, degrades at 50% sparsity

A fixed compression budget is always wrong.

---

## 4. Feasibility Assessment for Our Framework

### Can attention prediction solve our causal limitation?

**Answer: Partially, but not to 30-50x.**

| Approach | Max Realistic Compression | NIAH-Safe? | Training Required? | Applicable to Us? |
|----------|--------------------------|------------|--------------------|--------------------|
| Expected Attention | 2x (50%), 12x (decode-time) | Yes | No | **Yes — complement to quantization** |
| LookaheadKV | ~14x eviction cost reduction | Yes | Yes (LoRA) | **No — requires training, post-training constraint** |
| OracleKV guidance | Additive gains | Yes | No | **Yes — can boost existing eviction** |
| KVzip reconstruction | 3-4x | Yes | No | **Maybe — complementary to quant** |

### Key Insight: Expected Attention + Quantization

The most promising combination for our post-training constraint:

1. **Quantize all KV to 6-bit** (NIAH-safe for GQA ≥ 8, our Theorem 2): **2.67x**
2. **Expected Attention eviction at decode time** — use Gaussian query model to identify low-importance tokens and skip them: **additional 2-4x**
3. **Combined: 5-10x with NIAH preservation**

This is better than blind eviction but still below our 30-50x target.

### Why Pure Prediction Can't Reach 30-50x

The information-theoretic results are clear:
- Key vectors carry only **0.12 bits** of predictive information about future attention
- Position + prefill attention carry **0.28-0.31 bits**
- Even optimal prediction can't bridge the gap to 30-50x eviction while preserving NIAH
- The circular dependence (future attention ↔ generation ↔ cache) is fundamental

**Conclusion:** Attention prediction is a **useful ingredient** (especially Expected Attention as a training-free complement to quantization) but **cannot be the primary mechanism** for 30-50x compression.

---

## 5. Implications for Our Approach

### What This Survey Confirms

1. **Selective Recompute remains the most promising path to 30-50x.** No prediction-based eviction method achieves >12x while preserving NIAH.

2. **Expected Attention is a valuable addition** to our quantization tier — training-free, mathematically principled, and NIAH-safe. Can be layered on top of quantization for an additional 2-4x.

3. **GQA ratio is a critical variable** (confirmed by both Qwen failure and theoretical analysis — fewer KV heads = higher per-head information density = more sensitive to both quantization and eviction).

4. **Per-head adaptive budgets** are the right approach — TAPPA's q-similarity metric can identify which heads to compress aggressively vs. preserve.

### Recommended Integration Points

| Layer | Current | Enhancement |
|-------|---------|-------------|
| L1: Quantization | Uniform 6-bit | **GQA-aware**: n_kv≤4 → 6-8bit; n_kv≥8 → 4-6bit |
| L2: Eviction | Attention-score based | **Expected Attention scoring**: Gaussian query model for NIAH-safe eviction |
| L3: Recovery | Selective recompute | Unchanged — still the path to 30x+ |
| New: Head allocation | Uniform budget | **TAPPA q-similarity**: per-head adaptive budgets |

### Novel Contribution Opportunity

**No existing work combines:**
1. GQA-aware quantization thresholds (our empirical finding)
2. Expected Attention for NIAH-safe eviction (theoretical, training-free)
3. Selective recompute for recovery beyond information-theoretic limits

This three-layer stack is genuinely novel.

---

## 6. References

### Category A: Query-Aware
- SnapKV (2024): arXiv 2404.14469
- Quest (ICML 2024): arXiv 2406.10774
- RetrievalAttention (2024): arXiv 2409.10516
- HashEvict (2024): arXiv 2412.16187

### Category B: Learned Prediction
- LookaheadKV (2026): arXiv 2603.10899
- ForesightKV (2026): arXiv 2602.03203
- KVP (2026): arXiv 2602.10238
- DMC (2024): arXiv 2403.09636
- Attention-Gate (2024): arXiv 2410.12876
- Write-Gate KV (Dec 2025): arXiv 2512.17452
- TRIM-KV (Dec 2025): arXiv 2512.03324

### Category C: Statistical Modeling
- Expected Attention (Oct 2025): arXiv 2510.00636
- OracleKV (ICML 2025 Oral): OpenReview KHM2YOGgX9

### Category D: Information-Theoretic
- KVzip (NeurIPS 2025 Oral): arXiv 2505.23416
- SAGE-KV (2025): arXiv 2503.08879

### Theoretical
- TAPPA (2025): arXiv 2601.21709
- Sparse Frontier (2025): arXiv 2504.17768
- Limits of Learned Importance Scoring (Jan 2026): arXiv 2601.14279
- SpargeAttention (ICML 2025): arXiv 2502.18137
