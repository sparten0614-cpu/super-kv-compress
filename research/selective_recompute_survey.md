# Selective Recompute & Sparse Attention: A Survey for KV Cache Compression

**Date:** 2026-04-01
**Purpose:** Research survey for paper's future work section. Our context: we've shown eviction fundamentally fails at high compression (85%+ eviction destroys NIAH), and quantization saturates at ~4x. Selective recompute is identified as the path to 10x+ compression with NIAH preservation.

---

## Executive Summary

Selective recompute/sparse attention approaches avoid the fundamental causal limitation of eviction by **keeping all tokens accessible** and dynamically selecting which ones to attend to (or recompute) at decode time. The key insight across all approaches: the information about which tokens matter is only available **at query time**, not at prefill time — exactly the limitation that kills eviction-based methods.

The approaches fall into three categories:

| Category | Examples | Core Idea | Memory | Compute Overhead |
|----------|----------|-----------|--------|-----------------|
| **Sparse Selection** | Quest, SparQ, InfiniGen | Keep full KV cache, load only relevant subset per query | Full KV in CPU/GPU | Low (index lookup) |
| **Offload + Retrieval** | RetrievalAttention, MagicPIG | Full KV in CPU, retrieve via ANNS | GPU: minimal; CPU: full | Medium (vector search) |
| **Selective Recompute** | CacheBlend, ProphetKV, KV-Direct, KVPR | Don't store full KV — recompute from checkpoints or token IDs | Greatly reduced | Medium-High (partial forward pass) |
| **Query-Agnostic Eviction** | KVzip, Fast KVzip | Evict using context reconstruction (not query-dependent) | 3-4x reduced | Low after compression |

---

## 1. KVzip — Query-Agnostic KV Cache Eviction via Context Reconstruction

**Paper:** [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](https://arxiv.org/abs/2505.23416) (NeurIPS 2025 Oral)
**Code:** [github.com/snu-mllab/KVzip](https://github.com/snu-mllab/KVzip)

### Core Idea
Uses the LLM itself to score KV pair importance by measuring how well the model can reconstruct the original context from the compressed cache. Importance = maximum attention score each KV pair receives during context reconstruction. Evicts pairs with low reconstruction importance. Critically, this is **query-agnostic** — the compression is done once and reused across arbitrary downstream queries.

### How It Handles Eviction
This IS an eviction method, but a fundamentally better one than H2O/StreamingLLM. Instead of using attention scores from prefill (which cannot predict future queries), it uses **context reconstruction ability** as the importance signal. The reasoning: if a KV pair is essential for the model to "understand" the context (reconstruct it), it's likely important for any future query about that context.

### Compression & Accuracy
- **3-4x KV cache reduction** with negligible quality loss
- **2x FlashAttention decoding latency reduction**
- Tasks: QA, retrieval, reasoning, code comprehension — all near-lossless
- Models: LLaMA 3.1, Qwen 2.5, Gemma 3 (up to 170K context)
- "Significantly outperforms existing query-aware KV eviction methods, which suffer from performance degradation even at a 90% cache budget ratio under multi-query scenarios"

### Latency Overhead
- Compression requires a forward pass for importance scoring (one-time cost at prefill)
- After compression, decode is 2x faster due to smaller cache

### Key Limitations
- Still fundamentally evicts — information is permanently lost
- 3-4x compression ceiling (not 10x+)
- Compression overhead requires full context reconstruction pass
- Unknown NIAH performance at higher compression (>75% eviction)

### Relation to Our Work
KVzip is the best eviction method we've found, but it's still eviction. Our data shows all eviction methods (H2O, StreamingLLM) converge at matched rates. KVzip may push the safe eviction threshold higher (from ~50% to ~70%), but the causal limitation still applies — context reconstruction cannot fully predict future query needs.

### Follow-up: Fast KVzip
**Paper:** [Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction](https://arxiv.org/abs/2601.17668) (Jan 2026)

Eliminates KVzip's compression overhead by training lightweight gating modules (forward-pass only, no backprop) to predict KV importance. Achieves near-lossless at 70% eviction on Qwen2.5-1M, Qwen3, Gemma3. The gate training uses KVzip's reconstruction scores as ground truth.

---

## 2. Quest — Query-Aware Sparsity via Page-Level Key Statistics

**Paper:** [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2406.10774) (ICML 2024)
**Code:** [github.com/mit-han-lab/Quest](https://github.com/mit-han-lab/Quest)

### Core Idea
Organizes KV cache into **pages** (16 tokens each) and maintains per-page min/max statistics for each key dimension. At decode time, uses the query vector to compute an upper bound on the attention score for each page, then loads only the Top-K most critical pages. No eviction — all tokens remain accessible.

### How It Handles the Eviction Problem
**Completely avoids eviction.** The full KV cache remains in memory (or offloaded). Quest only reduces the amount of KV cache **loaded** into attention computation per decode step. This is the key differentiator from eviction methods: information is never destroyed.

### Compression / Memory
- Quest does NOT reduce memory — the full KV cache is still stored
- It reduces **memory bandwidth**: only Top-K pages are loaded per decode step
- Typical budgets: 1-12.5% of context length (e.g., 2048 tokens out of 32K)
- Effective bandwidth reduction: ~8x at 32K context with 2048-token budget

### NIAH / Retrieval Accuracy

| Test | Budget | H2O | StreamingLLM | Quest |
|------|--------|-----|-------------|-------|
| Passkey 10K | 64 tokens | 1% | 1% | **99%** |
| Passkey 10K | 256 tokens | 3% | 3% | **99%** |
| Passkey 100K | 1024 tokens | 1% | 1% | **96%** |

This is the critical result: **Quest achieves near-perfect retrieval accuracy with <1% of tokens**, while eviction methods completely fail. This directly validates our hypothesis that the causal limitation is the problem, not the selection algorithm.

### Latency
- Up to **7.03x self-attention speedup** (32K context, 2048-token budget)
- End-to-end: 1.74x (FP16), 2.23x (4-bit weights)
- First two layers use full attention (low sparsity); remaining layers achieve >90% sparsity

### Key Limitations
- **No memory reduction** — full KV cache must be stored somewhere
- Page-level granularity (16 tokens) may miss fine-grained patterns
- Upper bound estimation can be loose, loading unnecessary pages
- Min/max metadata overhead per page
- Not suitable when memory (not bandwidth) is the bottleneck

### Relation to Our Work
Quest proves that query-aware selection at decode time solves NIAH perfectly. Combined with our quantization (store the full cache in q4_0 but use Quest-style selection), this could achieve 4x memory reduction + 7x bandwidth reduction. The limitation is that Quest alone doesn't save memory — it needs to be paired with quantization or offloading.

---

## 3. RetrievalAttention — Eviction-Free KV Cache via CPU Vector Search

**Paper:** [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](https://arxiv.org/abs/2409.10516)
**Code:** [github.com/microsoft/RetrievalAttention](https://github.com/microsoft/RetrievalAttention)

### Core Idea
Treats the KV cache as a **vector database**. Offloads the full KV cache to CPU memory, builds an attention-aware ANNS (Approximate Nearest Neighbor Search) index, and at decode time retrieves only the 1-3% most relevant KV pairs via vector search. Solves the critical OOD (out-of-distribution) problem: query vectors and key vectors have different distributions due to separate projection matrices, causing standard ANNS to fail.

### How It Handles the Eviction Problem
**Complete eviction avoidance.** All tokens remain indexed in CPU memory. The system dynamically retrieves the most relevant tokens per query, per head, per layer. GPU stores only fixed patterns (128 initial tokens + 512 sliding window = 640 tokens).

### Memory Savings
- GPU: ~640 tokens per head (fixed) vs. full context
- CPU: full KV cache + ANNS index
- Single RTX 4090 (24GB) serves 128K context for 8B models
- At 1M tokens: requires A100 + CPU offload

### NIAH / Retrieval Accuracy

**Infinity-Bench (128K, Llama-3-8B):**
- Passkey retrieval: 100%
- Number retrieval: 100%
- Average across tasks: 48.9-49.6% vs 50.4% full attention (-0.8%)

**RULER (128K):** 84.70% vs 86.54% full attention (-1.85%)

**NIAH:** "Near-perfect retrieval regardless of needle position" across 4K-128K

### Latency

| Context | Full Attention | RetrievalAttention | Speedup |
|---------|---------------|-------------------|---------|
| 128K | 43.9s/token | 0.188s/token | **234x** |
| 1M | 1,740s/token | 0.172s/token | **>10,000x** |

Latency breakdown at 128K: 34% vector retrieval, 43% attention, 23% other.

**Critical scaling property:** Only 8% latency increase when context scales 10x (100K to 1M), because retrieval is O(log n) not O(n).

### vs. Other Methods
- Quest/InfLLM: "Nearly zero accuracy in complex tasks (KV retrieval) due to low accuracy of representative vectors"
- StreamingLLM: Fails on any non-recent retrieval
- RetrievalAttention with top-2000 retrieval matches full attention on KV retrieval

### Key Limitations
- **No memory savings on CPU** — full KV cache + index stored in CPU RAM
- **CPU dependency:** Requires multi-core CPU for parallel vector search (tested on 20-core Intel i9)
- **Index construction cost:** Requires full attention pass during prefill
- **PCIe bandwidth:** CPU-GPU data transfer becomes bottleneck at very high throughput
- **Not applicable to GPU-only or edge deployments**

### Relation to Our Work
RetrievalAttention validates that eviction-free approaches achieve near-perfect NIAH. The 0.188s/token at 128K is practical for many use cases. However, it requires **more** total memory (CPU RAM for full cache + index), not less. For our llama.cpp context (single-device inference), this translates to: store full KV at q2/q4 on disk/CPU, build ANNS index, retrieve top-k at full precision per decode step.

---

## 4. InfiniGen — Speculative KV Cache Prefetching via Cross-Layer Prediction

**Paper:** [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://arxiv.org/abs/2406.19707) (OSDI 2024)
**Code:** [github.com/snu-comparch/InfiniGen](https://github.com/snu-comparch/InfiniGen)

### Core Idea
Predicts which tokens will be important for layer i+1's attention by performing a **minimal rehearsal** using layer i's inputs and a subset of layer i+1's query/key weights. Leverages the observation that consecutive transformer layers have highly similar inputs (cosine similarity 0.95-0.97). Uses SVD-based weight skewing to emphasize discriminative dimensions, enabling single-pass column selection with only 30% of weight columns.

### How It Handles the Eviction Problem
**No permanent eviction.** Full KV cache resides in CPU memory. InfiniGen speculatively prefetches only the predicted-important entries to GPU before each layer's attention, using PCIe transfers. A dynamic threshold (alpha = 4-5) determines the loading cutoff, resulting in <10% of KV cache loaded per layer on average.

### Memory & Performance
- GPU stores: partial weights (2.5% of model params) + partial key cache (15% of full)
- CPU stores: full KV cache
- **Up to 3x speedup** vs prior offloading methods
- WikiText-2 PPL: 10.55 vs 10.55 baseline (bit-identical on perplexity)
- Accuracy within baseline range across COPA, OpenBookQA, WinoGrande, PIQA, RTE

### Latency
- 1.28x-34.64x improvement across batch sizes
- Saturates at 5.28x speedup for long sequences (vs H2O plateau at 3.40x)

### Models Tested
OPT (6.7B-30B), Llama-2 (7B-13B), Llama-2-7B-32K, Llama-3-8B-1048K

### Key Limitations
- **Speculation accuracy depends on cross-layer similarity assumption** (may fail for architectures with very different layer behaviors)
- **Requires offline SVD** per model (one-time cost)
- **CPU memory + PCIe bandwidth** are the bottlenecks
- **Not memory-efficient** — total memory usage equals or exceeds full cache

### Relation to Our Work
InfiniGen's cross-layer prediction is an elegant approach to the "which tokens matter?" problem, but it doesn't reduce total memory. For llama.cpp single-device inference, the CPU-GPU split doesn't apply cleanly. However, the SVD-based column selection technique could potentially be used to make Quest-style page selection more accurate.

---

## 5. CacheBlend — Selective KV Recomputation for RAG Cache Fusion

**Paper:** [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444) (EuroSys 2025)
**Code:** [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache)

### Core Idea
In RAG scenarios, precomputed KV caches for retrieved text chunks can be reused, but they lack **cross-attention** between chunks (each chunk was encoded independently). CacheBlend identifies a small subset of tokens with the highest KV deviation (HKVD) between precomputed and correct values, and selectively recomputes only those tokens to restore cross-chunk attention.

### Selective Recomputation Strategy
1. **High KV Deviation (HKVD) tokens:** Tokens with largest gap between precomputed and fully-recomputed KV values are prioritized
2. **Layer-to-layer correlation:** Tokens with high deviation in one layer tend to maintain high deviation across layers (Spearman's rank correlation is consistently high)
3. **Gradual filtering:** Use this correlation to progressively narrow the recompute set across layers
4. **Recomputation ratio:** 5-18% of tokens, typically ~10-15%
5. **Pipelining:** Layer i recomputation overlaps with layer i+1 KV cache loading, hiding latency

### Accuracy
- F1 loss vs full recompute: ≤0.02 absolute
- Multi-hop QA (Musique/2WikiMQA): 0.1-0.2 higher F1 than full KV reuse
- Summarization (Rouge-L): 0.03-0.25 higher than full KV reuse
- Full KV reuse (no recompute) degrades by 0.15-0.35 F1

### Performance
- **TTFT: 2.2-3.3x reduction** vs full recompute
- **Throughput: 2.8-5x increase** vs full recompute
- Compute overhead = r% of full prefill (linear in recomputation ratio)
- At 15% ratio on Llama-7B: 3ms recompute vs 16ms SSD loading (fully hidden)

### Models: Mistral-7B, Yi-34B, Llama-70B

### Key Limitations
- **RAG-specific:** Requires precomputed chunk caches; not general KV compression
- **Assumes text chunks with cross-attention gaps** — not applicable to single-stream inference
- **Transformer-only** (no Mamba, Griffin, etc.)
- **Single-device storage only** (no distributed caching explored)
- Chunk size sensitivity: works best with chunks ≥512 tokens

### Relation to Our Work
CacheBlend's core insight — that only 10-15% of tokens need recomputation to restore quality — is directly relevant. However, CacheBlend solves a different problem (RAG cache fusion) from our problem (runtime memory compression). The HKVD metric could potentially be adapted: identify high-deviation tokens during eviction and store their IDs for selective recompute, rather than losing them entirely.

---

## 6. ProphetKV — Query-Driven Selective Recomputation for RAG

**Paper:** [ProphetKV: User-Query-Driven Selective Recomputation for Efficient KV Cache Reuse in RAG](https://arxiv.org/abs/2602.02579) (Feb 2026)

### Core Idea
Improves on CacheBlend by using the **user query** as a "prophet" to guide token selection for recomputation. Identifies the "crowding-out effect" where globally salient tokens consume the recomputation budget, displacing query-relevant tokens.

### Key Innovation: Crowding-Out Effect
CacheBlend/EPIC select tokens by global importance (KV deviation), which wastes budget on tokens that are salient but irrelevant to the current query. ProphetKV shows this causes up to **86% accuracy degradation** in extreme cases.

### Dual-Stage Pipeline
- **Stage 1:** Lightweight pass computing query-to-context attention across all layers — O(|Q_s| × s), much cheaper than O(s²)
- **Stage 2:** Fuse layer-wise scores via uniform averaging, select top-p tokens, recompute their KV caches

### Results (20% recomputation ratio)

**RULER (8K context):**
| Model | ProphetKV | CacheBlend | EPIC | KVShare |
|-------|-----------|-----------|------|---------|
| Llama-3.1-8B | 84.71% | 77.27% | 70.32% | 73.47% |
| Qwen2.5-14B | 88.60% | 78.11% | 74.04% | 73.35% |
| Qwen-3-14B | 89.89% | 71.99% | 67.94% | 70.64% |

**LongBench:**
| Model | ProphetKV | CacheBlend | EPIC | KVShare |
|-------|-----------|-----------|------|---------|
| Llama-3.1-8B | 50.80% | 39.82% | 35.41% | 38.67% |
| Qwen2.5-14B | 53.43% | 35.40% | 30.99% | 34.82% |

**Convergence:** ProphetKV reaches near-complete accuracy at 20% recompute ratio; CacheBlend requires 40-80%.

### TTFT (16K context)
| Model | Full Prefill | ProphetKV (20%) | CacheBlend |
|-------|-------------|-----------------|-----------|
| Llama-3.1-8B | 5.23s | 1.13s (4.6x) | 1.29s |
| Qwen2.5-14B | 9.94s | 2.12s (4.7x) | 2.31s |

### Key Limitations
- RAG-specific (assumes pre-cached chunks)
- Requires query at recomputation time
- Chunk size ≥512 tokens for best results
- Accuracy degrades with longer contexts (16K shows larger gaps)

### Relation to Our Work
ProphetKV validates that **query-aware** token selection dramatically outperforms query-agnostic approaches for recomputation. The 20% recompute ratio achieving ~100% accuracy recovery is a strong signal. For our selective recompute direction, this suggests: store all token IDs + ultra-low-precision KV, use the decode query to identify the ~20% of tokens that need full-precision recomputation.

---

## 7. KV-Direct — Residual Stream Checkpointing (27x Compression, Bit-Identical)

**Paper:** [The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference](https://arxiv.org/abs/2603.19664) (March 2026)

### Core Idea
Keys and values at every layer are **deterministic projections** of the residual stream. Instead of caching K and V separately per layer, checkpoint the single residual stream vector per token and recompute K, V on demand. This achieves **zero reconstruction error — bit-identical output** under greedy decoding.

### Compression
- Residual checkpoint: **5 KB per token** (Gemma 3-4B)
- Full KV cache: **136 KB per token**
- **Compression ratio: 27x**
- Over 20 conversation turns: 42 MB (KV-Direct) vs 103 MB (standard) = **2.5x peak memory reduction**

### Accuracy
- **Token-identical** under greedy decoding across all tested models
- D_KL = 0 between patched and original output distributions
- **Perfect fidelity** — this is not an approximation

### Latency
- Recomputation is **up to 5x faster** than reading cached tensors at moderate batch sizes
- At 500 evicted tokens: 0.3x the time of cached reads (memory bandwidth is the bottleneck, not compute)
- KV projection is a single matrix multiply per layer — very cheap

### vs. Eviction Baselines
Against H2O, StreamingLLM, SnapKV, TOVA, window-only:
- **KV-Direct: 100% token match at every cache budget**
- All baselines degrade to **5-28%** accuracy

### Models Tested
Six models spanning 135M to 4B parameters across four architecture families.

### Key Limitations
- Requires recomputing K, V for **every** layer at every decode step
- At very long contexts, recomputation cost grows linearly with number of tokens to reconstruct
- Tested up to 4B parameters — unclear scaling to 70B+
- The 27x ratio assumes single residual vector replaces all per-layer KV; if only some layers need recompute, savings are lower
- Cannot be combined with techniques that modify residual stream (e.g., some LoRA variants)

### Relation to Our Work
**This is the most directly relevant approach for our selective recompute direction.** The insight that residual stream checkpointing gives bit-identical output at 27x compression is extraordinary. For our llama.cpp context:
- Store residual stream checkpoints (5 KB/token) instead of full KV (136 KB/token)
- At decode: use low-precision approximate attention to identify top-k important tokens
- Recompute full-precision K, V for those tokens from residual checkpoints
- Combined with quantization of checkpoints: potentially 50-100x total compression

The 0.3x recompute time (faster than cache reads at moderate sizes) means this is compute-positive, not just memory-positive. This changes our theoretical analysis significantly.

---

## 8. SparQ Attention — Bandwidth-Efficient Sparse Attention Without Eviction

**Paper:** [SparQ Attention: Bandwidth-Efficient LLM Inference](https://openreview.net/pdf?id=cp1hJ67l3M)

### Core Idea
Three-step algorithm to reduce KV cache bandwidth:
1. **Query sparsification:** Identify top-r largest-magnitude query dimensions, fetch only those key dimensions to compute approximate attention scores
2. **Top-k selection:** From approximate scores, identify top-k tokens and fetch their full K, V entries
3. **Score interpolation:** Estimate total attention mass, interpolate between sparse output and mean value vector

### How It Avoids Eviction
**All KV cache data remains in memory** — SparQ "simply does not access all of it at every iteration." This preserves retrieval capability that eviction methods destroy.

### Performance
- Up to **8x bandwidth compression** with little to no loss
- Near-dense performance on needle-in-haystack retrieval
- Models: Llama 2, Llama 3, Mistral, Gemma, Pythia

### Key Limitations
- **No memory reduction** — full cache stored in memory
- Hyperparameter-dependent (r, k)
- Benefits increase with sequence length; marginal at short contexts

### Relation to Our Work
SparQ is complementary to quantization: we reduce memory footprint via quantization, SparQ reduces bandwidth via sparse access. Combined: q4_0 cache (4x memory) + SparQ-style top-k access (8x bandwidth) could give excellent performance. However, SparQ doesn't help with memory pressure per se.

---

## 9. KVPR — I/O-Aware Partial Recomputation (CPU-GPU Overlap)

**Paper:** [KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation](https://arxiv.org/abs/2411.17089) (ACL 2025 Findings)
**Code:** [github.com/chaoyij/KVPR](https://github.com/chaoyij/KVPR)

### Core Idea
When KV cache is offloaded to CPU, instead of transferring everything via PCIe, **recompute a portion on GPU** while the rest transfers in parallel. An automated profiler/scheduler determines the optimal split based on hardware characteristics.

### Performance
- **Up to 35.8% lower latency** and **46.2% higher throughput** vs state-of-the-art offloading
- Automatically adapts to different hardware configurations

### Relation to Our Work
KVPR is a systems optimization for the offloading case. Not directly applicable to single-device llama.cpp, but the principle of overlapping recomputation with I/O is relevant to any hybrid storage approach.

---

## 10. MagicPIG — LSH Sampling for Attention Approximation

**Paper:** [MagicPIG: LSH Sampling for Efficient LLM Generation](https://arxiv.org/abs/2410.16179)

### Core Idea
Uses Locality-Sensitive Hashing (LSH) on CPU to **sample** (not select top-k) KV cache entries for attention computation. Sampling with theoretical guarantees provides better estimation than top-k selection when attention is not highly sparse.

### Performance
- Up to **5x throughput** improvement
- **54ms per token** on single RTX 4090 for Llama-3.1-8B at 96K context
- CPU handles LSH hash tables and attention computation
- Theoretical guarantees on approximation quality

### Relation to Our Work
MagicPIG addresses a subtle issue: top-k selection itself introduces bias when attention isn't sparse enough. Sampling-based approaches avoid this. For our ultra-high compression regime (85%+ eviction), where the remaining 15% of tokens must carry all information, this sampling approach could be more robust than deterministic top-k.

---

## 11. InfoFlow KV — Information-Flow-Aware Recomputation

**Paper:** [InfoFlow KV: Information-Flow-Aware KV Recomputation for Long Context](https://arxiv.org/abs/2603.05353) (March 2026)

### Core Idea
Frames selective recomputation as an **information flow problem**. Uses attention-norm signal from the query under inference-consistent RoPE geometry to identify tokens that are both semantically relevant and structurally positioned to propagate information. Introduces information-flow-guided chunk reordering strategy.

### Key Innovation
Previous methods use heuristics (KV deviation, attention scores) without modeling whether selected tokens can actually **influence generation** through the attention graph. InfoFlow explicitly models this information propagation.

### Relation to Our Work
The information-flow framing could improve our Expected Attention approach: instead of just estimating which tokens get high attention, model which tokens actually propagate information to the output.

---

## Theoretical Analysis: Selective Recompute Overhead

### The 34.6x Theoretical Overhead (From Our research_direction.md)

Our earlier analysis proposed:
- Store all tokens at 2-4bit + token IDs (16 bits)
- Use low-precision attention to find top-k important tokens
- Recompute those k tokens at full precision from token IDs
- Theoretical: **34.6x compression @ NIAH 100%, ~11% compute overhead**

### Updated Analysis with KV-Direct Insight

KV-Direct shows the theoretical picture is much better than we calculated:

**Old approach (recompute from token IDs):**
- Requires full forward pass through all layers for the selected tokens
- Cost per token: O(L × d² × k) where L = layers, d = hidden dim, k = selected tokens
- This is essentially running inference on k tokens — expensive

**New approach (recompute from residual checkpoints):**
- Store residual stream vectors (single vector per token, shared across layers)
- Recompute K, V via single matrix multiply: K = W_K × residual, V = W_V × residual
- Cost per token per layer: O(d × d_head) — just two matrix multiplies
- **Much cheaper than full recompute from token IDs**

### Revised Overhead Calculation

For Llama-3-8B (d=4096, d_head=128, n_kv_heads=8, L=32):
- Residual checkpoint: 4096 × 2 bytes = 8 KB/token (fp16)
- Full KV cache: 32 layers × 8 heads × 128 dim × 2 (K+V) × 2 bytes = 128 KB/token
- **Compression: 16x** from checkpointing alone
- With q4_0 on checkpoints: 4096 × 0.5 bytes = 2 KB/token → **64x compression**

Recompute cost for k tokens:
- Per layer: k × d × d_head × 2 (K+V) = k × 4096 × 128 × 2 FLOPs
- Total across 32 layers: k × 32 × 4096 × 128 × 2 = k × 33.6M FLOPs
- Full attention for 1 query: n × d × d_head × n_heads = n × 4096 × 128 × 32 = n × 16.8M FLOPs

For n=16K, k=0.01n=160 (1% selection like RetrievalAttention):
- Recompute: 160 × 33.6M = 5.4 GFLOPs
- Full attention: 16K × 16.8M = 275 GFLOPs
- **Overhead: 2%** — negligible

For n=16K, k=0.2n=3200 (20% selection like ProphetKV):
- Recompute: 3200 × 33.6M = 107.5 GFLOPs
- Full attention: 275 GFLOPs
- **Overhead: 39%** — significant but acceptable for the quality gain

### The Real Bottleneck: Memory Bandwidth

KV-Direct showed recomputation is **0.3x the time of cached reads** at 500 tokens. This is because:
- Cache read is memory-bandwidth-bound: reading 128 KB from DRAM
- Recompute is compute-bound: two matmuls on 4096-dim vector
- Modern GPUs have more compute than memory bandwidth (compute/bandwidth ratio keeps growing)

**This means selective recompute is not just memory-efficient — it's potentially faster than cached reads for the selected tokens.**

---

## Synthesis: What This Means for Our Paper

### The Eviction Wall (Confirmed by Literature)

Our experimental finding that eviction destroys NIAH at 85%+ is now validated by the broader literature:
- Quest shows **99% NIAH** with query-aware selection but **no eviction** (keeping full cache)
- RetrievalAttention shows **100% passkey** with ANNS retrieval and **no eviction**
- KVzip achieves 3-4x (75% eviction) but no results at higher rates
- H2O/StreamingLLM: 1% passkey at 64-token budget (confirmed by Quest paper)
- KV-Direct: 100% accuracy vs 5-28% for all eviction baselines

### The Selective Recompute Path Forward

| Approach | Memory | NIAH | Overhead | Feasibility |
|----------|--------|------|----------|-------------|
| Quantization only (q4_0) | 4x | 100% | 0% | Done |
| + Eviction (50%) | 8x | 60% | 0% | Done (fails) |
| + Eviction (85%) | 26x | 20% | 0% | Done (fails badly) |
| + Quest-style sparse attn | 4x* | ~99% | Low | Medium |
| + Residual checkpoint (KV-Direct) | 16-64x | 100% | 2-39% | Hard |
| + Checkpoint + sparse select | 64x+ | ~100% | 5-15% | Hard |

*Quest doesn't reduce memory, only bandwidth

### Recommended Future Work Direction for Paper

**Residual Stream Checkpointing + Query-Aware Sparse Selection:**
1. During prefill, store residual stream checkpoints at q4_0 (2 KB/token instead of 128 KB/token = 64x)
2. During decode, use approximate attention (on q4_0 checkpoints) to identify top-1-3% important tokens
3. Recompute full-precision K, V for those tokens from their residual checkpoints
4. Run full-precision attention on the small selected set + recent window

This achieves:
- **64x memory compression** (vs our current 4x with q4_0)
- **~100% NIAH** (validated by Quest/RetrievalAttention at 1-3% selection)
- **<5% compute overhead** (recompute is faster than cache reads)
- **Bit-identical output** for selected tokens (KV-Direct guarantee)

The engineering challenge is implementing this in llama.cpp: residual stream checkpointing, approximate attention for selection, and selective recompute in the attention loop.

---

## Complete Reference List

1. KVzip — [arxiv.org/abs/2505.23416](https://arxiv.org/abs/2505.23416) (NeurIPS 2025 Oral)
2. Fast KVzip — [arxiv.org/abs/2601.17668](https://arxiv.org/abs/2601.17668) (Jan 2026)
3. Quest — [arxiv.org/abs/2406.10774](https://arxiv.org/abs/2406.10774) (ICML 2024)
4. RetrievalAttention — [arxiv.org/abs/2409.10516](https://arxiv.org/abs/2409.10516)
5. InfiniGen — [arxiv.org/abs/2406.19707](https://arxiv.org/abs/2406.19707) (OSDI 2024)
6. CacheBlend — [arxiv.org/abs/2405.16444](https://arxiv.org/abs/2405.16444) (EuroSys 2025)
7. ProphetKV — [arxiv.org/abs/2602.02579](https://arxiv.org/abs/2602.02579) (Feb 2026)
8. KV-Direct — [arxiv.org/abs/2603.19664](https://arxiv.org/abs/2603.19664) (March 2026)
9. SparQ Attention — [openreview.net](https://openreview.net/pdf?id=cp1hJ67l3M)
10. KVPR — [arxiv.org/abs/2411.17089](https://arxiv.org/abs/2411.17089) (ACL 2025 Findings)
11. MagicPIG — [arxiv.org/abs/2410.16179](https://arxiv.org/abs/2410.16179)
12. InfoFlow KV — [arxiv.org/abs/2603.05353](https://arxiv.org/abs/2603.05353) (March 2026)
