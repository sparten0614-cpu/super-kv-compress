# Research Direction: KV Cache Compression

## Goal
Find maximum KV cache compression ratio while maintaining:
- PPL degradation < 5% vs F16 baseline
- NIAH retrieval accuracy >= 80% (ideally 100%)

## Target
- Minimum: 5x compression @ NIAH 100%
- Stretch: 10x+ compression @ NIAH >= 80%
- Moon shot: 30x+ (requires novel approach)

## Verified Findings (Experimentally Confirmed)

### Quantization Results (Cross-Model Verified)

| Config | Compression | PPL Δ% | NIAH | Models Tested |
|--------|-------------|--------|------|---------------|
| f16 (baseline) | 1.0x | 0% | 100% | All |
| q8_0 | 1.88x | ~0% | 100% | Llama-8B, Mistral-7B, 70B |
| q4_0 | 3.56x | +1-3% | 100% | Llama-8B (+3.1%), Mistral-7B (+1.2%), 70B (+3.2%) |
| q4_0 on Qwen | 3.56x | **CRASH** (PPL=6615) | 0% | Qwen-7B, Qwen-3B |

### Qwen Outlier Root Cause (Confirmed)
- **Root cause:** Layer 0 K values have extreme outlier (K_max=93)
- q4_0 symmetric quantization: range [-8,7]×scale → outlier dominates scale → other values zeroed
- **K is sole culprit:** V=q4_0 is safe (PPL=5.58), K=q4_0 crashes (PPL=9694)
- **Fix:** Skip Layer 0 (TQKV_SKIP_LAYERS=0): PPL recovers from +104% to +7.1%
- **Best Qwen config:** K=q8_0, V=q4_0 (asymmetric)

### Eviction Results (StreamingLLM, Position-Based)

**4K Context (Llama-3.1-8B):**
| Eviction | PPL Δ% | NIAH |
|----------|--------|------|
| 0% | 0% | 100% |
| 50% | +0.46% | 60% |
| 70% | +3.39% | 40% |
| 85% | — | 20% |

**16K Context (Llama-3.1-8B):**
| Eviction | PPL Δ% | NIAH |
|----------|--------|------|
| 0% | 0% | 100% |
| 50% | -0.03% | — |
| 70% | +1.11% | — |
| 80% | +3.36% | — |
| 85% | +7.50% | 20% |

**Key insight:** Longer context = more eviction-tolerant (50% at 16K is lossless!)

### H2O Attention-Aware Eviction
- H2O 50% NIAH = 60% (same as StreamingLLM) — **did not improve**
- Possible cause: H2O accumulates historical scores, but needle only matters for future queries
- Eval callback confirmed working (scores accumulated), but scoring strategy insufficient

### Information-Theoretic Limits (from Literature)
- Key vectors carry only 0.12 bits of predictive info about future attention
- Position + prefill attention carry 0.28-0.31 bits
- Trained 1.7M-param scorer showed NO improvement over heuristics
- Circular dependence: future attention ↔ generation ↔ cache ↔ future attention

## Failed Directions (Documented)
1. **QJL (Quantized Johnson-Lindenstrauss):** Random Gaussian projection harmful — variance too high, softmax amplifies noise. Both us and Tom independently confirmed.
2. **StreamingLLM eviction + NIAH:** Position-based eviction destroys retrieval at >50%. NIAH measures worst case.
3. **H2O on prefill-only scoring:** Cumulative softmax scores don't predict future query relevance. Information-theoretic bound confirms this.
4. **KVTC (NVIDIA):** Storage-only compression, not runtime memory. 20x headline number is misleading for our use case.
5. **RocketKV 400x:** Token selection ratio, not memory compression. Real memory saving = 32.6%. Real speedup = 3.7x. Still impressive but not magic.

## Promising Unexplored Directions

### 1. Asymmetric K/V + Per-Layer Quantization
- K=q8_0, V=q4_0 works for Qwen → can we push V even lower?
- Per-layer: outlier layers at q8_0, middle layers at q4_0, safe layers at q2_K
- --kv-type-map flag already implemented: "0:q8_0,1-30:q4_0,31:q8_0"
- Potential: 4-5x with quality close to uniform q8_0

### 2. Expected Attention (arXiv 2510.00636)
- Gaussian query model: z_i = exp(μ^T k_i / √d + k_i^T Σ k_i / 2d)
- Training-free, doesn't need eval callback, can work with flash attn
- Paper claims NIAH-safe to 125K context
- Phase 1 implemented (using H2O scores), Phase 2 (pure K-based) pending

### 3. SnapKV One-Shot Eviction
- Evict at end of prefill, re-enable flash attn for decode
- Zero decode overhead (one-time cost)
- Implemented in our fork, pending cloud testing

### 4. Multi-Tier KV Cache
- HOT tier (recent, q8_0) + COLD tier (old, q4_0/q2_K)
- Temporal decay: tokens migrate to lower precision as they age
- Design doc complete, implementation pending
- Combined with eviction: potentially 9x

### 5. Sparse V Dequant (Tom's Innovation)
- Skip V dequantization for low-attention positions
- +22.8% decode speed, zero quality loss
- Orthogonal to compression — speed optimization
- Could combine with our quantization

### 6. Novel: Dynamic Register KV
- Maintain tiny "register" of 32-128 high-attention tokens
- Rest at ultra-low precision (2-bit)
- Per-step: select which tokens to load into register
- Extreme compression potential (100x+)
- Engineering challenge: selection overhead

## Constraints
- Post-training only (no fine-tuning or continued pre-training)
- Must work in llama.cpp (C/C++ implementation)
- Target hardware: NVIDIA GPU (cloud) + Apple Silicon (local)
- Evaluation: PPL (wikitext) + NIAH (needle-in-a-haystack retrieval)
