# Research Direction: KV Cache Compression — v2

**Last updated:** 2026-04-01
**Purpose:** Input context for the AutoResearch AI agent (Phase 3). The more
detail here, the smarter the agent's proposals.

---

## Goal

Find maximum KV cache compression ratio while maintaining quality:
- **Lossless:** PPL < +0.5%, NIAH = 100%
- **Near-lossless:** PPL < +1%, NIAH ≥ 80%
- **Aggressive:** PPL < +5%, NIAH ≥ 60%
- **Moon shot:** 30x+ compression (requires novel approach beyond quant+eviction)

## Constraints
- Post-training only (no fine-tuning or continued pre-training)
- Must work in llama.cpp (C/C++ implementation)
- Asymmetric K/V types supported via `--cache-type-k` / `--cache-type-v`
- Flash attention requires `-DGGML_CUDA_FA_ALL_QUANTS=ON` for mixed K≠V types
- Target hardware: NVIDIA RTX 5880 (cloud, 48GB) + Apple M4 (local, 16GB)
- Evaluation: PPL (wikitext-2, 16K) + NIAH (5-position) + LongBench v2

---

## Complete Experimental Data

### 1. Quantization (Cross-Model, 16K Context)

| Model | GQA | f16 PPL | q8_0 PPL (Δ%) | q4_0 PPL (Δ%) | q8_0 NIAH | q4_0 NIAH |
|-------|-----|---------|---------------|---------------|-----------|-----------|
| Llama-3.1-8B | 4:1 | 6.159 | 6.154 (-0.09%) | 6.348 (+3.06%) | 100% | 100% |
| Mistral-7B | 4:1 | 5.038 | 5.037 (-0.03%) | 5.101 (+1.25%) | 100% | 100% |
| Qwen2.5-7B | 7:1 | 5.660 | 5.548 (-1.98%) | **6615 (+116,796%)** | 100% | **0%** |
| Ministral-8B | 4:1 | 9.334 | — | — | 100% | — |
| Llama-3.3-70B | 8:1 | — | 3.720 (baseline) | 3.838 (+3.17%) | 100% | 100% |

**Key finding:** q4_0 PPL delta is ~3% and **model-size-independent** (8B +3.06%, 70B +3.17%).
TQKV_6 (TurboQuant 6-bit): Llama +0.08%, Ministral +0.18% — near-lossless.

### 2. Qwen Failure Analysis (Complete Asymmetric Diagnostic)

| Config | PPL | Δ% | Status |
|--------|-----|-----|--------|
| K=f16, V=f16 | 5.660 | baseline | OK |
| K=q8_0, V=q8_0 | 5.548 | -1.98% | OK |
| K=f16, V=q4_0 | 5.577 | -1.47% | OK |
| K=q8_0, V=q4_0 | 5.578 | -1.45% | OK, NIAH 100% ✅ |
| K=q4_0, V=f16 | 9694 | catastrophic | **K alone kills it** |
| K=q4_0, V=q4_0 | 6616 | catastrophic | |
| q4_1 (K+V) | 8392 | catastrophic | q4_1 equally bad |

**Root causes (two factors):**
1. **GQA ratio 7:1** — each KV head shared by 7 query heads, amplifying K error through softmax
2. **Layer 0 outlier K_max=93** — dominates q4_0 quantization range, zeroing normal values
3. Skipping Layer 0: +104% → +7.1% (dramatic improvement but still too high)
4. **V is universally robust to q4_0.** K is fragile in high-GQA models.

**GQA-aware rule:**
- n_kv ≤ 4 (GQA ≥ 7): K ≥ q8_0, V ≥ q4_0
- n_kv ≥ 8 (GQA ≤ 4): K ≥ q4_0, V ≥ q4_0
- General: K precision ≥ V precision, gap increases with GQA ratio

### 3. Eviction PPL Gradient (StreamingLLM)

| Eviction % | Llama 4K | Llama 16K | Mistral 16K | 70B 16K |
|------------|----------|-----------|-------------|---------|
| 0% | 0% | 0% | 0% | 0% |
| 10% | -0.07% | — | — | — |
| 30% | -0.20% | -0.17% | — | — |
| 50% | +0.46% | -0.09% | +0.36% | **-0.32%** |
| 60% | +1.40% | -0.003% | +0.30% | — |
| 70% | +3.39% | +1.04% | +0.26% | **-0.27%** |
| 80% | +9.22% | +3.36% | — | — |
| 85% | — | +7.50% | +4.66% | — |

**Critical insight: PPL can IMPROVE under eviction.** 70B at 16K: PPL improves at 50% AND 70%. Evicted tokens were adding noise.

**Cliff positions (1% PPL threshold):**
- Llama-8B 4K: ~53%
- Llama-8B 16K: ~67%
- Mistral-7B 16K: ~85%
- Llama-70B 16K: >70% (no cliff observed)
- Formula: f_safe(n) = 0.277 + 0.0405 × ln(n)

**70B at 4K vs 16K:** Same model shows +26% at 4K but -0.3% at 16K for 50% eviction. Context length is the dominant factor for eviction tolerance.

### 4. NIAH Retrieval (The Causal Limitation)

| Method | Eviction | Compression | NIAH 16K | PPL Δ |
|--------|----------|-------------|----------|-------|
| f16 | 0% | 1x | 100% | 0% |
| q8_0 | 0% | 2x | 100% | -0.09% |
| q4_0 (Llama) | 0% | 4x | 100% | +3.06% |
| q4_0 (Qwen) | 0% | 4x | **0%** | crash |
| K8V4 (Qwen) | 0% | 2.5x | 100% | -1.45% |
| TQKV_6 | 0% | 2.67x | 100% | +0.08% |
| StreamingLLM | 50% | 2x | **60%** | -0.09% |
| StreamingLLM | 70% | 3.3x | **40%** | +1.04% |
| StreamingLLM | 85% | 6.7x | **20%** | +7.50% |
| H2O | 50% | 2x | **60%** | TBD |

**Quantization is NIAH-safe** (for compatible architectures).
**Eviction destroys NIAH** regardless of method — H2O = StreamingLLM at matched rates.

**The causal limitation:** Eviction decisions at prefill cannot anticipate future queries. The needle receives low attention during prefill but is critical for the future retrieval query. This is information-theoretic, not algorithmic:
- Key vectors carry only 0.12 bits of predictive info about future attention
- Trained 1.7M-param scorer showed NO improvement over simple heuristics
- Circular dependence: future attention ↔ generation ↔ cache retention

### 5. LongBench v2

| Config | Score | Correct | Cache Size | Compression |
|--------|-------|---------|------------|-------------|
| Llama f16 | 35% | 7/20 | ~400 MiB | 1x |
| Llama TQKV_6 | 35% | 7/20 | ~200 MiB | 2x |

18/20 predictions identical; 2 differ but cancel out.

### 6. The Metric Gap (PPL ≠ NIAH)

| Compression | PPL Δ | NIAH | Verdict |
|-------------|-------|------|---------|
| 2.67x (TQKV_6) | +0.08% | 100% | All agree |
| 4x (q4_0 Llama) | +3.06% | 100% | PPL degrades, NIAH holds |
| 2x (50% evict) | **-0.09%** | **60%** | **PPL improves, NIAH drops!** |
| 3.3x (70% evict) | +1.04% | 40% | PPL mild, NIAH severe |
| 6.7x (85% evict) | +7.50% | 20% | Both degrade |

**PPL is misleading for eviction.** Three failure modes:
1. False positive: PPL improves under eviction that destroys retrieval
2. Architecture blindness: Qwen q4_0 PPL=6615, Llama q4_0 PPL=6.348
3. Orthogonality: quant and eviction produce opposite PPL-NIAH signatures

---

## Failed Directions (Learn From These)

### 1. QJL (Quantized Johnson-Lindenstrauss) for K
- Random Gaussian projection adds variance to K
- Softmax amplifies the variance: softmax(QK^T) → K errors hit exponentially
- Community reproductions (tonbistudio) confirmed: plain MSE > QJL for K
- **Lesson:** Anything that adds variance to K before softmax is dangerous

### 2. H2O for NIAH Preservation
- H2O accumulates historical attention scores to identify "heavy hitters"
- But heavy hitters during prefill ≠ tokens needed for future queries
- H2O 50% NIAH = 60% = identical to position-based StreamingLLM
- **Lesson:** Retrospective scoring cannot solve the causal limitation

### 3. KVTC (NVIDIA ICLR 2026) for Runtime Compression
- PCA + VQ + DEFLATE achieves 20-40x compression — but for **storage only**
- Decode requires PCA inverse + VQ lookup + DEFLATE decompress per step
- Not suitable for online GPU inference
- **Lesson:** Distinguish storage compression from runtime memory compression

### 4. RocketKV "400x"
- 400x is the token selection ratio, not memory compression
- Real memory saving = 32.6%, real speedup = 3.7x
- **Lesson:** Read compression claims carefully; check what metric is being compressed

### 5. Tiny Autoencoder (Encoder/Decoder per KV entry)
- Idea: encode head_dim=128 → latent_dim=16, decode at attention time
- **Killed by latency:** Decode is memory-bandwidth-bound. MLP decoder overhead = 10.5x the savings. Linear decoder = 4.7x too expensive at batch size 1.
- PALU-style fused linear (absorb decoder into attention GEMM) is the only viable learned compression.
- **Lesson:** Any decoder in the attention hot path must be fused or free

### 6. Delta Encoding Across Layers
- Hypothesis: K/V vectors similar across adjacent layers → store deltas
- Analysis showed K_l and K_{l+1} use different W_K projection matrices → vectors not meaningfully similar
- **Lesson:** KV similarity is superficial; the projection matrices decorrelate them

---

## Information-Theoretic Bounds

### Shannon Rate-Distortion
- After Hadamard rotation, KV entries ≈ N(0, σ²)
- Minimum bits/dim: R(D) = 0.5 × log2(σ²/D) = 3.62 bits/dim at our distortion level
- TurboQuant 6-bit → coding efficiency η = 3.62/6 = 60.3%
- **PPL ceiling:** 13x at practical efficiency (matches our measured data)
- **Task ceiling:** 64x (only task-relevant tokens matter)

### Attention Predictability (TAPPA Framework)
- Governed by query self-similarity
- High q-similarity (factual QA, NIAH): predictable → safe to compress
- Low q-similarity (multi-hop reasoning): unpredictable → dangerous
- Per-head q-similarity can be used as adaptive allocation signal

### Key Vector Predictive Information
- Keys carry only 0.12 bits about future attention
- Position carries 0.28-0.31 bits
- Prefill attention carries 0.28-0.31 bits
- **Maximum prediction accuracy bounded by ~0.31 bits**

---

## Promising Directions (Ranked by Feasibility)

### Tier 1: Immediately Testable (llama.cpp flags exist)

**1a. Asymmetric K/V Per-Layer Mapping**
- `--kv-type-map "0:q8_0,1-30:q4_0,31:q8_0"` (skip outlier layers)
- Potential: 3.5-4x at PPL < +1%, NIAH 100%
- Test: sweep layer-group assignments

**1b. q5_0 / q5_1 for K on Qwen**
- Qwen cliff is between q4_0 (crash) and q8_0 (safe)
- q5_0/q5_1 should be in the transition zone
- If q5 works → Qwen can get 3x+ instead of 2.5x

**1c. Longer Context Eviction (32K, 64K)**
- 70B at 16K shows no cliff at 70% → test longer contexts
- Formula predicts f_safe(32K) ≈ 70%, f_safe(128K) ≈ 75%
- Combined with quant: potentially 10x+ at 128K context

### Tier 2: Requires Custom Implementation

**2a. Expected Attention Eviction**
- Model future queries as Gaussian → closed-form expected attention score
- z_i = exp(μ_q^T k_i / √d + k_i^T Σ_q k_i / 2d)
- Training-free, NIAH-safe to 125K context (arXiv 2510.00636)
- Could replace H2O/StreamingLLM as NIAH-preserving eviction
- Potential: 2-4x additional compression on top of quantization

**2b. SnapKV One-Shot Eviction**
- Use instruction/query in prompt tail as observation window
- Evict at end of prefill → re-enable flash attention for decode
- Zero decode overhead (one-time prefill cost)
- NIAH-safe when query is present at prefill time
- Potential: 3-8x KV cache reduction

**2c. Multi-Tier Temporal Decay**
- HOT tier (recent tokens, q8_0) + COLD tier (old tokens, q4_0 or q2_K)
- Tokens migrate from HOT → COLD as they age
- Recent tokens matter more (sliding window effect)
- Potential: 5-7x combined with quant

### Tier 3: Novel Approaches (High Risk, High Reward)

**3a. Selective Recompute**
- Store all tokens at 2-4bit + token IDs (16 bits, negligible)
- At decode: use low-precision attention to find top-k important tokens
- Recompute those k tokens at full precision from stored token IDs
- **34.6x compression @ NIAH 100%, ~11% compute overhead** (theoretical)
- Trades compute for memory — the only approach that can break 10x + NIAH 100%
- Engineering challenge: efficient prefix recompute for k tokens

**3b. Dynamic Register KV**
- Maintain tiny "register" of 32-128 high-attention tokens at high precision
- Everything else at ultra-low precision (2-bit)
- Per-step: query-aware selection of which tokens to load into register
- Extreme compression potential (50-100x)
- Challenge: selection overhead must be < attention savings

**3c. PALU-Style Low-Rank + Quantization Stacking**
- SVD decomposition: cache rank-r latents instead of full head_dim
- Fuse decoder into attention GEMM (zero overhead): latent_K @ (U @ Q^T)
- Stack with quantization: 1.5-2x (low-rank) × 2.67x (quant) = 4-5x
- Post-training (SVD calibration), no gradient training
- Potential: adds ~1.5x multiplier to any quant config for free

---

## Compression Achievability Summary

| Method | Compression | PPL Δ | NIAH | Status |
|--------|-------------|-------|------|--------|
| TQKV_6 (6-bit) | 2.67x | +0.08% | 100% | ✅ Verified |
| K8V4 asymmetric | 2.5x | -1.45% | 100% | ✅ Verified (Qwen) |
| q4_0 (GQA ≤ 4) | 4x | +3% | 100% | ✅ Verified |
| q8_0 + 50% evict | 3.8x | ~0% | 60% | ✅ Verified (NIAH limited) |
| TQKV_6 + 50% evict | 5.3x | ~+0.4% | 60% | ✅ Verified (NIAH limited) |
| + Expected Attention | 5-10x | TBD | predicted safe | 🔄 Proposed |
| + Low-rank stacking | 6-8x | TBD | predicted safe | 🔄 Proposed |
| + Selective recompute | 30-50x | TBD | 100% (theory) | 🔄 Proposed |

---

## Agent Instructions

When proposing experiments:
1. **Always test both PPL and NIAH** — PPL alone is misleading for eviction configs
2. **Respect the GQA-aware rule** — if the model is Qwen (GQA 7:1), K must be ≥ q8_0
3. **Asymmetric K/V is always worth trying** — K precision > V precision is universally better
4. **Try per-layer configurations** — outlier layers need higher precision
5. **Don't retry failed directions** — QJL for K, H2O for NIAH, autoencoder for online inference are all dead ends
6. **Prefer configurations near the Pareto frontier** — small perturbations of good configs are more likely to yield improvements than random exploration
7. **Longer context = more eviction tolerance** — if testing eviction, use the longest context the hardware supports
8. **70B models are more eviction-robust** — what fails on 8B at 4K may work on 70B at 16K
