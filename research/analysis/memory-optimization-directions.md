# LLM Memory Optimization: Four Research Directions

**Date:** 2026-03-31
**Author:** 宁宁
**Context:** TurboQuant (KV cache 6-bit compression, 2x reduction, lossless) is Phase 1. This document maps four directions for a comprehensive LLM memory optimization framework.

---

## Direction 1: Adaptive Layer-Aware Quantization

### Problem Definition

TurboQuant currently applies uniform bit-width across all layers. We discovered that Qwen2.5-3B has severe outlier layers (Layer 0: K_max=92.8, Layer 27: K_max=92.8, normal layers: ~12), causing PPL to spike from +4.4% to +8631% at 4-bit. Our binary fix (skip outlier layers → FP16) recovers 99% of quality at 5.6% extra memory. Can we do better?

### Why Layer 0 Is Always an Outlier

Four mechanisms converge to create the Layer 0 outlier:

**1. Attention Sinks (StreamingLLM, ICLR 2024; "When Attention Sink Emerges", ICLR 2025 Spotlight)**
Token 0 receives surplus attention probability due to causal masking — it's visible to all subsequent tokens. The Key vector at position 0 learns large magnitude so that small cosine similarity with any query yields a large logit. Layer 0 sees raw embeddings (no upstream LayerNorm dampening), making its Key magnitudes inherently larger.

**2. Massive Activations (Sun et al., 2024)**
A small number of fixed feature dimensions develop extreme magnitudes (20-60x normal) that are input-agnostic. These function as implicit bias terms, not information carriers. They emerge abruptly at specific layers and persist — explaining why Layer 27 (75% depth in 36-layer Qwen) is also an outlier.

**3. QKV Bias (Qwen-specific)**
Qwen2/2.5 retains QKV biases (unlike Llama). These additive offsets seed outlier features in specific layers. Quick heuristic: `model.config.attention_bias == True` → expect outlier layers.

**4. RoPE Amplification**
Qwen uses ABF RoPE with base=1,000,000 (vs Llama's 500,000). More aggressive rotation scrambles channel outliers, making post-RoPE Key distributions harder to quantize. KVQuant (NeurIPS 2024) showed that pre-RoPE Key quantization avoids 3.82 PPL degradation.

**5. GQA Multiplier**
Qwen2.5-3B uses 2 KV heads shared across 16 query heads (8x sharing). One outlier KV head contaminates 8 query heads simultaneously.

### Cross-Model Evidence

| Model Family | Layer 0 K Outlier | Other Outliers | Root Cause |
|---|---|---|---|
| Llama-2/3 | Mild (~12-30) | None prominent | No QKV bias, standard RoPE |
| Mistral/Ministral | Mild | None prominent | Similar to Llama |
| Qwen2/2.5 | Severe (92.8) | Yes (e.g. L27) | QKV bias + high-base RoPE + GQA |
| TinyLlama | Mild (~30) | None prominent | Llama architecture |

### Our Cut-In: From Binary Skip to Continuous Bit Allocation

**Current state:** Binary {FP16, TQKV_6} per layer.

**Evolution path:**
1. **Per-layer continuous bit allocation** — Each layer gets (k_bits, v_bits) from {2,3,4,5,6,FP16} based on K_max profile. Outlier layers get 6-bit+, normal layers can drop to 4-bit. Total memory budget constraint solved via greedy allocation (KVTuner approach but simpler — single K_max threshold vs multi-objective optimization).

2. **Channel-sparse FP16** — Instead of keeping the entire outlier layer at FP16, quantize all channels except the ~1% outlier channels (KVQuant-style sparse format). Reduces overhead from 5.6% to ~1-2%.

3. **Sink-token isolation** — Store only token 0's KV at FP16 in all layers (near-zero memory cost). Complementary to layer-level skip.

### Proposed Experiment

```
Qwen2.5-3B with budget=5-bit average:
  Config A: Uniform 5-bit (all layers)
  Config B: Binary skip (outlier=FP16, rest=TQKV_6) [current]
  Config C: Continuous (outlier=6-bit, normal=4-bit, weighted avg=5-bit)
  Config D: Channel-sparse (outlier channels=FP16, rest=4-bit)
Measure: PPL, NIAH 4K, memory
```

### Key References
- StreamingLLM (Xiao et al., ICLR 2024, arXiv:2309.17453)
- "When Attention Sink Emerges" (Gu et al., ICLR 2025 Spotlight)
- Massive Activations (Sun et al., 2024, arXiv:2402.17762)
- "A Unified View of Attention and Residual Sinks" (arXiv:2601.22966)
- KVQuant (Hooper et al., NeurIPS 2024) — per-channel K quantization
- KVTuner (Li et al., ICML 2025, arXiv:2502.04420)
- "Understanding and Minimising Outlier Features" (He et al., NeurIPS 2024)
- "Rethinking RoPE Scaling in Quantized LLMs" (arXiv:2510.00028)

---

## Direction 2: KV Cache Compression + Eviction Unified Framework

### Problem Definition

Compression (quantization) and eviction (pruning) are currently treated as independent problems. But they address the same goal — reduce KV cache memory. The key insight: **important tokens should be stored at high precision; unimportant tokens should be stored at low precision or evicted entirely.** This is a unified token-importance-to-precision mapping.

### Literature Survey

#### Eviction Methods

| Method | Venue | Core Mechanism | Quality | Key Limitation |
|---|---|---|---|---|
| **H2O** | NeurIPS 2023 | Cumulative attention "heavy hitters" + recent window | 80% cache reduction, 29x throughput | Noisy under quantization (Q-Hitter, MLSys 2024) |
| **StreamingLLM** | ICLR 2024 | Attention sinks (token 0-3) + sliding window | Unbounded generation, 22.2x speedup | No mid-context retrieval |
| **ScissorHands** | NeurIPS 2023 | "Persistence of Importance" — pivotal tokens tracked over bounded history | 5x cache reduction; 20x with 4-bit | Bounded history may miss long deps |
| **SnapKV** | NeurIPS 2024 | Observation window voting + 1D pooling | 3.6x speed, 8.2x memory at 16K | One-shot selection at prefill end |
| **PyramidKV** | TMLR 2025 | Layer-adaptive budget: more tokens in lower layers, fewer in upper | 88% cache reduction | Schedule is empirical |
| **Quest** | ICML 2024 | Query-aware page-level sparsity (min/max Key stats per page) | 7x self-attention speedup | Page granularity may miss fine details |
| **RazorAttention** | arXiv 2024 | Retrieval heads get full cache; non-retrieval drop remote | Head-type-dependent | Requires head classification |

#### Existing Unified Frameworks

| Paper | Venue | Approach |
|---|---|---|
| **MiniKV** | ACL Findings 2025 | PyramidKV selection + 2-bit for unimportant tokens; FlashAttention-compatible |
| **MiKV ("No Token Left Behind")** | arXiv 2024 | Soft eviction: "evicted" = INT2 residual, not deleted |
| **ScissorHands + 4-bit** | NeurIPS 2023 | 20x via eviction + 4-bit stack |
| **EVICPRESS** | arXiv 2024 | Joint utility function over compression quality + eviction delay |
| **"More Tokens, Lower Precision"** | arXiv 2024 | Explicit trade-off: more tokens at lower bits vs fewer at higher bits |

### The Convergent Architecture

All methods point to the same multi-tier design:

```
Tier 0: Sink tokens (position 0-3)     → FP16, never evict
Tier 1: Heavy hitters / high attention  → FP16 or INT8
Tier 2: Medium importance (retained)    → INT4 (TurboQuant)
Tier 3: Low importance                  → INT2 residual ("soft eviction")
Tier 4: Below threshold                 → Hard evict

Layer budget: lower layers → more tokens; upper layers → fewer tokens (PyramidKV)
```

### Our Cut-In: TurboQuant as the Compression Backend

TurboQuant's current 6-bit (2.56x compression) sits at Tier 2. The framework extension:

1. **Importance scoring** — Use attention accumulation (H2O) or observation window (SnapKV) to rank tokens. We already compute per-token statistics during compression — extend to importance scores at near-zero cost.

2. **Multi-precision TurboQuant** — The codebook + rotation pipeline supports any bit-width (2/3/4/5/6). Assign different bit-widths to different importance tiers:
   - Tier 1: TQKV_6 (lossless, our proven sweet spot)
   - Tier 2: TQKV_4 (+1.16-1.73% PPL, acceptable for medium tokens)
   - Tier 3: TQKV_2 (aggressive, for near-eviction tokens)

3. **MiKV-style soft eviction** — Instead of hard-deleting tokens, compress to TQKV_2 (2-bit residual). Prevents hallucination/safety issues that pure eviction causes.

### Proposed Experiment

```
Llama-3.1-8B, 16K context:
  Config A: Full KV (baseline)
  Config B: TurboQuant TQKV_6 uniform (current, 2.56x)
  Config C: H2O eviction only (keep top-20% + recent-128)
  Config D: Unified (top-20% at TQKV_6, next-30% at TQKV_4, rest at TQKV_2)
Measure: PPL, NIAH 16K, total memory, throughput
Target: 5-8x compression with <1% PPL degradation
```

### Key References
- H2O (Zhang et al., NeurIPS 2023, arXiv:2306.14048)
- SnapKV (Li et al., NeurIPS 2024, arXiv:2404.14469)
- PyramidKV (Cai et al., TMLR 2025, arXiv:2406.02069)
- StreamingLLM (Xiao et al., ICLR 2024, arXiv:2309.17453)
- ScissorHands (Liu et al., NeurIPS 2023, arXiv:2305.17118)
- MiniKV (ACL Findings 2025, arXiv:2411.18077)
- MiKV (arXiv:2402.18096)
- Quest (ICML 2024, arXiv:2406.10774)

---

## Direction 3: Learned Rotation Matrices

### Problem Definition

TurboQuant uses a fixed random Walsh-Hadamard Transform (WHT) to rotate KV vectors before quantization. This makes coordinates approximately i.i.d., enabling optimal scalar quantization. Could a data-driven rotation improve quality?

### Literature Survey

| Method | Venue | Rotation Learned? | Applies to KV? | Training Cost | Quality vs Random |
|---|---|---|---|---|---|
| **QuIP#** | ICML 2024 | No (random Hadamard) | No (weights only) | N/A | N/A |
| **QuaRot** | NeurIPS 2024 | No (random Hadamard) | Yes (online per-token) | N/A | Baseline; +0.28-1.35 PPL vs random orthogonal |
| **SpinQuant** | ICLR 2025 | R1/R2 learned; R3(KV) fixed | KV rotation = fixed Hadamard | 30min-3.5hr | Eliminates up to 13-point downstream variance |
| **RotateKV** | IJCAI 2025 | Channel sort from calibration | Yes (2-bit KV) | Minutes | <0.3 PPL at 2-bit on Llama-2-13B |
| **DartQuant** | arXiv 2024 | R1/R2 via Whip loss; R3 fixed | KV rotation = fixed Hadamard | ~1 GPU-hr for 70B | Comparable to SpinQuant, less overfit |
| **KVLinC** | arXiv 2024 | Linear correction adapters | Values rotated; keys NOT | Small FT | Beats KIVI/GEAR by 6.4% on GSM8K |

### The Consensus Finding

**The KV cache rotation itself is NOT learned in any production method.** It remains a fixed online Hadamard. Learning is applied to offline rotations (R1/R2 in SpinQuant/DartQuant) that reshape weight matrices, indirectly improving the distribution KV vectors are drawn from. These are fully amortized: computed once per model, baked into weights, zero per-token cost.

**What can be data-driven cheaply:**
- **Channel reordering permutation** (RotateKV): sort channels by magnitude from calibration stats, apply fixed permutation before WHT. Minutes of calibration, zero online overhead, significant gain at 2-bit.
- **Per-layer/per-head rotation parameters**: statistics-based, not gradient-based.

**What is too expensive online:**
- Genuinely learned per-prompt rotation is infeasible — cannot be amortized.

### KVLinC Counter-Finding

KVLinC found that rotating Keys can *hurt* quality — contradicting QuaRot's approach. Their solution: rotate Values only, add lightweight trainable linear correction adapters for attention scores. This suggests the optimal approach may be asymmetric: V rotation + K-specific handling.

### Our Cut-In: RotateKV-Style Channel Reordering

**Highest ROI addition to current TurboQuant:**

1. **Channel reordering** — Compute per-channel magnitude statistics over calibration set. Sort channels by magnitude. Apply fixed permutation before WHT. This prevents large-magnitude channels from being "hidden" by the random rotation where they interfere with low-magnitude channels.

2. **Pre-RoPE quantization** — Quantize Keys before RoPE application. Requires storing the RoPE angles separately and applying them at attention time. Avoids position-dependent distribution shifts that degrade quantization quality. KVQuant showed 3.82 PPL improvement from this alone.

3. **DartQuant-style weight rotation** — If pursuing aggressive compression (2-4 bit), learn R1/R2 rotations via Whip loss (~1 GPU-hr for 70B). Bake into weights. This reshapes the input distribution to KV projections, making the fixed KV rotation more effective.

### Proposed Experiment

```
Qwen2.5-3B at TQKV_4 (4-bit, currently +4.4% PPL):
  Config A: Random WHT (current)
  Config B: RotateKV channel reordering + WHT
  Config C: Pre-RoPE quantization + WHT
  Config D: B + C combined
Measure: PPL delta vs F16
Target: Reduce +4.4% to <+1% at 4-bit without adaptive layer skip
```

### Key References
- QuaRot (Ashkboos et al., NeurIPS 2024, arXiv:2404.00456)
- SpinQuant (Liu et al., ICLR 2025, arXiv:2405.16406)
- RotateKV (arXiv:2501.16383, IJCAI 2025)
- DartQuant (arXiv:2511.04063)
- KVLinC (arXiv:2510.05373)
- QuIP# (Tseng et al., ICML 2024, arXiv:2402.04396)

---

## Direction 4: Asymmetric K/V Quantization

### Problem Definition

TurboQuant uses the same bit-width for both K and V (`total_bits` parameter). Multiple papers show Keys are ~2x more sensitive to quantization than Values. Can asymmetric allocation (more bits for K, fewer for V) improve quality at the same memory budget?

### Why K Is More Sensitive: Three Explanations

**1. Softmax Nonlinearity (KVTuner, QJL)**
Attention logits are `QK^T/√d`. K quantization noise enters the softmax — an exponential function that amplifies small logit perturbations into large probability shifts. V quantization noise enters a linear combination `Σ aᵢvᵢ`, bounded by noise magnitude regardless of attention sharpness.

**2. Structural Outlier Asymmetry (KVQuant)**
K has channel-wise outliers (specific dimensions consistently large across tokens). V has token-wise outliers (specific tokens extreme across dimensions). Channel-wise outliers are harder for per-tensor quantization.

**3. Computational Role (QJL)**
K participates in an inner product feeding a nonlinear function (softmax). V participates in a linear aggregation. Nonlinear functions are generically more sensitive to input perturbations.

### Literature Evidence

| Source | K bits | V bits | Avg | Context |
|---|---|---|---|---|
| KVTuner (ICML 2025) | 4-6 | 2-4 | varies | Optimal per-layer search, K sensitivity 1.8-2.4x V |
| KIVI (NeurIPS 2024 ws) | 2 | 4 | ~3 | K=2 works with per-channel group quant (group=16) |
| WKVQuant (2024) | 4 (some 8) | 2 | ~3 | Clearest statement: K needs 2x more bits |
| MagR (2025) | 4 | 2 | 3 | K SNR degrades ~2x faster than V |
| KVQuant (NeurIPS 2024) | 4 (per-channel) | 4 (per-token) | 4 | Different granularity, not bit-width |

**Note on KIVI contradiction:** KIVI's K=2-bit works because they use per-channel group quantization with groups of 16 — effectively much higher precision than 2-bit per-element. TurboQuant's scalar quantization operates differently; KIVI's finding doesn't directly apply.

### Layer Variation of K/V Sensitivity

- **Layers 0-3:** K sensitivity is lower than average — V may need as many bits as K (attention still diffuse)
- **Layers 4 to N-10:** K sensitivity is 2-3x V. Classic asymmetric allocation applies
- **Layers N-10 to N:** Both K and V sensitivity peak. Late layers have sharpest attention — 6-bit+ K recommended even at tight budgets

### Our Cut-In: Split `total_bits` into `k_bits` and `v_bits`

**Implementation is straightforward:**

```python
# Current
config = TurboQuantConfig(head_dim=128, total_bits=6)  # Both K and V

# Proposed
config = TurboQuantConfig(head_dim=128, k_bits=6, v_bits=4)  # Asymmetric
```

In ggml: already have separate `--cache-type-k` and `--cache-type-v` flags. Just use different TQKV types:
```bash
./llama-server --cache-type-k tqkv_6 --cache-type-v tqkv_4
```

**Combined with per-layer adaptive:**
Each layer gets `(k_bits_l, v_bits_l)` instead of a single fallback to FP16. Example optimal allocation for Qwen2.5-3B:
- Layer 0: K=FP16, V=6-bit (outlier layer, K is critical)
- Layers 1-26: K=6-bit, V=4-bit (standard asymmetric)
- Layer 27: K=FP16, V=6-bit (outlier layer)
- Layers 28-35: K=6-bit, V=4-bit (late layers, could go K=8-bit for safety)

### Proposed Experiment

```
Llama-3.1-8B at 5-bit average budget:
  Config A: Uniform K=5, V=5
  Config B: Asymmetric K=6, V=4 (same memory)
  Config C: Asymmetric K=6, V=3 (less memory)
  Config D: Per-layer asymmetric (KVTuner-style search)
Measure: PPL, NIAH, LongBench v2
Target: K=6+V=4 beats uniform 5-bit by 0.5-2 PPL points
```

### Key References
- KVTuner (Li et al., ICML 2025, arXiv:2502.04420) — K 1.8-2.4x more sensitive
- KIVI (Liu et al., 2024) — 2-bit K + 4-bit V
- KVQuant (Hooper et al., NeurIPS 2024) — per-channel K vs per-token V
- WKVQuant (2024) — K=4-bit, V=2-bit
- MagR (2025) — K SNR degrades 2x faster
- QJL (Google, ICLR 2026) — K-centric correction design

---

## Priority Ranking

| Direction | Impact | Difficulty | Our Advantage |
|---|---|---|---|
| **D4: Asymmetric K/V** | High | Low | Already have separate cache-type flags in ggml |
| **D1: Adaptive Layer** | High | Medium | Already have binary skip working |
| **D2: Compression+Eviction** | Very High | High | TurboQuant as multi-precision backend |
| **D3: Learned Rotation** | Medium | Medium | RotateKV channel reorder is cheap |

**Recommended sequence:**
1. **D4 first** — lowest effort, highest immediate gain. Just run existing binaries with different K/V cache types.
2. **D1 next** — extend binary skip to continuous bit allocation. Requires new ggml per-layer config.
3. **D3 then** — RotateKV channel reordering. Requires calibration pipeline + ggml permutation support.
4. **D2 last** — most ambitious, requires importance scoring + multi-tier storage architecture.

---

## Summary: The Vision

TurboQuant Phase 1 proved that 6-bit KV cache quantization is lossless (2x compression). The four directions above extend this to a comprehensive framework:

```
Phase 1: Uniform TQKV_6                    → 2x compression (DONE)
Phase 2: + Asymmetric K/V                  → 2.5-3x compression
Phase 3: + Per-layer adaptive              → 3-4x compression
Phase 4: + Channel reordering              → 4-5x at 4-bit (currently lossy)
Phase 5: + Importance-based eviction        → 5-10x compression
```

Each phase builds on the previous, and each has a clear experimental validation path. The end goal: **10x KV cache reduction with <1% quality loss**, enabling long-context inference on consumer hardware.
