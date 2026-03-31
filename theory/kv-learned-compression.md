# KV Cache as Learned Compression: Tiny Autoencoder Feasibility Analysis

**Author:** 宁宁
**Date:** 2026-04-01
**Status:** Literature survey + feasibility analysis

---

## 1. The Idea

Compress KV cache entries using a tiny neural autoencoder:
- **Encoder** (prefill): KV vector (head_dim=128) → compressed latent (dim=16-32)
- **Decoder** (decode): compressed latent → reconstructed KV vector for attention

Analogous to neural image compression (JPEG-AI) but for KV vectors.

---

## 2. Prior Art — Someone Already Did This

### KV-CAR (Dec 2024, arXiv 2512.06727)
Exactly this idea. Per-layer lightweight MLPs compress K/V from dim D to d.
- Architecture: Linear(D→D/2) + ReLU + Linear(D/2→d)
- **Compression: ~2x** (not 8x)
- Quality: Minimal PPL impact on GPT-2/TinyLLaMA
- **Only tested on small models**

### PALU (ICLR 2025, arXiv 2407.21118) — The Winner
Low-rank SVD projection, but the **critical trick**: fuses the decoder into the Q·K^T GEMM:
```
Instead of: full_K @ Q^T
Do:          latent_K @ (U @ Q^T)    [U applied to Q, not to K]
```
Decoder is algebraically absorbed — never materializes full-dim KV.
- **Compression: 2x** with **1.89x speedup** (2.91x combined with 4-bit quant)
- Post-training (SVD calibration, no gradient training)
- **This is the correct abstraction for online inference**

### KVTC (ICLR 2026, NVIDIA, arXiv 2511.01815)
JPEG-style: PCA decorrelation → adaptive bit allocation → DEFLATE entropy coding.
- **Compression: 20-40x** — but for **storage** (disk/DRAM offload), not online GPU inference
- PCA alone captures most gain; entropy coding adds ~1.23x on top

### Other Methods
| Method | Compression | Approach | Use Case |
|--------|-------------|----------|----------|
| CacheGen (SIGCOMM 2024) | 3.5-4.3x | Entropy coding for network transfer | Distributed serving |
| Lexico (ICML 2025) | 4-6.7x | Sparse coding (OMP over learned dictionary) | Aggressive compression |
| CommVQ (2025) | 8-16x (1-2 bit) | Learned vector quantization codebooks | Extreme compression |
| KVSculpt (2025) | Configurable | Optimize synthetic KV pairs via L-BFGS | Offline/repeated queries |
| MLKV (2024) | 2-6x | Cross-layer KV sharing | Requires retraining |

---

## 3. Intrinsic Dimensionality of KV Vectors

Literature consensus on the effective rank of KV cache vectors:
- **Loki (NeurIPS 2024):** Top 12.5-25% of PCA components sufficient for accurate token selection
- **KV-CoRE (Feb 2026):** Normalized effective rank = **20-40% of head_dim** across models
- **KVTC:** 20x compression achievable → ~5-6 effective dimensions out of 128 after decorrelation
- **PALU:** SVD shows rank ~10-30 captures 90% variance for head_dim=128

**Conclusion:** For head_dim=128, intrinsic content ≈ 16-40 dimensions.
- 4x compression (→32 dims): reliable, low quality loss
- 8x compression (→16 dims): aggressive but plausible, model/layer-dependent
- 20x+: requires entropy coding, not pure dimension reduction

---

## 4. The Latency Problem — Why Naive Autoencoders Fail

### Core Issue: Decode is Memory-Bandwidth Bound

LLM decode is already memory-bandwidth limited. Adding a decoder that is ALSO memory-bandwidth bound makes things worse.

**Arithmetic intensity analysis (A100, head_dim=128, compressed_dim=16):**

| Operation | FLOPs/byte | A100 Ridge Point | Regime |
|-----------|-----------|-------------------|--------|
| Full KV attention (GEMV) | ~4 | 156 | Memory-bound |
| Linear decoder (16→128) | 14.2 | 156 | Memory-bound |
| MLP decoder (16→64→128) | ~18 | 156 | Memory-bound |

### Breakeven Analysis (S=8192, H=32, L=32, A100)

```
Full KV memory read per decode step:       4.30 GB  →  2.15 ms
Compressed KV memory read (8x):            0.54 GB  →  0.27 ms
Memory bandwidth savings:                            1.88 ms

MLP decoder overhead (16→64→128):                    19.8 ms  ← 10.5x too expensive!
Linear decoder overhead (16→128):                    8.8 ms   ← 4.7x too expensive!
Max affordable decoder FLOPs per token/head:         3,494
```

**At batch size 1, no naive decoder architecture is cost-effective.**

### When It Works

| Scenario | Verdict |
|----------|---------|
| PALU-style fused linear decoder | **YES** — decoder absorbed into attention GEMM |
| Large batch (B≥16) + linear decoder | Feasible — GPU utilization improves |
| Off-GPU storage (disk/DRAM/network) | **YES** — I/O bottleneck >> decode overhead |
| Nonlinear MLP decoder, online, B=1 | **NO** — 10x too slow |
| Modified attention on compressed latents | Active research, no production system |

---

## 5. Feasibility Verdict

### For Our 30-50x Goal

| Approach | Achievable? | Why/Why Not |
|----------|-------------|-------------|
| Tiny AE (encode/decode per step) | **No** | Decoder overhead >> bandwidth savings at B=1 |
| PALU-style linear (fused) | **Partial (2-3x)** | Only linear compression fusible; limited ratio |
| VQ codebook (CommVQ-style) | **Maybe (8-16x)** | Lookup is cheap but requires training; quality degrades at 1-bit |
| KVTC-style (storage) | **Yes (20-40x)** for storage | Not for online GPU inference |
| Combined: quant + low-rank + sparse | **5-10x realistic** | Layer cake of cheap operations |

### The Neural Image Compression Analogy Breaks Down

JPEG-AI works because:
1. Encoder runs once per image (amortized)
2. Decoder is off the critical path of downstream computation

For KV cache:
1. Decoder runs **every decode step × every layer × every head × every token**
2. Decoder IS on the critical path of attention computation

The cost multiplier is enormous. The analogy is misleading for online inference.

### What Actually Works for Online Inference

The winning approaches all share one property: **they avoid materializing full-dim KV during attention.**

1. **PALU:** Fuse linear projection into attention GEMM → decoder disappears
2. **Quantization:** Dequant is trivially cheap (bit manipulation, no matrix multiply)
3. **Eviction/sparsity:** Skip tokens entirely → no decoding needed
4. **Selective recompute:** Only recompute top-k → amortized over sequence length

---

## 6. Implications for Our Framework

### Does learned compression add a useful layer?

**Not as a standalone approach, but PALU-style low-rank projection can complement quantization:**

| Current Stack | Enhanced Stack |
|---------------|----------------|
| L1: GQA-aware quantization (2.67x) | L1: Low-rank projection (PALU, 1.5-2x) + quantization (2.67x) = **4-5x** |
| L2: Expected Attention eviction (2-4x) | L2: Same (2-4x) |
| L3: Selective recompute (8-10x) | L3: Same (8-10x) |
| **Total: 5-10x realistic, 30x theoretical** | **Total: 8-20x realistic, 50x theoretical** |

Low-rank (PALU) is the only learned compression that genuinely helps online inference because it can be fused. Adding it as L0.5 before quantization is a free ~1.5x multiplier.

### Novel Insight: Low-Rank + Quantization Stacking

No published work systematically studies the interaction of:
1. Low-rank projection (reduce effective dimension)
2. Quantization (reduce bits per dimension)
3. Eviction (reduce number of tokens)

The three are orthogonal compression axes:
- **Dimension reduction:** D → r (PALU)
- **Bit reduction:** 16-bit → b-bit per element (TurboQuant, KIVI)
- **Token reduction:** N → N' tokens (eviction, attention prediction)
- **Recovery:** Selective recompute for quality guarantee

Total compression = (D/r) × (16/b) × (N/N') = potentially massive.

---

## 7. Recommendation

**Do NOT pursue the tiny autoencoder approach for online inference.** The latency analysis shows it cannot pay for itself at batch size 1.

**Instead, consider adding PALU-style low-rank as an additional compression axis:**
- Post-training SVD calibration (no gradient training needed)
- Fuses with attention GEMM (zero decode overhead)
- ~1.5-2x additional compression on top of quantization
- Well-studied (ICLR 2025), implementation available

**For the paper:** Position our approach as exploiting **three orthogonal compression axes** (dimension × bits × tokens) plus **selective recompute** for quality recovery. This framing is novel and provides a clear theoretical contribution.

---

## 8. References

- KV-CAR (Dec 2024): arXiv 2512.06727
- PALU (ICLR 2025): arXiv 2407.21118
- KVTC (ICLR 2026): arXiv 2511.01815
- CacheGen (SIGCOMM 2024): arXiv 2310.07240
- Lexico (ICML 2025): arXiv 2412.08890
- CommVQ (2025): arXiv 2506.18879
- RVQ (2024): arXiv 2410.15704
- KVSculpt (2025): arXiv 2603.27819
- MLKV (2024): arXiv 2406.09297
- KV-CoRE (Feb 2026): arXiv 2602.05929
- Loki (NeurIPS 2024): proceedings.neurips.cc/paper/2024/1e027da6
