# PCA-Rotated Adaptive KV Cache Quantization

**Date:** 2026-04-01
**Status:** Mathematical framework — awaiting experimental validation

---

## 1. Problem Statement

Uniform quantization (q4_0, q8_0) assigns the same number of bits to every dimension of the K and V vectors. This is suboptimal because:
- K vectors have highly non-uniform energy distribution (condition number ~16M)
- A few "outlier" dimensions carry most of the energy
- Uniform quantization wastes precision on low-energy dimensions and lacks precision on high-energy dimensions

**Goal:** Achieve q8_0 quality at q4_0 bit budget by adaptive bit allocation across dimensions.

## 2. Mathematical Framework

### 2.1 PCA Rotation

For layer $\ell$, head $h$, collect K vectors from calibration data: $\{k_1, \ldots, k_N\} \in \mathbb{R}^d$ where $d = d_\text{head}$.

Compute empirical covariance: $\Sigma_K = \frac{1}{N} \sum_i k_i k_i^T$

Eigendecomposition: $\Sigma_K = P \Lambda P^T$ where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$, $\lambda_1 \geq \ldots \geq \lambda_d \geq 0$.

PCA rotation: $\tilde{k}_i = P^T k_i$ — the rotated K vector in decorrelated coordinates.

**Key property:** $\text{Cov}(\tilde{k}) = \Lambda$ is diagonal. Components of $\tilde{k}$ are uncorrelated.

### 2.2 Optimal Bit Allocation (Water-Filling)

Given a total bit budget $B$ (e.g., $B = 4d$ for average 4 bits/dimension), allocate $b_j$ bits to dimension $j$ to minimize total distortion.

**Quantization distortion for dimension $j$:**
Under uniform scalar quantization with $b_j$ bits, the distortion is approximately:

$$D_j(b_j) = c \cdot \lambda_j \cdot 2^{-2b_j}$$

where $c$ is a constant depending on the distribution (c = π√3/2 for uniform, c ≈ 1.2 for Gaussian).

**Optimization problem:**

$$\min_{\{b_j\}} \sum_{j=1}^d D_j(b_j) \quad \text{s.t.} \quad \sum_{j=1}^d b_j = B, \quad b_j \geq 0$$

**Solution (Lagrange multiplier / water-filling):**

Using KKT conditions:

$$b_j^* = \frac{B}{d} + \frac{1}{2} \log_2\left(\frac{\lambda_j}{\bar{\lambda}_g}\right)$$

where $\bar{\lambda}_g = \left(\prod_{j=1}^d \lambda_j\right)^{1/d}$ is the geometric mean of eigenvalues.

**Interpretation:** Dimensions with above-average variance get more bits; dimensions with below-average variance get fewer bits. The allocation is proportional to $\log_2(\lambda_j)$.

**With integer constraints:** Round $b_j^*$ to nearest integer (or allowable quantization level).

### 2.3 Available Quantization Levels

In llama.cpp, the available cache types map to specific bit widths:
- FP16: 16 bits
- q8_0: 8 bits
- q5_1: 5.5 bits
- q5_0: 5 bits
- q4_1: 4.5 bits
- q4_0: 4 bits

For PCA-adaptive, we can group dimensions into buckets by their optimal bit allocation and assign each bucket a llama.cpp quantization type.

### 2.4 Practical Implementation: Tiered Quantization

Instead of per-dimension bit allocation (impractical), group dimensions into tiers:

**Tier 1 (High energy, top 10-20% of eigenvalues):** q8_0 or FP16
**Tier 2 (Medium energy, middle 30-40%):** q5_0 or q4_1
**Tier 3 (Low energy, bottom 40-50%):** q4_0 or q2 (if available)

Example for Llama-8B K (d=128, 8 KV heads = 1024 total dims):
- Tier 1: top 13 dims × 8 bits = 104 bits
- Tier 2: middle 51 dims × 5 bits = 255 bits
- Tier 3: bottom 64 dims × 3 bits = 192 bits
- Total: 551 bits / 128 dims = **4.3 bits avg** (vs uniform q4_0 = 4.0 bits)

At 4.3 bits avg, we use only 7.5% more bits than q4_0 but the distortion should be significantly lower (approaching optimal allocation).

### 2.5 Attention-Aware Allocation

The standard water-filling minimizes MSE reconstruction error. But for attention, what matters is preserving the softmax ranking: $\text{argmax}_i (q \cdot k_i)$.

The attention-relevant distortion is:

$$D_\text{attn} = \mathbb{E}_{q}\left[\sum_i |\text{rank}(q \cdot k_i) - \text{rank}(q \cdot \hat{k}_i)|\right]$$

This depends on how quantization error projects onto the query direction:

$$\delta(q \cdot k) = q \cdot (k - \hat{k}) = \sum_j q_j \cdot \delta_j$$

where $\delta_j$ is the quantization error in dimension $j$.

If queries also concentrate energy on the high-eigenvalue dimensions of K (which they do, because $Q = W_Q h$ and $K = W_K h$ share the same input $h$), then the query-weighted distortion is even more biased toward high-eigenvalue dimensions — further justifying allocating more bits there.

## 3. Theoretical Distortion Comparison

### 3.1 Uniform q4_0 Distortion

$$D_\text{uniform} = c \sum_{j=1}^d \lambda_j \cdot 2^{-8} = c \cdot 2^{-8} \cdot \text{tr}(\Sigma_K)$$

### 3.2 Optimal Water-Filling Distortion (same avg bits)

$$D_\text{optimal} = d \cdot c \cdot \bar{\lambda}_g \cdot 2^{-2B/d}$$

### 3.3 Ratio (Gain from Adaptive Allocation)

$$\frac{D_\text{uniform}}{D_\text{optimal}} = \frac{\text{tr}(\Sigma_K) / d}{\bar{\lambda}_g} = \frac{\bar{\lambda}_a}{\bar{\lambda}_g}$$

where $\bar{\lambda}_a$ is the arithmetic mean and $\bar{\lambda}_g$ is the geometric mean of eigenvalues.

**By the AM-GM inequality, this ratio is always ≥ 1**, with equality only when all eigenvalues are equal (perfectly uniform distribution).

For Llama-8B K with condition number ~77:
- The ratio $\bar{\lambda}_a / \bar{\lambda}_g$ is potentially very large (depends on full spectrum)
- This means adaptive allocation can reduce distortion by a large factor compared to uniform

### 3.4 Computing the Gain from Our Data

From the eigenspectrum analysis:
- Llama-8B K: σ_max = 18.69, σ_min = 0.046 (layer 0)
- λ_max = 349.3, λ_min = 0.002
- If spectrum roughly follows power law: λ_j ∝ j^{-α}

For power-law spectrum with α > 0:
$$\frac{\bar{\lambda}_a}{\bar{\lambda}_g} = \frac{\frac{1}{d}\sum j^{-\alpha}}{\exp(\frac{1}{d}\sum \ln j^{-\alpha})} = \frac{\bar{j^{-\alpha}}}{\exp(\overline{\ln j^{-\alpha}})}$$

This ratio grows with α (steeper spectrum = more gain from adaptive allocation).

## 4. Implementation Plan

### 4.1 Offline (One-Time per Model)

1. Run calibration data through model (WikiText-2, 1000 tokens)
2. Collect per-layer, per-head K vectors
3. Compute PCA basis P and eigenvalues λ
4. Compute optimal bit allocation per dimension
5. Group into tiers, save tier assignments + P matrix

Storage: P matrix (128×128 × 2 bytes = 32 KB per head) + tier assignments (128 bytes)
Total: ~8 MB for Llama-8B (negligible vs model size)

### 4.2 Runtime

**Encode (during prefill):**
1. k_rotated = P^T · k  (rotate to PCA basis)
2. Quantize each tier with its assigned type
3. Store quantized k_rotated

**Decode (during generation):**
1. Dequantize k_rotated per tier
2. k_approx = P · k_rotated  (rotate back to original basis)
3. Compute attention normally: score = q · k_approx

**Compute overhead:** Two matrix multiplies per token per head (P^T · k and P · k_rotated). Each is d×d matmul = 128² = 16K FLOPs. Negligible compared to attention computation.

### 4.3 llama.cpp Integration

The main challenge is that llama.cpp's KV cache has a single quantization type per layer. Supporting per-dimension tiers requires either:

**Option A:** Pack tiered data into a custom GGML type (e.g., TQKV_PCA)
**Option B:** Store each tier in a separate cache buffer

Option A is cleaner but requires new GGML type implementation. Option B is hacky but faster to prototype.

## 5. Expected Results

| Config | Avg Bits | Distortion (relative) | NIAH | Compression vs FP16 |
|--------|---------|----------------------|------|---------------------|
| FP16 | 16.0 | 1.0 (baseline) | 100% | 1.0× |
| q8_0 uniform | 8.0 | ~0.004 | 100% | 2.0× |
| **PCA + adaptive (4.3 avg)** | 4.3 | ~0.004 (≈ q8_0) | 100%? | **3.7×** |
| q4_0 uniform | 4.0 | ~0.016 | 100% (Llama) | 4.0× |
| q4_0 uniform | 4.0 | catastrophic | 0% (Qwen) | 4.0× |
| **PCA + adaptive (5.2 avg, Qwen)** | 5.2 | ~0.004 | 100%? | **3.1×** |

The key prediction: **PCA + adaptive at 4.3 bits achieves q8_0 quality, saving 46% bits vs q8_0. For Qwen, PCA + adaptive at 5.2 bits may solve the catastrophic q4_0 failure while still achieving 3.1× compression.**

## 6. Relation to Existing Work

- **PALU** (2407.21118): Low-rank KV projection via SVD. Similar PCA idea but they reduce head dimension (d_head → d_head/2), losing information. We rotate but keep all dimensions, varying precision.
- **QuIP#** (weight quantization): Hadamard rotation + incoherence processing for weights. Same spirit (rotate to better basis), different target (weights vs KV cache).
- **SqueezeLLM** (weight quantization): Isolates outliers via sensitivity-weighted clustering. Similar insight but for weights.
- **TurboQuant / TQKV**: WHT rotation + Lloyd-Max quantization for KV cache. Closest existing work — but uses Walsh-Hadamard (data-independent rotation), not PCA (data-dependent rotation).

**Our contribution: PCA-rotated adaptive bit allocation for KV cache, with per-layer optimal allocation guided by eigenspectrum analysis.**
