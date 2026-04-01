# Novel KV Cache Compression: Mathematical Explorations

**Date:** 2026-04-01
**Status:** Active research — angles 2, 3, 4 from first-principles mathematical analysis

---

## Angle 2: Frequency-Domain Compression Along Position Axis

### 2.1 Core Insight

The KV cache for a single head is a matrix $K \in \mathbb{R}^{n \times d}$ where $n$ = sequence length, $d$ = head dimension. Most compression treats each row (token) independently. But there is structure **across positions** — adjacent tokens often have similar representations because:

1. Natural language has local coherence (sentences about the same topic)
2. Residual connections ensure gradual representation changes
3. RoPE adds smooth positional structure

This suggests the KV cache is **smooth along the position axis** and compressible in the frequency domain.

### 2.2 Mathematical Formulation

Apply the Discrete Cosine Transform (DCT) along the position axis:

$$\tilde{K} = \text{DCT}(K, \text{axis}=0) \in \mathbb{R}^{n \times d}$$

The DCT decorrelates the signal. If K is smooth, energy concentrates in low-frequency coefficients.

**Compression:** Keep only the first $m$ frequency coefficients:

$$\tilde{K}_m = \tilde{K}[0:m, :] \in \mathbb{R}^{m \times d}$$

**Reconstruction:** $\hat{K} = \text{IDCT}(\text{pad}(\tilde{K}_m, n), \text{axis}=0)$

**Compression ratio:** $n/m$ (e.g., keep 25% of coefficients = 4× compression)

### 2.3 Attention with Frequency-Domain KV

At decode time, query $q \in \mathbb{R}^d$ needs to compute:

$$\text{attn}(q) = \text{softmax}(qK^T / \sqrt{d}) \cdot V$$

Can this be done efficiently in the frequency domain?

$$qK^T = q \cdot \text{IDCT}(\tilde{K}_m)^T$$

The IDCT is a linear operation: $\hat{K} = C^T \tilde{K}_m$ where $C$ is the DCT matrix.

$$q \hat{K}^T = q (\tilde{K}_m^T C) = (q \cdot C^T_m) \cdot \tilde{K}_m^T$$

Wait — this doesn't simplify because we need the full $n$-length attention vector before softmax.

**Alternative:** Reconstruct K fully, then compute attention normally. The IDCT costs O(n·d·log n) which is comparable to the O(n·d) attention itself. So no speedup, only memory savings.

### 2.4 Critical Issue: Autoregressive Incompatibility

In autoregressive decoding, the KV cache grows by one token per step. DCT requires the complete signal — it cannot be efficiently updated incrementally.

**Workarounds:**
1. **Block DCT:** Divide the sequence into blocks of B tokens. DCT within each block. Supports incremental: only the current block needs updating.
2. **Sliding window DCT:** Apply DCT to a sliding window of recent tokens + compressed older tokens.
3. **Wavelet transform:** Wavelets support multi-resolution and are more amenable to incremental updates than DCT.

### 2.5 Orthogonality with Quantization

Key advantage: frequency-domain compression is **orthogonal** to quantization.

Pipeline: Tokens → KV cache → DCT (compress positions) → Quantize (compress precision)

Combined: 4× (position) × 4× (quantization) = **16× compression**

### 2.6 Potential Issues

1. **RoPE interaction:** RoPE adds position-dependent rotation to K vectors. This creates high-frequency components in the position axis, potentially reducing DCT compressibility.
2. **Attention spikes:** If attention is concentrated on specific positions (sink tokens, needle), their DCT representation requires all frequencies — compression would blur these crucial positions.
3. **The NIAH problem:** A single needle token at position $p$ has energy spread across all frequencies in DCT domain. Truncating high frequencies would blur the needle, degrading retrieval. This is the same fundamental issue as eviction.

### 2.7 Verdict

**Partially promising, but NIAH-incompatible at high compression.** The same information-theoretic limitation applies: compressing the position axis necessarily loses position-specific information, which is exactly what NIAH tests. At low compression (2×), it might work. At high compression (4×+), it will exhibit the same NIAH degradation as eviction.

**The fundamental issue:** Any compression along the position axis is a form of "soft eviction" — instead of hard-deleting tokens, you blur them. The result for NIAH is the same.

**Worth experimenting:** Measure the actual spectral energy distribution of KV caches to quantify compressibility. Even if NIAH-incompatible, combined with selective recompute it could work (compress everything, recompute the needle when found).

---

## Angle 3: Attention Function Compression

### 3.1 Core Insight

At decode time, we don't need K and V individually. We need the **attention function**:

$$F(q) = \text{softmax}\left(\frac{qK^T}{\sqrt{d}}\right) V$$

This is a mapping $F: \mathbb{R}^d \to \mathbb{R}^d$ from query space to output space.

If this function has low effective dimensionality, we can store a compressed representation of $F$ instead of storing $K$ and $V$ separately.

### 3.2 Analysis of the Function Space

$F(q)$ is a weighted sum of value vectors:

$$F(q) = \sum_{i=1}^{n} \alpha_i(q) \cdot v_i, \quad \alpha_i(q) = \frac{\exp(q \cdot k_i / \sqrt{d})}{\sum_j \exp(q \cdot k_j / \sqrt{d})}$$

**Properties:**
- $F$ is continuous and infinitely differentiable (softmax is smooth)
- $F(q) \in \text{conv}(v_1, \ldots, v_n)$ — the output always lies in the convex hull of value vectors
- The Jacobian $\nabla_q F$ encodes how the output changes with the query

### 3.3 Low-Rank Approximation

If attention is sparse (weights concentrate on k tokens), then:

$$F(q) \approx \sum_{i \in S(q)} \alpha_i(q) \cdot v_i, \quad |S(q)| = k \ll n$$

But $S(q)$ varies with $q$ — the "important" tokens change per query. This is the same query-dependence that kills eviction.

**Alternative: Static approximation.** If we cluster the key vectors into $m$ centroids $\{c_1, \ldots, c_m\}$ and precompute the average value for each cluster:

$$\bar{v}_j = \frac{\sum_{i \in C_j} v_i}{|C_j|}$$

Then: $\hat{F}(q) = \text{softmax}(q \cdot [c_1, \ldots, c_m]^T / \sqrt{d}) \cdot [\bar{v}_1, \ldots, \bar{v}_m]$

This reduces storage from $(K \in \mathbb{R}^{n \times d}, V \in \mathbb{R}^{n \times d})$ to $(C \in \mathbb{R}^{m \times d}, \bar{V} \in \mathbb{R}^{m \times d})$.

**Compression ratio:** $n/m$ (same as position compression)

### 3.4 When This Works and Fails

**Works:** When tokens form natural clusters (e.g., tokens in a paragraph are semantically similar, so their K vectors are close).

**Fails:** When individual token identity matters — specifically NIAH. The needle is a unique point that cannot be merged into any cluster without information loss.

### 3.5 Two-Level Hierarchy

A more sophisticated approach:

1. **Level 1 (coarse):** m centroids + average values (always in GPU memory)
2. **Level 2 (fine):** Full K/V cache (in CPU memory or quantized on GPU)

At decode time:
1. Query matches against Level 1 centroids → identify promising clusters
2. Load Level 2 entries for those clusters only
3. Compute exact attention on the loaded subset

This is essentially what Quest/RetrievalAttention do, but with a learned clustering structure.

### 3.6 Mathematical Connection to Kernel Methods

The attention function $F(q)$ can be viewed through the lens of **kernel methods**:

$$F(q) = \sum_i \alpha_i(q) v_i = V^T \alpha(q)$$

where $\alpha(q) = \text{softmax}(Kq/\sqrt{d})$ is a position in the probability simplex.

The mapping $q \mapsto \alpha(q)$ is a **softmax kernel**. If this kernel has low effective rank (attention distributions are similar for many queries), then $F$ can be approximated by a low-rank matrix.

Specifically, if we compute the SVD of the "query-to-attention" mapping over a representative set of queries, we can identify the principal modes of attention variation and discard the rest.

### 3.7 Verdict

**Theoretically elegant, practically equivalent to existing sparse attention methods.** The two-level hierarchy is essentially Quest/RetrievalAttention with extra steps. The direct function compression via clustering faces the same NIAH issue as eviction.

**Novel contribution potential:** The kernel-method perspective is new and could yield theoretical insights about attention compressibility bounds.

---

## Angle 4: Information-Theoretic Lower Bound

### 4.1 The Fundamental Question

Given that we want the output of attention to match the full-precision output within tolerance $\epsilon$, what is the **minimum number of bits** required to represent the KV cache?

This is Shannon's rate-distortion problem applied to attention.

### 4.2 Formal Setup

**Source:** The KV cache $(K, V) \in \mathbb{R}^{n \times d} \times \mathbb{R}^{n \times d}$ generated by a specific model on a specific input.

**Distortion measure:** For a query distribution $Q$, the distortion is:

$$D(\hat{K}, \hat{V}) = \mathbb{E}_{q \sim Q}\left[\left\|F(q; K, V) - F(q; \hat{K}, \hat{V})\right\|^2\right]$$

where $F(q; K, V) = \text{softmax}(qK^T/\sqrt{d})V$.

**Rate-distortion function:**

$$R(D) = \min_{\hat{K}, \hat{V}: D(\hat{K},\hat{V}) \leq D} I((K,V); (\hat{K},\hat{V}))$$

This gives the minimum bits needed to represent the KV cache such that attention output distortion stays below $D$.

### 4.3 The Gaussian Lower Bound

If $(K, V)$ entries are approximately Gaussian (reasonable after layer normalization), we can use the Gaussian rate-distortion function:

$$R(D) = \frac{1}{2} \sum_{i: \lambda_i > D} \log_2\left(\frac{\lambda_i}{D}\right)$$

where $\lambda_i$ are the eigenvalues of the covariance matrix of $(K, V)$ entries.

This tells us: **the number of "significant" eigenvalues determines the compressibility.**

### 4.4 Empirical Approach

We can measure this:

1. Run inference on a corpus, collect KV cache entries
2. Compute the covariance matrix of the KV cache (or per-layer)
3. Compute eigenvalue spectrum
4. Apply the Gaussian rate-distortion formula
5. Compare theoretical minimum bits vs. current methods (q4_0 = 4 bits/entry)

If the eigenvalue spectrum decays rapidly (many small eigenvalues), there's a large gap between current methods and the theoretical minimum — meaning much better compression is possible.

If the spectrum is flat (all eigenvalues similar), current methods are near-optimal.

### 4.5 Attention-Aware Distortion

The standard rate-distortion treats all entries equally. But attention is selective — some entries matter enormously (high attention weight), others are irrelevant.

**Attention-weighted distortion:**

$$D_\text{attn}(\hat{K}, \hat{V}) = \mathbb{E}_{q}\left[\sum_i \alpha_i(q) \left\|v_i - \hat{v}_i\right\|^2 + \beta \sum_i \left|\alpha_i(q) - \hat{\alpha}_i(q)\right|\right]$$

The first term weights V errors by attention weight. The second term penalizes attention distribution distortion.

Tokens with low attention weight (most tokens most of the time) can be compressed more aggressively. Tokens with high attention weight need high precision. This is exactly the intuition behind eviction — but the rate-distortion framework makes it rigorous.

### 4.6 The NIAH Information Content

A single needle token at position $p$ contains $\log_2(n)$ bits of positional information (which token is it among $n$) plus $d \cdot b$ bits of content information (the value vector at $b$ bits precision).

For NIAH to succeed at 100%, the compressed cache must preserve:
1. The ability to distinguish position $p$ from all others (requires preserving the key vector's uniqueness)
2. The full content at position $p$ (requires preserving the value vector)

**Minimum bits for NIAH:** At least $\log_2(n) + d \cdot b_\text{min}$ bits must be preserved for the needle position, where $b_\text{min}$ is the minimum bits for meaningful value reconstruction.

### 4.7 The Compression Hierarchy

This analysis suggests a natural hierarchy:

| Bits/token | What's preserved | Method |
|-----------|-----------------|--------|
| $d \times 16$ | Everything | FP16 (no compression) |
| $d \times 8$ | Most info | q8_0 (2×) |
| $d \times 4$ | Core info (GQA 4:1) | q4_0 (4×) |
| $d \times 2$ | Rough direction | q2 (8×) — too lossy? |
| $\log_2(n)$ | Only position | Token ID (extreme) |
| 0 | Nothing | Eviction |

The gap between "token ID" ($\log_2(n) \approx 15$ bits for 32K context) and q4_0 ($d \times 4 = 512$ bits) is enormous — **34× gap**. This is the space that KV-Direct operates in: residual checkpoint ≈ $d_\text{model} \times b \approx 4096 \times 4 = 16384$ bits, but shared across all layers.

### 4.8 Novel Bound: Layer-Sharing Entropy

Since KV-Direct stores one vector per token shared across all layers, the key question is: **how much independent information does each layer add?**

If layers are highly correlated (similar K/V patterns), then per-layer storage is wasteful. If layers are independent, per-layer storage is necessary.

We can quantify this with the **conditional entropy:**

$$H(K^{(\ell)} | K^{(\ell-1)}, \ldots, K^{(1)}) \text{ for layer } \ell$$

If this conditional entropy is small, later layers' KV caches are predictable from earlier layers — supporting cross-layer compression (CLA, LCKV approaches) or residual stream checkpointing.

### 4.9 Experiment Plan

1. **Eigenvalue spectrum:** Extract KV cache from model, compute per-layer covariance eigenvalues, plot spectrum
2. **Cross-layer MI:** Compute mutual information between layers' KV caches
3. **Rate-distortion curve:** Vary quantization precision, measure actual distortion vs. theoretical Gaussian bound
4. **NIAH-aware bound:** Compute minimum bits needed to preserve needle distinguishability

### 4.10 Verdict

**High academic value, paper-worthy.** No one has computed the information-theoretic lower bound for KV cache compression. If we show a 10× gap between current methods and the bound, it motivates an entire research agenda. Even a 2× gap is significant.

**This could be a standalone paper or a strong addition to our current paper's theoretical section.**

---

## Summary and Priority

| Angle | Feasibility | Novelty | Paper Value | Priority |
|-------|------------|---------|-------------|----------|
| 1. K→V Reconstruction | ❌ Disproven | Low (Slim Attention exists) | Supplementary | Done |
| 2. Frequency-Domain | ⚠️ NIAH-limited | Medium | Medium | Experiment |
| 3. Attention Function | ⚠️ Reduces to existing | Medium-High (kernel perspective) | Medium | Theoretical |
| 4. Info-Theory Bound | ✅ Feasible | **High** | **High** | **Priority** |

**Recommended order:** 4 (info-theory bound) → 2 (frequency experiment) → 3 (theoretical framework).

---

## Angles 5-10: Wild Brainstorming (2026-04-01)

See telegram conversation for full descriptions. Key angles:

- **Angle 5 (Manifold learning):** K vectors may live on a low-dim manifold. Store manifold coordinates.
- **Angle 6 (Holographic/SVD):** Store U,Σ,V^T decomposition of KV matrix.
- **Angle 7 (Thermodynamic analogy):** d↑ → effective temperature↓ → attention more sparse → more compressible. Explains why large models tolerate more eviction.
- **Angle 8 (Error-correcting codes):** Add redundancy (syndrome) to detect/correct quantization errors.
- **Angle 9 (Attention distillation):** Per-context MLP replaces KV cache. **ALREADY EXISTS as TTT (Test-Time Training, ICML 2025).** Synthetic validation positive (CosSim 0.99, NIAH 5/5), but not novel.
- **Angle 10 (Periodic KV reuse):** Detect periodicity in KV cache, store one period.

## Literature Discoveries (2026-04-01)

### Attention Matching (arxiv 2602.16284, Feb 2026)
- Closed-form least squares to fit compact KV pairs
- **50x compression in seconds** (vs MLP training in minutes)
- Most practical approach found

### KVSculpt (arxiv 2603.27819, March 2026)
- L-BFGS optimization for keys + ridge regression for values
- Per-context optimization

### TTT — Test-Time Training (Sun et al., ICML 2025)
- Per-context MLP training = our Angle 9
- 2.7x speedup at 128K, 35x at 2M context

### LESS (ICML 2024)
- Tiny MLPs (<2% params) at each attention layer
- Compresses evicted KV pairs into constant-size learned cache

### LoLCATs (ICLR 2025)
- Post-hoc linearization of softmax attention
- Applied to 70B and 405B models

## Updated Priority (after literature review)

| Direction | Status | Next Step |
|-----------|--------|-----------|
| Attention Matching + our quantization | 🆕 Most promising | Read paper, prototype |
| PCA-Quant (from eigenspectrum) | Feasible | Prototype on real model |
| Info-theory lower bound for paper | Ready | Write into paper theory section |
| K→V reconstruction | ❌ Disproven | Archive |
| MLP attention distillation | ❌ Not novel (TTT exists) | Archive |
