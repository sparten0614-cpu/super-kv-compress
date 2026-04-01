# Angle 9: Attention Distillation — MLP Replaces KV Cache

**Date:** 2026-04-01
**Status:** Active research

---

## 1. The Idea

Instead of storing the KV cache and computing attention at decode time, train a small MLP to approximate the attention function for the current context. Then discard the KV cache and use the MLP for all subsequent decode steps.

### 1.1 Setup

For a single attention head at a single layer, given context of n tokens:
- KV cache: $K \in \mathbb{R}^{n \times d}$, $V \in \mathbb{R}^{n \times d}$ — stored
- At decode, for query $q \in \mathbb{R}^d$:

$$F(q) = \text{softmax}\left(\frac{qK^T}{\sqrt{d}}\right) V \in \mathbb{R}^d$$

$F: \mathbb{R}^d \to \mathbb{R}^d$ is a fixed function once the context is determined.

### 1.2 Replace F with MLP

Train $\hat{F}_\theta(q) \approx F(q)$ where $\theta$ are MLP parameters.

**During prefill:**
1. Compute KV cache normally
2. Generate training data: sample M queries, compute $F(q_i)$ for each
3. Train MLP $\hat{F}_\theta$ via SGD to minimize $\sum_i \|F(q_i) - \hat{F}_\theta(q_i)\|^2$
4. Discard KV cache, keep only $\theta$

**During decode:**
- Each decode step uses $\hat{F}_\theta(q)$ instead of full attention
- No KV cache needed!

### 1.3 Memory Comparison

KV cache: $2nd$ floats per head per layer (K and V)
MLP parameters: depends on architecture

For n=32K, d=128:
- KV cache: 2 × 32768 × 128 × 2 bytes = 16 MB per head per layer
- MLP (3 layers, 256 hidden): ~(128×256 + 256×256 + 256×128) × 2 bytes = ~200 KB

**Compression: 80×**

Total across 8 heads × 32 layers = 256 MLPs × 200 KB = ~51 MB vs 4 GB (FP16 KV) = **80× compression**

## 2. Mathematical Analysis

### 2.1 Universal Approximation

By the universal approximation theorem, a sufficiently wide single-hidden-layer MLP can approximate any continuous function on a compact domain to arbitrary precision.

$F(q)$ is continuous (softmax is infinitely differentiable). On the domain $\{q : \|q\| \leq R\}$ (bounded queries), $F$ is uniformly continuous.

**Guarantee:** For any $\epsilon > 0$, there exists an MLP width $w$ such that $\sup_{\|q\| \leq R} \|F(q) - \hat{F}_\theta(q)\| < \epsilon$.

But: the required width $w$ may be exponential in $d$ for worst-case functions.

### 2.2 Structure of F

$F$ is not a worst-case function — it has specific structure:

$$F(q) = V^T \text{softmax}(Kq/\sqrt{d}) = V^T \sigma(Kq/\sqrt{d})$$

where $\sigma$ is the softmax function.

This is a composition of:
1. Linear map: $q \mapsto Kq/\sqrt{d}$ ($\mathbb{R}^d \to \mathbb{R}^n$) — but $n$ can be huge
2. Softmax: $\mathbb{R}^n \to \Delta^{n-1}$ (probability simplex)
3. Linear map: $\alpha \mapsto V^T\alpha$ ($\mathbb{R}^n \to \mathbb{R}^d$)

The bottleneck is the softmax nonlinearity in high dimensions.

### 2.3 Effective Rank of F

The Jacobian of F is:

$$J_F(q) = \frac{\partial F}{\partial q} = \frac{1}{\sqrt{d}} V^T \text{diag}(\alpha) (I - \alpha \alpha^T) K$$

where $\alpha = \text{softmax}(Kq/\sqrt{d})$.

If attention is sparse (k tokens dominate), then $\alpha$ has at most k significant entries, and:
- $\text{diag}(\alpha)$ has rank k
- $(I - \alpha\alpha^T)$ has rank at most k
- $J_F$ has rank at most $\min(k, d)$

For typical sparse attention with k ≈ 2-3 active tokens, **J_F has rank ~2-3**!

This means F maps from $\mathbb{R}^{128}$ but only varies along ~2-3 directions. An MLP with a 2-3 dimensional bottleneck layer should suffice.

### 2.4 The NIAH Challenge

For NIAH, the attention is maximally sparse: one needle token gets ~100% attention weight. So k=1 and:

$$F(q_\text{needle}) \approx v_\text{needle}$$

The MLP needs to learn: "for queries similar to $k_\text{needle}$, output $v_\text{needle}$."

This is a **one-shot memorization** task. A simple MLP can do this — it just needs to represent a Gaussian bump centered at $k_\text{needle}$ with value $v_\text{needle}$.

But: the MLP must simultaneously represent attention for ALL query types, not just the needle query. The question is interference.

### 2.5 Training Data: Where Do Queries Come From?

At prefill time, we don't know future queries. We need to sample representative queries.

**Option A: Random queries.** Sample $q \sim \mathcal{N}(0, I)$. Covers the space uniformly but may not represent actual decode queries.

**Option B: Model-aware sampling.** Use W_Q weight statistics to sample from the likely query distribution. Better coverage of actual queries.

**Option C: Self-supervised.** Use the context tokens themselves as query sources (each token's Q projection). Guarantees coverage for in-context queries.

**Option D: Online refinement.** Start with a coarse MLP from prefill, refine it during decode as actual queries arrive. But this adds decode latency.

### 2.6 Training Cost

Training the MLP during prefill adds latency. How much?

For M training samples, B SGD steps, MLP with P parameters:
- Forward pass: O(M × P)
- Backward pass: O(M × P)
- Total: O(B × M × P)

With M=1000, B=100, P=100K: 10^10 FLOPs per head per layer
= 10^10 × 256 (heads × layers) = 2.56 × 10^12 total FLOPs

For comparison, full prefill of 32K tokens: O(n² × d × L × H) ≈ 10^13 FLOPs

So MLP training adds ~25% to prefill time. Significant but not prohibitive, especially if decode savings are large.

## 3. Prototype Design

### 3.1 Architecture

```
MLP_attention:
  Input:  q ∈ R^128 (query vector)
  Hidden: ReLU(W1 @ q + b1), W1 ∈ R^{256×128}    — 33K params
  Hidden: ReLU(W2 @ h + b2), W2 ∈ R^{256×256}     — 66K params
  Output: W3 @ h + b3,       W3 ∈ R^{128×256}     — 33K params
  Total:  ~132K params = 264 KB (FP16)
```

### 3.2 Experiment Plan

Phase 1: Synthetic validation
1. Generate random K, V matrices (n=1000, d=128)
2. Define F(q) = softmax(qK^T/√d) V
3. Train MLP to approximate F
4. Measure: approximation error vs MLP size
5. Test with NIAH-like setup: one special K/V pair

Phase 2: Real model validation
1. Extract actual KV cache from llama.cpp inference
2. Train MLP per head per layer
3. Plug back into inference, measure PPL and NIAH

### 3.3 Key Metrics
- MSE: mean squared error of attention output
- Cosine similarity: direction preservation
- Top-1 accuracy: does the MLP identify the same "most attended" token?
- NIAH score: can it retrieve the needle?

## 4. Potential Issues

### 4.1 Per-Context Training
The MLP must be retrained for every new context. This is fundamentally different from normal model inference — it's a meta-learning problem.

### 4.2 Generalization to Unseen Queries
The MLP is trained on sampled queries but must generalize to actual decode queries. If the query distribution at decode time differs significantly from training queries, the MLP will fail.

### 4.3 Error Accumulation
Transformer layers are stacked — errors in early layers propagate and amplify through later layers. Even small MLP approximation errors might compound across 32 layers.

### 4.4 Autoregressive Dependency
In standard transformers, the KV cache grows with each new token. An MLP trained on n tokens' KV cache cannot incorporate token n+1 without retraining. Solutions:
- Retrain MLP every k tokens (amortized cost)
- Use a sliding window: MLP for old context + exact attention for recent tokens
- Online learning: update MLP incrementally per new token

### 4.5 Multi-Head Interference
Each head needs its own MLP (different K/V matrices = different function). 256 separate MLPs is manageable but adds complexity.

## 5. Related Work to Check

- Performer / Random Feature Attention (kernel approximation of softmax)
- Linear attention variants (cosFormer, etc.)
- Hyper-attention (locality-sensitive hashing for attention)
- Knowledge distillation for attention (student-teacher in a single layer)
- Neural Process family (function approximation for meta-learning)
