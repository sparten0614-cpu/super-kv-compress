# K→V Reconstruction: Mathematical Analysis

**Date:** 2026-04-01
**Purpose:** Investigate whether V can be reconstructed from K, exploiting their shared residual stream origin.

---

## 1. Setup

In a standard transformer layer, for token at position $t$ with residual stream vector $h_t \in \mathbb{R}^{d_\text{model}}$:

$$K_t = W_K h_t, \quad V_t = W_V h_t$$

where $W_K, W_V \in \mathbb{R}^{d_\text{head} \times d_\text{model}}$ are the key and value projection matrices.

**Key dimensions (Llama-3.1-8B):**
- $d_\text{model} = 4096$
- $d_\text{head} = 128$
- $n_\text{kv\_heads} = 8$ (GQA 4:1)
- 32 layers

So $W_K$ and $W_V$ each map from 4096-dim to 128-dim — a massive dimensionality reduction (32×).

## 2. Reconstruction Formula

Given $K_t = W_K h_t$, we want to recover $h_t$ (or enough of it) to compute $V_t = W_V h_t$.

### 2.1 Pseudoinverse Approach

$$\hat{h}_t = W_K^+ K_t = W_K^T (W_K W_K^T)^{-1} K_t$$

This recovers the component of $h_t$ in the row space of $W_K$ (i.e., $\text{col}(W_K^T)$).

Then:
$$\hat{V}_t = W_V \hat{h}_t = W_V W_K^+ K_t = M \cdot K_t$$

where $M = W_V W_K^+ \in \mathbb{R}^{d_\text{head} \times d_\text{head}}$ is a **fixed matrix per layer per head**, computable offline.

### 2.2 Reconstruction Error

The exact V is:
$$V_t = W_V h_t = W_V (P_K h_t + P_K^\perp h_t) = W_V P_K h_t + W_V P_K^\perp h_t$$

where $P_K = W_K^+ W_K$ is the projection onto the row space of $W_K$.

The reconstruction gives: $\hat{V}_t = W_V P_K h_t$

So the error is:
$$\epsilon_t = V_t - \hat{V}_t = W_V P_K^\perp h_t$$

This is the component of $h_t$ that V needs but K doesn't capture — the information in $h_t$ that lies in the **null space of $W_K$** but NOT in the null space of $W_V$.

### 2.3 Error Bound

$$\|\epsilon_t\| = \|W_V P_K^\perp h_t\| \leq \|W_V P_K^\perp\| \cdot \|h_t\|$$

The operator norm $\|W_V P_K^\perp\|$ is a property of the model weights alone — it can be precomputed for every layer and every head.

**If $\text{col}(W_V^T) \subseteq \text{col}(W_K^T)$** (V's row space is contained in K's row space), then $W_V P_K^\perp = 0$ and **reconstruction is exact**.

**If the row spaces are orthogonal**, then $W_V P_K = 0$ and reconstruction is useless.

**In practice:** We expect partial overlap. The key empirical question is: what fraction of V's information lives in K's subspace?

### 2.4 Quantifying Subspace Overlap

Define the **subspace overlap** between $W_K$ and $W_V$:

$$\text{overlap}(\ell, h) = \frac{\|W_V P_K\|_F^2}{\|W_V\|_F^2} = \frac{\sum_i \sigma_i^2(W_V P_K)}{\sum_i \sigma_i^2(W_V)}$$

This gives the fraction of V's "energy" that lies in K's subspace.

- overlap = 1.0: perfect reconstruction
- overlap = 0.5: half the information recoverable
- overlap = 0.0: no information shared

### 2.5 The Reconstruction Matrix M

$M = W_V W_K^+ \in \mathbb{R}^{128 \times 128}$ is a small fixed matrix.

**Storage:** 128 × 128 × 2 bytes = 32 KB per head per layer = 32 × 8 × 32 KB = 8 MB total.
This is negligible compared to the KV cache savings.

**Compute:** One 128×128 matrix-vector multiply per token per head per layer at decode time.
FLOPs: 128² × 2 = 32K FLOPs per head → 32K × 8 heads × 32 layers = 8.2M FLOPs per token.
Compare to full attention: d × n FLOPs per head = 128 × n, so for n > 64K tokens, attention dominates.

## 3. Practical Scheme

### 3.1 Full K→V Reconstruction (aggressive)
- Store only K cache (quantized)
- At decode: V̂ = M · K for each needed position
- Compression: 2× from dropping V, × quantization
- Risk: depends entirely on subspace overlap

### 3.2 Hybrid: K + Residual for V (moderate)
- Store K (quantized) + residual error r_t = V_t - M·K_t (quantized at lower precision)
- At decode: V̂ = M·K + r
- If residual is small, it can be quantized more aggressively (e.g., 2-bit)
- Compression: K at q4_0 (4 bits) + residual at q2 (2 bits) = 6 bits/dim vs 32 bits = 5.3×
- But still stores both, so gain is marginal over just quantizing V

### 3.3 Adaptive per-layer (practical)
- Precompute overlap(ℓ,h) for every layer and head
- Layers with overlap > 0.95: store only K (save 50% on that layer)
- Layers with overlap < 0.95: store both K and V
- Expected: some layers may have high overlap (especially deeper layers where K and V become more correlated)

## 4. Experiment Plan

### 4.1 Extract W_K and W_V from GGUF model
Using llama.cpp or Python with gguf library:
1. Load Llama-3.1-8B-Instruct Q4_K_M
2. For each layer ℓ, head h: extract W_K[ℓ,h] and W_V[ℓ,h]
3. Compute M[ℓ,h] = W_V[ℓ,h] · pinv(W_K[ℓ,h])
4. Compute overlap(ℓ,h)

### 4.2 Validate on actual KV cache
1. Run inference, dump actual K and V vectors
2. Compute V̂ = M · K
3. Measure ‖V - V̂‖ / ‖V‖ per layer per head
4. Correlate with overlap metric

### 4.3 End-to-end PPL test
If overlap is high enough:
1. Modify llama.cpp to reconstruct V from K at decode time
2. Measure PPL and NIAH impact

## 5. Why This Might Work (Intuition)

In many transformer models, K and V projections are learned to extract related but different aspects of the residual stream:
- K captures "what am I about?" (for matching with queries)
- V captures "what do I contribute?" (for output aggregation)

But both need to understand the token's role in context. In deeper layers, representations become more abstract and K/V may converge toward encoding similar information.

**GQA provides additional evidence:** In GQA, K and V heads are shared across query heads, meaning the model already learns K and V to be "general purpose" representations. This generality may increase their mutual information.

## 6. Relation to Existing Work

- **MQA (Multi-Query Attention):** Uses a single KV head for all query heads — reduces KV count but still stores both K and V
- **GQA:** Groups of query heads share KV heads — same
- **PALU:** Low-rank K/V projection — reduces d_head but still stores both
- **KV-Direct:** Stores residual stream checkpoint — replaces both K and V with h
- **Slim Attention (2503.05840):** V = K · W_KV where W_KV = W_K⁻¹ · W_V — **exact for MHA, does NOT work for GQA/MQA**
- **Our approach:** Pseudoinverse generalization for GQA — DISPROVEN by experiment

## 7. Experimental Results — DISPROVEN

### 7.1 Subspace Overlap (2026-04-01)

Extracted W_K and W_V from GGUF models, computed overlap = ||W_V P_K||²_F / ||W_V||²_F.

**Llama-3.1-8B (32 layers × 8 KV heads = 256):**
- Mean overlap: 12.9%, Median: 11.5%, Max: 35.9%
- 0/256 heads with overlap > 50%

**Qwen2.5-3B (36 layers × 2 KV heads = 72):**
- Mean overlap: 19.4%, Median: 18.3%, Max: 40.1%
- 0/72 heads with overlap > 50%

**Random baseline:** 3.1% (128/4096)

### 7.2 Interpretation

K and V subspaces are nearly orthogonal — V's information mostly lives outside K's subspace. This is expected: K learns "what matches queries" (softmax compatibility), V learns "what to contribute" (output content). Training specializes them into different subspaces.

### 7.3 Prior Art: Slim Attention

Slim Attention (March 2025) already solved this for MHA models (V = K · W_KV, exact). But it explicitly states it does NOT work for GQA/MQA — our experiment confirms why: the pseudoinverse approach fails because W_K is not square and subspace overlap is too low.

### 7.4 Conclusion

**K→V reconstruction is NOT viable for modern GQA/MQA models.** This validates KV-Direct's approach of storing the full residual stream h (which contains ALL information for both K and V).
