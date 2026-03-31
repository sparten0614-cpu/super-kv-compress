# Delta Encoding for KV Cache Compression: Theoretical Analysis

**Author:** 宁宁
**Date:** 2026-03-31
**Status:** Hypothesis — awaiting empirical validation

---

## 1. Core Idea

Instead of storing each KV cache entry independently, store the **difference (delta)** from a predicted value. If the prediction is good, the delta has low variance and compresses dramatically.

Three candidate delta dimensions:
- **Layer delta:** $\delta_l = \text{KV}^{(l+1)} - f(\text{KV}^{(l)})$ (adjacent layers)
- **Token delta:** $\delta_t = \text{KV}_t - \text{KV}_{t-1}$ (adjacent tokens within a layer)
- **Head delta:** $\delta_h = \text{KV}^{(h)} - \text{KV}^{(h-1)}$ (adjacent heads within a layer)

The compression gain from delta encoding is:

$$\rho_{\delta} = \frac{\text{Var}[\text{KV}]}{\text{Var}[\delta]} = \frac{\sigma^2}{\sigma_\delta^2}$$

If $\text{cos\_sim} = \rho$ between adjacent entries, then:

$$\sigma_\delta^2 = 2\sigma^2(1 - \rho) \quad \Rightarrow \quad \rho_{\delta} = \frac{1}{2(1-\rho)}$$

| Cosine similarity $\rho$ | Delta variance ratio | Compression gain |
|--------------------------|---------------------|-----------------|
| 0.5 | 1.0 | 1x (no gain) |
| 0.7 | 0.6 | 1.7x |
| 0.9 | 0.2 | 5x |
| 0.95 | 0.1 | 10x |
| 0.99 | 0.02 | 50x |

**Threshold for viability: cos_sim > 0.8 needed for >2.5x gain.**

---

## 2. Layer Delta Analysis

### 2.1 Mathematical Model

For layer $l$, the Key cache is $K^{(l)} = X \cdot W_K^{(l)}$ where $X$ is the (common) input to all layers and $W_K^{(l)}$ is the layer-specific projection.

**Actually, $X$ is NOT common across layers.** Each layer transforms the input:
$$X^{(l+1)} = \text{LayerNorm}(X^{(l)} + \text{Attn}^{(l)}(X^{(l)}) + \text{FFN}^{(l)}(\cdot))$$

So:
$$K^{(l)} = X^{(l)} W_K^{(l)}, \quad K^{(l+1)} = X^{(l+1)} W_K^{(l+1)}$$

The layer delta is:
$$\delta_K^{(l)} = K^{(l+1)} - K^{(l)} = X^{(l+1)} W_K^{(l+1)} - X^{(l)} W_K^{(l)}$$

This involves BOTH different inputs AND different projections. The correlation depends on:
1. How similar $X^{(l+1)}$ is to $X^{(l)}$ (residual connections help — $X^{(l+1)} \approx X^{(l)} + \epsilon$)
2. How similar $W_K^{(l+1)}$ is to $W_K^{(l)}$ (no guarantee — learned independently)

### 2.2 Residual Connection Effect

Thanks to residual connections, $X^{(l+1)} = X^{(l)} + \Delta^{(l)}$ where $\Delta^{(l)}$ is the attention + FFN output. If $\|\Delta^{(l)}\| \ll \|X^{(l)}\|$ (which is true in well-trained transformers due to LayerNorm scaling):

$$K^{(l+1)} = (X^{(l)} + \Delta^{(l)}) W_K^{(l+1)} = X^{(l)} W_K^{(l+1)} + \Delta^{(l)} W_K^{(l+1)}$$

But $W_K^{(l+1)} \neq W_K^{(l)}$ in general, so even with $\Delta^{(l)} = 0$:
$$K^{(l+1)} - K^{(l)} = X^{(l)} (W_K^{(l+1)} - W_K^{(l)})$$

**This is nonzero unless $W_K^{(l+1)} = W_K^{(l)}$.** Standard transformers do NOT share KV projection weights across layers.

### 2.3 Prediction: Layer Delta is Unlikely to Be Small

Unless the model has weight-sharing across layers (like Universal Transformers or ALBERT), different layers project into fundamentally different subspaces. The delta will be $O(\sigma_K)$ — roughly the same magnitude as the original, giving no compression gain.

**Exception: GQA (Grouped Query Attention)**
In GQA, multiple query heads share one KV head. If we predict across the shared dimension, the correlation might be higher within a GQA group. But this is already exploited by GQA's parameter sharing.

### 2.4 Verdict: Layer delta likely NOT viable (cos_sim probably < 0.3)

Awaiting empirical validation from kv_layer_similarity.py.

---

## 3. Token Delta Analysis

### 3.1 Mathematical Model

For adjacent tokens $t$ and $t+1$ at layer $l$:
$$K_t^{(l)} = x_t^{(l)} W_K^{(l)}, \quad K_{t+1}^{(l)} = x_{t+1}^{(l)} W_K^{(l)}$$

The delta is:
$$\delta_{K,t}^{(l)} = K_{t+1}^{(l)} - K_t^{(l)} = (x_{t+1}^{(l)} - x_t^{(l)}) W_K^{(l)}$$

**Key difference from layer delta: SAME projection matrix $W_K^{(l)}$.** The delta depends only on how similar adjacent hidden states are.

### 3.2 Why Adjacent Hidden States are Correlated

1. **Sub-word tokenization:** Adjacent tokens often belong to the same word (e.g., "trans" + "former"). Their embeddings share morphological features.
2. **Residual connections:** After many layers, $x_t^{(l)} \approx x_t^{(0)} + \sum \Delta_t^{(l')}$. The base embedding $x_t^{(0)}$ varies by token, but the accumulated residuals smooth out local differences.
3. **Contextual smoothing:** Attention itself creates smoothing — each position is a weighted average of surrounding positions.

### 3.3 Compression Analysis

If $\text{cos\_sim}(x_t^{(l)}, x_{t+1}^{(l)}) = \rho_x$, then since $W_K^{(l)}$ is a linear projection:

$$\text{cos\_sim}(K_t, K_{t+1}) \geq \rho_x - O(\kappa(W_K^{(l)}) \cdot (1 - \rho_x))$$

where $\kappa(W_K)$ is the condition number of $W_K$. For well-conditioned projections, cosine similarity is approximately preserved.

**Expected range:** $\rho_x \in [0.6, 0.9]$ depending on layer depth and tokenization. Deeper layers should show higher similarity (more contextual smoothing). Early layers may show lower similarity (closer to raw embeddings).

### 3.4 Practical Compression with Token Delta

If average cos_sim ≈ 0.8 across tokens:
- Delta variance: $\sigma_\delta^2 = 2\sigma^2(1-0.8) = 0.4\sigma^2$
- Bits for delta: $R(\sigma_\delta^2) = R(\sigma^2) - \frac{1}{2}\log_2(1/0.4) = R(\sigma^2) - 0.66$ bits/dim
- Saving: ~0.66 bits/dim → at 6 bits/dim baseline, ~11% saving

This is modest. **Token delta is worth ~1.1-2x compression, not 5-10x**, unless cosine similarity is >0.95.

### 3.5 Verdict: Token delta moderately promising (1.5-2x if cos_sim > 0.8)

---

## 4. Head Delta Analysis

### 4.1 Mathematical Model

For GQA, KV heads are already shared across query heads — there's nothing to delta-encode.

For MHA (multi-head attention), different heads use different $W_K^{(h)}$ projections:
$$K^{(h)} = X W_K^{(h)}$$

Same issue as layer delta — different projection matrices → likely low correlation.

### 4.2 Verdict: Head delta likely NOT viable for MHA, N/A for GQA

---

## 5. Alternative: Predictive Coding (Not Simple Delta)

The simple delta $K_{t+1} - K_t$ uses a trivial predictor ($\hat{K}_{t+1} = K_t$). Better predictors could reduce the residual further:

### 5.1 Linear Predictor
$$\hat{K}_{t+1} = A \cdot K_t + b$$

Learned from calibration data. This is essentially AR(1) prediction. If KV values have temporal structure beyond simple similarity, a linear predictor captures it.

**Expected gain:** If $K$ follows an AR(1) process with coefficient $\phi$, the prediction residual variance is $(1-\phi^2)\sigma^2$. For $\phi = 0.8$: residual = $0.36\sigma^2$ → ~2.8x compression of the residual vs storing K directly.

### 5.2 Cross-Layer Predictor
Even though $W_K^{(l+1)} \neq W_K^{(l)}$, there may be a learnable mapping:
$$\hat{K}^{(l+1)} = M \cdot K^{(l)}$$

where $M = W_K^{(l+1)} (W_K^{(l)})^{-1}$ (if $W_K$ is square and invertible, or pseudo-inverse otherwise).

If $M$ is near-identity (layers learn similar features), the residual is small. If not, $M$ captures the transformation and the residual is still smaller than $K^{(l+1)}$ itself.

**This is a learned delta — more like video compression's motion compensation than simple subtraction.**

### 5.3 Practical Consideration
Predictive coding adds computational overhead: the predictor must be evaluated during inference. For a simple linear predictor, this is a matrix-vector multiply — same cost as the original projection. Probably too expensive.

For the trivial predictor (simple subtraction), overhead is negligible — just a vector subtract.

---

## 6. Summary and Recommendations

| Dimension | Expected cos_sim | Delta compression | Viability |
|-----------|-----------------|-------------------|-----------|
| Layer delta | <0.3 (different $W_K$) | ~1x | ❌ Unlikely |
| Token delta | 0.6-0.9 | 1.5-2x | ⚠️ Moderate |
| Head delta | <0.3 (different $W_K^{(h)}$) | ~1x | ❌ Unlikely |
| Linear predictor (token) | N/A | 2-3x | ⚠️ If overhead acceptable |
| Learned cross-layer | N/A | 2-5x | ⚠️ High complexity |

**Honest assessment:** Simple delta encoding is unlikely to deliver the 5-10x originally hypothesized. The 90% correlation assumption was too optimistic for layer and head dimensions due to different projection matrices. Token-level delta is more promising but limited to 1.5-2x.

**The real opportunity may be elsewhere:**
1. **RVQ/Product Quantization** (§12.1 of unified-error-bound.md) — proven 1.66x improvement from closing the scalar-VQ gap, no additional assumptions needed.
2. **Predictive coding with learned predictor** — if calibration overhead is acceptable, 2-3x on top of quantization is possible.

**Priority: Run kv_layer_similarity.py to validate or refute these predictions before investing implementation effort.**
