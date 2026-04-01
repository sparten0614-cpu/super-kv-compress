# Rate-Distortion Analysis: Theoretical Compression Bounds

**Date:** 2026-04-01

## Per-Layer Analysis (1% distortion tolerance)

### Llama-3.1-8B (32 layers, 8 KV heads, d_head=128)

| Quantization | Bits/token/layer | Theoretical Min | Compression vs FP16 | Headroom vs Theory |
|-------------|-----------------|-----------------|---------------------|-------------------|
| FP16 (K+V)  | 32,768          | 2,957           | 1.0×                | 11.1×             |
| q8_0 (K+V)  | 16,384          | 2,957           | 2.0×                | 5.5×              |
| q4_0 (K+V)  | 8,192           | 2,957           | 4.0×                | 2.8×              |
| q4_0 K + q8_0 V | 6,144       | 2,957           | 5.3×                | 2.1×              |
| q8_0 K + q4_0 V | 6,144       | 2,957           | 5.3×                | 2.1×              |
| Theoretical  | 2,957           | 2,957           | 11.1×               | 1.0×              |

### Qwen2.5-3B (36 layers, 2 KV heads, d_head=128)

| Quantization | Bits/token/layer | Theoretical Min | Compression vs FP16 | Headroom vs Theory |
|-------------|-----------------|-----------------|---------------------|-------------------|
| FP16 (K+V)  | 8,192           | 1,072           | 1.0×                | 7.6×              |
| q8_0 (K+V)  | 4,096           | 1,072           | 2.0×                | 3.8×              |
| q4_0 (K+V)  | 2,048           | 1,072           | 4.0×                | 1.9×              |
| Theoretical  | 1,072           | 1,072           | 7.6×                | 1.0×              |

## Total Across All Layers

### Llama-3.1-8B

| Quantization | Bytes/token | Theoretical Min | Compression | Headroom |
|-------------|-------------|-----------------|-------------|----------|
| FP16        | 131,072     | 11,827          | 1.0×        | 11.1×    |
| q8_0        | 65,536      | 11,827          | 2.0×        | 5.5×     |
| q4_0        | 32,768      | 11,827          | 4.0×        | 2.8×     |
| q8_0/q4_0   | 24,576      | 11,827          | 5.3×        | 2.1×     |
| Theory      | 11,827      | 11,827          | 11.1×       | 1.0×     |

### Qwen2.5-3B

| Quantization | Bytes/token | Theoretical Min | Compression | Headroom |
|-------------|-------------|-----------------|-------------|----------|
| FP16        | 36,864      | 4,824           | 1.0×        | 7.6×     |
| q8_0        | 18,432      | 4,824           | 2.0×        | 3.8×     |
| q4_0        | 9,216       | 4,824           | 4.0×        | 1.9×     |
| Theory      | 4,824       | 4,824           | 7.6×        | 1.0×     |

## Key Insights

1. **Llama has more headroom than Qwen** (2.8× vs 1.9× from q4_0). This explains why Llama tolerates aggressive quantization while Qwen doesn't — Qwen is already closer to its information-theoretic limit.

2. **The 2.8× gap in Llama** means there's room for ~2.8× more compression beyond q4_0 without crossing the theoretical distortion limit. PCA-Quant + non-uniform bit allocation could capture this.

3. **Qwen's 1.9× gap** is much smaller — Qwen's high GQA ratio (7:1) means each KV head carries more information per dimension. There's less "waste" to compress away.

4. **Asymmetric K/V quantization** (q8_0/q4_0) is already 5.3× — only 2.1× from theory. This validates that our asymmetric approach is near-optimal for uniform quantization.

## Method

Theoretical minimum computed via Gaussian rate-distortion: R(D) = 0.5 × Σ log2(λ_i / D) for λ_i > D, where λ_i are squared singular values of the projection weight matrices W_K and W_V, and D = 0.01 × λ_max (1% distortion threshold).

Caveat: This uses W_K/W_V SVD as proxy for actual KV cache covariance. The true bound depends on input distribution (Σ_K = W_K Σ_x W_K^T). Empirical analysis on actual K vectors may show different (likely tighter) bounds.
