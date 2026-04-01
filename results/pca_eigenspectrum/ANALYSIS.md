# PCA Eigenspectrum Analysis: Actual KV Cache vs W_K SVD Proxy

## Summary

We compare the AM/GM ratio (arithmetic mean / geometric mean of eigenvalues) between:
1. **Actual KV cache** vectors extracted during inference (empirical covariance eigenspectrum)
2. **W_K SVD proxy** — singular value spectrum of the K projection weight matrix

AM/GM ratio determines the theoretical coding gain from PCA-based adaptive bit allocation via water-filling: `coding_gain = 0.5 * log2(AM/GM)` bits.

## Key Findings

### 1. Proxy validity depends on layer type

| Layer Type | Actual K AM/GM | W_K SVD AM/GM | Proxy Error | Verdict |
|-----------|---------------|---------------|-------------|---------|
| Normal (L2-21) | ~1.85 avg | ~1.80 avg | <10% | **Proxy valid** |
| Semi-outlier (L1) | 3.78 | 11.33 | +200% | Proxy overestimates |
| Extreme outlier (L0) | 56.8 | 148.2 | +161% | Proxy overestimates |

### 2. Why proxy overestimates outlier layers

W_K weight matrix absorbs pre-LayerNorm scaling. During inference, LayerNorm normalizes the input activation before K projection, compressing the dynamic range. Outlier layers have extreme weight singular values that are partially cancelled by this normalization — the actual KV cache eigenspectrum is less extreme than the weight matrix predicts.

### 3. V cache is robust — PCA unnecessary

| Cache | Avg AM/GM (excl L0) | Coding Gain | Implication |
|-------|-------------------|-------------|-------------|
| K | 1.97 | +0.49 bits | PCA beneficial |
| V | 1.43 | +0.26 bits | Uniform quant sufficient |
| K/V ratio | 1.38x | — | K is the bottleneck |

### 4. Correct paper claim

**Wrong:** "W_K SVD AM/GM is a conservative lower bound for actual coding gain"  
**Correct:** "W_K SVD is a reliable proxy for normal layers (±10% error), but overestimates outlier layers due to LayerNorm absorption of activation dynamic range"

## Data Files

| File | Description |
|------|-------------|
| `eigenspectrum_full.json` | TinyLlama per-layer, per-head full eigenvalue spectrum (K + V) |
| `eigenspectrum_summary.json` | Compression ratios and condition numbers summary |
| `kv_amgm_ratios.json` | Per-layer K and V AM/GM ratios with coding gain |
| `wk_svd_amgm.json` | W_K SVD proxy AM/GM per layer |
| `actual_vs_proxy_scatter.png` | Appendix figure: proxy vs actual scatter + bar chart |

## Model

TinyLlama-1.1B-Chat-v1.0, 22 layers, 4 KV heads, head_dim=64, seq_len=395 tokens.
