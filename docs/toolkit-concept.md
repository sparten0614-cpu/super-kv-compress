# GQA-Aware KV Cache Compression Toolkit — Concept

**Status:** Concept draft
**Date:** 2026-04-01

---

## Vision

A practical, pip-installable toolkit that automatically selects the best KV cache compression configuration for any model, based on architecture analysis (GQA ratio, head dimensions, outlier detection).

**Name candidates:** `kvcache-toolkit`, `kvcompress`, `gqa-compress`, `smartkv`

## Core Components

### 1. Model Analyzer (`analyze`)
```bash
kvcompress analyze meta-llama/Llama-3.1-8B-Instruct
```
Output:
```
Architecture: GQA 4:1 (n_kv=8, n_q=32, head_dim=128)
Outlier layers: None detected
Recommended K precision: q4_0 (safe)
Recommended V precision: q4_0 (safe)
Expected compression: 4x
Expected PPL delta: +3.1%
NIAH safety: HIGH
```

### 2. Configuration Generator (`config`)
```bash
kvcompress config --model qwen2.5-7b --target-compression 2.5x --niah-safe
```
Output:
```yaml
k_type: q8_0
v_type: q4_0
skip_layers: [0]  # outlier detected (K_max=93)
compression: 2.5x
estimated_ppl_delta: -1.5%
```

### 3. Benchmark Runner (`bench`)
```bash
kvcompress bench --model llama-3.1-8b --configs auto --metrics ppl,niah
```
Runs the full diagnostic matrix (f16, q8, q4, asymmetric K/V, eviction gradients) and produces a report.

### 4. Integration Adapters
- **llama.cpp:** Generate `--cache-type-k` / `--cache-type-v` flags + build config
- **vLLM:** When asymmetric support lands, generate config
- **HuggingFace:** Monkey-patch KV cache quantization in transformers

## Key Algorithms

### GQA-Aware Bit Floor
```python
def min_k_bits(n_kv_heads, n_q_heads, outlier_max):
    gqa_ratio = n_q_heads / n_kv_heads
    if gqa_ratio >= 7 or outlier_max > 50:
        return 8  # q8_0 minimum for K
    elif gqa_ratio >= 4:
        return 4  # q4_0 safe for K
    else:  # MHA
        return 4
        
def min_v_bits(n_kv_heads):
    return 4  # V is universally robust to q4_0
```

### Outlier Layer Detection
```python
def detect_outlier_layers(model, calibration_data, threshold=50):
    """Run calibration, find layers where K_max > threshold."""
    outliers = []
    for layer_idx, layer in enumerate(model.layers):
        k_proj = layer.self_attn.k_proj
        k_max = compute_activation_max(k_proj, calibration_data)
        if k_max > threshold:
            outliers.append((layer_idx, k_max))
    return outliers
```

## Data Assets (from our experiments)

The toolkit ships with pre-computed profiles for common models:

| Model | GQA Ratio | K Floor | V Floor | Outlier Layers | Optimal Config |
|-------|-----------|---------|---------|----------------|----------------|
| Llama-3.1-8B | 4:1 | q4_0 | q4_0 | None | K4V4 (4x) |
| Llama-3.3-70B | 8:1 | q4_0 | q4_0 | None | K4V4 (4x) |
| Mistral-7B | 4:1 | q4_0 | q4_0 | None | K4V4 (4x) |
| Qwen2.5-7B | 7:1 | q8_0 | q4_0 | Layer 0 (K=93) | K8V4 (2.5x) |
| Ministral-8B | 4:1 | q4_0 | q4_0 | None | K4V4 (4x) |

## Differentiation from Existing Tools

| Feature | Our Toolkit | llama.cpp | vLLM | KIVI |
|---------|-------------|-----------|------|------|
| GQA-aware config | **Yes** | Manual | No | No |
| Outlier detection | **Yes** | No | No | No |
| Asymmetric K/V | **Yes** | Yes (manual) | No | No |
| Per-layer config | **Yes** | No (uniform) | No | Yes |
| Auto-benchmark | **Yes** | No | No | No |
| NIAH safety check | **Yes** | No | No | No |

## Implementation Plan

1. **Phase 1 (1 week):** Model analyzer + config generator (pure Python, reads HF model configs)
2. **Phase 2 (1 week):** Benchmark runner (wraps llama.cpp perplexity + NIAH scripts)
3. **Phase 3 (2 weeks):** HuggingFace integration + web demo
4. **Phase 4:** vLLM integration when asymmetric support lands

## Release Strategy

- GitHub: open source (Apache 2.0)
- PyPI: `pip install kvcompress`
- Paper companion: reference implementation for all experiments
- Blog post: "Stop Using Uniform KV Cache Quantization"
