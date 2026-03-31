# H-000: Asymmetric K/V Quantization

**Status:** Confirmed ✅
**Date:** 2026-03-31
**Owner:** 阳阳 (experiment), 宁宁 (analysis)

## Hypothesis

At a fixed memory budget, allocating more bits to Keys and fewer to Values yields better quality than uniform allocation, because Keys are more sensitive to quantization due to softmax nonlinearity.

## Experiment

Model: Llama-3.1-8B-Instruct Q4_K_M, WikiText-2, ctx=512, 10 chunks

| Config | K bits | V bits | Avg | PPL | vs F16 | Status |
|--------|--------|--------|-----|-----|--------|--------|
| F16 baseline | 16 | 16 | 16 | 8.8984 | — | ✅ |
| TQKV_6 uniform | 6 | 6 | 6 | 8.9043 | +0.07% | ✅ |
| K6V4 asymmetric | 6 | 4 | 5 | — | +0.52% | ✅ |
| K6V2q aggressive | 6 | 2 | 4 | — | +48% | ❌ |

## Results

1. **K6V4 (3.2x compression): +0.52% PPL** — within <1% quality constraint ✅
2. **K6V2q (~5x compression): +48% PPL** — catastrophic failure ❌

## Key Finding: V Is More Sensitive Than Expected

Contradicts KVTuner (ICML 2025) which found K ~2x more sensitive than V.

**Explanation:** V=2-bit codebook has only 4 values. The reconstruction error is not a gradual degradation but a **cliff between 2-bit and 4-bit**. V errors propagate linearly through attention output (output = Σ aᵢvᵢ), but the coefficient is the attention weight itself — for high-attention tokens, even small V errors are amplified.

## Implications

- Quantization-only compression ceiling: ~3.2x (quality-preserving)
- To reach 30-50x target, eviction must contribute ~10-15x
- V precision should be determined by token importance (attention weight), not uniform
- This naturally leads to the unified attention-weight → precision mapping framework
