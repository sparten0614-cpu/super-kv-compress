# Super KV Compress

**Post-training KV cache compression framework targeting 30-50x with quality preservation.**

No one has achieved post-training 50x lossless KV cache compression. We aim to be the first.

## Target

- **Compression:** 30-50x post-training KV cache reduction
- **Quality:** PPL degradation < 1% (hard constraint)
- **Scope:** Any pretrained model, no retraining required
- **Validation:** Multi-model (1B-70B), multi-benchmark (PPL, NIAH, LongBench)

## Three-Layer Architecture

```
Layer 1: Adaptive Asymmetric Quantization
  K=6-bit, V=4-bit (per-layer adaptive, outlier layers FP16)
  Compression: ~3.2x
  Status: Validated (K6V4 +0.52% PPL)

Layer 2: Attention-Aware Eviction
  Token importance → precision mapping:
    high attention   → retain, V=6-bit
    medium attention → retain, V=4-bit
    low attention    → evict (contribution < quantization noise floor)
  Compression: ~10x additional
  Status: Math framework in progress

Layer 3: Sparse V Skip (acceleration)
  Skip dequantization for low-attention V entries
  No compression gain, inference speedup only
  Status: Implemented (阳阳)
```

**Combined target: 3.2x × 10x = 32x (within 30-50x range)**

## Key Insight

Attention weight is the single signal that determines both quantization precision AND eviction:

```
a_t > τ_high  →  retain + high precision (6-bit V)
τ_low < a_t < τ_high  →  retain + low precision (4-bit V)
a_t < τ_low  →  evict (signal below quantization noise floor)
```

Thresholds τ_low and τ_high are derived from quantization error bounds — not heuristics.

## Current State of the Art

| Method | Compression | Quality | Post-training? | Online inference? |
|--------|------------|---------|----------------|-------------------|
| GQA + FP8 | 16x | <0.1% | Architecture | Yes |
| TurboQuant 6-bit | 2.56x | +0.07% PPL | Yes | Yes |
| KVTC (NVIDIA) | 20x | <1 point | Yes | Storage only |
| MLA (DeepSeek) | 28-93x | Lossless | No (retrain) | Yes |
| **Ours (target)** | **30-50x** | **<1% PPL** | **Yes** | **Yes** |

## Validated Results (Phase 1: TurboQuant)

| Model | TQKV_6 vs F16 | NIAH | LongBench v2 |
|-------|--------------|------|-------------|
| TinyLlama 1.1B | +0.04% PPL | — | — |
| Llama-2-7B | +0.09% PPL | — | — |
| Llama-3.1-8B | +0.07% PPL | 100% | 35%=35% (identical) |
| Ministral-8B | +0.18% PPL | 100% | — |
| Qwen2.5-3B (adaptive) | +0.04% PPL | — | — |

## Repository Structure

```
research/
  hypotheses/    — Hypothesis-driven research (H-000, H-001, ...)
  results/       — Experiment data and analysis
  literature/    — Paper surveys and comparisons
docs/
  memory-optimization-directions.md  — 4 research directions
  d4-asymmetric-kv-experiment-plan.md — Asymmetric K/V experiment
```

## Team

- **SZ**: Coordination, deep paper research, QA
- **宁宁**: Math framework, theory, Python validation
- **阳阳**: ggml/Metal engineering, benchmarks

## License

Apache-2.0
