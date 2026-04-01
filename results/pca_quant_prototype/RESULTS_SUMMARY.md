# PCA-Quant Prototype Results Summary

## TinyLlama-1.1B (CPU, 5 diverse texts, 128 tokens each)

| Config | Mean ΔPPL | Std | PPL Reduction vs q4 |
|--------|----------|-----|-------------------|
| FP16 baseline | — | — | — |
| Uniform q4_0 | +2.20% | ±1.35% | — |
| PCA-Quant (4-bit avg) | +0.82% | ±1.02% | **63%** |

- 4/5 texts show PCA-Quant advantage
- Cosine similarity error reduction: 51.6% (consistent with PPL)
- PCA-Quant has lower variance across texts

## Qwen2.5-3B (GPU, RTX 5880 Ada)

| Config | PPL Effect | NIAH |
|--------|-----------|------|
| FP16 baseline | — | 100% |
| Uniform q4_0 (block-32) | +39.1% | 100% |
| PCA rotation + uniform q4 (block-32) | +5.7% (**85% reduction**) | 80%* |
| PCA rotation only (no quantization) | ~0% | 100% |

*NIAH 80% caused by block-32 cross-dim scale mismatch in PCA space.

## Key Findings

1. **PCA rotation is the primary contributor** to quality improvement, not adaptive bit allocation
2. **Rotation preserves NIAH perfectly** (100%) — the attention structure is invariant under orthogonal transforms
3. **Block-32 quantization in PCA space** introduces scale mismatch: high-eigenvalue and low-eigenvalue dims share block scale, crushing low-variance information → NIAH stuttering artifacts
4. **Qwen benefits more than TinyLlama** (85% vs 63% PPL reduction) — consistent with Qwen's higher AM/GM ratios (7.5-26.8 vs 1.5-2.3)

## Known Limitations

1. **Block-32 scale mismatch**: Standard block quantization groups 32 consecutive PCA dims, but their magnitudes differ by orders of magnitude after PCA rotation. The block scale is dominated by high-variance dims.
2. **Per-dim quantization instability**: Quantizing each PCA dim independently requires enough tokens for stable scale estimation. At 128 tokens, per-dim scales are too noisy.
3. **Adaptive tiering budget constraint**: At avg_bits=4.0 with tiers [4,8,16], zero budget headroom for any upgrade. Sub-4-bit tiers (2-bit) are too aggressive.

## Open Problems

1. **PCA-aware block quantization**: Group PCA dims by eigenvalue magnitude into blocks of similar scale
2. **Longer calibration for per-dim**: Use 1K+ tokens for stable per-dim scale estimation
3. **Finer tier granularity**: 3-bit and 5-bit quantization to enable meaningful adaptive allocation at 4-bit average budget
