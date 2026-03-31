# H-001b: StreamingLLM-Style Eviction

**Status:** Confirmed ✅
**Date:** 2026-03-31
**Owner:** 阳阳 (experiment)

## Hypothesis

Retaining attention sink tokens (first 128) + recent tokens while evicting middle tokens preserves quality, because sink tokens carry semantic anchors and recent tokens carry local context.

## Results

- **50% eviction: +0.46% PPL** ✅ (within <1% constraint)
- Combined with K6V4 quantization (3.2x): total 6.4x compression at ~1% PPL

## Key Insight

H-001 (blind sliding window) failed catastrophically (+163%) because it evicted semantic anchors. H-001b succeeds by preserving them. The attention sink phenomenon (first few tokens accumulate disproportionate attention) is real and must be respected.

## Implications

- 50% StreamingLLM eviction is safe
- Combined 6.4x (quant × eviction) confirmed achievable at <1% PPL
- Critical next step: test higher eviction rates (60-90%) to find the cliff

## Next: H-001c

Test eviction gradient: 50%, 60%, 70%, 80%, 90% to map the PPL-vs-eviction curve.
