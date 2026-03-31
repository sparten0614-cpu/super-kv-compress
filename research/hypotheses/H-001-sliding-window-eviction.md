# H-001: Sliding Window Eviction

**Status:** Refuted ❌
**Date:** 2026-03-31
**Owner:** 阳阳 (implementation + experiment)

## Hypothesis

A sliding window eviction strategy can safely discard KV entries for tokens outside the recent window, achieving significant memory reduction with minimal quality impact. Combined with quantization (Layer 1), this can push total compression beyond 10x.

## Experiment Design

- Model: Llama-3.1-8B-Instruct Q4_K_M
- Context: 4096 tokens
- 5 eviction points: retain 100%, 50%, 25%, 12.5%, 6.25% of KV entries
- Baseline: full KV (no eviction)
- Metric: PPL, NIAH retrieval accuracy

## Expected Outcome

- 50% retention (2x eviction): <0.5% PPL increase
- 25% retention (4x eviction): <2% PPL increase
- Combined with K6V4 (3.2x): 25% retention → 3.2 × 4 = 12.8x total

## Results (2026-03-31)

**Catastrophic failure.** 10% eviction → PPL +163%.

### Root Cause
Oldest tokens contain semantic anchors (titles, subjects, discourse markers). Blind temporal eviction removes exactly the tokens with highest long-range attention weight — the worst possible eviction strategy.

### Implications
- Time-based eviction is fundamentally flawed for PPL evaluation
- PPL uses geometric mean — even one badly predicted token from missing context spikes the metric
- Attention-aware eviction (H-001c) may still work, but needs worst-case guarantees, not average-case bounds

### Next: H-001b (StreamingLLM)
阳阳 testing: retain first 128 + recent tokens, evict middle. Expected to perform better (preserves semantic anchors) but mid-range dependencies (coreference, etc.) will still degrade.

## Dependencies

- H-000 confirmed: K6V4 as quantization baseline
- llama-server/llama-perplexity with eviction support needed
