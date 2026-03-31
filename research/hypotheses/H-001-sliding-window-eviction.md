# H-001: Sliding Window Eviction

**Status:** Active 🔄
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

## Current Status

- 阳阳 implementing sliding window eviction in llama.cpp
- 5 eviction configurations ready to test
- Awaiting implementation completion

## Dependencies

- H-000 confirmed: K6V4 as quantization baseline
- llama-server/llama-perplexity with eviction support needed
