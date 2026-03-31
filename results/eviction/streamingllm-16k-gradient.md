# StreamingLLM Eviction Gradient — 16K Context

**Model:** Llama-3.1-8B-Instruct Q4_K_M
**Context:** 16384, chunks=2, sink=128
**Hardware:** Mac Mini M4, ngl=20 (12GB not enough for ngl=99 at 16K)

| Eviction | Cache Budget  | PPL    | Δ% vs Baseline |
|----------|--------------|--------|----------------|
| 0%       | 16384/16384  | 5.7202 | —              |
| 50%      | 8192/16384   | 5.7035 | -0.03%         |
| 70%      | 4915/16384   | 5.7835 | +1.11%         |
| 80%      | 3276/16384   | 5.9126 | +3.36%         |
| 85%      | 2457/16384   | 6.1490 | +7.50%         |
| 90%      | 1638/16384   | FAIL   | decode error    |

## 4K vs 16K Comparison

| Eviction | 4K Δ%  | 16K Δ% | Improvement |
|----------|--------|--------|-------------|
| 50%      | +0.46% | -0.03% | ∞ (better)  |
| 70%      | +3.39% | +1.11% | 3.1x        |
| 80%      | +9.22% | +3.36% | 2.7x        |

**Key findings:**
- Longer context = more eviction-tolerant (confirmed)
- 1% PPL cliff at ~67% (vs 53% at 4K)
- Safe zone: ≤70% eviction at 16K (+1.11%)
- Combined with TQKV_6 (2.67x): 2.67x × 3.33x = 8.9x at ~+1.18% PPL

## 32K Baseline (partial)

**Context:** 32768, chunks=1, ngl=10
**PPL = 7.2508** (higher than 16K — likely chunks=1 instability)
