# StreamingLLM Eviction Gradient — 4K Context

**Model:** Llama-3.1-8B Q4_K_M
**Context:** 4096, chunks=5, sink=128
**Hardware:** Mac Mini M4, ngl=99

| Eviction | Cache Budget | PPL    | Δ% vs Baseline |
|----------|-------------|--------|----------------|
| 0%       | 4096/4096   | 6.0280 | —              |
| 10%      | 3686/4096   | 6.0240 | -0.07%         |
| 30%      | 2867/4096   | 6.0160 | -0.20%         |
| 50%      | 2048/4096   | 6.0556 | +0.46%         |
| 55%      | 1843/4096   | 6.0940 | +1.09%         |
| 60%      | 1638/4096   | 6.1125 | +1.40%         |
| 65%      | 1433/4096   | 6.1500 | +2.02%         |
| 70%      | 1228/4096   | 6.2322 | +3.39%         |
| 75%      | 1024/4096   | 6.3474 | +5.30%         |
| 80%      | 819/4096    | 6.5841 | +9.22%         |

**Key findings:**
- PPL 1% cliff at ~53% eviction
- PPL curve is convex (accelerating degradation)
- Safe zone: ≤50% eviction (+0.46%)
