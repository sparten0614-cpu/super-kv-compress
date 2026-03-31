# KV Cache Compression Method Landscape

**Last updated:** 2026-03-31
**Maintained by:** 宁宁

## Comprehensive Comparison

| Method | Venue | Type | Compression | Quality (PPL/Task) | Runtime/Storage | Post-training? | Open Source | Notes |
|--------|-------|------|-------------|---------------------|-----------------|----------------|-------------|-------|
| **Quantization** | | | | | | | | |
| KIVI | ICML 2024 | K2V2 uniform quant | ~8x | +1-3% PPL | Runtime | Yes | Yes | Per-channel INT2, asymmetric |
| KVQuant | 2024 | Per-channel quant + outlier | 4-8x | <1% PPL | Runtime | Yes | Yes | Outlier-aware |
| QJL | 2024 | JL projection + quant (K only) | 2-4x (K) | <1% PPL | Runtime | Yes | Yes | Theoretical guarantees via JL lemma |
| KVTuner | ICML 2025 | Sensitivity-guided bit alloc | 4-8x | <1% PPL | Runtime | Yes | ? | Claims K ~2x more sensitive than V |
| TurboQuant | 2025 | Hadamard rotation + Lloyd-Max | 2.67-5.3x | <0.2% PPL | Runtime | Yes | Yes (ours) | 6-bit near-lossless, basis for our L1 |
| **Ours (K6V4)** | — | Asymmetric rotated quant | **3.2x** | **+0.52% PPL** | **Runtime** | **Yes** | **Yes** | K=6, V=4. V more sensitive (2-4bit cliff) |
| **Eviction** | | | | | | | | |
| H₂O | NeurIPS 2023 | Heavy hitter + recent | ~5x | <1% PPL | Runtime | Yes | Yes | Attention-score-based retention |
| StreamingLLM | 2024 | Sink + sliding window | ~2x | <0.5% PPL | Runtime | Yes | Yes | PPL good but NIAH fails (position-based) |
| ScissorHands | 2024 | Importance + pivot token | 5-10x | ~1% PPL | Runtime | Yes | Yes | Pivot token mechanism |
| FastGen | 2024 | Adaptive eviction policy | 2-4x | <1% PPL | Runtime | Yes | ? | Per-head adaptive strategy |
| **Ours (H2O eviction)** | — | Attention-aware eviction | **2-4x** | **<1% PPL** | **Runtime** | **Yes** | **Yes** | Noise-floor threshold from quant error |
| **SVD / Low-Rank** | | | | | | | | |
| xKV | 2025 (arxiv 2503.18893) | Cross-layer SVD | 2-3x (usable) | ~0.5% acc drop | Runtime | Yes (zero calib) | Yes (abdelfattah-lab/xKV) | Shared basis across layer groups. Reconstruct before attention. |
| GEAR | ICML 2024 | Low-rank + sparse outlier | 4-8x | <1% PPL | Runtime | Yes | Yes | SVD + outlier correction |
| **Transform Coding** | | | | | | | | |
| KVTC | ICLR 2026 (arxiv 2511.01815) | PCA + adaptive quant + entropy | 20-40x | <1 pt benchmark | **Storage** | Yes (10min calib) | ? | NVIDIA. Compress-store-decompress. NOT runtime memory. |
| **Hybrid / Advanced** | | | | | | | | |
| RocketKV | ICML 2025 | Token selection (2-stage: coarse→fine) | **~1.5x actual** (32.6% mem saved) | NIAH 100% @256 budget | Runtime | Yes | ? | NVIDIA. "400x" is title vs full cache, actual mem saving modest. Supports PPL-task gap. |
| DeltaKV | Feb 2026 | Cross-token residual | 3.4x | ? | Runtime? | Yes | ? | Token-level delta encoding |
| CLA/LCKV | 2024-2025 | Cross-layer sharing | ~2x | minimal | Runtime | **No (retrain)** | Yes | Requires architecture modification |
| MiniCache | 2024 | KV cache merging | ~2x | <1% | Runtime | Yes | Yes | Token merging in KV space |
| **Token Merging** | | | | | | | | |
| ToMe (Vision) | 2023 | Bipartite token merging | 2-4x | minimal | Runtime | Yes | Yes | Vision transformers, not LLM KV |
| Attention Matching (MIT) | 2024 | Virtual token optimization | ~50x | task-dependent | Runtime | Partial | ? | Optimize m virtual tokens to match attention |

## Summary by Approach

| Approach | Best Achievable (Runtime, PPL<1%) | Key Limitation |
|----------|-----------------------------------|----------------|
| Quantization alone | 3-8x | Bit-width floor (2-bit cliff) |
| Eviction alone | 2-5x | PPL sensitive to eviction rate |
| SVD alone | 2-3x | Reconstruction overhead |
| Quant + Eviction (ours) | **6-13x** | Log(n) eviction scaling |
| Quant + Eviction + SVD | **12-25x** (theoretical) | Quality loss stacking |
| Transform coding (storage) | 20-40x | NOT runtime — decompress required |
| RocketKV | 400x? | Unverified — under investigation |

## Key Insights from Landscape

1. **No existing method exceeds 10x runtime compression under PPL<1%.** Our two-knife approach (K6V4 + H2O) at 6-13x would be SOTA.

2. **Storage vs Runtime is critical.** KVTC's 20x is impressive but doesn't reduce GPU VRAM during inference. Papers often conflate these.

3. **Eviction methods fail on retrieval tasks.** StreamingLLM +0.46% PPL but 20% NIAH. Attention-aware (H2O) should fix this.

4. **Most methods are orthogonal.** Quantization, eviction, and SVD compress along different dimensions and can stack multiplicatively.

5. **Retraining is a dealbreaker for deployment.** Only post-training methods (marked "Yes") are practical for existing model deployments.

6. **Task-based evaluation gap.** Under task metrics, compression tolerance is 2-5x higher than PPL suggests (our §7 contribution).
