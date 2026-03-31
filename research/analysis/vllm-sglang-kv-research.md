# KV Cache Quantization in Production Serving: vLLM, SGLang, and Deployment

## Current State: FP8 is the Only Production-Ready Option

| Framework | KV Quant Support | Production Status |
|-----------|-----------------|-------------------|
| **vLLM** | FP8 E4M3 only | ✅ Production |
| **SGLang** | FP8 E4M3 (FP4 exists but 55x slower) | ✅ FP8 only |
| **TensorRT-LLM** | FP8 + NVFP4 (Blackwell) | ✅ FP8, ✅ FP4 on B200 |
| **LMDeploy** | INT4, INT8 (KIVI-based) | ✅ Production since 2024 |
| **llama.cpp** | Q8_0, Q4_0, Q4_1, Q5_x + asymmetric + per-layer | ✅ CPU/Metal |

## Our Unique Capabilities (No Other Framework Has These)

### 1. Asymmetric K/V Quantization
- **What:** K=q8_0, V=q4_0 (different precision for keys vs values)
- **Why:** K is more sensitive (outliers in attention weights), V is more tolerant
- **Impact:** Enables Qwen-family models to use 4-bit V without K crash
- **Status in frameworks:** vLLM/SGLang both use single `kv_cache_dtype` for K and V. No separate control exists.

### 2. Per-Layer KV Precision
- **What:** `--kv-type-map "0:q8_0,1-30:q4_0,31:q8_0"` (outlier layers at high precision)
- **Why:** Layer 0 has extreme K outliers (K_max=93 on Qwen). Middle layers are safe.
- **Impact:** Recovers from PPL +104% to +7.1% on Qwen without losing compression on safe layers
- **Status in frameworks:** All frameworks quantize all layers identically. vLLM issue #22195 requests this but unimplemented.

### 3. Additive Error Law
- **Finding:** ΔPPL(K_q, V_q) ≈ ΔPPL(K_q, V_f16) + ΔPPL(K_f16, V_q)
- **Verified:** 3.1% predicted vs 3.14% actual (0.04pp error)
- **Impact:** Enables independent K/V optimization (search space N_k + N_v instead of N_k × N_v)
- **Status:** Novel empirical finding, not reported in literature

## Framework Details

### vLLM
- PagedAttention: K and V stored in separate block tensors (architecturally ready for asymmetric)
- FP8: per-tensor or per-head scales, auto-calibrated at warmup
- TurboQuant plugin exists (PR #38280, 49 upvotes) but sub-4-bit produces garbage
- INT4/INT8 feature requests open (#33480, #28538) but unimplemented

### SGLang
- RadixAttention: KV cache in radix tree for prefix sharing
- FP4 E2M1 supported but 55x throughput regression (dequant not fused with attention)
- TurboQuant feature request (#21618) WIP

### LMDeploy
- Only GPU framework with production INT4 KV cache
- KIVI-based: per-head, per-token asymmetric quantization
- ~40% RPS improvement vs FP16
- Our GQA-aware rules could be contributed as best practices

## Deployment Roadmap

### Phase 1 (Now): llama.cpp
- Already done: q4_0, asymmetric K/V, per-layer, --kv-type-map
- Target: edge/local inference on Apple Silicon

### Phase 2 (Post-Paper): vLLM Contribution
- Asymmetric K/V support → high-impact PR
- Per-layer precision → medium-impact (deeper architectural change)
- GQA-aware auto-configuration → configuration guide

### Phase 3 (Future): SGLang
- Fix FP4 throughput (fuse dequant kernel)
- Port asymmetric K/V

## References
- vLLM KV cache docs: docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- SGLang KV cache: docs.sglang.io/advanced_features/quantized_kv_cache.html
- vLLM TurboQuant PR: github.com/vllm-project/vllm/pull/38280
- LMDeploy INT4: lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html
- KVTuner per-layer: arxiv.org/abs/2502.04420
