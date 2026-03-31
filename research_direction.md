# Research Direction: KV Cache Compression

## Goal
Find maximum KV cache compression ratio while maintaining:
- PPL degradation < 5% vs F16 baseline
- NIAH retrieval accuracy >= 80% (ideally 100%)

## Target
- Minimum: 5x compression @ NIAH 100%
- Stretch: 10x+ compression @ NIAH >= 80%

## Known Findings
- q4_0 gives 3.56x with NIAH 100% on Llama/Mistral (verified across 8B/7B/70B)
- Qwen models crash on q4_0 K due to Layer 0 outlier (K_max=93) — skip layer 0 fixes it
- K is more sensitive to quantization than V (asymmetric K=q8_0/V=q4_0 works for Qwen)
- StreamingLLM eviction at 50%: +0.46% PPL (safe), but NIAH drops to 60%
- At 16K context, eviction tolerance is higher than 4K (70% eviction = +1.1%)
- Longer context = more eviction-tolerant

## Directions to Explore
1. Asymmetric K/V quantization (K at higher precision, V aggressive)
2. Per-layer quantization (skip outlier layers, aggressive on middle)
3. Combining quantization + eviction for multiplicative compression
4. Attention-aware eviction (H2O, SnapKV, Expected Attention) for NIAH preservation
5. Ultra-low-bit (Q2_K) for cold/old tokens in multi-tier architecture

## Constraints
- Post-training only (no fine-tuning)
- Must work in llama.cpp (C/C++ implementation)
- Target hardware: NVIDIA GPU (cloud) + Apple Silicon (local)
