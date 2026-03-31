# Q2_K as KV Cache Type — Feasibility Analysis

## Goal

Enable `--cache-type-k q2_K --cache-type-v q2_K` in llama.cpp for 2-bit KV cache,
the lowest tier in the multi-tier architecture.

## Current State

Q2_K is defined as `GGML_TYPE_Q2_K = 10` in ggml but is **NOT** in the KV cache
allowed types list (`kv_cache_types` in `common/arg.cpp`).

## What Q2_K Already Has

| Component | Status | Details |
|-----------|--------|---------|
| ggml type enum | ✅ | `GGML_TYPE_Q2_K = 10` |
| Block structure | ✅ | `block_q2_K`, QK_K=256, ~2.5 bits/val |
| CPU from_float | ✅ | `quantize_row_q2_K_ref()` in ggml-quants.c |
| CPU to_float | ✅ | `dequantize_row_q2_K()` in ggml-quants.c |
| CPU mul_mv | ✅ | Full CPU matmul support |
| Metal dequantize | ✅ | `dequantize_q2_K()` in ggml-metal.metal |
| Metal mul_mv | ✅ | `kernel_mul_mv_q2_K_f32` |
| CUDA support | ✅ | Full CUDA kernels exist |
| Metal set_rows | ❌ | **Missing** — block size 256 not handled |
| CPU set_rows | ✅ | Uses `from_float_ref` (generic path) |
| Flash attention | ❓ | Need to check `flash_attn_ext` kernel |

## Key Blocker: Metal set_rows for Q2_K

The KV cache write path uses `ggml_set_rows()` to quantize and store new KV values.

On Metal, `set_rows` has two kernel variants:
1. `kernel_set_rows_f` — for float/half/bfloat (element-wise)
2. `kernel_set_rows_q32` — for block_size=32 types (q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl)

Q2_K has block_size=256 (QK_K). The q32 kernel's quantize function template expects
32-element blocks. A new kernel template is needed for 256-element blocks.

### Solution: kernel_set_rows_q256

```metal
// New kernel for QK_K=256 block types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
template <typename index_t, typename block_t, void (*quantize_func)(...)>
kernel void kernel_set_rows_q256(
    device const char * src0,           // f32 source
    device       char * dst,            // quantized destination
    device const char * src1,           // indices
    constant  ggml_metal_kargs_set_rows & args,
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Similar to kernel_set_rows_q32 but:
    // - Block size = 256 instead of 32
    // - quantize_func signature matches Q2_K's quantize function
    // - Need 256 elements of shared memory for the block
}
```

### Files to Modify

1. **common/arg.cpp** — Add `GGML_TYPE_Q2_K` to `kv_cache_types` vector
2. **ggml/src/ggml-metal/ggml-metal.metal** — Add `kernel_set_rows_q256` template + Q2_K instantiation
3. **ggml/src/ggml-metal/ggml-metal.m** — Register the new kernel dispatch
4. **ggml/src/ggml-cuda/** — Verify CUDA set_rows supports Q2_K (likely already works via generic path)

### Q2_K Block Structure

```c
typedef struct {
    ggml_half d;           // super-block scale (delta)
    ggml_half dmin;        // super-block min
    uint8_t scales[QK_K/16]; // sub-block scales (4-bit each, packed)
    uint8_t qs[QK_K/4];     // 2-bit quantized values (packed)
} block_q2_K;
// sizeof = 2 + 2 + 16 + 64 = 84 bytes per 256 elements
// = 2.625 bits per value
```

### Metal quantize_q2_K Function

Need to write a Metal-side quantize function matching the CPU `quantize_row_q2_K_ref`:
1. Find min/max per 16-element sub-block
2. Compute scales and mins
3. Quantize to 2-bit per element
4. Pack into block_q2_K layout

This is moderate complexity — the sub-block structure makes it more involved than q4_0.

### Flash Attention Compatibility

Need to check if `flash_attn_ext` has Q2_K instantiation in Metal:

```
grep "Q2_K" ggml-metal-flash.metal
```

If missing, flash attention won't work with Q2_K KV cache (falls back to non-flash path).
This is acceptable for Phase 3a testing but needs to be added for production.

## Estimated Work

| Task | Effort | Priority |
|------|--------|----------|
| Add Q2_K to kv_cache_types | 1 line | P0 (5 min) |
| CPU path (already works) | 0 lines | ✅ Done |
| Metal kernel_set_rows_q256 | ~80 lines | P0 (2-3 hours) |
| Metal quantize_q2_K function | ~60 lines | P0 (included above) |
| Metal dispatch registration | ~10 lines | P0 (included above) |
| CUDA verification | ~30 min | P1 |
| Flash attention Q2_K | ~40 lines | P2 |
| **Total** | **~150 lines** | **~3-4 hours** |

## Quick Test Path (CPU-only, no Metal kernel needed)

For immediate testing without Metal kernel work:

1. Add `GGML_TYPE_Q2_K` to `kv_cache_types`
2. Run with `-ngl 0` (CPU only) — CPU `set_rows` uses generic `from_float_ref` path
3. This works TODAY, just slow

```bash
./llama-perplexity -m model.gguf -ngl 0 -c 4096 --chunks 5 \
    --cache-type-k q2_K --cache-type-v q2_K -f wiki.test.raw
```

This gives us PPL data immediately while Metal kernel is being developed.

## Expected Results

Based on the general trend (q8_0 ~0%, q4_0 +1-3%):
- Q2_K PPL: estimated +10-20% (significant but functional)
- Q2_K NIAH: unclear — 2-bit may corrupt needle embeddings
- Q2_K compression: 6.1x (16/2.625)
- Best use: COLD tier only (old tokens that matter less)
