# Per-Layer KV Cache Precision — Analysis

## Question

Can different layers use different KV cache quantization levels?
(e.g., attention-heavy layers at q8_0, others at q4_0)

## Answer: Already Supported

**llama.cpp already supports per-layer KV precision** — we implemented this
in TurboQuant Phase 2 via `TQKV_SKIP_LAYERS`.

### Current Mechanism (Our Fork)

```cpp
// src/llama-kv-cache.cpp:141-144
ggml_type layer_type_k = tqkv_is_skip_layer(il) ? GGML_TYPE_F16 : type_k;
ggml_type layer_type_v = tqkv_is_skip_layer(il) ? GGML_TYPE_F16 : type_v;
ggml_tensor * k = ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa, kv_size, n_stream);
```

Each layer's KV tensors are allocated independently with potentially different types.
The skip_layer check uses `TQKV_SKIP_LAYERS` env var (e.g., "0,27" for Qwen).

### Generalizing to Arbitrary Per-Layer Types

The current system is binary (skip layer → F16, else → configured type).
Generalizing to arbitrary per-layer types requires:

#### Option A: Layer-Type Config String (Simplest)

```bash
# New CLI flag
--kv-type-map "0-3:q8_0,4-27:q4_0,28-31:q8_0"
```

Implementation:
```cpp
// In common_params
std::vector<ggml_type> kv_type_per_layer;  // [n_layer], populated from map string

// In llama-kv-cache.cpp allocation
ggml_type layer_type_k = params.kv_type_per_layer.empty()
    ? type_k
    : params.kv_type_per_layer[il];
```

**Effort: ~50 lines** (arg parsing + allocation change)

#### Option B: Auto-Calibrate Based on Layer Sensitivity

Run a calibration pass:
1. For each layer, compute KV cache quantization error (MSE between F16 and quantized)
2. Assign higher precision to layers with higher error
3. Budget constraint: total memory ≤ target

This is what Tom's "Boundary V" does manually (first/last 2 layers at q8_0).
Our TQKV_SKIP_LAYERS is the same idea.

**Effort: ~200 lines** (calibration script + auto-assignment)

#### Option C: Eval Callback Approach (Not Needed)

The eval callback (`cb_eval`) runs during graph execution. It CAN'T change
tensor types (types are fixed at allocation time). So eval callback is the
**wrong tool** for per-layer precision — the allocation-time approach is correct.

The eval callback IS useful for:
- Monitoring attention patterns (H2O scoring)
- Collecting calibration statistics
- But NOT for changing KV cache quantization levels

### What Matters: Which Layers Need High Precision?

From our experiments and Tom's findings:
- **First layer (layer 0):** Often has extreme outliers (Qwen K_max=93). Needs F16 or q8_0.
- **Last 2-4 layers:** Highest sensitivity to quantization (Tom: 100% of quality loss from last 20%)
- **Middle layers:** Most tolerant, can use aggressive quantization (q4_0 or lower)

Optimal layer-type assignment (32-layer model):
```
Layer 0:     q8_0  (outlier protection)
Layers 1-27: q4_0  (bulk, aggressive)
Layers 28-31: q8_0  (output-sensitive)
```

Memory: 4 layers × q8_0 + 28 layers × q4_0 = weighted average ~4.0 bits/val
Quality: Close to uniform q8_0 (protecting sensitive layers) at q4_0 memory cost

### Interaction with Multi-Tier

Per-layer precision is ORTHOGONAL to multi-tier (per-age precision):
- Per-layer: layer 0 always q8_0, layer 15 always q4_0
- Multi-tier: recent tokens q8_0, old tokens q4_0 (within each layer)
- Combined: layer 0 recent at q8_0, layer 15 old at q2_K

The multi-tier architecture naturally supports this — each tier in each layer
can have its own type. The kv_layer_tiered struct already allows this.

## Summary

| Approach | Mechanism | Effort | Status |
|----------|-----------|--------|--------|
| Binary skip layers | TQKV_SKIP_LAYERS env var | Done | ✅ Implemented |
| Layer-type config string | --kv-type-map CLI flag | 50 lines | Ready to implement |
| Auto-calibrate | Sensitivity analysis | 200 lines | Phase 3b |
| Eval callback | N/A | N/A | ❌ Wrong tool |

**Recommendation:** Implement Option A (--kv-type-map) as a quick generalization
of the existing TQKV_SKIP_LAYERS mechanism. This unblocks per-layer experiments
without calibration complexity.
