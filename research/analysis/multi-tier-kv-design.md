# Multi-Tier KV Cache Design Document

## Phase 3: Temporal Decay with Per-Tier Quantization

### 1. Motivation

Current llama.cpp KV cache uses a single quantization level for all tokens:
- q4_0 gives 3.56x compression with NIAH 100% and PPL +1-3%
- But all tokens get equal precision, regardless of age or importance

**Insight:** Recent tokens contribute more to output quality than old tokens.
Older tokens can tolerate lower precision. By gradually reducing precision
as tokens age, we can achieve higher effective compression without quality loss.

**Target:**
- Recent tokens (last ~20%): q8_0 (high quality, 1.88x)
- Mid-age tokens (next ~30%): q4_0 (good quality, 3.56x)
- Old tokens (remaining ~50%): q2_K or lower (acceptable, 5-8x)
- Effective compression: weighted average ≈ 4-5x (vs uniform q4_0 = 3.56x)

### 2. Current Architecture (Barriers)

```
llama_kv_cache
  └── layers: vector<kv_layer>
        └── kv_layer
              ├── k: ggml_tensor [n_embd_k_gqa, kv_size, n_stream]  ← single type
              ├── v: ggml_tensor [n_embd_v_gqa, kv_size, n_stream]  ← single type
              ├── k_stream: vector<ggml_tensor*>  (2D views)
              └── v_stream: vector<ggml_tensor*>  (2D views)
```

**Barrier 1:** `ggml_tensor` has a single `type` field → all rows same format
**Barrier 2:** `ggml_row_size()` returns fixed stride → no variable-width rows
**Barrier 3:** GPU kernels compiled per-type → can't mix in one matmul
**Barrier 4:** `get_k()` creates views with fixed strides

### 3. Proposed Architecture: Multi-Tier KV Cache

```
llama_kv_cache_tiered (extends llama_kv_cache)
  └── layers: vector<kv_layer_tiered>
        └── kv_layer_tiered
              ├── tiers[0] (HOT):  k/v tensors, type=q8_0, capacity=hot_size
              ├── tiers[1] (WARM): k/v tensors, type=q4_0, capacity=warm_size
              ├── tiers[2] (COLD): k/v tensors, type=q2_K, capacity=cold_size
              └── cell_tier_map: vector<uint8_t>  // which tier each cell is in
```

Each tier is a separate homogeneous tensor. Existing ggml ops work unchanged
because each tier is a standard typed tensor.

#### 3.1 Data Structures

```cpp
struct kv_tier {
    ggml_type     type;       // e.g., GGML_TYPE_Q8_0
    uint32_t      capacity;   // max tokens in this tier
    uint32_t      used;       // current occupancy
    ggml_tensor * k;          // [n_embd_k_gqa, capacity, n_stream]
    ggml_tensor * v;          // [n_embd_v_gqa, capacity, n_stream]

    // Cell tracking: maps tier-local slot index → global position
    std::vector<llama_pos> positions;  // [capacity]
    std::vector<bool>      occupied;   // [capacity]
};

struct kv_layer_tiered {
    uint32_t il;  // model layer index

    static constexpr int N_TIERS = 3;
    kv_tier tiers[N_TIERS];  // HOT=0, WARM=1, COLD=2

    // Reverse map: global position → (tier_idx, slot_idx)
    struct cell_loc { uint8_t tier; uint32_t slot; };
    std::unordered_map<llama_pos, cell_loc> pos_to_loc;
};
```

#### 3.2 Tier Configuration (Defaults)

```cpp
struct tier_config {
    ggml_type type;
    float     fraction;  // fraction of total kv_size
    int       max_age;   // tokens older than this decay to next tier (-1 = unlimited)
};

// Default: 20% hot + 30% warm + 50% cold
static const tier_config DEFAULT_TIERS[3] = {
    { GGML_TYPE_Q8_0, 0.20f, 512  },  // HOT: recent, high precision
    { GGML_TYPE_Q4_0, 0.30f, 4096 },  // WARM: mid-age, medium precision
    { GGML_TYPE_Q2_K, 0.50f, -1   },  // COLD: old, low precision
};
```

#### 3.3 Memory Budget

For ctx=16384, n_embd_k_gqa=4096 (Llama-3.1-8B, 8 KV heads × 128 dim):

| Tier | Type | Tokens | Per-token bytes | Total |
|------|------|--------|-----------------|-------|
| HOT  | q8_0 | 3276   | 4352 (8.5 bits/val) | 14.3 MB |
| WARM | q4_0 | 4915   | 2304 (4.5 bits/val) | 11.3 MB |
| COLD | q2_K | 8193   | 1280 (2.5 bits/val) | 10.5 MB |
| **Total** | | **16384** | | **36.1 MB** |

Uniform q4_0: 16384 × 2304 = 37.7 MB → multi-tier saves ~4% more
Uniform f16: 16384 × 8192 = 134 MB → multi-tier = **3.7x compression**

The real win isn't memory at a single context length — it's **quality at the same compression level**. At 3.7x average compression, multi-tier preserves recent token quality (q8_0) that uniform q4_0 sacrifices.

### 4. Operations

#### 4.1 Write (cpy_k / cpy_v) — New tokens always go to HOT tier

```
Token arrives → quantize to q8_0 → write to tiers[0].k[next_slot]
               Update pos_to_loc[pos] = {tier=0, slot=next_slot}

If tiers[0] full:
  trigger_decay(layer, tier=0)  // move oldest HOT tokens to WARM
```

No change to the write path logic — just target the HOT tier tensor.

#### 4.2 Read (get_k / get_v) — Concatenate tier views for attention

```
For attention computation:
  k_hot  = view(tiers[0].k, [0:hot_used])
  k_warm = view(tiers[1].k, [0:warm_used])
  k_cold = view(tiers[2].k, [0:cold_used])

  K_full = concat(k_cold, k_warm, k_hot)  // cold first, hot last

  // Build position mapping for RoPE correctness
  positions = concat(cold_positions, warm_positions, hot_positions)
```

**Critical:** RoPE positional encoding must match the original positions,
not the tier-local slot indices. The attention mask needs the global positions.

#### 4.3 Decay Pass — Migrate aged tokens between tiers

```
trigger_decay(layer, source_tier):
  for each occupied slot in tiers[source_tier]:
    if token_age(slot) > tier_config[source_tier].max_age:
      1. Dequantize: k_f32 = dequant(tiers[source_tier].k[slot])
      2. Requantize: k_lower = quant(k_f32, tiers[source_tier+1].type)
      3. Write:      tiers[source_tier+1].k[next_slot] = k_lower
      4. Free:       tiers[source_tier].occupied[slot] = false
      5. Update:     pos_to_loc[pos] = {tier=source+1, slot=next_slot}

  If tiers[source_tier+1] full:
    trigger_decay(layer, source_tier+1)  // cascade

  // For COLD tier overflow: evict lowest-score tokens (H2O) or oldest
```

**Frequency:** Decay runs when a tier is full, not every token.
For HOT tier (3276 slots at 20%), decay triggers every ~3276 tokens.

#### 4.4 Eviction — When COLD tier is also full

When even the COLD tier is full and needs space:
- Option A: StreamingLLM — evict oldest cold tokens
- Option B: H2O — evict lowest attention-score cold tokens
- Option C: Drop — simply discard (lossy but simplest)

This integrates naturally with the existing eviction work (StreamingLLM/H2O).

### 5. Integration with Attention Graph

The main change is in `build_attn()` where K and V tensors are obtained:

```cpp
// Current (single tensor):
ggml_tensor * k = kv->get_k(ctx, il, n_kv, sinfo);

// Multi-tier (concatenated):
ggml_tensor * k = kv_tiered->get_k_concat(ctx, il, sinfo);
// Internally: concat views from all tiers + reorder by position
```

**Key constraint:** The attention kernel expects K as a contiguous tensor
with consistent type. Options:

**Option A: Dequant to F16 before concat**
- Each tier dequants to F16 → concat → single F16 tensor for attention
- Pro: Simple, works with flash attention
- Con: Loses quantization benefit during attention (memory peak at F16)

**Option B: Per-tier attention + merge**
- Compute attention separately for each tier: attn_hot, attn_warm, attn_cold
- Merge with softmax correction: `output = softmax_merge(attn_hot, attn_warm, attn_cold)`
- Pro: Each tier stays quantized, no F16 blowup
- Con: Complex softmax merge, 3x attention passes
- Note: This is similar to how flash attention handles chunked computation

**Option C: Uniform cold quant (simplest practical)**
- HOT stays separate at q8_0
- WARM + COLD share q4_0 (no separate cold tier)
- Only 2 tiers: HOT(q8_0, 20%) + REST(q4_0, 80%)
- Decay: HOT→REST is just requantize q8_0→q4_0
- Attention: 2 concat views or per-tier attention

**Recommendation: Start with Option C (2 tiers).** Simplest to implement,
captures 80% of the benefit. Extend to 3 tiers later if needed.

### 6. Implementation Plan

#### Phase 3a: 2-Tier Prototype (HOT q8_0 + REST q4_0)

1. **Extend kv_layer struct** with second k/v tensor pair
2. **New CLI flags:** `--kv-tier-hot-ratio 0.2 --kv-tier-hot-type q8_0 --kv-tier-cold-type q4_0`
3. **Write path:** always to HOT tier
4. **Decay trigger:** when HOT tier full, migrate oldest to REST
5. **Read path:** concat HOT + REST views (dequant to F16 for now — Option A)
6. **Test:** PPL comparison vs uniform q4_0 and uniform q8_0

**Expected result:** PPL close to q8_0 (thanks to recent tokens at high precision)
but memory close to q4_0 (80% of tokens at low precision).

#### Phase 3b: 3-Tier + Per-Tier Attention

1. Add COLD tier with q2_K
2. Implement per-tier attention (Option B) to avoid F16 decompression
3. Integrate H2O scores for COLD tier eviction

#### Phase 3c: Adaptive Tier Sizing

1. Monitor attention score distribution to auto-size tiers
2. Layers with more "diffuse" attention → larger HOT tier
3. Layers with "concentrated" attention → smaller HOT tier

### 7. Files to Modify

| File | Change |
|------|--------|
| `src/llama-kv-cache.h` | Add `kv_tier` struct, extend `kv_layer` |
| `src/llama-kv-cache.cpp` | Tier allocation, decay pass, concat views |
| `src/llama-graph.cpp` | `build_attn_inp_kv()` to handle multi-tier K/V |
| `common/common.h` | Tier config params |
| `common/arg.cpp` | CLI flags |
| `tools/perplexity/perplexity.cpp` | Pass-through (uses llama API) |

### 8. Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| RoPE position mismatch after decay | Broken attention | Explicit position tracking per-slot, validate in tests |
| Decay latency spike | Jitter in generation | Amortize: decay small batches per step, not all-at-once |
| F16 peak memory during concat (Option A) | Defeats purpose | Move to per-tier attention (Option B) in Phase 3b |
| GQA head mismatch between tiers | Incorrect attention | All tiers must have same n_embd_k_gqa |

### 9. Expected Results

| Config | Compression | Quality (PPL) |
|--------|-------------|---------------|
| Uniform q8_0 | 1.88x | ~0% loss |
| Uniform q4_0 | 3.56x | +1-3% |
| **2-tier (20% q8_0 + 80% q4_0)** | **~3.2x** | **<1%** (estimated) |
| **3-tier (20% q8_0 + 30% q4_0 + 50% q2_K)** | **~4.5x** | **<2%** (estimated) |

The key insight: by keeping recent tokens at q8_0, we protect the tokens
that matter most for next-token prediction, while aggressively compressing
old tokens that contribute less.

Combined with StreamingLLM 50% eviction (safe zone):
- 3-tier 4.5x × eviction 2x = **9x total compression**
- Quality: <2% PPL + NIAH preserved (recent tokens at q8_0 protect retrieval)
