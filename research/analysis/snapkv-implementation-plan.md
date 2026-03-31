# SnapKV Stage 1 Eviction — Implementation Plan

## Overview

SnapKV evicts low-importance KV tokens at the end of prefill using attention
scores from an "observation window" (last N tokens of the prompt).

Unlike H2O (which accumulates scores during every decode step), SnapKV is
**one-shot**: compute scores once after prefill, evict, then never evict again.

This is simpler, has zero decode overhead, and is the Stage 1 of RocketKV.

## Algorithm

```
After all prompt tokens are decoded (prefill complete):
  1. Let W = last 32 tokens (observation window)
  2. For each layer:
     a. Extract attention matrix A = softmax(Q_W @ K_all^T / sqrt(d))
        where Q_W = queries from window tokens, K_all = all cached keys
     b. Score[i] = sum(A[:, i]) across all window queries and heads
     c. Apply 1D avg pooling with kernel=63 to Score (preserves clusters)
     d. Keep top-K positions (by score) + sink tokens
     e. Mark rest for eviction
  3. Remove evicted tokens from KV cache: llama_memory_seq_rm()
```

## Code Changes

### File 1: tools/completion/completion.cpp

#### Change A: SnapKV state struct

```cpp
struct snapkv_state {
    int  observation_window = 32;  // tokens to use as queries
    int  pooling_kernel     = 63;  // avg pooling kernel size
    int  budget;                   // tokens to keep (from evict_ratio)
    int  n_sink;                   // sink tokens to always keep
    bool fired = false;            // only fire once (after prefill)
};
```

#### Change B: Trigger after prefill

In the main loop, find where prompt processing ends and generation begins.
Current code (line ~719):

```cpp
if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
    // This is where generation starts — prefill is done
    const llama_token id = common_sampler_sample(smpl, ctx, -1);
```

Insert SnapKV trigger BEFORE first sample:

```cpp
if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
    // SnapKV: one-shot eviction after prefill
    if (params.snapkv_eviction && !snapkv.fired) {
        snapkv_evict(ctx, mem, params, n_past);
        snapkv.fired = true;
    }
    const llama_token id = common_sampler_sample(smpl, ctx, -1);
```

#### Change C: snapkv_evict() function

This is the core function. Two approaches:

**Approach 1: Reuse eval callback (like H2O)**
- Set `cb_eval` to capture last layer's softmax during the final prefill batch
- Pro: Uses existing infrastructure
- Con: Only gets scores from one batch (the last one)

**Approach 2: Explicit re-computation (more accurate)**
- After prefill, re-run attention for observation window queries against all keys
- This gives exact attention scores for scoring
- Con: One extra forward pass for 32 tokens (fast)

**Approach 3: Use accumulated H2O scores (simplest)**
- If H2O eval callback is active, we already have cumulative scores
- Just use those scores for one-shot eviction
- Pro: Zero extra computation
- Con: Cumulative scores may differ from window-only scores

**Recommendation: Approach 3 first (reuse H2O scores), then Approach 1 for accuracy.**

```cpp
void snapkv_evict(llama_context * ctx, llama_memory_t mem,
                  const common_params & params, int & n_past) {
    const int budget = std::max(1, (int)(n_past * (1.0f - params.evict_ratio)));
    const int n_sink = std::min(params.evict_sink, budget / 2);
    const int n_evict = n_past - budget;

    if (n_evict <= 0) return;

    LOG_INF("SnapKV: evicting %d tokens (keeping %d/%d, sink=%d)\n",
            n_evict, budget, n_past, n_sink);

    // Use H2O accumulated scores for eviction decision
    auto victims = g_h2o.batch_evict(n_past, n_sink, n_evict);

    // Remove from KV cache back-to-front
    int remaining = n_past;
    for (int i = (int)victims.size() - 1; i >= 0; i--) {
        llama_memory_seq_rm(mem, 0, victims[i], victims[i] + 1);
        llama_memory_seq_add(mem, 0, victims[i] + 1, remaining, -1);
        remaining--;
    }
    n_past = remaining;

    LOG_INF("SnapKV: done, n_past=%d\n", n_past);
}
```

### File 2: common/common.h

Add flag:
```cpp
bool snapkv_eviction = false;  // SnapKV one-shot prefill eviction
```

### File 3: common/arg.cpp

Add CLI flag:
```cpp
add_opt(common_arg(
    {"--snapkv-eviction"},
    "Use SnapKV one-shot eviction after prefill (requires --evict-ratio)",
    [](common_params & params) {
        params.snapkv_eviction = true;
    }
).set_examples({LLAMA_EXAMPLE_COMPLETION}));
```

## Differences from H2O

| Aspect | H2O | SnapKV |
|--------|-----|--------|
| When | Every decode step | Once after prefill |
| Score | Cumulative over all steps | Window-only (last 32 tokens) |
| Overhead | Per-step score update | Zero after initial eviction |
| Decode speed | Slower (non-flash + callback) | Full speed (flash OK after eviction) |
| Adaptivity | Adapts during conversation | Fixed after prefill |

**Key advantage of SnapKV:** After the one-shot eviction, flash attention
can be re-enabled (the evicted tokens are gone, remaining cache is smaller,
no more score tracking needed). This means full decode speed.

## Integration with H2O

SnapKV and H2O can be combined:
1. SnapKV fires after prefill (one-shot, big eviction)
2. H2O continues during decode (incremental eviction as context grows)
3. Two-stage like RocketKV

## Testing

```bash
# SnapKV with 50% eviction after prefill
./llama-completion -m model.gguf -c 16384 -n 64 --temp 0 \
    --evict-ratio 0.50 --evict-sink 128 --snapkv-eviction \
    -f prompt.txt

# Compare NIAH: SnapKV vs StreamingLLM vs H2O
python3 niah_test.py --model model.gguf --ctx 16384 \
    --evict-ratio 0.85 --evict-sink 128 --snapkv-eviction
```

## Estimated Work

| Component | Lines | Time |
|-----------|-------|------|
| snapkv_state struct | 10 | 5 min |
| CLI flag | 10 | 5 min |
| Trigger in main loop | 5 | 5 min |
| snapkv_evict() using H2O scores | 30 | 30 min |
| NIAH script --snapkv flag | 10 | 10 min |
| Testing + debug | — | 1-2 hours |
| **Total** | **~65 lines** | **~3 hours** |

## Future: Window-Based Scoring (Approach 1)

For more accurate scoring without H2O dependency:
1. After prefill, create a mini-batch of last 32 tokens
2. Run one forward pass (attention only, no FFN needed)
3. Extract softmax output via eval callback
4. Use those scores for eviction

This is more complex but gives SnapKV-specific scores.
Effort: additional ~100 lines + 1 day testing.
