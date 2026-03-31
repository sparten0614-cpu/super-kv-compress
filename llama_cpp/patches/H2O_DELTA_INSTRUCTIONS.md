# H2O Delta — Manual Apply Instructions

Apply these changes ON TOP of the already-applied eviction patches.

## 1. common/common.h — Add 1 line

Find the line:
```cpp
    int     evict_sink      = 128;   // number of "attention sink" tokens to always keep at the beginning (mode 1)
```
Add AFTER it:
```cpp
    bool    h2o_eviction    = false; // use H2O attention-aware eviction (Heavy-Hitter Oracle)
```

## 2. common/arg.cpp — Add 7 lines

Find the block ending with:
```cpp
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-dt", "--defrag-thold"}, "N",
```
Insert BEFORE the `{"-dt"` line:
```cpp
    add_opt(common_arg(
        {"--h2o-eviction"},
        "Use H2O attention-aware eviction (Heavy-Hitter Oracle) instead of position-based",
        [](common_params & params) {
            params.h2o_eviction = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
```

## 3. tools/completion/completion.cpp — 3 changes

### Change A: Add H2O state struct + eval callback (top of file)

After the `#pragma warning` block and BEFORE `static llama_context ** g_ctx;`, insert:

```cpp
// ============================================================================
// H2O: Heavy-Hitter Oracle — Attention-aware KV cache eviction
// ============================================================================
struct h2o_state {
    std::vector<double> scores;
    int n_layers = 0;
    int n_kv_max = 0;
    int n_heads  = 0;
    bool enabled = false;

    void init(int layers, int kv_max, int heads) {
        n_layers = layers;
        n_kv_max = kv_max;
        n_heads  = heads;
        scores.assign(kv_max, 0.0);
        enabled  = true;
    }

    void accumulate(const float * attn_data, int n_kv, int n_tokens, int n_head) {
        if (!enabled || !attn_data) return;
        for (int h = 0; h < n_head; h++) {
            for (int t = 0; t < n_tokens; t++) {
                const float * row = attn_data + (h * n_tokens + t) * n_kv;
                for (int k = 0; k < n_kv && k < n_kv_max; k++) {
                    scores[k] += (double)row[k];
                }
            }
        }
    }

    int find_evict_target(int n_cached, int n_sink) const {
        if (!enabled || n_cached <= n_sink) return -1;
        int min_pos = -1;
        double min_score = 1e30;
        for (int i = n_sink; i < n_cached; i++) {
            if (scores[i] < min_score) {
                min_score = scores[i];
                min_pos = i;
            }
        }
        return min_pos;
    }

    void on_evict(int pos, int n_cached) {
        if (!enabled || pos < 0) return;
        for (int i = pos; i < n_cached - 1; i++) {
            scores[i] = scores[i + 1];
        }
        scores[n_cached - 1] = 0.0;
    }

    std::vector<int> batch_evict(int n_cached, int n_sink, int n_evict) {
        if (!enabled || n_evict <= 0) return {};
        std::vector<std::pair<double, int>> scored;
        scored.reserve(n_cached - n_sink);
        for (int i = n_sink; i < n_cached; i++) {
            scored.push_back({scores[i], i});
        }
        std::sort(scored.begin(), scored.end());
        int actual = std::min(n_evict, (int)scored.size());
        std::vector<int> victims;
        victims.reserve(actual);
        for (int i = 0; i < actual; i++) {
            victims.push_back(scored[i].second);
        }
        std::sort(victims.rbegin(), victims.rend());
        for (int pos : victims) {
            on_evict(pos, n_cached);
            n_cached--;
        }
        std::sort(victims.begin(), victims.end());
        return victims;
    }
};

static h2o_state g_h2o;

static bool h2o_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;
    if (!g_h2o.enabled) return true;
    const char * name = ggml_get_name(t);
    if (!name) return true;
    const bool is_attn = (strncmp(name, "kq_soft_max-", 12) == 0);
    if (ask) return is_attn;
    if (!is_attn) return true;
    const int n_kv     = t->ne[0];
    const int n_tokens = t->ne[1];
    const int n_head   = t->ne[2];
    const size_t data_size = ggml_nbytes(t);
    std::vector<float> buf(data_size / sizeof(float));
    ggml_backend_tensor_get(t, buf.data(), 0, data_size);
    g_h2o.accumulate(buf.data(), n_kv, n_tokens, n_head);
    return true;
}
```

### Change B: H2O initialization (in main(), after params parse, before common_init)

Find:
```cpp
    common_init();

    auto & sparams = params.sampling;
```
Insert BEFORE `common_init();`:
```cpp
    // H2O: set eval callback and force non-flash attention
    if (params.h2o_eviction) {
        if (params.evict_ratio <= 0.0f) {
            LOG_WRN("%s: --h2o-eviction requires --evict-ratio > 0, disabling H2O\n", __func__);
            params.h2o_eviction = false;
        } else {
            params.cb_eval = h2o_eval_callback;
            params.cb_eval_user_data = nullptr;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            LOG_INF("%s: H2O eviction enabled (ratio=%.0f%%, sink=%d, non-flash forced)\n",
                    __func__, params.evict_ratio * 100.0f, params.evict_sink);
        }
    }
```

And after model load (find `llama_memory_t mem = llama_get_memory(ctx);`), add:
```cpp
    // Initialize H2O state
    if (params.h2o_eviction) {
        const int n_layers = llama_model_n_layer(model);
        const int n_ctx    = llama_n_ctx(ctx);
        const int n_heads  = llama_model_n_head(model);
        g_h2o.init(n_layers, n_ctx, n_heads);
        LOG_INF("%s: H2O initialized: %d layers, %d ctx, %d heads\n",
                __func__, n_layers, n_ctx, n_heads);
    }
```

### Change C: H2O eviction branch (in the context shift section)

In the eviction section, find:
```cpp
                    if (has_eviction && params.evict_mode == 1) {
                        // StreamingLLM: keep first `sink` tokens
```
Insert BEFORE this block:
```cpp
                    if (has_eviction && params.h2o_eviction) {
                        // H2O: batch evict tokens with lowest accumulated attention score
                        const int sink = std::min(params.evict_sink, cache_budget / 2);
                        const int n_discard = n_past + (int) embd.size() - cache_budget;
                        if (n_discard > 0) {
                            auto victims = g_h2o.batch_evict(n_past, sink, n_discard);
                            int remaining = n_past;
                            for (int i = (int)victims.size() - 1; i >= 0; i--) {
                                const int pos = victims[i];
                                llama_memory_seq_rm (mem, 0, pos, pos + 1);
                                llama_memory_seq_add(mem, 0, pos + 1, remaining, -1);
                                remaining--;
                            }
                            n_past = remaining;
                            LOG_DBG("H2O batch evicted %d tokens (budget=%d, sink=%d), n_past=%d\n",
                                    (int)victims.size(), cache_budget, sink, n_past);
                        }
                    } else
```
And change the existing `if (has_eviction && params.evict_mode == 1)` to `if (has_eviction && params.evict_mode == 1)` (keep as-is, the `else` chains them).

## 4. Verify

```bash
cmake --build build --target llama-completion -j$(nproc)
./build/bin/llama-completion --help | grep h2o
# Should show: --h2o-eviction
```

## 5. Test

```bash
# Quick smoke test (small context)
echo "The quick brown fox" | ./build/bin/llama-completion -m model.gguf -c 512 -n 10 --temp 0 --evict-mode 1 --evict-ratio 0.50 --evict-sink 32 --h2o-eviction -f /dev/stdin
```
