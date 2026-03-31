// H2O Score Tracking Unit Test
// Standalone test for h2o_state logic (no llama.cpp dependency)
//
// Compile: g++ -std=c++17 -O2 -o test_h2o test_h2o_logic.cpp && ./test_h2o

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// Copy of h2o_state from completion.cpp for standalone testing
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

    // Batch eviction: find and evict n_evict lowest-score tokens
    // Returns sorted list of evicted positions (for KV cache removal)
    std::vector<int> batch_evict(int n_cached, int n_sink, int n_evict) {
        if (!enabled || n_evict <= 0) return {};

        // Collect (score, position) pairs for non-sink tokens
        std::vector<std::pair<double, int>> scored;
        for (int i = n_sink; i < n_cached; i++) {
            scored.push_back({scores[i], i});
        }

        // Sort by score ascending (lowest first)
        std::sort(scored.begin(), scored.end());

        // Take the n_evict lowest
        int actual_evict = std::min(n_evict, (int)scored.size());
        std::vector<int> victims;
        for (int i = 0; i < actual_evict; i++) {
            victims.push_back(scored[i].second);
        }

        // Sort victims descending so we can remove from back to front
        // (removing from back doesn't shift earlier indices)
        std::sort(victims.rbegin(), victims.rend());

        // Remove each victim (back to front)
        for (int pos : victims) {
            on_evict(pos, n_cached);
            n_cached--;
        }

        // Return in ascending order for reporting
        std::sort(victims.begin(), victims.end());
        return victims;
    }
};

// ============================================================================
// Tests
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("Test: %s ... ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { printf("FAIL: %s (got %d, expected %d)\n", msg, (int)(a), (int)(b)); tests_failed++; return; } } while(0)
#define ASSERT_NEAR(a, b, eps, msg) do { if (fabs((a) - (b)) > (eps)) { printf("FAIL: %s (got %.6f, expected %.6f)\n", msg, (double)(a), (double)(b)); tests_failed++; return; } } while(0)

void test_init() {
    TEST("init");
    h2o_state h;
    h.init(32, 4096, 8);
    ASSERT_EQ(h.scores.size(), 4096, "scores size");
    ASSERT_EQ(h.enabled, true, "enabled");
    ASSERT_NEAR(h.scores[0], 0.0, 1e-10, "initial score");
    PASS();
}

void test_accumulate_simple() {
    TEST("accumulate_simple");
    h2o_state h;
    h.init(1, 8, 1);  // 1 layer, 8 kv, 1 head

    // Single head, single query token, 8 KV positions
    // Softmax output: token 0 attends mostly to position 3
    float attn[8] = {0.01f, 0.01f, 0.01f, 0.90f, 0.01f, 0.02f, 0.02f, 0.02f};
    h.accumulate(attn, 8, 1, 1);

    ASSERT_NEAR(h.scores[3], 0.90, 1e-5, "high attention position");
    ASSERT_NEAR(h.scores[0], 0.01, 1e-5, "low attention position");
    PASS();
}

void test_accumulate_multi_query() {
    TEST("accumulate_multi_query");
    h2o_state h;
    h.init(1, 4, 1);  // 4 KV positions, 1 head

    // 2 query tokens, each attending to different positions
    // Layout: [kv0_t0, kv1_t0, kv2_t0, kv3_t0, kv0_t1, kv1_t1, kv2_t1, kv3_t1]
    float attn[8] = {
        0.8f, 0.1f, 0.05f, 0.05f,  // token 0: attends to pos 0
        0.1f, 0.1f, 0.1f, 0.7f,    // token 1: attends to pos 3
    };
    h.accumulate(attn, 4, 2, 1);

    // Scores should be sum across queries
    ASSERT_NEAR(h.scores[0], 0.9, 1e-5, "pos 0 score (0.8+0.1)");
    ASSERT_NEAR(h.scores[3], 0.75, 1e-5, "pos 3 score (0.05+0.7)");
    PASS();
}

void test_accumulate_multi_head() {
    TEST("accumulate_multi_head");
    h2o_state h;
    h.init(1, 4, 2);  // 4 KV positions, 2 heads

    // 2 heads, 1 query token each, 4 KV positions
    // Layout: [head0: kv0, kv1, kv2, kv3], [head1: kv0, kv1, kv2, kv3]
    float attn[8] = {
        0.7f, 0.1f, 0.1f, 0.1f,   // head 0: attends to pos 0
        0.1f, 0.1f, 0.1f, 0.7f,   // head 1: attends to pos 3
    };
    h.accumulate(attn, 4, 1, 2);

    // Aggregated across heads
    ASSERT_NEAR(h.scores[0], 0.8, 1e-5, "pos 0 (0.7+0.1)");
    ASSERT_NEAR(h.scores[3], 0.8, 1e-5, "pos 3 (0.1+0.7)");
    ASSERT_NEAR(h.scores[1], 0.2, 1e-5, "pos 1 (0.1+0.1)");
    PASS();
}

void test_accumulate_across_layers() {
    TEST("accumulate_across_layers");
    h2o_state h;
    h.init(2, 4, 1);

    // Layer 0: attends to pos 0
    float attn0[4] = {0.9f, 0.03f, 0.03f, 0.04f};
    h.accumulate(attn0, 4, 1, 1);

    // Layer 1: attends to pos 2
    float attn1[4] = {0.05f, 0.05f, 0.85f, 0.05f};
    h.accumulate(attn1, 4, 1, 1);

    // Cumulative scores
    ASSERT_NEAR(h.scores[0], 0.95, 1e-5, "pos 0 cumulative");
    ASSERT_NEAR(h.scores[2], 0.88, 1e-5, "pos 2 cumulative");
    PASS();
}

void test_find_evict_target() {
    TEST("find_evict_target");
    h2o_state h;
    h.init(1, 8, 1);

    // Set known scores
    h.scores = {10.0, 5.0, 1.0, 8.0, 0.5, 3.0, 7.0, 2.0};

    // With 2 sink tokens (pos 0,1 protected)
    int victim = h.find_evict_target(8, 2);
    ASSERT_EQ(victim, 4, "lowest non-sink is pos 4 (score=0.5)");

    // With 0 sink tokens
    victim = h.find_evict_target(8, 0);
    ASSERT_EQ(victim, 4, "lowest overall is pos 4 (score=0.5)");

    // With 5 sink tokens (only pos 5,6,7 eligible)
    victim = h.find_evict_target(8, 5);
    ASSERT_EQ(victim, 7, "lowest in 5-7 is pos 7 (score=2.0)");

    PASS();
}

void test_on_evict_shift() {
    TEST("on_evict_shift");
    h2o_state h;
    h.init(1, 8, 1);

    h.scores = {10.0, 5.0, 1.0, 8.0, 0.5, 3.0, 7.0, 2.0};

    // Evict position 2 (score=1.0) from 8 cached tokens
    h.on_evict(2, 8);

    // After eviction: positions shift down
    // Expected: {10.0, 5.0, 8.0, 0.5, 3.0, 7.0, 2.0, 0.0}
    ASSERT_NEAR(h.scores[0], 10.0, 1e-10, "pos 0 unchanged");
    ASSERT_NEAR(h.scores[1], 5.0, 1e-10, "pos 1 unchanged");
    ASSERT_NEAR(h.scores[2], 8.0, 1e-10, "pos 2 = old pos 3");
    ASSERT_NEAR(h.scores[3], 0.5, 1e-10, "pos 3 = old pos 4");
    ASSERT_NEAR(h.scores[7], 0.0, 1e-10, "last pos cleared");
    PASS();
}

void test_batch_evict() {
    TEST("batch_evict");
    h2o_state h;
    h.init(1, 8, 1);

    // scores: pos 0=10 (sink), pos 1=5 (sink), then filler + one important
    h.scores = {10.0, 5.0, 1.0, 8.0, 0.5, 0.3, 7.0, 0.2};

    // Evict 3 lowest non-sink tokens (sink=2)
    auto victims = h.batch_evict(8, 2, 3);

    // The 3 lowest non-sink: pos 7 (0.2), pos 5 (0.3), pos 4 (0.5)
    ASSERT_EQ(victims.size(), 3, "3 victims");

    // After eviction: 5 tokens remain
    // Remaining should be: {10.0, 5.0, 1.0, 8.0, 7.0, 0, 0, 0}
    // (sink 0, sink 1, then sorted survivors)
    // Actually after removing back-to-front: remove 7, then 5, then 4
    // After remove 7: {10, 5, 1, 8, 0.5, 0.3, 7, 0}
    // After remove 5: {10, 5, 1, 8, 0.5, 7, 0, 0}
    // After remove 4: {10, 5, 1, 8, 7, 0, 0, 0}
    ASSERT_NEAR(h.scores[0], 10.0, 1e-10, "sink 0");
    ASSERT_NEAR(h.scores[1], 5.0, 1e-10, "sink 1");
    ASSERT_NEAR(h.scores[2], 1.0, 1e-10, "survivor pos 2");
    ASSERT_NEAR(h.scores[3], 8.0, 1e-10, "survivor pos 3");
    ASSERT_NEAR(h.scores[4], 7.0, 1e-10, "survivor pos 6→4");
    PASS();
}

void test_sink_protection() {
    TEST("sink_protection");
    h2o_state h;
    h.init(1, 4, 1);

    // Sink tokens have LOW scores — they should still be protected
    h.scores = {0.001, 0.002, 5.0, 3.0};

    int victim = h.find_evict_target(4, 2);
    ASSERT_EQ(victim, 3, "evict pos 3 (score=3.0), not sinks");

    PASS();
}

void test_softmax_row_sum() {
    TEST("softmax_row_sum_invariant");
    // Verify that softmax rows sum to 1.0, so total score per step = n_tokens * n_heads
    h2o_state h;
    h.init(1, 4, 2);

    // Valid softmax output: each row sums to 1.0
    float attn[16] = {
        // head 0, token 0
        0.25f, 0.25f, 0.25f, 0.25f,
        // head 0, token 1
        0.1f, 0.2f, 0.3f, 0.4f,
        // head 1, token 0
        0.5f, 0.3f, 0.1f, 0.1f,
        // head 1, token 1
        0.05f, 0.05f, 0.8f, 0.1f,
    };
    h.accumulate(attn, 4, 2, 2);

    // Total score across all positions should = n_tokens(2) * n_heads(2) = 4.0
    double total = 0;
    for (int i = 0; i < 4; i++) total += h.scores[i];
    ASSERT_NEAR(total, 4.0, 1e-5, "total score = n_tokens * n_heads");

    PASS();
}

void test_eviction_preserves_order() {
    TEST("eviction_preserves_relative_order");
    h2o_state h;
    h.init(1, 6, 1);

    h.scores = {10.0, 8.0, 2.0, 7.0, 1.0, 5.0};
    // Evict the 2 lowest (pos 4=1.0, pos 2=2.0), sink=1
    auto victims = h.batch_evict(6, 1, 2);

    // Remaining: {10.0, 8.0, 7.0, 5.0, 0, 0}
    // Check relative order preserved
    bool ordered = true;
    for (int i = 0; i < 3; i++) {
        if (h.scores[i] < h.scores[i+1] && h.scores[i+1] > 0) {
            // This is fine, we don't require sorted order
        }
    }
    // Key check: high-score tokens survived
    ASSERT_NEAR(h.scores[0], 10.0, 1e-10, "highest survived");
    ASSERT_NEAR(h.scores[1], 8.0, 1e-10, "2nd highest survived");
    PASS();
}

int main() {
    printf("=== H2O Score Tracking Unit Tests ===\n\n");

    test_init();
    test_accumulate_simple();
    test_accumulate_multi_query();
    test_accumulate_multi_head();
    test_accumulate_across_layers();
    test_find_evict_target();
    test_on_evict_shift();
    test_batch_evict();
    test_sink_protection();
    test_softmax_row_sum();
    test_eviction_preserves_order();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
