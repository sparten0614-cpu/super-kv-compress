# AutoResearch: Automated KV Cache Compression Search

**Author:** 宁宁
**Date:** 2026-04-01
**Status:** System specification

---

## 1. Vision

An automated system that searches the space of KV cache compression configurations and discovers optimal Pareto-front solutions for any given model. Instead of manually testing configurations one by one, the system:

1. Defines a structured search space of compression techniques
2. Evaluates candidates using a tiered protocol (fast screening → full evaluation)
3. Maintains a Pareto front of (compression, quality) trade-offs
4. Proposes novel combinations and validates them automatically

Analogous to NAS (Neural Architecture Search) but for KV cache compression.

---

## 2. Search Space

### 2.1 Dimension 1: Quantization Configuration

Each layer $l$ has independent K and V type selection:

```
quant_config[l] = {
    k_type: {f16, q8_0, q6_0*, q5_0, q5_1, q4_0, q4_1, q2_0*},
    v_type: {f16, q8_0, q6_0*, q5_0, q5_1, q4_0, q4_1, q2_0*}
}
```
*Types marked with * require custom implementation in llama.cpp

**Constraints:**
- k_type ≥ v_type (K more sensitive, from our empirical finding)
- If GQA ratio ≥ 7: k_type ≥ q8_0 (Qwen finding)
- Outlier layers (K_max > 50): k_type ≥ q8_0 or skip

**Search complexity:** Per-layer: 8 × 8 = 64 options. For L=32 layers: 64^32 ≈ 10^57 (infeasible for exhaustive search → need intelligent sampling).

**Practical reduction:**
- Group layers by profile (uniform, first-N, last-N, alternating)
- Start with uniform config, then refine per-layer
- Use calibration data to identify outlier layers and treat them separately

### 2.2 Dimension 2: Eviction Strategy

```
eviction_config = {
    method: {none, streamingllm, h2o, snapkv, expected_attention, pyramid_kv},
    eviction_rate: [0.0, 0.95],  // fraction of tokens evicted
    sink_tokens: [4, 256],       // attention sink count
    recent_window: [64, 4096],   // recent token window size
    # Method-specific params:
    h2o_budget: [0.05, 0.50],    // heavy hitter fraction
    snapkv_observation_window: [32, 256],
    expected_attention_gaussian: {diagonal, full},  // query covariance model
}
```

**Constraints:**
- StreamingLLM/H2O: no query access at prefill
- SnapKV: requires query in prompt tail
- Expected Attention: training-free, needs calibration for μ_q and Σ_q

### 2.3 Dimension 3: Layer Skip Configuration

```
skip_config = {
    skip_layers: Set[int],       // layers that bypass compression
    skip_criterion: {manual, outlier_detection, sensitivity_profiling},
    outlier_threshold: float,    // K_max threshold for auto-detection
}
```

**Auto-detection algorithm:**
1. Run calibration (10 chunks × 512 tokens)
2. Compute K_max per layer
3. Layers with K_max > threshold → skip or use higher precision
4. Compute per-layer PPL sensitivity → layers with sensitivity > 2× median → skip

### 2.4 Dimension 4: Advanced Methods (Proposed)

```
advanced_config = {
    low_rank: {none, palu},           // SVD dimension reduction
    low_rank_ratio: [0.25, 1.0],      // fraction of head_dim retained
    vq_codebook: {none, rvq, commvq}, // vector quantization
    vq_bits: [1, 4],
    selective_recompute: {none, topk}, // recompute top-k at decode
    recompute_k: [1, 64],             // number of tokens to recompute
    token_merge: {none, cam, kvmerger},
    merge_ratio: [0.0, 0.5],
}
```

### 2.5 Composite Search Space Size

| Dimension | Parameters | Effective Options |
|-----------|-----------|-------------------|
| Quantization (uniform) | K_type × V_type | 64 |
| Quantization (per-layer group, 4 groups) | 64^4 | ~16M |
| Eviction | method × rate × params | ~500 |
| Layer skip | subset of L layers | 2^L (~4B for L=32) |
| Advanced | low_rank × vq × recompute | ~200 |

**Total (uniform quant):** ~64 × 500 × 32 × 200 ≈ 200M configurations
**Total (per-layer):** intractable → must use intelligent search

---

## 3. Evaluation Protocol

### 3.1 Tier 1: Ultra-Fast Screening (< 1 min)

**Purpose:** Eliminate obviously bad configurations before expensive evaluation.

```yaml
tier1:
  context: 512
  metric: PPL only
  dataset: wikitext-2 (first 2K tokens)
  pass_criterion: PPL_delta < 10%
  throughput: ~100 configs/hour
```

**Rationale:** 512-token PPL is a weak but fast signal. Configs that fail here will certainly fail at 16K.

### 3.2 Tier 2: Fast Evaluation (< 5 min)

**Purpose:** Dual-metric screening on short context.

```yaml
tier2:
  context: 4096
  metrics:
    - PPL (wikitext-2, 5 chunks)
    - NIAH (single needle, 50% position)
  pass_criterion: PPL_delta < 5% AND NIAH == FOUND
  throughput: ~12 configs/hour
```

**Key insight from our research:** NIAH at 50% position catches the most common eviction failure mode (middle tokens evicted). A single-point NIAH is sufficient for screening.

### 3.3 Tier 3: Full Evaluation (< 30 min)

**Purpose:** Complete characterization for Pareto-front candidates.

```yaml
tier3:
  context: 16384
  metrics:
    - PPL (wikitext-2, 2 chunks)
    - NIAH (5 positions: 10%, 25%, 50%, 75%, 90%)
    - LongBench v2 (20 questions)
  pass_criterion: Pareto-optimal in (compression, quality) space
  throughput: ~2 configs/hour
```

### 3.4 Tier 4: Stress Test (< 2 hours)

**Purpose:** Final validation before publication/deployment.

```yaml
tier4:
  contexts: [4096, 8192, 16384, 32768]
  metrics:
    - PPL gradient across contexts
    - NIAH at all contexts × all positions (25 data points)
    - LongBench v2
    - Latency benchmark (tokens/sec)
    - Memory measurement (peak GPU/CPU)
  pass_criterion: no regression vs tier3 results
```

---

## 4. Search Algorithm

### 4.1 Phase 1: Coarse Grid Search

Enumerate a small grid of canonical configurations:

```python
coarse_grid = [
    # Quantization only
    {"quant": "K8V8", "evict": "none"},
    {"quant": "K8V4", "evict": "none"},
    {"quant": "K6V4", "evict": "none"},
    {"quant": "K4V4", "evict": "none"},  # skip if GQA ≥ 7
    
    # Eviction only
    {"quant": "f16", "evict": "streamingllm_50"},
    {"quant": "f16", "evict": "h2o_50"},
    {"quant": "f16", "evict": "streamingllm_70"},
    {"quant": "f16", "evict": "h2o_70"},
    
    # Combined
    {"quant": "K8V4", "evict": "streamingllm_50"},
    {"quant": "K6V4", "evict": "h2o_50"},
    {"quant": "K8V4", "evict": "h2o_70"},
    
    # With layer skip
    {"quant": "K4V4", "evict": "none", "skip": "auto_outlier"},
]
```

Run all through Tier 1 → Tier 2 → select top-10 for Tier 3.

### 4.2 Phase 2: Bayesian Optimization

Use Tier 2 results to build a surrogate model:

```python
# Gaussian Process over search space
gp = GaussianProcess(
    X=evaluated_configs,
    y=[(compression, ppl_delta, niah_accuracy)],
    kernel=Matern52
)

# Acquisition function: Expected Hypervolume Improvement (EHVI)
# for multi-objective Pareto optimization
next_config = argmax(EHVI(gp, pareto_front))
```

**Multi-objective targets:**
- Maximize: compression ratio ρ
- Minimize: PPL_delta (subject to < X%)
- Maximize: NIAH accuracy (subject to > Y%)

EHVI naturally handles the Pareto trade-off.

### 4.3 Phase 3: Novelty Proposals

After exhausting the predefined search space, the system generates novel combinations:

1. **Interpolation:** If K8V4 and K6V4 are both on the Pareto front, try K7V4 (requires custom quant type)
2. **Per-layer specialization:** Use sensitivity profiling to assign different configs per layer
3. **Method combination:** If quantization and eviction are Pareto-complementary, try all cross-products
4. **Hypothesis testing:** Generate hypotheses from patterns in the search data (e.g., "layer 0 always benefits from higher K precision" → test on new models)

---

## 5. Pareto Front Specification

### 5.1 Objective Space

```
objectives = {
    compression: float,     # ρ = f16_memory / compressed_memory
    ppl_delta: float,       # % change from f16 baseline
    niah_accuracy: float,   # 0.0 to 1.0 (5-position average)
    latency_ratio: float,   # decode tokens/sec relative to f16
}
```

### 5.2 Constraint Profiles

Users select a profile that defines acceptable trade-offs:

| Profile | PPL Constraint | NIAH Constraint | Target Use Case |
|---------|---------------|-----------------|-----------------|
| **Lossless** | < 0.5% | = 100% | Production serving |
| **Near-lossless** | < 1% | ≥ 80% | Cost-optimized serving |
| **Aggressive** | < 5% | ≥ 60% | Batch inference, non-retrieval |
| **Maximum** | < 10% | any | Research exploration |

### 5.3 Expected Pareto Front (from our data)

```
Lossless region:
  K8V4 (GQA≥7) or K6V4 (GQA≤4): 2.5-3.2x, PPL<0.5%, NIAH=100%

Near-lossless region:
  K4V4 (GQA≤4): 4x, PPL~3%, NIAH=100%
  K8V4 + 50% eviction: 5x, PPL<1%, NIAH~60% (eviction limits NIAH)

Aggressive region:
  K6V4 + 70% eviction (16K): 8-9x, PPL~1%, NIAH~40%
  
Maximum region:
  K4V4 + 85% eviction: 21x+, PPL~8%, NIAH~20%
```

---

## 6. Implementation Architecture

```
┌─────────────────────────────────────────┐
│           AutoResearch Controller        │
│  - Search space definition              │
│  - Bayesian optimizer (EHVI)            │
│  - Pareto front manager                 │
│  - Result database (SQLite)             │
└───────────┬─────────────────────────────┘
            │
    ┌───────┴───────┐
    │  Config Gen   │ → generates llama.cpp CLI args
    └───────┬───────┘
            │
    ┌───────┴───────┐
    │  Evaluator    │ → runs llama.cpp benchmarks
    │  (Tiered)     │
    └───────┬───────┘
            │
    ┌───────┴───────┐
    │  Reporter     │ → Pareto plots, config recommendations
    └───────────────┘
```

### 6.1 Config → CLI Translation

```python
def config_to_cli(config, model_path):
    args = [
        f"--model {model_path}",
        f"--cache-type-k {config['k_type']}",
        f"--cache-type-v {config['v_type']}",
        f"--ctx-size {config['context']}",
    ]
    if config.get('eviction') == 'streamingllm':
        args.extend([
            f"--grp-attn-n 1",
            f"--grp-attn-w {int(config['context'] * (1 - config['eviction_rate']))}",
        ])
    return " ".join(args)
```

### 6.2 Result Schema

```sql
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    model TEXT,
    config_json TEXT,
    tier INTEGER,
    compression REAL,
    ppl REAL,
    ppl_delta REAL,
    niah_accuracy REAL,
    longbench_score REAL,
    latency_tps REAL,
    memory_mb REAL,
    timestamp TEXT,
    is_pareto BOOLEAN
);
```

---

## 7. Calibration Protocol

Before search begins, run a one-time calibration:

1. **Baseline PPL + NIAH** (f16, 16K): establishes ground truth
2. **Outlier profiling:** K_max per layer → identify skip candidates
3. **Sensitivity profiling:** Per-layer PPL contribution under q4_0 → rank layer sensitivity
4. **GQA analysis:** Extract n_q, n_kv from model config → set K bit floor

Calibration output feeds into search space constraints (narrowing infeasible regions before search).

---

## 8. Novel Method Proposal Engine

Beyond searching predefined configs, the system can propose and test novel approaches:

### 8.1 Hypothesis Generation
```python
hypotheses = [
    # From pattern: "outlier layers dominate PPL loss"
    H("Skip top-3 sensitive layers → same compression, lower PPL?"),
    
    # From pattern: "K>V sensitivity universally"
    H("K=q8, V=q2 → more compression than K=q4, V=q4?"),
    
    # From theory: "Expected Attention is NIAH-safe"
    H("Replace H2O with Expected Attention → same eviction rate, better NIAH?"),
    
    # From cross-model: "70B tolerates more eviction"
    H("Larger models: eviction rate can scale with model size?"),
    
    # Combination: "low-rank + quant stacking"
    H("PALU r=64 + q4_0 → 8x compression, better than q2_0 alone?"),
]
```

### 8.2 Automated Validation
Each hypothesis is converted to a config, evaluated through the tiered protocol, and the result is stored with the hypothesis for future reference.

---

## 9. Deliverables

1. **`autoresearch/` Python package** — search controller, evaluator, reporter
2. **`autoresearch/configs/` preset configs** — canonical grid from our experiments
3. **`autoresearch/results/` database** — all evaluation results (shareable across team)
4. **CLI interface:**
   ```bash
   # Run full search for a model
   python -m autoresearch search --model llama-3.1-8b --profile near-lossless
   
   # Evaluate a single config
   python -m autoresearch eval --model llama-3.1-8b --k q8_0 --v q4_0 --evict h2o_50
   
   # Show Pareto front
   python -m autoresearch pareto --model llama-3.1-8b
   ```
5. **Integration with our paper** — all paper results reproducible via `autoresearch eval`

---

## 10. Timeline

| Phase | Deliverable | Time |
|-------|-------------|------|
| 1 | Evaluator (Tier 1-3) + coarse grid | 3 days |
| 2 | Bayesian optimizer + Pareto manager | 3 days |
| 3 | Novelty proposal engine | 2 days |
| 4 | CLI + documentation + paper integration | 2 days |
