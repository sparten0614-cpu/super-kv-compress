# D4: Asymmetric K/V Quantization — Experiment Plan

**Date:** 2026-03-31
**Author:** 宁宁
**Status:** Ready to run (pending 阳阳 Sparse V completion)

---

## Hypothesis

At a fixed memory budget, allocating more bits to Keys and fewer to Values yields better quality than uniform allocation, because Keys are ~2x more sensitive to quantization (softmax nonlinearity amplifies K errors; V errors propagate linearly).

**Specific claim:** K=tqkv_6 + V=tqkv_4 at ~5-bit average will match or beat uniform tqkv_6 (6-bit) quality while using ~17% less KV cache memory.

---

## Experiment Matrix

### Experiment 1: Asymmetric vs Uniform (Same Average Budget)

**Models:** Llama-3.1-8B-Instruct Q4_K_M, Llama-2-7B Q4_K_M

| Config | K Cache | V Cache | Avg Bits | Expected KV Size (4096 ctx) | Notes |
|--------|---------|---------|----------|----------------------------|-------|
| A | F16 | F16 | 16 | ~400 MiB | Baseline |
| B | tqkv_6 | tqkv_6 | 6 | ~200 MiB | Current best (proven lossless) |
| C | tqkv_6 | tqkv_4 | 5 | ~167 MiB | **Primary test** |
| D | tqkv_4 | tqkv_4 | 4 | ~133 MiB | Uniform 4-bit reference |
| E | tqkv_6 | tqkv_2 | 4 | ~133 MiB | Aggressive asymmetric (same budget as D) |

**Commands:**
```bash
# Config A: F16 baseline
DYLD_LIBRARY_PATH=. ./llama-perplexity -m model.gguf -f wiki.test.raw --ctx-size 512 --chunks 10 -ngl 99

# Config B: Uniform tqkv_6
DYLD_LIBRARY_PATH=. ./llama-perplexity -m model.gguf -f wiki.test.raw --ctx-size 512 --chunks 10 -ngl 99 --cache-type-k tqkv_6 --cache-type-v tqkv_6

# Config C: Asymmetric K6/V4
DYLD_LIBRARY_PATH=. ./llama-perplexity -m model.gguf -f wiki.test.raw --ctx-size 512 --chunks 10 -ngl 99 --cache-type-k tqkv_6 --cache-type-v tqkv_4

# Config D: Uniform tqkv_4
DYLD_LIBRARY_PATH=. ./llama-perplexity -m model.gguf -f wiki.test.raw --ctx-size 512 --chunks 10 -ngl 99 --cache-type-k tqkv_4 --cache-type-v tqkv_4

# Config E: Asymmetric K6/V2
DYLD_LIBRARY_PATH=. ./llama-perplexity -m model.gguf -f wiki.test.raw --ctx-size 512 --chunks 10 -ngl 99 --cache-type-k tqkv_6 --cache-type-v tqkv_2
```

**Metrics:** PPL (WikiText-2), delta vs F16

### Experiment 2: NIAH Validation

Run NIAH 4K for configs B, C, E to verify retrieval isn't degraded.

```bash
# Start server for each config, run benchmarks/niah.py
DYLD_LIBRARY_PATH=. ./llama-server -m model.gguf -ngl 99 -c 4096 \
  --cache-type-k tqkv_6 --cache-type-v tqkv_4 --port 8080
```

### Experiment 3: LongBench Quick Eval (if PPL results are promising)

Same 20-sample quick eval (seed=42) for the best asymmetric config vs uniform tqkv_6.

---

## Expected Results

Based on literature (KVTuner, WKVQuant, MagR):

| Config | Expected PPL Delta | Memory vs F16 | Prediction |
|--------|-------------------|---------------|------------|
| B (K6/V6) | +0.07% | 50% (2x) | Known result |
| C (K6/V4) | +0.2-0.5% | 42% (2.4x) | Should be close to B |
| D (K4/V4) | +1.2-1.7% | 33% (3x) | Known result |
| E (K6/V2) | +0.5-1.5% | 33% (3x) | Should beat D significantly |

**Key comparison:** Config E vs D — same memory budget (3x compression), but E protects K quality. Literature predicts E wins by 0.5-2 PPL points.

---

## Success Criteria

1. **Config C (K6/V4):** PPL delta < +0.3% → validates asymmetric approach, 2.4x compression
2. **Config E (K6/V2) vs D (K4/V4):** E has lower PPL → confirms K sensitivity > V sensitivity in our framework
3. **NIAH:** All configs maintain 100% retrieval at 4K

---

## Dependencies

- [x] llama-perplexity binary with TQKV support (have it)
- [x] llama-server binary with TQKV support (have it, from 阳阳)
- [ ] Verify tqkv_4 and tqkv_2 cache types work in both binaries
- [ ] Download model GGUFs again (deleted per cleanup — ~4.6GB each)
- [ ] 阳阳 Sparse V implementation complete (SZ: run after this)

---

## Execution Plan

1. **Pre-flight check** — Verify tqkv_4 and tqkv_2 are recognized by the binary:
   ```bash
   DYLD_LIBRARY_PATH=. ./llama-server --help 2>&1 | grep tqkv
   ```

2. **Download models** — Llama-3.1-8B Q4_K_M (~4.6GB)

3. **Run Experiment 1** — All 5 configs, PPL (sequential, ~2 min each = ~10 min total)

4. **Analyze** — Build comparison table, compute deltas

5. **Run Experiment 2** — NIAH for top 3 configs (~5 min)

6. **Run Experiment 3** — LongBench if warranted (~8 min)

7. **Update results** — benchmarks/results.md, push to GitHub

**Total estimated time: ~30 minutes of compute**

---

## Future Extensions (if D4 validates)

1. **Per-layer asymmetric** — Different (k_bits, v_bits) per layer based on sensitivity profiling
2. **Combined D4+D1** — Asymmetric K/V + adaptive layer skip
3. **Three-way asymmetric** — K=6, V_important=4, V_unimportant=2 (bridges to D2 eviction framework)
