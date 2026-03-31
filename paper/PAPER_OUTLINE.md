# Paper Outline — Beyond Perplexity: Why KV Cache Compression Needs Dual-Metric Evaluation

Status: Draft sections written (930 lines LaTeX). Needs ExpAttn update + TBD fills + polish.

## Structure (agreed narrative: quantization=main, eviction=cautionary, dual-metric=methodology)

### Abstract
- 3 findings: (1) quantization safety is architecture-dependent, (2) eviction has fundamental causal limit, (3) PPL is unreliable proxy
- Additive error law enables O(2n) calibration
- GQA-aware asymmetric quantization recommendation
- Status: WRITTEN, needs ExpAttn sentence added

### 1. Introduction (sections/introduction.tex, 27 lines)
- Motivation: KV cache = memory bottleneck for long-context LLMs
- 3 contributions listed with section references
- Status: WRITTEN

### 2. Related Work (sections/related.tex, 59 lines)
- 2.1 Quantization approaches (KIVI, QJL, GEAR, etc.)
- 2.2 Eviction/sparsity (H2O, StreamingLLM, SnapKV, PyramidKV)
- 2.3 Information-theoretic (Expected Attention, KVzip)
- Status: WRITTEN. **UPDATE NEEDED**: ExpAttn related work should note our negative result

### 3. Method (sections/method.tex, 98 lines)
- 3.1 Experimental setup (models, metrics, framework)
- 3.2 Quantization configurations
- 3.3 Eviction configurations (StreamingLLM, H2O)
- 3.4 NIAH test protocol
- Status: WRITTEN

### 4. Theoretical Analysis (sections/theory.tex, 194 lines)
- 4.1 Additive error law (K→softmax, V→linear, independent paths)
- 4.2 Eviction error bound
- 4.3 Two-knife stack (quantization + eviction joint bound)
- 4.4 Selective recompute ceiling (34.6x theoretical max)
- Status: WRITTEN

### 5. The Metric Gap (sections/metric_gap.tex, 120 lines) ← CORE CONTRIBUTION
- 5.1 PPL-NIAH divergence at 50% eviction
- 5.2 Why PPL fails for eviction (information redistribution)
- 5.3 Dual-metric evaluation recommendation
- Status: WRITTEN

### 6. Experiments (sections/experiments.tex, 327 lines)
- 6.1 Quantization safety (Table 1: cross-model)
- 6.2 Qwen asymmetric diagnostic (Table 2: K vs V)
- 6.3 Additive error verification (Table 3)
- 6.4 Eviction gradient (Tables 4-5: 4K and 16K)
- 6.5 NIAH under eviction (Table 6: StreamingLLM + H2O)
- 6.6 Metric gap summary (Table 7)
- 6.7 Pareto frontier (Table 8)
- 6.8 Cliff scaling analysis
- Status: WRITTEN with TBD values. **UPDATE NEEDED**:
  - Add ExpAttn NIAH results (Table 6 expansion)
  - Fill LongBench TBDs (waiting for 宁宁)
  - Fill Llama-70B 85% eviction TBD

### 7. Conclusion (sections/conclusion.tex, 45 lines)
- Summary of findings
- Future work: selective recompute, Expected Attention, GQA toolkit, PALU stacking
- Status: WRITTEN. **UPDATE NEEDED**: Replace ExpAttn future work bullet with Scenario A (negative result = confirms fundamental limit)

### References (references.bib)
- 24 references, properly formatted
- Status: WRITTEN

---

## Remaining Work (Priority Order)

### P0: Must-do before submission
1. **ExpAttn negative result integration** — Use Scenario A narrative from expected-attn-narratives.tex. Add ExpAttn rows to eviction NIAH table. Update conclusion future work bullet.
2. **Fill remaining TBDs** — q4_1 NIAH, LongBench scores (blocked on 宁宁), 70B 85% eviction (blocked on GPU)
3. **Abstract update** — Add sentence about ExpAttn confirming eviction limit is fundamental (four scoring strategies tested)

### P1: Should-do
4. **Figures** — PPL-NIAH metric gap visualization, Pareto frontier plot, eviction gradient curves
5. **Polish** — Consistent terminology, grammar pass, ensure all table references are correct
6. **Formatting** — Conference template (ACL/EMNLP/NeurIPS style), page limits

### P2: Nice-to-have
7. **Additional experiments** — More models, longer contexts, other eviction methods
8. **Code release** — Clean up llama.cpp fork for artifact submission
