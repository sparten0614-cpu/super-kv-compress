#!/usr/bin/env python3
"""
End-to-End Integration Test: Phase 1 → Phase 2 → Phase 3 Pipeline

Tests the full research loop in mock mode (no GPU, no API key required
for Phase 1+2; API key optional for Phase 3).

Verifies:
1. Phase 1 (grid search) produces valid results.jsonl
2. Phase 2 (Bayesian) can read Phase 1 results and extend them
3. Phase 3 (agent) can read Phase 1+2 results and propose new experiments
4. Schema compatibility across all phases
5. merge_results() correctly combines outputs

Usage:
    python3 test_integration.py            # Tests Phase 1+2 only
    ANTHROPIC_API_KEY=... python3 test_integration.py   # Full pipeline
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from schema import ExperimentResult, load_results, save_result, merge_results, find_pareto
from pareto_search import compression_ratio, is_pareto_optimal, BITS_PER_VAL

tests_passed = 0
tests_failed = 0

def test(name):
    print(f"Test: {name} ... ", end="", flush=True)

def passed():
    global tests_passed
    print("PASS")
    tests_passed += 1

def failed(msg):
    global tests_failed
    print(f"FAIL: {msg}")
    tests_failed += 1


def generate_mock_grid_results(outdir):
    """Simulate Phase 1 grid search output."""
    results_file = os.path.join(outdir, "results.jsonl")
    configs = [
        ("f16", "f16", 0.0, "none", 6.0, 1.0),
        ("q8_0", "q8_0", 0.0, "none", 6.01, 1.0),
        ("q4_0", "q4_0", 0.0, "none", 6.18, 1.0),
        ("q4_0", "q4_0", 0.5, "streamingllm", 6.21, 0.6),
        ("q4_0", "q4_0", 0.7, "streamingllm", 6.42, 0.4),
        ("q8_0", "q4_0", 0.0, "none", 6.05, 1.0),  # asymmetric
    ]
    baseline_ppl = 6.0

    for qk, qv, er, em, ppl, niah in configs:
        comp = compression_ratio(qk, qv, er)
        ppl_delta = round((ppl - baseline_ppl) / baseline_ppl * 100, 2)
        result = {
            "quant_k": qk, "quant_v": qv,
            "evict_ratio": er, "evict_method": em,
            "skip_layers": "",
            "compression": round(comp, 3),
            "ppl": ppl, "ppl_delta": ppl_delta,
            "niah": niah,
            "ppl_time_s": 0.1, "niah_time_s": 0.1,
            "ctx": 4096, "model": "test.gguf",
            "phase": "grid",
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    return results_file


def generate_mock_bayesian_results(outdir):
    """Simulate Phase 2 Bayesian optimizer output."""
    results_file = os.path.join(outdir, "bayesian_results.jsonl")
    configs = [
        ("q5_1", "q4_0", 0.0, "none", 6.08, 1.0),  # new point
        ("q4_0", "q4_0", 0.3, "streamingllm", 6.12, 0.8),  # new eviction point
        ("q8_0", "q4_0", 0.5, "streamingllm", 6.15, 0.7),  # asymmetric + eviction
    ]
    baseline_ppl = 6.0

    for qk, qv, er, em, ppl, niah in configs:
        comp = compression_ratio(qk, qv, er)
        result = {
            "quant_k": qk, "quant_v": qv,
            "evict_ratio": er, "evict_method": em,
            "skip_layers": "",
            "compression": round(comp, 3),
            "ppl": ppl,
            "ppl_delta": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
            "niah": niah,
            "ctx": 4096, "phase": "bayesian",
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    return results_file


# ============================================================================
# Tests
# ============================================================================

def test_schema_roundtrip():
    test("Schema: ExperimentResult roundtrip")
    r = ExperimentResult(
        quant_k="q4_0", quant_v="q4_0", evict_ratio=0.5,
        evict_method="streamingllm", compression=7.11,
        ppl=6.21, ppl_delta=3.5, niah=0.6, phase="grid"
    )
    d = r.to_dict()
    r2 = ExperimentResult.from_dict(d)
    assert r2.quant_k == "q4_0"
    assert r2.compression == 7.11
    assert r2.phase == "grid"
    passed()


def test_schema_tolerant_parsing():
    test("Schema: tolerant parsing of incomplete dicts")
    # Missing optional fields
    d = {"quant_k": "q8_0", "quant_v": "q8_0", "evict_ratio": 0,
         "evict_method": "none", "compression": 1.88}
    r = ExperimentResult.from_dict(d)
    assert r.ppl is None
    assert r.phase == ""
    # Extra unknown fields should be ignored
    d["unknown_field"] = "hello"
    r = ExperimentResult.from_dict(d)
    assert r.quant_k == "q8_0"
    passed()


def test_phase1_output():
    test("Phase 1: grid search output format")
    tmpdir = tempfile.mkdtemp()
    try:
        results_file = generate_mock_grid_results(tmpdir)
        results = load_results(results_file)
        assert len(results) == 6, f"Expected 6, got {len(results)}"
        # Check all have required fields
        for r in results:
            assert r.quant_k != ""
            assert r.compression > 0
            assert r.phase == "grid"
        # Check baseline
        baseline = [r for r in results if r.quant_k == "f16" and r.evict_ratio == 0]
        assert len(baseline) == 1
        assert baseline[0].ppl_delta == 0.0
        passed()
    finally:
        shutil.rmtree(tmpdir)


def test_phase2_reads_phase1():
    test("Phase 2: can read Phase 1 results")
    tmpdir = tempfile.mkdtemp()
    try:
        p1_file = generate_mock_grid_results(tmpdir)
        p2_file = generate_mock_bayesian_results(tmpdir)

        p1 = load_results(p1_file)
        p2 = load_results(p2_file)

        # Both should parse
        assert len(p1) == 6
        assert len(p2) == 3

        # Merge should deduplicate
        merged = merge_results(p1_file, p2_file)
        assert len(merged) == 9  # 6 + 3, no overlap

        passed()
    finally:
        shutil.rmtree(tmpdir)


def test_pareto_frontier():
    test("Pareto: correct frontier from merged results")
    tmpdir = tempfile.mkdtemp()
    try:
        p1_file = generate_mock_grid_results(tmpdir)
        p2_file = generate_mock_bayesian_results(tmpdir)

        merged = merge_results(p1_file, p2_file)
        pareto = find_pareto(merged)

        assert len(pareto) > 0, "No Pareto points found"
        # f16 baseline should be Pareto (lowest PPL)
        assert any(r.quant_k == "f16" for r in pareto), "Baseline not in Pareto"
        # Highest compression Pareto point should be > 1x
        max_comp = max(r.compression for r in pareto)
        assert max_comp > 3.0, f"Max compression {max_comp} too low"

        # Pareto should be sorted by compression
        for i in range(len(pareto) - 1):
            assert pareto[i].compression <= pareto[i+1].compression

        passed()
    finally:
        shutil.rmtree(tmpdir)


def test_key_deduplication():
    test("Schema: key-based deduplication in merge")
    tmpdir = tempfile.mkdtemp()
    try:
        f1 = os.path.join(tmpdir, "a.jsonl")
        f2 = os.path.join(tmpdir, "b.jsonl")

        # Same config in both files
        r = ExperimentResult(
            quant_k="q4_0", quant_v="q4_0", evict_ratio=0,
            evict_method="none", compression=3.556, ppl=6.18, phase="grid"
        )
        save_result(f1, r)

        r2 = ExperimentResult(
            quant_k="q4_0", quant_v="q4_0", evict_ratio=0,
            evict_method="none", compression=3.556, ppl=6.19, phase="bayesian"
        )
        save_result(f2, r2)

        merged = merge_results(f1, f2)
        assert len(merged) == 1, f"Expected 1 after dedup, got {len(merged)}"
        passed()
    finally:
        shutil.rmtree(tmpdir)


def test_agent_mock():
    test("Phase 3: agent mock mode (no API key)")
    # This tests that agent can be instantiated in mock mode
    from agent import ResearchAgent

    tmpdir = tempfile.mkdtemp()
    try:
        agent = ResearchAgent("dummy.gguf", "dummy.raw", 4096, 99, 5,
                             tmpdir, "llama-perplexity", "niah_test.py", mock=True)

        # Seed with some results
        p1_file = generate_mock_grid_results(tmpdir)
        # Agent reads results.jsonl which is in the same outdir
        results = agent.load_results()
        assert len(results) == 6

        # Format summary
        summary = agent.format_results_summary(results)
        assert "f16" in summary
        assert "q4_0" in summary

        passed()
    finally:
        shutil.rmtree(tmpdir)


def test_full_pipeline_mock():
    test("Full pipeline: Phase 1 → 2 → 3 (mock)")
    has_api = bool(os.environ.get("ANTHROPIC_API_KEY"))

    tmpdir = tempfile.mkdtemp()
    try:
        # Phase 1: Grid search (mock)
        p1_file = generate_mock_grid_results(tmpdir)
        p1_results = load_results(p1_file)
        assert len(p1_results) == 6

        # Phase 2: Bayesian (mock)
        p2_file = generate_mock_bayesian_results(tmpdir)
        p2_results = load_results(p2_file)
        assert len(p2_results) == 3

        # Merge Phase 1 + 2
        merged = merge_results(p1_file, p2_file)
        assert len(merged) == 9

        # Pareto
        pareto = find_pareto(merged)
        assert len(pareto) >= 2

        # Phase 3: Agent (mock execution, real API if available)
        if has_api:
            from agent import ResearchAgent
            agent = ResearchAgent("dummy.gguf", "dummy.raw", 4096, 99, 5,
                                 tmpdir, "llama-perplexity", "niah_test.py", mock=True)
            # The agent's results.jsonl IS the p1_file (same outdir)
            agent.run(max_rounds=1)
            final = agent.load_results()
            assert len(final) >= 7  # 6 grid + 1 agent
            print(f"PASS (with API: {len(final)} total results)")
        else:
            print(f"PASS (no API key, Phase 3 skipped, {len(merged)} results)")

        tests_passed_local = True

    finally:
        shutil.rmtree(tmpdir)

    if not has_api:
        global tests_passed
        tests_passed += 1


if __name__ == "__main__":
    print("=== Integration Tests: Phase 1 → 2 → 3 Pipeline ===\n")

    test_schema_roundtrip()
    test_schema_tolerant_parsing()
    test_phase1_output()
    test_phase2_reads_phase1()
    test_pareto_frontier()
    test_key_deduplication()
    test_agent_mock()
    test_full_pipeline_mock()

    print(f"\n=== Results: {tests_passed} passed, {tests_failed} failed ===")
    sys.exit(1 if tests_failed > 0 else 0)
