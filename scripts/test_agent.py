#!/usr/bin/env python3
"""
Test the AutoResearch agent's Claude API integration and result parsing.

Tests:
1. Result loading and formatting
2. Claude API call (proposal generation)
3. Proposal JSON parsing
4. Mock execution loop (3 rounds)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 test_agent.py
"""

import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))
from agent import ResearchAgent
from pareto_search import compression_ratio, is_pareto_optimal

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


def test_compression_ratio():
    test("compression_ratio calculation")
    r = compression_ratio("f16", "f16", 0.0)
    assert abs(r - 1.0) < 0.01, f"f16/f16/0% should be 1.0x, got {r}"
    r = compression_ratio("q4_0", "q4_0", 0.0)
    assert abs(r - 3.556) < 0.1, f"q4_0/q4_0/0% should be ~3.56x, got {r}"
    r = compression_ratio("q4_0", "q4_0", 0.5)
    assert abs(r - 7.11) < 0.2, f"q4_0/q4_0/50% should be ~7.1x, got {r}"
    passed()


def test_pareto_optimal():
    test("Pareto frontier detection")
    results = [
        {"compression": 1.0, "ppl": 6.0, "quant_k": "f16", "quant_v": "f16", "evict_ratio": 0, "evict_method": "none"},
        {"compression": 3.5, "ppl": 6.2, "quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0, "evict_method": "none"},
        {"compression": 3.5, "ppl": 6.5, "quant_k": "q4_1", "quant_v": "q4_1", "evict_ratio": 0, "evict_method": "none"},
        {"compression": 7.0, "ppl": 7.0, "quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0.5, "evict_method": "streamingllm"},
    ]
    results = is_pareto_optimal(results)
    pareto = [r for r in results if r.get("pareto")]
    assert len(pareto) == 3, f"Expected 3 Pareto points, got {len(pareto)}"
    # q4_1/q4_1 should NOT be Pareto (q4_0/q4_0 dominates: same compression, better PPL)
    assert not results[2].get("pareto"), "q4_1 should be dominated"
    passed()


def test_result_formatting():
    test("Result summary formatting")
    tmpdir = tempfile.mkdtemp()
    try:
        agent = ResearchAgent("dummy.gguf", "dummy.raw", 4096, 99, 5,
                             tmpdir, "llama-perplexity", "niah_test.py", mock=True)
        # Write some mock results
        results = [
            {"quant_k": "f16", "quant_v": "f16", "evict_ratio": 0, "evict_method": "none",
             "compression": 1.0, "ppl": 6.0, "niah": 1.0},
            {"quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0, "evict_method": "none",
             "compression": 3.556, "ppl": 6.18, "niah": 1.0},
        ]
        summary = agent.format_results_summary(results)
        assert "f16" in summary, "Summary should contain f16"
        assert "q4_0" in summary, "Summary should contain q4_0"
        assert "3.56" in summary or "3.55" in summary, "Summary should show compression"
        passed()
    finally:
        shutil.rmtree(tmpdir)


def test_claude_api_call():
    test("Claude API call + proposal parsing")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("SKIP (no ANTHROPIC_API_KEY)")
        return

    tmpdir = tempfile.mkdtemp()
    try:
        agent = ResearchAgent("dummy.gguf", "dummy.raw", 4096, 99, 5,
                             tmpdir, "llama-perplexity", "niah_test.py", mock=True)

        # Provide some existing results
        results = [
            {"quant_k": "f16", "quant_v": "f16", "evict_ratio": 0, "evict_method": "none",
             "compression": 1.0, "ppl": 6.0, "niah": 1.0},
            {"quant_k": "q8_0", "quant_v": "q8_0", "evict_ratio": 0, "evict_method": "none",
             "compression": 1.88, "ppl": 6.01, "niah": 1.0},
            {"quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0, "evict_method": "none",
             "compression": 3.556, "ppl": 6.18, "niah": 1.0},
        ]

        proposal, text = agent.propose_experiment(results)

        assert proposal is not None, f"Failed to parse proposal from: {text[:200]}"
        assert "quant_k" in proposal, f"Proposal missing quant_k: {proposal}"
        assert "quant_v" in proposal, f"Proposal missing quant_v: {proposal}"
        assert "reasoning" in proposal, f"Proposal missing reasoning: {proposal}"

        print(f"PASS (proposed: {proposal.get('quant_k')}/{proposal.get('quant_v')} "
              f"evict={proposal.get('evict_ratio',0)})")
    finally:
        shutil.rmtree(tmpdir)


def test_mock_loop():
    test("Mock execution loop (3 rounds)")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("SKIP (no ANTHROPIC_API_KEY)")
        return

    tmpdir = tempfile.mkdtemp()
    try:
        agent = ResearchAgent("dummy.gguf", "dummy.raw", 4096, 99, 5,
                             tmpdir, "llama-perplexity", "niah_test.py", mock=True)

        # Seed with baseline
        baseline = {
            "quant_k": "f16", "quant_v": "f16", "evict_ratio": 0,
            "evict_method": "none", "skip_layers": "",
            "compression": 1.0, "ppl": 6.0, "niah": 1.0,
            "ppl_time_s": 0.1, "niah_time_s": 0.1, "ctx": 4096, "model": "dummy",
        }
        with open(agent.results_file, "w") as f:
            f.write(json.dumps(baseline) + "\n")

        agent.run(max_rounds=3)

        # Check results
        results = agent.load_results()
        assert len(results) >= 2, f"Expected >=2 results (1 seed + >=1 agent), got {len(results)}"

        # Check log exists
        assert os.path.exists(agent.log_file), "Agent log not created"

        passed()
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    print("=== AutoResearch Agent Tests ===\n")

    test_compression_ratio()
    test_pareto_optimal()
    test_result_formatting()
    test_claude_api_call()
    test_mock_loop()

    print(f"\n=== Results: {tests_passed} passed, {tests_failed} failed ===")
    sys.exit(1 if tests_failed > 0 else 0)
