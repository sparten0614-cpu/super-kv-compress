#!/usr/bin/env python3
"""
Needle-In-A-Haystack (NIAH) test for TurboQuant KV cache evaluation.

Tests retrieval accuracy: inserts a "needle" fact at various positions
in a "haystack" of filler text, then asks the model to recall the fact.

Usage:
    python3 niah_test.py --model path/to/model.gguf [--ctx 4096] [--cache-type tqkv_6]
"""

import subprocess
import argparse
import json
import os
import sys
import re

LLAMA_CLI = os.path.expanduser("~/zylos/workspace/turboquant/llama.cpp/build/bin/llama-completion")

# The needle: a specific fact to retrieve
NEEDLE = "The secret code for Project Alpha is 7492-BLUE-DELTA."

# The question that tests retrieval
QUESTION = "What is the secret code for Project Alpha?"
EXPECTED = "7492-BLUE-DELTA"

# Haystack filler: boring repetitive text that fills context
HAYSTACK_UNIT = """The city of Millbrook has a population of approximately 45,000 residents. It was founded in 1823 by settlers from the eastern seaboard. The town is known for its annual harvest festival, which takes place every October. The local economy is driven by agriculture and small manufacturing. The Millbrook River runs through the center of town, providing water for irrigation and recreation. Several parks line the riverbanks, offering walking trails and picnic areas. The public library, built in 1952, houses over 50,000 volumes and serves as a community gathering place. The school district operates three elementary schools, one middle school, and one high school. """

def build_prompt(needle, haystack_unit, target_ctx_tokens, needle_position_pct, model_path):
    """Build a prompt with needle inserted at specified position in haystack."""
    # Estimate: ~1.3 tokens per word, ~4 chars per token
    target_chars = target_ctx_tokens * 4

    # Build haystack
    haystack = ""
    while len(haystack) < target_chars:
        haystack += haystack_unit

    # Trim to approximate target
    haystack = haystack[:target_chars]

    # Insert needle at specified position
    insert_pos = int(len(haystack) * needle_position_pct)
    # Find a sentence boundary near the insert position
    period_pos = haystack.rfind('. ', 0, insert_pos)
    if period_pos == -1:
        period_pos = insert_pos
    else:
        period_pos += 2  # after the period and space

    prompt_text = haystack[:period_pos] + "\n" + needle + "\n" + haystack[period_pos:]

    # Add the question at the end
    prompt_text += f"\n\nBased on the information above, answer the following question:\n{QUESTION}\nAnswer:"

    return prompt_text


def run_niah_test(model_path, ctx_size, cache_type, needle_position_pct, n_gpu_layers=99, skip_layers=None, evict_mode=None, evict_ratio=None, evict_sink=None, h2o_eviction=False, snapkv_eviction=False):
    """Run a single NIAH test and return whether the needle was found."""

    # Reserve some tokens for the answer
    answer_tokens = 64
    prompt_ctx = ctx_size - answer_tokens

    prompt = build_prompt(NEEDLE, HAYSTACK_UNIT, prompt_ctx, needle_position_pct, model_path)

    # Build command
    # Write prompt to temp file (avoids shell escaping issues with long prompts)
    prompt_file = "/tmp/niah_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt)

    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size),
        "-n", str(answer_tokens),
        "-fa", "on",
        "--temp", "0",  # greedy decoding
        "-f", prompt_file,
    ]

    if cache_type and cache_type != "f16":
        cmd.extend(["--cache-type-k", cache_type, "--cache-type-v", cache_type])

    if evict_mode is not None:
        cmd.extend(["--evict-mode", str(evict_mode)])
    if evict_ratio is not None:
        cmd.extend(["--evict-ratio", str(evict_ratio)])
    if evict_sink is not None:
        cmd.extend(["--evict-sink", str(evict_sink)])

    if h2o_eviction:
        cmd.append("--h2o-eviction")

    if snapkv_eviction:
        cmd.append("--snapkv-eviction")

    env = os.environ.copy()
    if skip_layers:
        env["TQKV_SKIP_LAYERS"] = skip_layers

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        # llama-completion prints the full prompt+completion to stderr
        # stdout may be empty; check both
        full_output = result.stdout + "\n" + result.stderr

        # Extract answer: look for text after "Answer:" in full output
        answer_start = full_output.rfind("Answer:")
        if answer_start >= 0:
            answer = full_output[answer_start + 7:answer_start + 300].strip()
        else:
            answer = full_output[-300:]

        # Check if the expected code appears in the answer (NOT the full output,
        # which includes the prompt echo containing the needle itself)
        found = EXPECTED.lower() in answer.lower()

        return {
            "found": found,
            "answer": answer[:200],  # truncate for display
            "needle_pos": needle_position_pct,
            "ctx_size": ctx_size,
            "cache_type": cache_type or "f16",
        }
    except subprocess.TimeoutExpired:
        return {"found": False, "answer": "[TIMEOUT]", "needle_pos": needle_position_pct,
                "ctx_size": ctx_size, "cache_type": cache_type or "f16"}
    except Exception as e:
        return {"found": False, "answer": f"[ERROR: {e}]", "needle_pos": needle_position_pct,
                "ctx_size": ctx_size, "cache_type": cache_type or "f16"}


def main():
    parser = argparse.ArgumentParser(description="NIAH test for TurboQuant")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--ctx", type=int, default=512, help="Context size (default: 512)")
    parser.add_argument("--positions", type=str, default="0.1,0.25,0.5,0.75,0.9",
                        help="Needle positions as fractions (comma-separated)")
    parser.add_argument("--cache-types", type=str, default="f16,tqkv_6",
                        help="Cache types to test (comma-separated)")
    parser.add_argument("--skip-layers", type=str, default=None,
                        help="TQKV_SKIP_LAYERS value for outlier layers")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers")
    parser.add_argument("--evict-mode", type=int, default=None, help="Eviction mode (0=sliding, 1=StreamingLLM)")
    parser.add_argument("--evict-ratio", type=float, default=None, help="Eviction ratio (0.0-0.9)")
    parser.add_argument("--evict-sink", type=int, default=None, help="Attention sink tokens")
    parser.add_argument("--h2o-eviction", action="store_true", help="Use H2O attention-aware eviction")
    parser.add_argument("--snapkv-eviction", action="store_true", help="Use SnapKV one-shot prefill eviction")
    args = parser.parse_args()

    positions = [float(p) for p in args.positions.split(",")]
    cache_types = args.cache_types.split(",")

    evict_label = ""
    if args.evict_ratio is not None:
        method = "H2O" if args.h2o_eviction else "StreamingLLM"
        evict_label = f" + {method} {args.evict_ratio:.0%}"

    print(f"NIAH Test Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Context: {args.ctx}")
    print(f"  Positions: {positions}")
    print(f"  Cache types: {cache_types}")
    if args.evict_mode is not None:
        method = "H2O" if args.h2o_eviction else "StreamingLLM"
        print(f"  Eviction: mode={args.evict_mode} ratio={args.evict_ratio} sink={args.evict_sink} method={method}")
    print(f"  Needle: {NEEDLE}")
    print(f"  Question: {QUESTION}")
    print(f"  Expected: {EXPECTED}")
    print()

    results = {}
    for ct in cache_types:
        results[ct] = []
        for pos in positions:
            label = f"{ct}{evict_label}"
            print(f"Testing {label} @ position {pos:.0%}...", end=" ", flush=True)
            r = run_niah_test(args.model, args.ctx, ct if ct != "f16" else None,
                              pos, args.ngl, args.skip_layers,
                              args.evict_mode, args.evict_ratio, args.evict_sink,
                              args.h2o_eviction, args.snapkv_eviction)
            results[ct].append(r)
            status = "FOUND" if r["found"] else "MISS"
            print(f"{status} — {r['answer'][:80]}")

    # Summary table
    print("\n" + "="*60)
    print("NIAH Results Summary")
    print("="*60)
    header = f"{'Cache Type':>12} | " + " | ".join(f"{p:.0%}" for p in positions) + " | Accuracy"
    print(header)
    print("-"*len(header))
    for ct in cache_types:
        hits = [r["found"] for r in results[ct]]
        row = f"{ct:>12} | " + " | ".join("  ✓ " if h else "  ✗ " for h in hits)
        accuracy = sum(hits) / len(hits) * 100
        row += f" | {accuracy:.0f}%"
        print(row)


if __name__ == "__main__":
    main()
