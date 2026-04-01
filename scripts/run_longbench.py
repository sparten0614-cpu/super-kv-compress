#!/usr/bin/env python3
"""
LongBench v2 evaluation for KV cache compression experiments.

Downloads LongBench v2 dataset from HuggingFace, runs inference with llama-cli,
and scores results. Supports quantization and eviction configurations.

Usage:
    python3 run_longbench.py --model /path/to/model.gguf [--cache-type-k q4_0] [--cache-type-v q4_0] [--evict-pct 50]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Default paths (cloud GPU)
DEFAULT_MODEL = "/root/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEFAULT_BINARY = "/root/llama.cpp/build/bin/llama-cli"
RESULTS_DIR = Path(__file__).parent.parent / "benchmarks" / "longbench" / "results"

# LongBench v2 — 20 multiple-choice questions with long contexts
# We use the official dataset from HuggingFace: THUDM/LongBench-v2
DATASET_URL = "https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json"
DATASET_PATH = Path(__file__).parent.parent / "benchmarks" / "longbench" / "data.json"


def download_dataset():
    """Download LongBench v2 dataset if not cached."""
    if DATASET_PATH.exists():
        print(f"Dataset cached at {DATASET_PATH}")
        return

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LongBench v2 dataset...")

    try:
        import urllib.request
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print(f"Downloaded to {DATASET_PATH}")
    except Exception as e:
        # Fallback: try curl
        result = subprocess.run(
            ["curl", "-sL", "-o", str(DATASET_PATH), DATASET_URL],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Failed to download dataset: {e}")
            print("Try manually: curl -L -o benchmarks/longbench/data.json " + DATASET_URL)
            sys.exit(1)


def load_dataset(limit=20):
    """Load first `limit` questions from LongBench v2."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    # LongBench v2 format: list of dicts with 'context', 'question', 'answer', etc.
    if isinstance(data, dict):
        # Some versions nest under a key
        for key in ['data', 'examples', 'test']:
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        print(f"Unexpected dataset format: {type(data)}")
        sys.exit(1)

    return data[:limit]


def run_inference(binary, model, context, question, args):
    """Run llama-cli inference and return the generated text."""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Answer the question based on the given context. Be concise.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # Write prompt to temp file (can be very long)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    cmd = [
        binary,
        "-m", model,
        "-f", prompt_file,
        "-n", "256",           # max generation tokens
        "--temp", "0",         # greedy decoding
        "-c", "0",             # auto context size
        "-ngl", "99",          # offload all layers to GPU
        "--no-display-prompt",
    ]

    # Add KV cache type args
    if args.cache_type_k:
        cmd.extend(["--cache-type-k", args.cache_type_k])
    if args.cache_type_v:
        cmd.extend(["--cache-type-v", args.cache_type_v])

    # Add eviction args (StreamingLLM)
    if args.evict_pct and args.evict_pct > 0:
        cmd.extend(["--kv-evict-pct", str(args.evict_pct / 100.0)])
        cmd.extend(["--kv-evict-type", "streamingllm"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )
        os.unlink(prompt_file)

        if result.returncode != 0:
            print(f"  llama-cli error: {result.stderr[:200]}")
            return ""

        # Extract generated text (after prompt)
        output = result.stdout.strip()
        return output

    except subprocess.TimeoutExpired:
        os.unlink(prompt_file)
        print("  Timeout (1800s)")
        return ""
    except Exception as e:
        if os.path.exists(prompt_file):
            os.unlink(prompt_file)
        print(f"  Error: {e}")
        return ""


def score_answer(generated, reference):
    """Score generated answer against reference.

    Simple scoring: check if the reference answer (or key parts) appear in generated text.
    For multiple-choice, check if the correct letter appears.
    """
    if not generated or not reference:
        return 0

    gen_lower = generated.lower().strip()
    ref_lower = reference.lower().strip()

    # Multiple choice: reference is a single letter (A/B/C/D)
    if len(ref_lower) == 1 and ref_lower in 'abcd':
        # Check if the letter appears prominently
        if ref_lower in gen_lower[:50].lower():
            return 1
        return 0

    # Free-form: check keyword overlap
    ref_words = set(ref_lower.split())
    gen_words = set(gen_lower.split())

    if len(ref_words) == 0:
        return 0

    overlap = len(ref_words & gen_words) / len(ref_words)
    return 1 if overlap > 0.5 else 0


def main():
    parser = argparse.ArgumentParser(description="LongBench v2 evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model path")
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="llama-cli binary path")
    parser.add_argument("--cache-type-k", default=None, help="K cache quantization type")
    parser.add_argument("--cache-type-v", default=None, help="V cache quantization type")
    parser.add_argument("--evict-pct", type=int, default=0, help="Eviction percentage (0-90)")
    parser.add_argument("--limit", type=int, default=20, help="Number of questions to evaluate")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Config label
    config_parts = []
    if args.cache_type_k:
        config_parts.append(f"K={args.cache_type_k}")
    if args.cache_type_v:
        config_parts.append(f"V={args.cache_type_v}")
    if args.evict_pct > 0:
        config_parts.append(f"evict={args.evict_pct}%")
    config_label = "_".join(config_parts) if config_parts else "fp16_baseline"

    print(f"=== LongBench v2 Evaluation ===")
    print(f"Config: {config_label}")
    print(f"Model: {args.model}")
    print(f"Questions: {args.limit}")
    print()

    # Download dataset
    download_dataset()

    # Load questions
    questions = load_dataset(args.limit)
    print(f"Loaded {len(questions)} questions")

    # Run evaluation
    correct = 0
    total = len(questions)
    results = []

    for i, item in enumerate(questions):
        # Extract fields (handle different dataset formats)
        context = item.get('context', item.get('input', ''))
        question = item.get('question', item.get('query', ''))
        answer = item.get('answer', item.get('answers', ''))
        task = item.get('task', item.get('type', 'unknown'))

        if isinstance(answer, list):
            answer = answer[0] if answer else ''

        print(f"[{i+1}/{total}] Task: {task} ... ", end="", flush=True)

        generated = run_inference(args.binary, args.model, context, question, args)
        score = score_answer(generated, answer)
        correct += score

        results.append({
            "id": i + 1,
            "task": task,
            "score": score,
            "answer": answer[:100],
            "generated": generated[:200],
        })

        status = "✓" if score else "✗"
        print(f"{status} ({correct}/{i+1})")

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n=== Results ===")
    print(f"Config: {config_label}")
    print(f"Score: {accuracy:.0f}% ({correct}/{total})")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(RESULTS_DIR / f"longbench_{config_label}.json")

    output_data = {
        "config": config_label,
        "model": os.path.basename(args.model),
        "score_pct": accuracy,
        "correct": correct,
        "total": total,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "evict_pct": args.evict_pct,
        "results": results,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")
    return accuracy


if __name__ == "__main__":
    main()
