#!/usr/bin/env python3
"""
Cross-Model Benchmark Harness

Runs a KV cache compression config across multiple models and generates
a comparison table. Used for cross-model validation of findings.

Usage:
    python3 benchmark_harness.py --config config.json --models-dir /path/to/models/

Config JSON format:
{
    "quant_k": "q4_0",
    "quant_v": "q4_0",
    "evict_ratio": 0.0,
    "evict_method": "none",
    "skip_layers": ""
}

Or test multiple configs:
    python3 benchmark_harness.py --configs-file multi_configs.json --models-dir /path/to/models/
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from pareto_search import run_ppl, run_niah, compression_ratio, BITS_PER_VAL
from schema import ExperimentResult, save_result

# ============================================================================
# Default model registry
# ============================================================================

DEFAULT_MODELS = {
    "llama-8b": {
        "pattern": "*Llama*3.1*8B*Q4_K_M*.gguf",
        "name": "Llama-3.1-8B",
        "ctx": 4096,
        "known_issues": [],
    },
    "mistral-7b": {
        "pattern": "*Mistral*7B*Q4_K_M*.gguf",
        "name": "Mistral-7B",
        "ctx": 4096,
        "known_issues": [],
    },
    "llama-70b": {
        "pattern": "*Llama*70B*Q4_K_M*.gguf",
        "name": "Llama-3.1-70B",
        "ctx": 4096,
        "known_issues": ["requires 48GB+ VRAM"],
    },
    "qwen-7b": {
        "pattern": "*Qwen*7B*Q4_K_M*.gguf",
        "name": "Qwen2.5-7B",
        "ctx": 4096,
        "known_issues": ["K outlier layer 0 — may need skip_layers=0 for q4_0"],
    },
}


def find_model(models_dir, pattern):
    """Find model file matching glob pattern."""
    import glob
    matches = glob.glob(os.path.join(models_dir, pattern))
    if not matches:
        matches = glob.glob(os.path.join(models_dir, "**", pattern), recursive=True)
    return matches[0] if matches else None


def run_single_model(model_path, model_name, wiki_path, config, ctx, ngl,
                     chunks, perplexity_bin, niah_script, skip_niah):
    """Run PPL + NIAH for a single model + config combo."""
    qk = config.get("quant_k", "f16")
    qv = config.get("quant_v", "f16")
    er = config.get("evict_ratio", 0.0)
    em = config.get("evict_method", "none")
    sl = config.get("skip_layers", "")

    comp = compression_ratio(qk, qv, er)

    # PPL
    t0 = time.time()
    ppl = run_ppl(model_path, wiki_path, ctx, chunks, ngl,
                  qk, qv, er, em, sl, perplexity_bin)
    ppl_time = time.time() - t0

    # NIAH
    niah_acc = None
    niah_time = 0
    if not skip_niah and ppl is not None:
        t0 = time.time()
        niah_acc = run_niah(model_path, ctx, ngl, qk, qv, er, em, sl, niah_script)
        niah_time = time.time() - t0

    return {
        "model": model_name,
        "model_path": os.path.basename(model_path),
        "quant_k": qk, "quant_v": qv,
        "evict_ratio": er, "evict_method": em,
        "skip_layers": sl,
        "compression": round(comp, 3),
        "ppl": round(ppl, 4) if ppl else None,
        "niah": round(niah_acc, 2) if niah_acc is not None else None,
        "ppl_time_s": round(ppl_time, 1),
        "niah_time_s": round(niah_time, 1),
        "ctx": ctx,
        "phase": "benchmark",
    }


def print_comparison_table(results, config_label):
    """Print cross-model comparison table."""
    print(f"\n{'='*70}")
    print(f"  Cross-Model Comparison: {config_label}")
    print(f"{'='*70}")
    print(f"{'Model':>20} | {'PPL':>8} | {'PPL Δ%':>8} | {'NIAH':>6} | {'Comp':>6} | {'Time':>6}")
    print(f"{'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    # Find baseline PPLs per model (f16 results if available)
    for r in results:
        ppl_str = f"{r['ppl']:.4f}" if r.get('ppl') else "FAIL"
        niah_str = f"{r['niah']:.0%}" if r.get('niah') is not None else "N/A"
        delta_str = f"{r['ppl_delta']:+.2f}%" if r.get('ppl_delta') is not None else "N/A"
        time_str = f"{r.get('ppl_time_s', 0):.0f}s"
        print(f"{r['model']:>20} | {ppl_str:>8} | {delta_str:>8} | {niah_str:>6} | {r['compression']:>5.2f}x | {time_str:>6}")

    # Summary
    ppls = [r['ppl'] for r in results if r.get('ppl')]
    niahs = [r['niah'] for r in results if r.get('niah') is not None]
    if ppls:
        print(f"\n  PPL range: {min(ppls):.4f} - {max(ppls):.4f}")
    if niahs:
        print(f"  NIAH range: {min(niahs):.0%} - {max(niahs):.0%}")
        if all(n >= 0.99 for n in niahs):
            print(f"  ✅ NIAH 100% across all models")
        elif any(n < 0.5 for n in niahs):
            print(f"  ❌ NIAH failure on some models")


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Benchmark Harness")
    parser.add_argument("--config", type=str, help="Single config JSON string or file")
    parser.add_argument("--configs-file", type=str, help="File with multiple configs (JSON array)")
    parser.add_argument("--models-dir", required=True, help="Directory containing GGUF models")
    parser.add_argument("--wiki", required=True, help="Path to wiki.test.raw")
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated model keys (default: all found)")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--outdir", default="results/benchmark")
    parser.add_argument("--perplexity-bin", default="llama-perplexity")
    parser.add_argument("--niah-script", default="scripts/niah_test.py")
    parser.add_argument("--skip-niah", action="store_true")
    parser.add_argument("--baseline-first", action="store_true",
                       help="Run f16 baseline for each model first")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results_file = os.path.join(args.outdir, "benchmark_results.jsonl")

    # Parse config(s)
    configs = []
    if args.config:
        try:
            cfg = json.loads(args.config)
        except json.JSONDecodeError:
            with open(args.config) as f:
                cfg = json.load(f)
        configs.append(cfg)
    elif args.configs_file:
        with open(args.configs_file) as f:
            configs = json.load(f)
    else:
        # Default: test a few key configs
        configs = [
            {"quant_k": "f16", "quant_v": "f16", "evict_ratio": 0, "evict_method": "none"},
            {"quant_k": "q8_0", "quant_v": "q8_0", "evict_ratio": 0, "evict_method": "none"},
            {"quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0, "evict_method": "none"},
        ]

    # Find models
    model_keys = args.models.split(",") if args.models else list(DEFAULT_MODELS.keys())
    found_models = {}
    for key in model_keys:
        if key not in DEFAULT_MODELS:
            print(f"WARNING: unknown model key '{key}', skipping")
            continue
        info = DEFAULT_MODELS[key]
        path = find_model(args.models_dir, info["pattern"])
        if path:
            found_models[key] = {"path": path, **info}
            print(f"  Found {info['name']}: {os.path.basename(path)}")
        else:
            print(f"  Not found: {info['name']} ({info['pattern']})")

    if not found_models:
        print("ERROR: No models found!")
        sys.exit(1)

    print(f"\n  Models: {len(found_models)}")
    print(f"  Configs: {len(configs)}")
    print(f"  Total runs: {len(found_models) * len(configs)}")

    # Run baselines first if requested
    baselines = {}
    if args.baseline_first:
        print(f"\n--- Running f16 baselines ---")
        baseline_cfg = {"quant_k": "f16", "quant_v": "f16", "evict_ratio": 0, "evict_method": "none"}
        for key, model in found_models.items():
            print(f"  Baseline: {model['name']}...")
            r = run_single_model(model["path"], model["name"], args.wiki, baseline_cfg,
                                args.ctx, args.ngl, args.chunks,
                                args.perplexity_bin, args.niah_script, args.skip_niah)
            baselines[key] = r.get("ppl")
            with open(results_file, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"    PPL={r.get('ppl', 'FAIL')}")

    # Run each config across all models
    for cfg_idx, cfg in enumerate(configs):
        label = f"{cfg.get('quant_k','f16')}/{cfg.get('quant_v','f16')}"
        if cfg.get("evict_ratio", 0) > 0:
            label += f"+{cfg.get('evict_method','none')}{cfg.get('evict_ratio',0):.0%}"

        print(f"\n--- Config {cfg_idx+1}/{len(configs)}: {label} ---")
        config_results = []

        for key, model in found_models.items():
            print(f"  {model['name']}...", end=" ", flush=True)
            r = run_single_model(model["path"], model["name"], args.wiki, cfg,
                                args.ctx, args.ngl, args.chunks,
                                args.perplexity_bin, args.niah_script, args.skip_niah)

            # Add ppl_delta
            if r.get("ppl") and key in baselines and baselines[key]:
                r["ppl_delta"] = round((r["ppl"] - baselines[key]) / baselines[key] * 100, 2)

            config_results.append(r)
            with open(results_file, "a") as f:
                f.write(json.dumps(r) + "\n")

            status = f"PPL={r.get('ppl','FAIL')}"
            if r.get("niah") is not None:
                status += f" NIAH={r['niah']:.0%}"
            print(status)

        print_comparison_table(config_results, label)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
