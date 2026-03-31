#!/usr/bin/env python3
"""
Pareto Search: Automatic KV Cache Compression Configuration Optimizer

Searches across quantization levels, eviction ratios, layer maps, and
eviction methods to find Pareto-optimal configurations (compression vs PPL vs NIAH).

Usage:
    python3 pareto_search.py --model path/to/model.gguf --wiki path/to/wiki.test.raw [options]

Output:
    results/pareto/results.jsonl   — one JSON line per config
    results/pareto/pareto.csv      — Pareto frontier only
    results/pareto/pareto.png      — visualization (if matplotlib available)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SEARCH_SPACE = {
    "quant_k":      ["f16", "q8_0", "q4_0"],
    "quant_v":      ["f16", "q8_0", "q4_0"],
    "evict_ratio":  [0.0, 0.5, 0.7, 0.85],
    "evict_method": ["none", "streamingllm"],
    "skip_layers":  [""],  # empty = no skip
}

# Bits per value for compression ratio calculation
BITS_PER_VAL = {
    "f16": 16.0, "bf16": 16.0, "f32": 32.0,
    "q8_0": 8.5, "q5_1": 6.0, "q5_0": 5.5,
    "q4_1": 5.0, "q4_0": 4.5, "iq4_nl": 4.5,
    "q3_K": 3.4375, "q2_K": 2.625,
    "tqkv_6": 6.0, "tqkv_4": 4.25, "tqkv_2": 2.5,
}

# ============================================================================
# Helpers
# ============================================================================

def compression_ratio(quant_k, quant_v, evict_ratio):
    """Calculate effective compression ratio vs F16 baseline."""
    bits_k = BITS_PER_VAL.get(quant_k, 16.0)
    bits_v = BITS_PER_VAL.get(quant_v, 16.0)
    avg_bits = (bits_k + bits_v) / 2.0
    quant_ratio = 16.0 / avg_bits
    evict_ratio_mult = 1.0 / (1.0 - evict_ratio) if evict_ratio < 1.0 else float('inf')
    return quant_ratio * evict_ratio_mult


def run_ppl(model, wiki, ctx, chunks, ngl, quant_k, quant_v, evict_ratio,
            evict_method, skip_layers, perplexity_bin):
    """Run perplexity evaluation. Returns PPL or None on failure."""
    cmd = [
        perplexity_bin, "-m", model, "-ngl", str(ngl),
        "-c", str(ctx), "--chunks", str(chunks), "-fa", "on", "-f", wiki,
    ]

    if quant_k != "f16":
        cmd.extend(["--cache-type-k", quant_k])
    if quant_v != "f16":
        cmd.extend(["--cache-type-v", quant_v])

    if evict_ratio > 0 and evict_method != "none":
        cmd.extend(["--evict-mode", "1", "--evict-ratio", str(evict_ratio), "--evict-sink", "128"])

    env = os.environ.copy()
    if skip_layers:
        env["TQKV_SKIP_LAYERS"] = skip_layers

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        output = result.stdout + result.stderr
        match = re.search(r'Final estimate: PPL = ([\d.]+)', output)
        if match:
            return float(match.group(1))
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  PPL failed: {e}")

    return None


def run_niah(model, ctx, ngl, quant_k, quant_v, evict_ratio, evict_method,
             skip_layers, niah_script):
    """Run NIAH test. Returns accuracy (0.0-1.0) or None on failure."""
    cmd = [
        sys.executable, niah_script,
        "--model", model, "--ctx", str(ctx), "--ngl", str(ngl),
        "--positions", "0.1,0.25,0.5,0.75,0.9",
    ]

    cache_type = quant_k  # NIAH uses same type for K and V
    cmd.extend(["--cache-types", cache_type])

    if evict_ratio > 0 and evict_method != "none":
        cmd.extend(["--evict-mode", "1", "--evict-ratio", str(evict_ratio), "--evict-sink", "128"])
        if evict_method == "snapkv":
            cmd.append("--snapkv-eviction")
        elif evict_method == "h2o":
            cmd.append("--h2o-eviction")
        elif evict_method == "expected_attn":
            cmd.append("--expected-attn-eviction")

    env = os.environ.copy()
    if skip_layers:
        env["TQKV_SKIP_LAYERS"] = skip_layers

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        output = result.stdout + result.stderr

        # Count FOUND/MISS
        found = output.count("FOUND")
        miss = output.count("MISS")
        total = found + miss
        if total > 0:
            return found / total
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  NIAH failed: {e}")

    return None


def is_pareto_optimal(results):
    """Mark Pareto-optimal points (higher compression + lower PPL is better)."""
    for i, r in enumerate(results):
        r["pareto"] = True
        if r.get("ppl") is None:
            r["pareto"] = False
            continue
        for j, other in enumerate(results):
            if i == j or other.get("ppl") is None:
                continue
            # `other` dominates `r` if it has better compression AND better PPL
            if (other["compression"] >= r["compression"] and
                other["ppl"] <= r["ppl"] and
                (other["compression"] > r["compression"] or other["ppl"] < r["ppl"])):
                r["pareto"] = False
                break
    return results


def plot_pareto(results, output_path):
    """Generate Pareto frontier plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    pareto = [r for r in results if r.get("pareto")]
    non_pareto = [r for r in results if not r.get("pareto") and r.get("ppl")]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    if non_pareto:
        ax.scatter([r["compression"] for r in non_pareto],
                   [r["ppl"] for r in non_pareto],
                   c='gray', alpha=0.5, s=30, label='Non-Pareto')

    if pareto:
        pareto_sorted = sorted(pareto, key=lambda r: r["compression"])
        ax.plot([r["compression"] for r in pareto_sorted],
                [r["ppl"] for r in pareto_sorted],
                'ro-', markersize=8, linewidth=2, label='Pareto Frontier')

        for r in pareto_sorted:
            label = f"{r['quant_k']}/{r['quant_v']}"
            if r['evict_ratio'] > 0:
                label += f"+{r['evict_method']}{r['evict_ratio']:.0%}"
            ax.annotate(label, (r["compression"], r["ppl"]),
                       textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Compression Ratio (vs F16)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("KV Cache Compression Pareto Frontier", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KV Cache Compression Pareto Search")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--wiki", required=True, help="Path to wiki.test.raw for PPL")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--chunks", type=int, default=5, help="PPL chunks")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers")
    parser.add_argument("--outdir", default="results/pareto", help="Output directory")
    parser.add_argument("--perplexity-bin", default="llama-perplexity",
                       help="Path to llama-perplexity binary")
    parser.add_argument("--niah-script", default="scripts/niah_test.py",
                       help="Path to niah_test.py")
    parser.add_argument("--skip-niah", action="store_true", help="Skip NIAH tests (PPL only)")
    parser.add_argument("--search-space", default=None,
                       help="JSON file with custom search space")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results_file = os.path.join(args.outdir, "results.jsonl")

    # Load search space
    if args.search_space:
        with open(args.search_space) as f:
            space = json.load(f)
    else:
        space = DEFAULT_SEARCH_SPACE

    # Load existing results (for resume)
    existing = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    key = f"{r['quant_k']}_{r['quant_v']}_{r['evict_ratio']}_{r['evict_method']}_{r.get('skip_layers','')}"
                    existing.add(key)
                except:
                    pass
        print(f"Loaded {len(existing)} existing results (resume mode)")

    # Generate all configs
    configs = []
    for qk, qv, er, em, sl in product(
        space["quant_k"], space["quant_v"], space["evict_ratio"],
        space["evict_method"], space["skip_layers"]
    ):
        # Skip invalid combos
        if er > 0 and em == "none":
            continue
        if er == 0 and em != "none":
            continue

        key = f"{qk}_{qv}_{er}_{em}_{sl}"
        if key in existing:
            continue

        configs.append({
            "quant_k": qk, "quant_v": qv,
            "evict_ratio": er, "evict_method": em,
            "skip_layers": sl,
        })

    total = len(configs)
    print(f"\n{'='*60}")
    print(f"  Pareto Search: {total} configs to test")
    print(f"  Model: {args.model}")
    print(f"  Ctx: {args.ctx}, Chunks: {args.chunks}")
    print(f"  Output: {results_file}")
    print(f"{'='*60}\n")

    results_all = []
    baseline_ppl = None

    # Try to find baseline from existing results
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("quant_k") == "f16" and r.get("quant_v") == "f16" and r.get("evict_ratio", 0) == 0:
                        baseline_ppl = r.get("ppl")
                except:
                    pass

    for i, cfg in enumerate(configs):
        qk, qv = cfg["quant_k"], cfg["quant_v"]
        er, em = cfg["evict_ratio"], cfg["evict_method"]
        sl = cfg["skip_layers"]

        comp = compression_ratio(qk, qv, er)
        label = f"{qk}/{qv}"
        if er > 0:
            label += f"+{em}{er:.0%}"
        if sl:
            label += f"+skip({sl})"

        print(f"[{i+1}/{total}] {label} (compression={comp:.2f}x)")

        # Run PPL
        t0 = time.time()
        ppl = run_ppl(args.model, args.wiki, args.ctx, args.chunks, args.ngl,
                      qk, qv, er, em, sl, args.perplexity_bin)
        ppl_time = time.time() - t0

        # Run NIAH
        niah_acc = None
        niah_time = 0
        if not args.skip_niah and ppl is not None:
            t0 = time.time()
            niah_acc = run_niah(args.model, args.ctx, args.ngl, qk, qv, er, em, sl,
                               args.niah_script)
            niah_time = time.time() - t0

        # Compute PPL delta vs baseline
        ppl_delta = None
        if ppl is not None and baseline_ppl is not None and baseline_ppl > 0:
            ppl_delta = round((ppl - baseline_ppl) / baseline_ppl * 100, 2)

        result = {
            "quant_k": qk, "quant_v": qv,
            "evict_ratio": er, "evict_method": em,
            "skip_layers": sl,
            "compression": round(comp, 3),
            "ppl": round(ppl, 4) if ppl else None,
            "ppl_delta": ppl_delta,
            "niah": round(niah_acc, 2) if niah_acc is not None else None,
            "ppl_time_s": round(ppl_time, 1),
            "niah_time_s": round(niah_time, 1),
            "ctx": args.ctx,
            "model": os.path.basename(args.model),
            "phase": "grid",
        }

        # Track baseline PPL from f16/f16/0% result
        if qk == "f16" and qv == "f16" and er == 0 and ppl is not None and baseline_ppl is None:
            baseline_ppl = ppl

        results_all.append(result)

        # Append to file
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = f"PPL={ppl:.4f}" if ppl else "PPL=FAIL"
        if niah_acc is not None:
            status += f" NIAH={niah_acc:.0%}"
        print(f"  → {status} ({ppl_time:.0f}s + {niah_time:.0f}s)")

    # Load all results (including previously cached)
    all_results = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line.strip()))
                except:
                    pass

    # Find Pareto frontier
    all_results = is_pareto_optimal(all_results)

    pareto = [r for r in all_results if r.get("pareto")]
    pareto.sort(key=lambda r: r["compression"])

    # Write Pareto CSV
    csv_path = os.path.join(args.outdir, "pareto.csv")
    with open(csv_path, "w") as f:
        f.write("compression,ppl,niah,quant_k,quant_v,evict_ratio,evict_method,skip_layers\n")
        for r in pareto:
            f.write(f"{r['compression']},{r.get('ppl','')},{r.get('niah','')},{r['quant_k']},{r['quant_v']},{r['evict_ratio']},{r['evict_method']},{r.get('skip_layers','')}\n")
    print(f"\nPareto CSV: {csv_path} ({len(pareto)} points)")

    # Plot
    plot_path = os.path.join(args.outdir, "pareto.png")
    plot_pareto(all_results, plot_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Pareto Frontier ({len(pareto)} optimal configs)")
    print(f"{'='*60}")
    print(f"{'Compression':>12} | {'PPL':>8} | {'NIAH':>6} | Config")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*30}")
    for r in pareto:
        label = f"{r['quant_k']}/{r['quant_v']}"
        if r['evict_ratio'] > 0:
            label += f"+{r['evict_method']}{r['evict_ratio']:.0%}"
        niah_str = f"{r['niah']:.0%}" if r.get('niah') is not None else "N/A"
        print(f"{r['compression']:>11.2f}x | {r.get('ppl','N/A'):>8} | {niah_str:>6} | {label}")


if __name__ == "__main__":
    main()
