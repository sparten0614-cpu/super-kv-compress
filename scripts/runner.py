#!/usr/bin/env python3
"""
AutoResearch Runner: Three-phase KV cache compression search pipeline.

Phase 1: Grid search (canonical configs, ~1hr)
Phase 2: Bayesian optimization (EHVI, ~2hr)
Phase 3: AI agent creative proposals (~1hr)

Outputs unified Pareto frontier across all phases.

Usage:
    python3 runner.py --model model.gguf --wiki wiki.test.raw

    # Skip phases
    python3 runner.py --model model.gguf --wiki wiki.test.raw --skip-phase 1  # skip grid
    python3 runner.py --model model.gguf --wiki wiki.test.raw --only-phase 2  # only Bayesian

    # Resume from previous run
    python3 runner.py --model model.gguf --wiki wiki.test.raw --resume
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from pareto_search import (
    run_ppl, run_niah, compression_ratio, is_pareto_optimal,
    plot_pareto, BITS_PER_VAL,
)

# ============================================================================
# Unified Result Format
# ============================================================================

RESULT_KEYS = [
    "quant_k", "quant_v", "evict_ratio", "evict_method", "skip_layers",
    "compression", "ppl", "ppl_delta", "niah", "longbench",
    "ppl_time_s", "niah_time_s", "ctx", "model", "phase", "reasoning",
    "timestamp",
]


def normalize_result(raw: dict, phase: str, baseline_ppl: float = None) -> dict:
    """Normalize a result dict to the unified format."""
    r = {k: raw.get(k) for k in RESULT_KEYS}
    r["phase"] = phase
    r["timestamp"] = r.get("timestamp") or datetime.now().isoformat()
    r.setdefault("evict_ratio", 0.0)
    r.setdefault("evict_method", "none")
    r.setdefault("skip_layers", "")

    # Compute ppl_delta if missing
    if r.get("ppl") and baseline_ppl and r.get("ppl_delta") is None:
        r["ppl_delta"] = round((r["ppl"] - baseline_ppl) / baseline_ppl * 100, 3)

    # Compute compression if missing
    if r.get("compression") is None:
        r["compression"] = round(compression_ratio(
            r.get("quant_k", "f16"), r.get("quant_v", "f16"),
            r.get("evict_ratio", 0.0)
        ), 3)

    return r


def result_key(r: dict) -> str:
    """Unique key for deduplication."""
    return (f"{r.get('quant_k','f16')}_{r.get('quant_v','f16')}_"
            f"{r.get('evict_ratio',0.0)}_{r.get('evict_method','none')}_"
            f"{r.get('skip_layers','')}")


# ============================================================================
# SQLite Backend
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    quant_k TEXT NOT NULL,
    quant_v TEXT NOT NULL,
    evict_ratio REAL NOT NULL DEFAULT 0.0,
    evict_method TEXT NOT NULL DEFAULT 'none',
    skip_layers TEXT DEFAULT '',
    ctx INTEGER NOT NULL,
    compression REAL,
    ppl REAL,
    ppl_delta REAL,
    niah REAL,
    longbench REAL,
    ppl_time_s REAL,
    niah_time_s REAL,
    phase TEXT DEFAULT 'manual',
    reasoning TEXT,
    timestamp TEXT DEFAULT (datetime('now')),
    UNIQUE(model, quant_k, quant_v, evict_ratio, evict_method, skip_layers, ctx)
);
"""


class ResultDB:
    """SQLite-backed result storage with deduplication."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert(self, r: dict):
        """Insert or ignore (dedup) a result."""
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO experiments
                (model, quant_k, quant_v, evict_ratio, evict_method, skip_layers,
                 ctx, compression, ppl, ppl_delta, niah, longbench,
                 ppl_time_s, niah_time_s, phase, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.get("model", ""), r.get("quant_k", "f16"), r.get("quant_v", "f16"),
                r.get("evict_ratio", 0.0), r.get("evict_method", "none"),
                r.get("skip_layers", ""), r.get("ctx", 4096),
                r.get("compression"), r.get("ppl"), r.get("ppl_delta"),
                r.get("niah"), r.get("longbench"),
                r.get("ppl_time_s"), r.get("niah_time_s"),
                r.get("phase", "manual"), r.get("reasoning"),
                r.get("timestamp"),
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"  DB insert error: {e}")

    def get_all(self, model: str = None) -> list[dict]:
        """Get all results, optionally filtered by model."""
        if model:
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE model = ? AND ppl IS NOT NULL ORDER BY compression",
                (model,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE ppl IS NOT NULL ORDER BY compression"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_tested_keys(self, model: str) -> set[str]:
        """Get set of already-tested config keys."""
        rows = self.conn.execute(
            "SELECT quant_k, quant_v, evict_ratio, evict_method, skip_layers FROM experiments WHERE model = ?",
            (model,)
        ).fetchall()
        return {f"{r[0]}_{r[1]}_{r[2]}_{r[3]}_{r[4]}" for r in rows}

    def count(self, model: str = None, phase: str = None) -> int:
        sql = "SELECT COUNT(*) FROM experiments WHERE 1=1"
        params = []
        if model:
            sql += " AND model = ?"
            params.append(model)
        if phase:
            sql += " AND phase = ?"
            params.append(phase)
        return self.conn.execute(sql, params).fetchone()[0]

    def close(self):
        self.conn.close()


# ============================================================================
# Phase 1: Grid Search
# ============================================================================

CANONICAL_GRID = [
    # Quantization only
    {"quant_k": "f16",  "quant_v": "f16",  "evict_ratio": 0.0, "evict_method": "none"},
    {"quant_k": "q8_0", "quant_v": "q8_0", "evict_ratio": 0.0, "evict_method": "none"},
    {"quant_k": "q8_0", "quant_v": "q4_0", "evict_ratio": 0.0, "evict_method": "none"},
    {"quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0.0, "evict_method": "none"},
    {"quant_k": "q5_0", "quant_v": "q4_0", "evict_ratio": 0.0, "evict_method": "none"},
    # Eviction only
    {"quant_k": "f16",  "quant_v": "f16",  "evict_ratio": 0.5, "evict_method": "streamingllm"},
    {"quant_k": "f16",  "quant_v": "f16",  "evict_ratio": 0.7, "evict_method": "streamingllm"},
    {"quant_k": "f16",  "quant_v": "f16",  "evict_ratio": 0.85,"evict_method": "streamingllm"},
    # Combined
    {"quant_k": "q8_0", "quant_v": "q4_0", "evict_ratio": 0.5, "evict_method": "streamingllm"},
    {"quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0.5, "evict_method": "streamingllm"},
    {"quant_k": "q8_0", "quant_v": "q4_0", "evict_ratio": 0.7, "evict_method": "streamingllm"},
    {"quant_k": "q4_0", "quant_v": "q4_0", "evict_ratio": 0.7, "evict_method": "streamingllm"},
]


def run_phase1(db: ResultDB, model, wiki, ctx, chunks, ngl,
               perplexity_bin, niah_script, skip_niah, model_name):
    """Phase 1: Grid search over canonical configurations."""
    tested = db.get_tested_keys(model_name)
    configs = [c for c in CANONICAL_GRID if result_key(c) not in tested]

    print(f"\n{'='*60}")
    print(f"  Phase 1: Grid Search ({len(configs)} new / {len(CANONICAL_GRID)} total)")
    print(f"{'='*60}\n")

    for i, cfg in enumerate(configs):
        qk, qv = cfg["quant_k"], cfg["quant_v"]
        er, em = cfg["evict_ratio"], cfg["evict_method"]
        comp = compression_ratio(qk, qv, er)
        label = f"{qk}/{qv}"
        if er > 0:
            label += f"+{em}{er:.0%}"

        print(f"  [{i+1}/{len(configs)}] {label} ({comp:.2f}x) ...", end=" ", flush=True)

        t0 = time.time()
        ppl = run_ppl(model, wiki, ctx, chunks, ngl, qk, qv, er, em, "",
                      perplexity_bin)
        ppl_time = time.time() - t0

        niah_acc = None
        niah_time = 0
        if not skip_niah and ppl is not None:
            t0 = time.time()
            niah_acc = run_niah(model, ctx, ngl, qk, qv, er, em, "", niah_script)
            niah_time = time.time() - t0

        result = normalize_result({
            **cfg, "skip_layers": "",
            "compression": round(comp, 3),
            "ppl": round(ppl, 4) if ppl else None,
            "niah": round(niah_acc, 2) if niah_acc is not None else None,
            "ppl_time_s": round(ppl_time, 1),
            "niah_time_s": round(niah_time, 1),
            "ctx": ctx, "model": model_name,
        }, phase="grid")

        db.insert(result)

        status = f"PPL={ppl:.4f}" if ppl else "FAIL"
        if niah_acc is not None:
            status += f" NIAH={niah_acc:.0%}"
        print(f"{status} ({ppl_time:.0f}s)")

    return db.get_all(model_name)


# ============================================================================
# Phase 2: Bayesian Optimization
# ============================================================================

def run_phase2(db: ResultDB, model, wiki, ctx, chunks, ngl,
               perplexity_bin, niah_script, skip_niah, model_name,
               budget, baseline_ppl):
    """Phase 2: Bayesian optimization with EHVI."""
    try:
        from bayesian_optimizer import BayesianKVOptimizer
    except ImportError:
        print("\n  Phase 2 skipped: bayesian_optimizer.py not importable")
        print("  (Ensure botorch is installed: pip install botorch)")
        return db.get_all(model_name)

    print(f"\n{'='*60}")
    print(f"  Phase 2: Bayesian Optimization ({budget} rounds)")
    print(f"{'='*60}\n")

    optimizer = BayesianKVOptimizer(
        model_path=model, wiki_path=wiki, ctx=ctx, ngl=ngl, chunks=chunks,
        perplexity_bin=perplexity_bin, niah_script=niah_script,
        outdir=os.path.dirname(db.db_path), baseline_ppl=baseline_ppl,
        skip_niah=skip_niah,
    )

    # Seed with existing results
    existing = db.get_all(model_name)
    for r in existing:
        from bayesian_optimizer import Config, Result as BOResult
        cfg = Config(
            quant_k=r["quant_k"], quant_v=r["quant_v"],
            evict_ratio=r["evict_ratio"], evict_method=r["evict_method"],
            skip_layers=r.get("skip_layers", ""),
        )
        bo_result = BOResult(
            config=cfg, compression=r["compression"],
            ppl=r["ppl"], ppl_delta=r.get("ppl_delta"), niah=r.get("niah"),
        )
        optimizer.results.append(bo_result)
        optimizer.tested_keys.add(cfg.key())

    if baseline_ppl:
        optimizer.baseline_ppl = baseline_ppl

    # Run Bayesian loop
    optimizer.run_bayesian(budget=budget)

    # Save new results to DB
    for r in optimizer.results:
        if r.config.key() not in db.get_tested_keys(model_name):
            result = normalize_result({
                "quant_k": r.config.quant_k, "quant_v": r.config.quant_v,
                "evict_ratio": r.config.evict_ratio,
                "evict_method": r.config.evict_method,
                "skip_layers": r.config.skip_layers,
                "compression": r.compression,
                "ppl": r.ppl, "ppl_delta": r.ppl_delta,
                "niah": r.niah,
                "ctx": ctx, "model": model_name,
            }, phase="bayesian", baseline_ppl=baseline_ppl)
            db.insert(result)

    return db.get_all(model_name)


# ============================================================================
# Phase 3: AI Agent
# ============================================================================

def run_phase3(db: ResultDB, model, wiki, ctx, chunks, ngl,
               perplexity_bin, niah_script, skip_niah, model_name,
               budget):
    """Phase 3: Claude-powered creative proposals."""
    try:
        from agent import ResearchAgent
    except ImportError:
        print("\n  Phase 3 skipped: agent.py not importable or anthropic not installed")
        return db.get_all(model_name)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  Phase 3 skipped: ANTHROPIC_API_KEY not set")
        return db.get_all(model_name)

    print(f"\n{'='*60}")
    print(f"  Phase 3: AI Agent Creative Proposals ({budget} rounds)")
    print(f"{'='*60}\n")

    outdir = os.path.dirname(db.db_path)
    agent = ResearchAgent(
        model_path=model, wiki_path=wiki, ctx=ctx, ngl=ngl, chunks=chunks,
        outdir=outdir, perplexity_bin=perplexity_bin,
        niah_script=niah_script, skip_niah=skip_niah,
    )

    agent.run(max_rounds=budget)

    # Import agent results into DB
    agent_results_file = os.path.join(outdir, "results.jsonl")
    if os.path.exists(agent_results_file):
        with open(agent_results_file) as f:
            for line in f:
                try:
                    raw = json.loads(line.strip())
                    result = normalize_result(raw, phase="agent")
                    result["model"] = model_name
                    result["ctx"] = ctx
                    db.insert(result)
                except (json.JSONDecodeError, KeyError):
                    pass

    return db.get_all(model_name)


# ============================================================================
# Report
# ============================================================================

def print_final_report(db: ResultDB, model_name: str, outdir: str):
    """Print final Pareto frontier and generate plots."""
    all_results = db.get_all(model_name)
    if not all_results:
        print("\nNo results to report.")
        return

    # Mark Pareto optimal
    all_results = is_pareto_optimal(all_results)
    pareto = [r for r in all_results if r.get("pareto")]
    pareto.sort(key=lambda r: r["compression"])

    # Phase breakdown
    phase_counts = {}
    for r in all_results:
        p = r.get("phase", "unknown")
        phase_counts[p] = phase_counts.get(p, 0) + 1

    print(f"\n{'='*70}")
    print(f"  AutoResearch Complete")
    print(f"{'='*70}")
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Phase breakdown: {phase_counts}")
    print(f"  Pareto-optimal: {len(pareto)}")
    print(f"\n  {'Comp':>6} | {'PPL':>8} | {'Δ%':>8} | {'NIAH':>6} | {'Phase':>8} | Config")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*30}")

    for r in pareto:
        label = f"{r['quant_k']}/{r['quant_v']}"
        if r.get("evict_ratio", 0) > 0:
            label += f"+{r['evict_method']}{r['evict_ratio']:.0%}"
        if r.get("skip_layers"):
            label += f"+skip({r['skip_layers']})"
        niah_str = f"{r['niah']:.0%}" if r.get("niah") is not None else "N/A"
        delta_str = f"{r['ppl_delta']:+.2f}%" if r.get("ppl_delta") is not None else "N/A"
        phase = r.get("phase", "?")[:8]
        print(f"  {r['compression']:>5.2f}x | {r['ppl']:>8.4f} | {delta_str:>8} | "
              f"{niah_str:>6} | {phase:>8} | {label}")

    # Generate plot
    try:
        plot_path = os.path.join(outdir, "pareto_final.png")
        plot_pareto(all_results, plot_path)
    except Exception as e:
        print(f"\n  Plot generation failed: {e}")

    # Write Pareto CSV
    csv_path = os.path.join(outdir, "pareto_final.csv")
    with open(csv_path, "w") as f:
        f.write("compression,ppl,ppl_delta,niah,quant_k,quant_v,evict_ratio,evict_method,skip_layers,phase\n")
        for r in pareto:
            f.write(f"{r['compression']},{r.get('ppl','')},{r.get('ppl_delta','')},{r.get('niah','')},")
            f.write(f"{r['quant_k']},{r['quant_v']},{r.get('evict_ratio',0)},{r.get('evict_method','none')},")
            f.write(f"{r.get('skip_layers','')},{r.get('phase','')}\n")

    print(f"\n  Results DB: {db.db_path}")
    print(f"  Pareto CSV: {csv_path}")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch: Three-phase KV cache compression search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all 3 phases)
  python3 runner.py --model model.gguf --wiki wiki.test.raw

  # Only Bayesian optimization (skip grid + agent)
  python3 runner.py --model model.gguf --wiki wiki.test.raw --only-phase 2

  # Custom budgets
  python3 runner.py --model model.gguf --wiki wiki.test.raw --bo-budget 50 --agent-budget 20
        """
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--wiki", required=True, help="Path to wiki.test.raw")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--chunks", type=int, default=5, help="PPL chunks")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers")
    parser.add_argument("--outdir", default="results/autoresearch", help="Output directory")
    parser.add_argument("--perplexity-bin", default="llama-perplexity")
    parser.add_argument("--niah-script", default="scripts/niah_test.py")
    parser.add_argument("--skip-niah", action="store_true")
    parser.add_argument("--baseline-ppl", type=float, default=None,
                        help="Known F16 baseline PPL (skips F16 evaluation)")

    # Phase control
    parser.add_argument("--skip-phase", type=int, nargs="+", default=[],
                        help="Phase numbers to skip (1, 2, or 3)")
    parser.add_argument("--only-phase", type=int, default=None,
                        help="Run only this phase")

    # Budget control
    parser.add_argument("--bo-budget", type=int, default=20,
                        help="Phase 2: Bayesian optimization iterations")
    parser.add_argument("--agent-budget", type=int, default=10,
                        help="Phase 3: AI agent rounds")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_name = os.path.basename(args.model)
    db = ResultDB(os.path.join(args.outdir, "experiments.db"))

    print(f"\n{'='*70}")
    print(f"  AutoResearch Pipeline")
    print(f"  Model: {model_name}")
    print(f"  Context: {args.ctx}, GPU layers: {args.ngl}")
    print(f"  Output: {args.outdir}")
    print(f"  Existing results: {db.count(model_name)}")
    print(f"{'='*70}")

    phases_to_run = {1, 2, 3}
    if args.only_phase:
        phases_to_run = {args.only_phase}
    for s in args.skip_phase:
        phases_to_run.discard(s)

    baseline_ppl = args.baseline_ppl
    t_start = time.time()

    # Phase 1: Grid
    if 1 in phases_to_run:
        results = run_phase1(db, args.model, args.wiki, args.ctx, args.chunks,
                             args.ngl, args.perplexity_bin, args.niah_script,
                             args.skip_niah, model_name)
        # Extract baseline PPL
        if baseline_ppl is None:
            for r in results:
                if r.get("quant_k") == "f16" and r.get("evict_ratio", 0) == 0 and r.get("ppl"):
                    baseline_ppl = r["ppl"]
                    print(f"\n  Baseline PPL: {baseline_ppl:.4f}")
                    break

    # Phase 2: Bayesian
    if 2 in phases_to_run:
        run_phase2(db, args.model, args.wiki, args.ctx, args.chunks,
                   args.ngl, args.perplexity_bin, args.niah_script,
                   args.skip_niah, model_name, args.bo_budget, baseline_ppl)

    # Phase 3: AI Agent
    if 3 in phases_to_run:
        run_phase3(db, args.model, args.wiki, args.ctx, args.chunks,
                   args.ngl, args.perplexity_bin, args.niah_script,
                   args.skip_niah, model_name, args.agent_budget)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Final report
    print_final_report(db, model_name, args.outdir)
    db.close()


if __name__ == "__main__":
    main()
