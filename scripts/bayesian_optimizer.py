#!/usr/bin/env python3
"""
Phase 2: Bayesian Multi-Objective Optimization for KV Cache Compression

Uses Expected Hypervolume Improvement (EHVI) to efficiently search the
configuration space for Pareto-optimal (compression, PPL, NIAH) trade-offs.

Integrates with pareto_search.py infrastructure for experiment execution.

Requirements:
    pip install botorch gpytorch torch

Usage:
    python3 bayesian_optimizer.py --model model.gguf --wiki wiki.test.raw --budget 50
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

# Import evaluation infrastructure from pareto_search
sys.path.insert(0, os.path.dirname(__file__))
from pareto_search import run_ppl, run_niah, compression_ratio, BITS_PER_VAL

# ============================================================================
# Search Space Encoding
# ============================================================================

# Discrete options encoded as integers for GP
QUANT_TYPES = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
EVICT_METHODS = ["none", "streamingllm", "h2o"]

# Bounds: [quant_k_idx, quant_v_idx, evict_ratio, evict_method_idx]
# quant indices: 0-5, evict_ratio: 0.0-0.9, method: 0-2
DIMS = 4
BOUNDS = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],   # lower
    [5.0, 5.0, 0.9, 2.0],   # upper
])


@dataclass
class Config:
    quant_k: str
    quant_v: str
    evict_ratio: float
    evict_method: str
    skip_layers: str = ""

    def to_tensor(self) -> Tensor:
        return torch.tensor([
            QUANT_TYPES.index(self.quant_k),
            QUANT_TYPES.index(self.quant_v),
            self.evict_ratio,
            EVICT_METHODS.index(self.evict_method),
        ], dtype=torch.double)

    @staticmethod
    def from_tensor(t: Tensor) -> "Config":
        qk_idx = int(t[0].round().clamp(0, len(QUANT_TYPES) - 1).item())
        qv_idx = int(t[1].round().clamp(0, len(QUANT_TYPES) - 1).item())
        er = round(t[2].clamp(0.0, 0.9).item(), 2)
        em_idx = int(t[3].round().clamp(0, len(EVICT_METHODS) - 1).item())

        # Enforce constraint: K precision >= V precision
        if qk_idx > qv_idx:
            qk_idx = qv_idx

        # Enforce: evict_ratio=0 ↔ method=none
        if er < 0.05:
            er = 0.0
            em_idx = 0
        elif em_idx == 0:
            em_idx = 1  # default to streamingllm

        return Config(
            quant_k=QUANT_TYPES[qk_idx],
            quant_v=QUANT_TYPES[qv_idx],
            evict_ratio=er,
            evict_method=EVICT_METHODS[em_idx],
        )

    def key(self) -> str:
        return f"{self.quant_k}_{self.quant_v}_{self.evict_ratio}_{self.evict_method}_{self.skip_layers}"

    def label(self) -> str:
        s = f"{self.quant_k}/{self.quant_v}"
        if self.evict_ratio > 0:
            s += f"+{self.evict_method}{self.evict_ratio:.0%}"
        return s


@dataclass
class Result:
    config: Config
    compression: float
    ppl: Optional[float]
    ppl_delta: Optional[float]
    niah: Optional[float]


# ============================================================================
# Bayesian Optimizer
# ============================================================================

class BayesianKVOptimizer:
    """Multi-objective Bayesian optimization for KV cache compression."""

    def __init__(self, model_path, wiki_path, ctx, ngl, chunks,
                 perplexity_bin, niah_script, outdir, baseline_ppl=None,
                 skip_niah=False):
        self.model = model_path
        self.wiki = wiki_path
        self.ctx = ctx
        self.ngl = ngl
        self.chunks = chunks
        self.perplexity_bin = perplexity_bin
        self.niah_script = niah_script
        self.outdir = outdir
        self.skip_niah = skip_niah
        self.baseline_ppl = baseline_ppl

        self.results: list[Result] = []
        self.tested_keys: set[str] = set()
        self.db_path = os.path.join(outdir, "bayesian_results.jsonl")

        os.makedirs(outdir, exist_ok=True)

    def evaluate(self, config: Config) -> Result:
        """Run PPL + NIAH for a config."""
        comp = compression_ratio(config.quant_k, config.quant_v, config.evict_ratio)
        print(f"  Eval: {config.label()} (compression={comp:.2f}x)")

        ppl = run_ppl(self.model, self.wiki, self.ctx, self.chunks, self.ngl,
                      config.quant_k, config.quant_v, config.evict_ratio,
                      config.evict_method, config.skip_layers, self.perplexity_bin)

        niah_acc = None
        if not self.skip_niah and ppl is not None:
            niah_acc = run_niah(self.model, self.ctx, self.ngl,
                               config.quant_k, config.quant_v,
                               config.evict_ratio, config.evict_method,
                               config.skip_layers, self.niah_script)

        ppl_delta = None
        if ppl is not None and self.baseline_ppl is not None:
            ppl_delta = (ppl - self.baseline_ppl) / self.baseline_ppl * 100

        result = Result(config=config, compression=comp,
                        ppl=ppl, ppl_delta=ppl_delta, niah=niah_acc)
        self.results.append(result)
        self.tested_keys.add(config.key())

        # Persist
        entry = {
            "quant_k": config.quant_k, "quant_v": config.quant_v,
            "evict_ratio": config.evict_ratio, "evict_method": config.evict_method,
            "skip_layers": config.skip_layers,
            "compression": round(comp, 3),
            "ppl": round(ppl, 4) if ppl else None,
            "ppl_delta": round(ppl_delta, 2) if ppl_delta is not None else None,
            "niah": round(niah_acc, 2) if niah_acc is not None else None,
            "ctx": self.ctx,
            "phase": "bayesian",
        }
        with open(self.db_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return result

    def get_training_data(self) -> tuple[Tensor, Tensor]:
        """Convert results to tensors for GP fitting.

        Returns:
            X: (n, 4) config tensor
            Y: (n, 3) objective tensor [compression, -ppl_delta, niah]
              (all objectives are MAXIMIZED)
        """
        valid = [r for r in self.results if r.ppl is not None]
        if not valid:
            return torch.empty(0, DIMS, dtype=torch.double), torch.empty(0, 3, dtype=torch.double)

        X = torch.stack([r.config.to_tensor() for r in valid])
        Y = torch.tensor([
            [
                r.compression,
                -(r.ppl_delta if r.ppl_delta is not None else 100.0),  # negate: lower is better
                r.niah if r.niah is not None else 0.0,
            ]
            for r in valid
        ], dtype=torch.double)

        return X, Y

    def suggest_next(self) -> Config:
        """Use EHVI acquisition to suggest next config."""
        X, Y = self.get_training_data()

        if len(X) < 5:
            # Not enough data for GP — use random sampling
            return self._random_config()

        try:
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Standardize, Normalize
            from botorch.fit import fit_gpytorch_mll
            from botorch.acquisition.multi_objective import (
                qExpectedHypervolumeImprovement,
            )
            from botorch.utils.multi_objective.pareto import is_non_dominated
            from botorch.utils.multi_objective.hypervolume import Hypervolume
            from botorch.optim import optimize_acqf
            from gpytorch.mlls import ExactMarginalLogLikelihood

            # Normalize inputs
            train_X = (X - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])

            # Fit GP
            model = SingleTaskGP(
                train_X, Y,
                outcome_transform=Standardize(m=3),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Reference point for hypervolume (worst acceptable values)
            ref_point = torch.tensor([1.0, -20.0, 0.0], dtype=torch.double)

            # Pareto front
            pareto_mask = is_non_dominated(Y)
            pareto_Y = Y[pareto_mask]

            # EHVI acquisition
            acqf = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                partitioning=None,  # auto
                Y=pareto_Y,
            )

            # Optimize acquisition function
            candidates, value = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * DIMS, [1.0] * DIMS], dtype=torch.double),
                q=1,
                num_restarts=10,
                raw_samples=256,
            )

            # Denormalize
            raw = candidates[0] * (BOUNDS[1] - BOUNDS[0]) + BOUNDS[0]
            config = Config.from_tensor(raw)

            # Avoid duplicates
            for _ in range(10):
                if config.key() not in self.tested_keys:
                    return config
                # Perturb slightly
                noise = torch.randn(DIMS, dtype=torch.double) * 0.5
                raw_perturbed = (raw + noise).clamp(BOUNDS[0], BOUNDS[1])
                config = Config.from_tensor(raw_perturbed)

            return config

        except ImportError:
            print("  botorch not available, falling back to random sampling")
            return self._random_config()
        except Exception as e:
            print(f"  GP/EHVI failed ({e}), falling back to random")
            return self._random_config()

    def _random_config(self) -> Config:
        """Generate a random config not yet tested."""
        for _ in range(100):
            raw = BOUNDS[0] + torch.rand(DIMS, dtype=torch.double) * (BOUNDS[1] - BOUNDS[0])
            config = Config.from_tensor(raw)
            if config.key() not in self.tested_keys:
                return config
        # Fallback
        return Config(quant_k="q8_0", quant_v="q4_0", evict_ratio=0.0, evict_method="none")

    def run_initial_grid(self):
        """Phase 1: Run canonical configs to seed the GP."""
        canonical = [
            Config("f16", "f16", 0.0, "none"),
            Config("q8_0", "q8_0", 0.0, "none"),
            Config("q4_0", "q4_0", 0.0, "none"),
            Config("q8_0", "q4_0", 0.0, "none"),
            Config("f16", "f16", 0.5, "streamingllm"),
            Config("f16", "f16", 0.7, "streamingllm"),
            Config("q8_0", "q4_0", 0.5, "streamingllm"),
            Config("q4_0", "q4_0", 0.5, "streamingllm"),
        ]

        print(f"\n{'='*60}")
        print(f"  Phase 1: Initial Grid ({len(canonical)} configs)")
        print(f"{'='*60}\n")

        for i, cfg in enumerate(canonical):
            if cfg.key() in self.tested_keys:
                print(f"  [{i+1}/{len(canonical)}] {cfg.label()} — already tested, skipping")
                continue
            print(f"  [{i+1}/{len(canonical)}] ", end="")
            result = self.evaluate(cfg)
            status = f"PPL={result.ppl:.4f}" if result.ppl else "FAIL"
            if result.niah is not None:
                status += f" NIAH={result.niah:.0%}"
            print(f"    → {status} (comp={result.compression:.2f}x)")

        # Set baseline PPL if not already set
        if self.baseline_ppl is None:
            f16_results = [r for r in self.results
                           if r.config.quant_k == "f16" and r.config.evict_ratio == 0.0
                           and r.ppl is not None]
            if f16_results:
                self.baseline_ppl = f16_results[0].ppl
                print(f"\n  Baseline PPL (f16): {self.baseline_ppl:.4f}")
                # Retroactively compute ppl_delta
                for r in self.results:
                    if r.ppl is not None:
                        r.ppl_delta = (r.ppl - self.baseline_ppl) / self.baseline_ppl * 100

    def run_bayesian(self, budget: int):
        """Phase 2: Bayesian optimization loop."""
        print(f"\n{'='*60}")
        print(f"  Phase 2: Bayesian Optimization ({budget} iterations)")
        print(f"{'='*60}\n")

        for i in range(budget):
            print(f"\n--- Iteration {i+1}/{budget} ---")
            config = self.suggest_next()
            print(f"  Suggested: {config.label()}")
            result = self.evaluate(config)

            status = f"PPL={result.ppl:.4f}" if result.ppl else "FAIL"
            if result.ppl_delta is not None:
                status += f" (Δ{result.ppl_delta:+.2f}%)"
            if result.niah is not None:
                status += f" NIAH={result.niah:.0%}"
            print(f"    → {status} (comp={result.compression:.2f}x)")

    def print_pareto_front(self):
        """Print the current Pareto front."""
        valid = [r for r in self.results if r.ppl is not None]
        if not valid:
            print("No valid results.")
            return

        # Find Pareto-optimal points (maximize compression, minimize PPL, maximize NIAH)
        pareto = []
        for r in valid:
            dominated = False
            for other in valid:
                if other is r:
                    continue
                better_comp = other.compression >= r.compression
                better_ppl = (other.ppl or float('inf')) <= (r.ppl or float('inf'))
                better_niah = (other.niah or 0) >= (r.niah or 0)
                strictly_better = (other.compression > r.compression or
                                   (other.ppl or float('inf')) < (r.ppl or float('inf')) or
                                   (other.niah or 0) > (r.niah or 0))
                if better_comp and better_ppl and better_niah and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)

        pareto.sort(key=lambda r: r.compression)

        print(f"\n{'='*60}")
        print(f"  Pareto Front ({len(pareto)} configs out of {len(valid)} tested)")
        print(f"{'='*60}")
        print(f"{'Comp':>6} | {'PPL':>8} | {'Δ%':>7} | {'NIAH':>6} | Config")
        print("-" * 60)
        for r in pareto:
            niah_str = f"{r.niah:.0%}" if r.niah is not None else "N/A"
            delta_str = f"{r.ppl_delta:+.2f}%" if r.ppl_delta is not None else "N/A"
            print(f"{r.compression:>5.2f}x | {r.ppl:>8.4f} | {delta_str:>7} | "
                  f"{niah_str:>6} | {r.config.label()}")


# ============================================================================
# SQLite Schema (for future migration from JSONL)
# ============================================================================

SQLITE_SCHEMA = """
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
    latency_tps REAL,
    memory_mb REAL,
    phase TEXT DEFAULT 'manual',
    reasoning TEXT,
    timestamp TEXT DEFAULT (datetime('now')),
    UNIQUE(model, quant_k, quant_v, evict_ratio, evict_method, skip_layers, ctx)
);

CREATE INDEX IF NOT EXISTS idx_model ON experiments(model);
CREATE INDEX IF NOT EXISTS idx_compression ON experiments(compression);
CREATE INDEX IF NOT EXISTS idx_phase ON experiments(phase);

-- View: Pareto front (approximation via SQL)
CREATE VIEW IF NOT EXISTS pareto_front AS
SELECT * FROM experiments e1
WHERE NOT EXISTS (
    SELECT 1 FROM experiments e2
    WHERE e2.model = e1.model
      AND e2.compression >= e1.compression
      AND e2.ppl <= e1.ppl
      AND (e2.niah >= e1.niah OR e1.niah IS NULL)
      AND (e2.compression > e1.compression OR e2.ppl < e1.ppl
           OR (e2.niah > e1.niah AND e1.niah IS NOT NULL))
)
AND e1.ppl IS NOT NULL;
"""


def init_sqlite(db_path: str):
    """Initialize SQLite database with schema."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()
    conn.close()
    print(f"SQLite DB initialized: {db_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bayesian KV Cache Compression Optimizer")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--wiki", required=True, help="Path to wiki.test.raw")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--budget", type=int, default=30, help="Bayesian optimization iterations")
    parser.add_argument("--outdir", default="results/bayesian")
    parser.add_argument("--perplexity-bin", default="llama-perplexity")
    parser.add_argument("--niah-script", default="scripts/niah_test.py")
    parser.add_argument("--skip-niah", action="store_true")
    parser.add_argument("--baseline-ppl", type=float, default=None,
                        help="F16 baseline PPL (skip f16 eval if provided)")
    parser.add_argument("--init-sqlite", action="store_true",
                        help="Initialize SQLite DB and exit")
    args = parser.parse_args()

    if args.init_sqlite:
        init_sqlite(os.path.join(args.outdir, "experiments.db"))
        return

    optimizer = BayesianKVOptimizer(
        model_path=args.model,
        wiki_path=args.wiki,
        ctx=args.ctx,
        ngl=args.ngl,
        chunks=args.chunks,
        perplexity_bin=args.perplexity_bin,
        niah_script=args.niah_script,
        outdir=args.outdir,
        baseline_ppl=args.baseline_ppl,
        skip_niah=args.skip_niah,
    )

    # Phase 1: Initial grid
    optimizer.run_initial_grid()

    # Phase 2: Bayesian optimization
    optimizer.run_bayesian(budget=args.budget)

    # Summary
    optimizer.print_pareto_front()


if __name__ == "__main__":
    main()
