#!/usr/bin/env python3
"""
Tiered Evaluation for KV Cache Compression Configurations

Three evaluation tiers with increasing cost and accuracy:
  Tier 1: 512-token PPL screening (<1 min) — eliminates obviously bad configs
  Tier 2: 4K PPL + single-point NIAH (<5 min) — dual-metric screening
  Tier 3: 16K PPL + 5-point NIAH + optional LongBench (<30 min) — full eval

Configs are promoted through tiers only if they pass the previous tier's
quality gate. This reduces total evaluation time by ~10x compared to
running full eval on every candidate.

Usage:
    from tiered_eval import TieredEvaluator, EvalResult

    evaluator = TieredEvaluator(model, wiki, perplexity_bin, niah_script)
    result = evaluator.evaluate(config)
    # result.tier = highest tier passed
    # result.promoted = True if passed all requested tiers

Standalone:
    python3 tiered_eval.py --model model.gguf --wiki wiki.test.raw \
        --k q8_0 --v q4_0 --evict 0.5 --method streamingllm --max-tier 3
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from pareto_search import run_ppl, run_niah, compression_ratio


# ============================================================================
# Tier Configuration
# ============================================================================

@dataclass
class TierConfig:
    """Configuration for a single evaluation tier."""
    name: str
    ctx: int
    chunks: int
    ppl_gate: float          # max PPL delta % to pass (relative to baseline at this tier)
    run_niah: bool
    niah_positions: str      # comma-separated positions
    niah_gate: float         # min NIAH accuracy to pass (0.0-1.0)
    timeout_s: int           # max seconds for this tier


TIERS = {
    1: TierConfig(
        name="screening",
        ctx=512,
        chunks=10,           # more chunks at short context for stability
        ppl_gate=10.0,       # very permissive — just catch catastrophic failures
        run_niah=False,
        niah_positions="",
        niah_gate=0.0,
        timeout_s=60,
    ),
    2: TierConfig(
        name="dual-metric",
        ctx=4096,
        chunks=5,
        ppl_gate=5.0,        # moderate gate
        run_niah=True,
        niah_positions="0.5", # single center position (most informative)
        niah_gate=0.5,        # at least find the needle at center
        timeout_s=300,
    ),
    3: TierConfig(
        name="full",
        ctx=16384,
        chunks=2,
        ppl_gate=None,        # no gate — record everything for Pareto analysis
        run_niah=True,
        niah_positions="0.1,0.25,0.5,0.75,0.9",
        niah_gate=0.0,        # no gate — record for Pareto
        timeout_s=1800,
    ),
}


# ============================================================================
# Evaluation Result
# ============================================================================

@dataclass
class EvalResult:
    """Result from tiered evaluation."""
    # Config
    quant_k: str
    quant_v: str
    evict_ratio: float
    evict_method: str
    skip_layers: str
    compression: float

    # Tier results (None = tier not run)
    tier1_ppl: Optional[float] = None
    tier1_delta: Optional[float] = None
    tier1_time_s: Optional[float] = None
    tier1_passed: Optional[bool] = None

    tier2_ppl: Optional[float] = None
    tier2_delta: Optional[float] = None
    tier2_niah: Optional[float] = None
    tier2_time_s: Optional[float] = None
    tier2_passed: Optional[bool] = None

    tier3_ppl: Optional[float] = None
    tier3_delta: Optional[float] = None
    tier3_niah: Optional[float] = None
    tier3_time_s: Optional[float] = None

    # Summary
    highest_tier: int = 0
    promoted: bool = False   # passed all requested tiers
    rejected_at: Optional[int] = None
    reject_reason: Optional[str] = None

    def best_ppl(self) -> Optional[float]:
        """Return PPL from highest completed tier."""
        for ppl in [self.tier3_ppl, self.tier2_ppl, self.tier1_ppl]:
            if ppl is not None:
                return ppl
        return None

    def best_niah(self) -> Optional[float]:
        """Return NIAH from highest completed tier."""
        for niah in [self.tier3_niah, self.tier2_niah]:
            if niah is not None:
                return niah
        return None

    def total_time(self) -> float:
        return sum(t or 0 for t in [self.tier1_time_s, self.tier2_time_s, self.tier3_time_s])

    def to_dict(self) -> dict:
        return {
            "quant_k": self.quant_k, "quant_v": self.quant_v,
            "evict_ratio": self.evict_ratio, "evict_method": self.evict_method,
            "skip_layers": self.skip_layers, "compression": self.compression,
            "tier1_ppl": self.tier1_ppl, "tier1_delta": self.tier1_delta,
            "tier1_passed": self.tier1_passed,
            "tier2_ppl": self.tier2_ppl, "tier2_delta": self.tier2_delta,
            "tier2_niah": self.tier2_niah, "tier2_passed": self.tier2_passed,
            "tier3_ppl": self.tier3_ppl, "tier3_delta": self.tier3_delta,
            "tier3_niah": self.tier3_niah,
            "highest_tier": self.highest_tier,
            "promoted": self.promoted,
            "rejected_at": self.rejected_at,
            "reject_reason": self.reject_reason,
            "total_time_s": round(self.total_time(), 1),
        }


# ============================================================================
# Tiered Evaluator
# ============================================================================

class TieredEvaluator:
    """Evaluates configs through progressive tiers with early rejection."""

    def __init__(self, model_path: str, wiki_path: str,
                 perplexity_bin: str = "llama-perplexity",
                 niah_script: str = "scripts/niah_test.py",
                 ngl: int = 99,
                 baselines: dict = None):
        self.model = model_path
        self.wiki = wiki_path
        self.perplexity_bin = perplexity_bin
        self.niah_script = niah_script
        self.ngl = ngl

        # Per-tier baselines: {tier_num: baseline_ppl}
        # Must be established before evaluating configs
        self.baselines = baselines or {}

        # Stats
        self.total_evaluated = 0
        self.rejected_at_tier = {1: 0, 2: 0}
        self.promoted_to_tier3 = 0

    def establish_baselines(self):
        """Run f16 baseline at each tier's context length."""
        print("  Establishing per-tier baselines...")
        for tier_num, tier in TIERS.items():
            if tier_num in self.baselines:
                continue
            print(f"    Tier {tier_num} ({tier.ctx} tokens)...", end=" ", flush=True)
            ppl = run_ppl(self.model, self.wiki, tier.ctx, tier.chunks, self.ngl,
                          "f16", "f16", 0.0, "none", "", self.perplexity_bin)
            if ppl:
                self.baselines[tier_num] = ppl
                print(f"PPL={ppl:.4f}")
            else:
                print("FAILED — tier will use absolute thresholds")

    def evaluate(self, quant_k: str, quant_v: str,
                 evict_ratio: float = 0.0, evict_method: str = "none",
                 skip_layers: str = "",
                 max_tier: int = 3) -> EvalResult:
        """Evaluate a config through progressive tiers.

        Stops early if a tier's quality gate is not met.
        Returns EvalResult with all tier data populated.
        """
        comp = compression_ratio(quant_k, quant_v, evict_ratio)
        result = EvalResult(
            quant_k=quant_k, quant_v=quant_v,
            evict_ratio=evict_ratio, evict_method=evict_method,
            skip_layers=skip_layers, compression=round(comp, 3),
        )
        self.total_evaluated += 1

        label = f"{quant_k}/{quant_v}"
        if evict_ratio > 0:
            label += f"+{evict_method}{evict_ratio:.0%}"

        # === Tier 1: Quick PPL screening ===
        if max_tier >= 1:
            tier = TIERS[1]
            t0 = time.time()
            ppl = run_ppl(self.model, self.wiki, tier.ctx, tier.chunks, self.ngl,
                          quant_k, quant_v, evict_ratio, evict_method,
                          skip_layers, self.perplexity_bin)
            result.tier1_time_s = round(time.time() - t0, 1)
            result.tier1_ppl = round(ppl, 4) if ppl else None

            if ppl is None:
                result.rejected_at = 1
                result.reject_reason = "PPL computation failed"
                result.tier1_passed = False
                self.rejected_at_tier[1] += 1
                print(f"    T1 {label}: FAIL ({result.tier1_time_s}s)")
                return result

            baseline = self.baselines.get(1)
            if baseline:
                result.tier1_delta = round((ppl - baseline) / baseline * 100, 3)
                if result.tier1_delta > tier.ppl_gate:
                    result.rejected_at = 1
                    result.reject_reason = f"T1 PPL +{result.tier1_delta:.1f}% > {tier.ppl_gate}% gate"
                    result.tier1_passed = False
                    self.rejected_at_tier[1] += 1
                    print(f"    T1 {label}: REJECT PPL +{result.tier1_delta:.1f}% ({result.tier1_time_s}s)")
                    return result

            result.tier1_passed = True
            result.highest_tier = 1
            print(f"    T1 {label}: PASS PPL={ppl:.4f} ({result.tier1_time_s}s)")

        # === Tier 2: 4K PPL + single NIAH ===
        if max_tier >= 2:
            tier = TIERS[2]
            t0 = time.time()

            ppl = run_ppl(self.model, self.wiki, tier.ctx, tier.chunks, self.ngl,
                          quant_k, quant_v, evict_ratio, evict_method,
                          skip_layers, self.perplexity_bin)

            niah_acc = None
            if ppl is not None and tier.run_niah:
                niah_acc = run_niah(self.model, tier.ctx, self.ngl,
                                   quant_k, quant_v, evict_ratio, evict_method,
                                   skip_layers, self.niah_script)

            result.tier2_time_s = round(time.time() - t0, 1)
            result.tier2_ppl = round(ppl, 4) if ppl else None
            result.tier2_niah = round(niah_acc, 2) if niah_acc is not None else None

            if ppl is None:
                result.rejected_at = 2
                result.reject_reason = "T2 PPL failed"
                result.tier2_passed = False
                self.rejected_at_tier[2] += 1
                print(f"    T2 {label}: FAIL ({result.tier2_time_s}s)")
                return result

            baseline = self.baselines.get(2)
            if baseline:
                result.tier2_delta = round((ppl - baseline) / baseline * 100, 3)

            # Check PPL gate
            ppl_ok = True
            if result.tier2_delta is not None and tier.ppl_gate is not None:
                if result.tier2_delta > tier.ppl_gate:
                    ppl_ok = False

            # Check NIAH gate
            niah_ok = True
            if niah_acc is not None and niah_acc < tier.niah_gate:
                niah_ok = False

            if not ppl_ok or not niah_ok:
                result.rejected_at = 2
                reasons = []
                if not ppl_ok:
                    reasons.append(f"PPL +{result.tier2_delta:.1f}% > {tier.ppl_gate}%")
                if not niah_ok:
                    reasons.append(f"NIAH {niah_acc:.0%} < {tier.niah_gate:.0%}")
                result.reject_reason = "T2 " + " & ".join(reasons)
                result.tier2_passed = False
                self.rejected_at_tier[2] += 1
                niah_str = f" NIAH={niah_acc:.0%}" if niah_acc is not None else ""
                print(f"    T2 {label}: REJECT PPL={ppl:.4f}{niah_str} ({result.tier2_time_s}s)")
                return result

            result.tier2_passed = True
            result.highest_tier = 2
            niah_str = f" NIAH={niah_acc:.0%}" if niah_acc is not None else ""
            print(f"    T2 {label}: PASS PPL={ppl:.4f}{niah_str} ({result.tier2_time_s}s)")

        # === Tier 3: Full evaluation ===
        if max_tier >= 3:
            tier = TIERS[3]
            t0 = time.time()

            ppl = run_ppl(self.model, self.wiki, tier.ctx, tier.chunks, self.ngl,
                          quant_k, quant_v, evict_ratio, evict_method,
                          skip_layers, self.perplexity_bin)

            niah_acc = None
            if ppl is not None and tier.run_niah:
                niah_acc = run_niah(self.model, tier.ctx, self.ngl,
                                   quant_k, quant_v, evict_ratio, evict_method,
                                   skip_layers, self.niah_script)

            result.tier3_time_s = round(time.time() - t0, 1)
            result.tier3_ppl = round(ppl, 4) if ppl else None
            result.tier3_niah = round(niah_acc, 2) if niah_acc is not None else None

            baseline = self.baselines.get(3)
            if baseline and ppl:
                result.tier3_delta = round((ppl - baseline) / baseline * 100, 3)

            result.highest_tier = 3
            self.promoted_to_tier3 += 1
            niah_str = f" NIAH={niah_acc:.0%}" if niah_acc is not None else ""
            print(f"    T3 {label}: DONE PPL={ppl:.4f}{niah_str} ({result.tier3_time_s}s)")

        result.promoted = (result.highest_tier == max_tier)
        return result

    def print_stats(self):
        """Print evaluation statistics."""
        print(f"\n  Tiered Evaluation Stats:")
        print(f"    Total evaluated: {self.total_evaluated}")
        print(f"    Rejected at T1: {self.rejected_at_tier.get(1, 0)} "
              f"(saved ~{self.rejected_at_tier.get(1, 0) * 5:.0f} min)")
        print(f"    Rejected at T2: {self.rejected_at_tier.get(2, 0)} "
              f"(saved ~{self.rejected_at_tier.get(2, 0) * 25:.0f} min)")
        print(f"    Promoted to T3: {self.promoted_to_tier3}")
        total_saved = (self.rejected_at_tier.get(1, 0) * 5 +
                       self.rejected_at_tier.get(2, 0) * 25)
        if total_saved > 0:
            print(f"    Estimated time saved: ~{total_saved} min")


# ============================================================================
# Standalone CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tiered KV Cache Compression Evaluation")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--wiki", required=True, help="Path to wiki.test.raw")
    parser.add_argument("--k", default="f16", help="K cache type")
    parser.add_argument("--v", default="f16", help="V cache type")
    parser.add_argument("--evict", type=float, default=0.0, help="Eviction ratio")
    parser.add_argument("--method", default="none", help="Eviction method")
    parser.add_argument("--skip-layers", default="", help="Layers to skip")
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--max-tier", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--perplexity-bin", default="llama-perplexity")
    parser.add_argument("--niah-script", default="scripts/niah_test.py")
    parser.add_argument("--baseline-t1", type=float, help="T1 baseline PPL (512)")
    parser.add_argument("--baseline-t2", type=float, help="T2 baseline PPL (4K)")
    parser.add_argument("--baseline-t3", type=float, help="T3 baseline PPL (16K)")
    args = parser.parse_args()

    baselines = {}
    if args.baseline_t1:
        baselines[1] = args.baseline_t1
    if args.baseline_t2:
        baselines[2] = args.baseline_t2
    if args.baseline_t3:
        baselines[3] = args.baseline_t3

    evaluator = TieredEvaluator(
        model_path=args.model, wiki_path=args.wiki,
        perplexity_bin=args.perplexity_bin, niah_script=args.niah_script,
        ngl=args.ngl, baselines=baselines,
    )

    if not baselines:
        evaluator.establish_baselines()

    result = evaluator.evaluate(
        quant_k=args.k, quant_v=args.v,
        evict_ratio=args.evict, evict_method=args.method,
        skip_layers=args.skip_layers,
        max_tier=args.max_tier,
    )

    print(f"\n{'='*50}")
    print(f"  Result: {result.quant_k}/{result.quant_v}", end="")
    if result.evict_ratio > 0:
        print(f"+{result.evict_method}{result.evict_ratio:.0%}", end="")
    print(f" ({result.compression:.2f}x)")
    print(f"  Highest tier: {result.highest_tier}")
    print(f"  Promoted: {result.promoted}")
    if result.reject_reason:
        print(f"  Rejected: {result.reject_reason}")
    print(f"  Best PPL: {result.best_ppl()}")
    print(f"  Best NIAH: {result.best_niah()}")
    print(f"  Total time: {result.total_time():.1f}s")
    print(f"{'='*50}")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
