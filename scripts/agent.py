#!/usr/bin/env python3
"""
KV Compression AutoResearch Agent

AI-driven research loop that autonomously discovers optimal KV cache
compression configurations. Uses Claude API to analyze experiment results
and propose next experiments.

Loop:
  1. Read research_direction.md + current results
  2. Claude analyzes Pareto frontier → proposes next experiment
  3. Execute experiment (PPL + NIAH via pareto_search.py infra)
  4. Record result
  5. Repeat

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 agent.py --model model.gguf --wiki wiki.test.raw --max-rounds 20

Requirements:
    pip install anthropic
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

# Import helpers from pareto_search
sys.path.insert(0, os.path.dirname(__file__))
from pareto_search import (
    run_ppl, run_niah, compression_ratio, is_pareto_optimal, BITS_PER_VAL
)

# ============================================================================
# Constants
# ============================================================================

SYSTEM_PROMPT = """You are a KV cache compression research agent. Your goal is to find
configurations that achieve maximum compression ratio while maintaining quality
(low PPL degradation, high NIAH retrieval accuracy).

You have access to these compression knobs:
- quant_k: KV cache K quantization (f16, q8_0, q5_1, q5_0, q4_1, q4_0, q2_K)
- quant_v: KV cache V quantization (same options)
- evict_ratio: fraction of tokens to evict (0.0 to 0.9)
- evict_method: none, streamingllm, snapkv, h2o, expected_attn
- skip_layers: comma-separated layer indices to keep at F16 (e.g. "0" or "0,31")

Key findings so far:
- K is more sensitive to quantization than V (especially Qwen models)
- Layer 0 often has extreme outliers — skipping it helps
- StreamingLLM eviction is position-based (bad for NIAH at >50%)
- NIAH = needle-in-a-haystack retrieval test (100% = perfect recall)

Your task: analyze the current results and propose ONE new experiment that is
likely to improve the Pareto frontier. Return your proposal as JSON:

{
  "quant_k": "q4_0",
  "quant_v": "q4_0",
  "evict_ratio": 0.5,
  "evict_method": "streamingllm",
  "skip_layers": "0",
  "reasoning": "brief explanation of why this config might be interesting"
}

Be creative — don't just try nearby points. Think about asymmetric configs,
layer-specific optimization, combining quantization with eviction, etc.
If all obvious configs have been tried, suggest a novel direction."""

# ============================================================================
# Agent Core
# ============================================================================

class ResearchAgent:
    def __init__(self, model_path, wiki_path, ctx, ngl, chunks,
                 outdir, perplexity_bin, niah_script, skip_niah=False, mock=False):
        self.model = model_path
        self.wiki = wiki_path
        self.ctx = ctx
        self.ngl = ngl
        self.chunks = chunks
        self.outdir = outdir
        self.perplexity_bin = perplexity_bin
        self.niah_script = niah_script
        self.skip_niah = skip_niah
        self.mock = mock
        self.results_file = os.path.join(outdir, "results.jsonl")
        self.log_file = os.path.join(outdir, "agent_log.md")

        self.client = anthropic.Anthropic()
        os.makedirs(outdir, exist_ok=True)

    def load_results(self):
        """Load all existing experiment results."""
        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file) as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except:
                        pass
        return results

    def load_direction(self):
        """Load research direction markdown if exists."""
        path = os.path.join(os.path.dirname(__file__), "..", "research_direction.md")
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        return "Find the maximum compression ratio that maintains PPL < +5% and NIAH >= 80%."

    def format_results_summary(self, results):
        """Format results for Claude to analyze."""
        if not results:
            return "No experiments run yet. Start with baseline (f16) and common configs."

        results = is_pareto_optimal(results)
        pareto = [r for r in results if r.get("pareto")]
        pareto.sort(key=lambda r: r.get("compression", 0))

        lines = [f"Total experiments: {len(results)}",
                 f"Pareto-optimal: {len(pareto)}", ""]

        lines.append("All results (sorted by compression):")
        lines.append(f"{'Comp':>6} | {'PPL':>8} | {'NIAH':>6} | {'K':>6} | {'V':>6} | {'Evict':>6} | {'Method':>12} | {'Skip':>6}")
        lines.append("-" * 75)

        for r in sorted(results, key=lambda x: x.get("compression", 0)):
            niah = f"{r['niah']:.0%}" if r.get("niah") is not None else "N/A"
            ppl = f"{r['ppl']:.4f}" if r.get("ppl") else "FAIL"
            pareto_mark = " *" if r.get("pareto") else ""
            lines.append(
                f"{r.get('compression',0):>5.2f}x | {ppl:>8} | {niah:>6} | "
                f"{r['quant_k']:>6} | {r['quant_v']:>6} | {r['evict_ratio']:>5.0%} | "
                f"{r['evict_method']:>12} | {r.get('skip_layers',''):>6}{pareto_mark}"
            )

        lines.append("\n(* = Pareto optimal)")
        return "\n".join(lines)

    def propose_experiment(self, results):
        """Ask Claude to propose the next experiment."""
        direction = self.load_direction()
        summary = self.format_results_summary(results)

        user_msg = f"""## Research Direction
{direction}

## Current Results
{summary}

## Available Quantization Types and Bits/Val
{json.dumps({k: v for k, v in BITS_PER_VAL.items() if k in ['f16','q8_0','q5_1','q5_0','q4_1','q4_0','q2_K']}, indent=2)}

Based on the current results, propose ONE new experiment configuration as JSON.
Focus on improving the Pareto frontier — either better compression at similar quality,
or better quality at similar compression."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        # Extract JSON from response
        text = response.content[0].text
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                proposal = json.loads(json_match.group())
                return proposal, text
            except json.JSONDecodeError:
                pass

        return None, text

    def execute_experiment(self, config):
        """Run PPL + NIAH for a single config (or mock)."""
        qk = config.get("quant_k", "f16")
        qv = config.get("quant_v", "f16")
        er = config.get("evict_ratio", 0.0)
        em = config.get("evict_method", "none")
        sl = config.get("skip_layers", "")

        comp = compression_ratio(qk, qv, er)
        print(f"  Executing: {qk}/{qv} evict={er:.0%} method={em} skip={sl} (comp={comp:.2f}x)")

        if self.mock:
            # Mock mode: generate plausible fake results for testing
            import random
            base_ppl = 6.0
            # Higher compression → worse PPL (roughly)
            ppl = base_ppl * (1.0 + 0.03 * comp + random.gauss(0, 0.01))
            # NIAH degrades with eviction
            niah_acc = max(0.0, 1.0 - er * 0.8 + random.gauss(0, 0.05))
            niah_acc = min(1.0, niah_acc)
            ppl_time, niah_time = 0.1, 0.1
            print(f"  [MOCK] PPL={ppl:.4f} NIAH={niah_acc:.0%}")
        else:
            # Real execution
            t0 = time.time()
            ppl = run_ppl(self.model, self.wiki, self.ctx, self.chunks, self.ngl,
                          qk, qv, er, em, sl, self.perplexity_bin)
            ppl_time = time.time() - t0

            niah_acc = None
            niah_time = 0
            if not self.skip_niah and ppl is not None:
                t0 = time.time()
                niah_acc = run_niah(self.model, self.ctx, self.ngl, qk, qv, er, em, sl,
                                   self.niah_script)
                niah_time = time.time() - t0

        result = {
            "quant_k": qk, "quant_v": qv,
            "evict_ratio": er, "evict_method": em,
            "skip_layers": sl,
            "compression": round(comp, 3),
            "ppl": round(ppl, 4) if ppl else None,
            "niah": round(niah_acc, 2) if niah_acc is not None else None,
            "ppl_time_s": round(ppl_time, 1),
            "niah_time_s": round(niah_time, 1),
            "ctx": self.ctx,
            "model": os.path.basename(self.model),
            "agent_proposed": True,
            "reasoning": config.get("reasoning", ""),
        }

        # Append to results
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        return result

    def log_round(self, round_num, proposal, proposal_text, result):
        """Append round to agent log."""
        with open(self.log_file, "a") as f:
            f.write(f"\n## Round {round_num}\n\n")
            f.write(f"**Agent reasoning:**\n{proposal_text}\n\n")
            if result:
                status = f"PPL={result.get('ppl','FAIL')}"
                if result.get('niah') is not None:
                    status += f" NIAH={result['niah']:.0%}"
                f.write(f"**Result:** {status} (compression={result.get('compression',0):.2f}x)\n\n")
            f.write("---\n")

    def run(self, max_rounds=20):
        """Main research loop."""
        print(f"\n{'='*60}")
        print(f"  KV Compression AutoResearch Agent")
        print(f"  Model: {self.model}")
        print(f"  Max rounds: {max_rounds}")
        print(f"{'='*60}\n")

        for round_num in range(1, max_rounds + 1):
            print(f"\n--- Round {round_num}/{max_rounds} ---")

            # Load current results
            results = self.load_results()
            print(f"  {len(results)} experiments so far")

            # Ask Claude for next experiment
            print(f"  Asking Claude for next experiment...")
            proposal, proposal_text = self.propose_experiment(results)

            if not proposal:
                print(f"  Failed to parse proposal. Agent said:\n{proposal_text[:200]}")
                self.log_round(round_num, None, proposal_text, None)
                continue

            reasoning = proposal.get("reasoning", "no reasoning")
            print(f"  Proposal: {proposal.get('quant_k')}/{proposal.get('quant_v')} "
                  f"evict={proposal.get('evict_ratio', 0):.0%} "
                  f"method={proposal.get('evict_method', 'none')}")
            print(f"  Reasoning: {reasoning[:100]}")

            # Check if already tested
            key = f"{proposal.get('quant_k')}_{proposal.get('quant_v')}_{proposal.get('evict_ratio',0)}_{proposal.get('evict_method','none')}_{proposal.get('skip_layers','')}"
            existing_keys = set()
            for r in results:
                ek = f"{r['quant_k']}_{r['quant_v']}_{r['evict_ratio']}_{r['evict_method']}_{r.get('skip_layers','')}"
                existing_keys.add(ek)

            if key in existing_keys:
                print(f"  Already tested! Asking for different experiment...")
                # Could retry, for now just log and continue
                self.log_round(round_num, proposal, proposal_text + "\n\n[DUPLICATE - skipped]", None)
                continue

            # Execute
            result = self.execute_experiment(proposal)

            status = f"PPL={result.get('ppl','FAIL')}"
            if result.get('niah') is not None:
                status += f" NIAH={result['niah']:.0%}"
            print(f"  Result: {status} (compression={result['compression']:.2f}x)")

            self.log_round(round_num, proposal, proposal_text, result)

            # Brief pause between rounds
            time.sleep(2)

        # Final summary
        results = self.load_results()
        results = is_pareto_optimal(results)
        pareto = [r for r in results if r.get("pareto")]

        print(f"\n{'='*60}")
        print(f"  AutoResearch Complete: {max_rounds} rounds")
        print(f"  Total experiments: {len(results)}")
        print(f"  Pareto-optimal: {len(pareto)}")
        print(f"  Results: {self.results_file}")
        print(f"  Agent log: {self.log_file}")
        print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KV Compression AutoResearch Agent")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--wiki", required=True, help="Path to wiki.test.raw")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--chunks", type=int, default=5, help="PPL chunks")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers")
    parser.add_argument("--max-rounds", type=int, default=20, help="Max research rounds")
    parser.add_argument("--outdir", default="results/pareto", help="Output directory")
    parser.add_argument("--perplexity-bin", default="llama-perplexity")
    parser.add_argument("--niah-script", default="scripts/niah_test.py")
    parser.add_argument("--skip-niah", action="store_true")
    parser.add_argument("--mock", action="store_true",
                       help="Mock mode: fake PPL/NIAH results for testing agent logic")
    args = parser.parse_args()

    agent = ResearchAgent(
        model_path=args.model,
        wiki_path=args.wiki,
        ctx=args.ctx,
        ngl=args.ngl,
        chunks=args.chunks,
        outdir=args.outdir,
        perplexity_bin=args.perplexity_bin,
        niah_script=args.niah_script,
        skip_niah=args.skip_niah,
        mock=args.mock,
    )
    agent.run(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
