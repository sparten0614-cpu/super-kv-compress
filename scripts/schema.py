"""
Shared Result Schema for KV Compression Experiments

All three phases (grid search, Bayesian optimizer, AI agent) use the same
result format for interoperability. Results are stored as JSON Lines (.jsonl).

Schema:
{
    # Config (required)
    "quant_k": str,          # KV cache K type: "f16", "q8_0", "q4_0", etc.
    "quant_v": str,          # KV cache V type: same options
    "evict_ratio": float,    # 0.0 - 0.9
    "evict_method": str,     # "none", "streamingllm", "snapkv", "h2o", "expected_attn"
    "skip_layers": str,      # "" or comma-sep layer indices: "0" or "0,31"

    # Metrics (null if failed)
    "compression": float,    # vs F16 baseline (e.g. 3.56 for q4_0)
    "ppl": float | null,     # perplexity (lower = better)
    "ppl_delta": float | null, # % change vs baseline (e.g. +3.1)
    "niah": float | null,    # 0.0 - 1.0 retrieval accuracy

    # Metadata (optional)
    "ctx": int,              # context size
    "model": str,            # model filename
    "phase": str,            # "grid", "bayesian", "agent"
    "ppl_time_s": float,     # seconds for PPL eval
    "niah_time_s": float,    # seconds for NIAH eval
    "reasoning": str,        # agent's reasoning (Phase 3 only)
    "agent_proposed": bool,  # true if proposed by AI agent
}
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import os


# Canonical result dict keys
REQUIRED_KEYS = ["quant_k", "quant_v", "evict_ratio", "evict_method", "compression"]
OPTIONAL_KEYS = ["skip_layers", "ppl", "ppl_delta", "niah", "ctx", "model",
                 "phase", "ppl_time_s", "niah_time_s", "reasoning", "agent_proposed"]


@dataclass
class ExperimentResult:
    """Canonical experiment result. All phases produce this."""
    quant_k: str
    quant_v: str
    evict_ratio: float
    evict_method: str
    compression: float
    skip_layers: str = ""
    ppl: Optional[float] = None
    ppl_delta: Optional[float] = None
    niah: Optional[float] = None
    ctx: int = 4096
    model: str = ""
    phase: str = ""
    ppl_time_s: float = 0.0
    niah_time_s: float = 0.0
    reasoning: str = ""
    agent_proposed: bool = False

    def key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.quant_k}_{self.quant_v}_{self.evict_ratio}_{self.evict_method}_{self.skip_layers}"

    def label(self) -> str:
        """Human-readable label."""
        s = f"{self.quant_k}/{self.quant_v}"
        if self.evict_ratio > 0:
            s += f"+{self.evict_method}{self.evict_ratio:.0%}"
        if self.skip_layers:
            s += f"+skip({self.skip_layers})"
        return s

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        # Round floats
        if d["ppl"] is not None:
            d["ppl"] = round(d["ppl"], 4)
        if d["ppl_delta"] is not None:
            d["ppl_delta"] = round(d["ppl_delta"], 2)
        if d["niah"] is not None:
            d["niah"] = round(d["niah"], 2)
        d["compression"] = round(d["compression"], 3)
        return d

    @staticmethod
    def from_dict(d: dict) -> "ExperimentResult":
        """Create from dict (tolerant of missing/extra keys)."""
        known = {f.name for f in ExperimentResult.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return ExperimentResult(**filtered)


def load_results(path: str) -> list[ExperimentResult]:
    """Load results from a JSONL file."""
    results = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    results.append(ExperimentResult.from_dict(d))
                except:
                    pass
    return results


def save_result(path: str, result: ExperimentResult):
    """Append a single result to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def merge_results(*paths: str) -> list[ExperimentResult]:
    """Merge results from multiple JSONL files, deduplicating by key."""
    seen = {}
    for path in paths:
        for r in load_results(path):
            key = r.key()
            if key not in seen or (r.ppl is not None and seen[key].ppl is None):
                seen[key] = r  # keep the one with data
    return list(seen.values())


def find_pareto(results: list[ExperimentResult]) -> list[ExperimentResult]:
    """Find Pareto-optimal results (max compression, min PPL)."""
    pareto = []
    for r in results:
        if r.ppl is None:
            continue
        dominated = False
        for other in results:
            if other.ppl is None or other is r:
                continue
            if (other.compression >= r.compression and
                other.ppl <= r.ppl and
                (other.compression > r.compression or other.ppl < r.ppl)):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return sorted(pareto, key=lambda r: r.compression)
