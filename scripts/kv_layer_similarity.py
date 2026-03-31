#!/usr/bin/env python3
"""
KV Cache Inter-Layer Similarity Analysis

Measures cosine similarity between KV cache vectors across adjacent layers
to validate the "delta encoding" hypothesis: if adjacent layers produce
highly correlated KV vectors, storing deltas could yield extreme compression.

Usage:
    python kv_layer_similarity.py --model <model_name_or_path> [--text <input_text_file>] [--ctx <context_length>]

Example:
    python kv_layer_similarity.py --model meta-llama/Llama-3.1-8B-Instruct --ctx 512
    python kv_layer_similarity.py --model Qwen/Qwen2.5-7B-Instruct --text input.txt

Output:
    - Per-layer-pair cosine similarity statistics (mean, std, min, max)
    - Overall summary with delta encoding viability assessment
    - JSON results saved to kv_similarity_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np


def load_model_and_tokenizer(model_name):
    """Load model with KV cache output enabled."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_kv_cache(model, tokenizer, text, max_length=512):
    """Run forward pass and extract KV cache."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    # outputs.past_key_values: tuple of (key, value) per layer
    # Each key/value shape: (batch, num_heads, seq_len, head_dim)
    return outputs.past_key_values, inputs["input_ids"].shape[1]


def compute_layer_similarity(kv_cache, seq_len):
    """Compute cosine similarity between adjacent layers' KV vectors."""
    num_layers = len(kv_cache)
    results = {
        "num_layers": num_layers,
        "seq_len": seq_len,
        "key_similarity": [],
        "value_similarity": [],
        "key_delta_norm_ratio": [],
        "value_delta_norm_ratio": [],
    }

    for l in range(num_layers - 1):
        k_curr = kv_cache[l][0].float()  # (batch, heads, seq, dim)
        k_next = kv_cache[l + 1][0].float()
        v_curr = kv_cache[l][1].float()
        v_next = kv_cache[l + 1][1].float()

        # Flatten heads: (batch, heads, seq, dim) -> (batch*heads*seq, dim)
        k_curr_flat = k_curr.reshape(-1, k_curr.shape[-1])
        k_next_flat = k_next.reshape(-1, k_next.shape[-1])
        v_curr_flat = v_curr.reshape(-1, v_curr.shape[-1])
        v_next_flat = v_next.reshape(-1, v_next.shape[-1])

        # Cosine similarity per vector pair
        k_cos = torch.nn.functional.cosine_similarity(k_curr_flat, k_next_flat, dim=-1)
        v_cos = torch.nn.functional.cosine_similarity(v_curr_flat, v_next_flat, dim=-1)

        # Delta norm ratio: ||delta|| / ||original||
        # If this is small, delta encoding is viable
        k_delta = k_next_flat - k_curr_flat
        v_delta = v_next_flat - v_curr_flat
        k_delta_ratio = torch.norm(k_delta, dim=-1) / (torch.norm(k_curr_flat, dim=-1) + 1e-8)
        v_delta_ratio = torch.norm(v_delta, dim=-1) / (torch.norm(v_curr_flat, dim=-1) + 1e-8)

        k_stats = {
            "layer_pair": f"{l}-{l+1}",
            "cos_mean": k_cos.mean().item(),
            "cos_std": k_cos.std().item(),
            "cos_min": k_cos.min().item(),
            "cos_max": k_cos.max().item(),
            "cos_median": k_cos.median().item(),
        }
        v_stats = {
            "layer_pair": f"{l}-{l+1}",
            "cos_mean": v_cos.mean().item(),
            "cos_std": v_cos.std().item(),
            "cos_min": v_cos.min().item(),
            "cos_max": v_cos.max().item(),
            "cos_median": v_cos.median().item(),
        }
        k_delta_stats = {
            "layer_pair": f"{l}-{l+1}",
            "ratio_mean": k_delta_ratio.mean().item(),
            "ratio_std": k_delta_ratio.std().item(),
            "ratio_median": k_delta_ratio.median().item(),
        }
        v_delta_stats = {
            "layer_pair": f"{l}-{l+1}",
            "ratio_mean": v_delta_ratio.mean().item(),
            "ratio_std": v_delta_ratio.std().item(),
            "ratio_median": v_delta_ratio.median().item(),
        }

        results["key_similarity"].append(k_stats)
        results["value_similarity"].append(v_stats)
        results["key_delta_norm_ratio"].append(k_delta_stats)
        results["value_delta_norm_ratio"].append(v_delta_stats)

    return results


def compute_token_similarity(kv_cache):
    """Compute cosine similarity between adjacent tokens within each layer."""
    num_layers = len(kv_cache)
    results = {"key_token_sim": [], "value_token_sim": [], "key_token_delta": [], "value_token_delta": []}

    for l in range(num_layers):
        k = kv_cache[l][0].float()  # (batch, heads, seq, dim)
        v = kv_cache[l][1].float()
        seq_len = k.shape[2]
        if seq_len < 2:
            continue

        # Adjacent token similarity: token[i] vs token[i+1], averaged over heads
        # Shape: (batch, heads, seq-1, dim)
        k_curr = k[:, :, :-1, :].reshape(-1, k.shape[-1])
        k_next = k[:, :, 1:, :].reshape(-1, k.shape[-1])
        v_curr = v[:, :, :-1, :].reshape(-1, v.shape[-1])
        v_next = v[:, :, 1:, :].reshape(-1, v.shape[-1])

        k_cos = torch.nn.functional.cosine_similarity(k_curr, k_next, dim=-1)
        v_cos = torch.nn.functional.cosine_similarity(v_curr, v_next, dim=-1)

        k_delta = k_next - k_curr
        v_delta = v_next - v_curr
        k_ratio = torch.norm(k_delta, dim=-1) / (torch.norm(k_curr, dim=-1) + 1e-8)
        v_ratio = torch.norm(v_delta, dim=-1) / (torch.norm(v_curr, dim=-1) + 1e-8)

        results["key_token_sim"].append({
            "layer": l, "cos_mean": k_cos.mean().item(), "cos_std": k_cos.std().item(),
            "cos_median": k_cos.median().item(),
        })
        results["value_token_sim"].append({
            "layer": l, "cos_mean": v_cos.mean().item(), "cos_std": v_cos.std().item(),
            "cos_median": v_cos.median().item(),
        })
        results["key_token_delta"].append({
            "layer": l, "ratio_mean": k_ratio.mean().item(), "ratio_median": k_ratio.median().item(),
        })
        results["value_token_delta"].append({
            "layer": l, "ratio_mean": v_ratio.mean().item(), "ratio_median": v_ratio.median().item(),
        })

    return results


def compute_head_similarity(kv_cache):
    """Compute cosine similarity between attention heads within each layer."""
    num_layers = len(kv_cache)
    results = {"key_head_sim": [], "value_head_sim": []}

    for l in range(num_layers):
        k = kv_cache[l][0].float()  # (batch, heads, seq, dim)
        v = kv_cache[l][1].float()
        num_heads = k.shape[1]
        if num_heads < 2:
            continue

        # Pairwise head similarity: average over all head pairs
        k_flat = k[0]  # (heads, seq, dim) -> flatten seq: (heads, seq*dim)
        v_flat = v[0]
        k_flat = k_flat.reshape(num_heads, -1)
        v_flat = v_flat.reshape(num_heads, -1)

        # Compute pairwise cosine similarity matrix
        k_norm = k_flat / (k_flat.norm(dim=-1, keepdim=True) + 1e-8)
        v_norm = v_flat / (v_flat.norm(dim=-1, keepdim=True) + 1e-8)
        k_sim_matrix = (k_norm @ k_norm.T)
        v_sim_matrix = (v_norm @ v_norm.T)

        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(num_heads, num_heads, dtype=torch.bool), diagonal=1)
        k_upper = k_sim_matrix[mask]
        v_upper = v_sim_matrix[mask]

        results["key_head_sim"].append({
            "layer": l, "cos_mean": k_upper.mean().item(), "cos_std": k_upper.std().item(),
            "cos_min": k_upper.min().item(), "cos_max": k_upper.max().item(),
        })
        results["value_head_sim"].append({
            "layer": l, "cos_mean": v_upper.mean().item(), "cos_std": v_upper.std().item(),
            "cos_min": v_upper.min().item(), "cos_max": v_upper.max().item(),
        })

    return results


def compute_cross_layer_matrix(kv_cache):
    """Compute full NxN cosine similarity matrix across ALL layer pairs (not just adjacent)."""
    num_layers = len(kv_cache)
    k_matrix = np.zeros((num_layers, num_layers))
    v_matrix = np.zeros((num_layers, num_layers))

    # Pre-flatten all layers
    k_flat = []
    v_flat = []
    for l in range(num_layers):
        k = kv_cache[l][0].float().reshape(-1, kv_cache[l][0].shape[-1])
        v = kv_cache[l][1].float().reshape(-1, kv_cache[l][1].shape[-1])
        k_flat.append(k)
        v_flat.append(v)

    for i in range(num_layers):
        for j in range(num_layers):
            k_cos = torch.nn.functional.cosine_similarity(k_flat[i], k_flat[j], dim=-1)
            v_cos = torch.nn.functional.cosine_similarity(v_flat[i], v_flat[j], dim=-1)
            k_matrix[i, j] = k_cos.mean().item()
            v_matrix[i, j] = v_cos.mean().item()

    return k_matrix, v_matrix


def print_results(results):
    """Print results in a readable format."""
    print("\n" + "=" * 70)
    print("KV Cache Inter-Layer Similarity Analysis")
    print(f"Layers: {results['num_layers']}, Sequence length: {results['seq_len']}")
    print("=" * 70)

    # Key similarity
    print("\n--- Key Cosine Similarity (adjacent layers) ---")
    print(f"{'Layers':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
    print("-" * 50)
    k_means = []
    for s in results["key_similarity"]:
        print(f"{s['layer_pair']:<10} {s['cos_mean']:>8.4f} {s['cos_std']:>8.4f} {s['cos_min']:>8.4f} {s['cos_max']:>8.4f} {s['cos_median']:>8.4f}")
        k_means.append(s["cos_mean"])

    # Value similarity
    print("\n--- Value Cosine Similarity (adjacent layers) ---")
    print(f"{'Layers':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
    print("-" * 50)
    v_means = []
    for s in results["value_similarity"]:
        print(f"{s['layer_pair']:<10} {s['cos_mean']:>8.4f} {s['cos_std']:>8.4f} {s['cos_min']:>8.4f} {s['cos_max']:>8.4f} {s['cos_median']:>8.4f}")
        v_means.append(s["cos_mean"])

    # Delta norm ratios
    print("\n--- Delta Norm Ratio ||k_{l+1} - k_l|| / ||k_l|| ---")
    print(f"{'Layers':<10} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("-" * 35)
    k_delta_means = []
    for s in results["key_delta_norm_ratio"]:
        print(f"{s['layer_pair']:<10} {s['ratio_mean']:>8.4f} {s['ratio_std']:>8.4f} {s['ratio_median']:>8.4f}")
        k_delta_means.append(s["ratio_mean"])

    print("\n--- Delta Norm Ratio ||v_{l+1} - v_l|| / ||v_l|| ---")
    print(f"{'Layers':<10} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("-" * 35)
    v_delta_means = []
    for s in results["value_delta_norm_ratio"]:
        print(f"{s['layer_pair']:<10} {s['ratio_mean']:>8.4f} {s['ratio_std']:>8.4f} {s['ratio_median']:>8.4f}")
        v_delta_means.append(s["ratio_mean"])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_k_cos = np.mean(k_means)
    avg_v_cos = np.mean(v_means)
    avg_k_delta = np.mean(k_delta_means)
    avg_v_delta = np.mean(v_delta_means)

    print(f"Average Key cosine similarity (adjacent):   {avg_k_cos:.4f}")
    print(f"Average Value cosine similarity (adjacent):  {avg_v_cos:.4f}")
    print(f"Average Key delta norm ratio:                {avg_k_delta:.4f}")
    print(f"Average Value delta norm ratio:              {avg_v_delta:.4f}")

    print("\n--- Delta Encoding Viability ---")
    if avg_k_cos > 0.9 and avg_v_cos > 0.9:
        print("HIGHLY VIABLE: cos > 0.9 → delta encoding could yield 5-10x additional compression")
    elif avg_k_cos > 0.7 and avg_v_cos > 0.7:
        print("MODERATELY VIABLE: cos 0.7-0.9 → delta encoding could yield 2-3x additional compression")
    elif avg_k_cos > 0.5 and avg_v_cos > 0.5:
        print("MARGINAL: cos 0.5-0.7 → delta encoding offers limited benefit")
    else:
        print("NOT VIABLE: cos < 0.5 → layers are too different for delta encoding")

    if avg_k_delta < 0.3:
        print(f"Key deltas are small (ratio {avg_k_delta:.2f}) → good for delta compression")
    else:
        print(f"Key deltas are large (ratio {avg_k_delta:.2f}) → delta encoding less effective for Keys")

    if avg_v_delta < 0.3:
        print(f"Value deltas are small (ratio {avg_v_delta:.2f}) → good for delta compression")
    else:
        print(f"Value deltas are large (ratio {avg_v_delta:.2f}) → delta encoding less effective for Values")


def main():
    parser = argparse.ArgumentParser(description="KV Cache Inter-Layer Similarity Analysis")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--text", type=str, default=None, help="Input text file (default: use sample text)")
    parser.add_argument("--ctx", type=int, default=512, help="Max context length (default: 512)")
    parser.add_argument("--full-matrix", action="store_true", help="Compute full NxN cross-layer similarity matrix")
    parser.add_argument("--output", type=str, default="kv_similarity_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Prepare input text
    if args.text:
        text = Path(args.text).read_text()
    else:
        # Sample text: WikiText-style paragraph for realistic KV cache
        text = (
            "The transformer architecture has become the dominant paradigm in natural language processing. "
            "At its core, the attention mechanism computes pairwise interactions between all tokens in a sequence, "
            "storing intermediate key and value representations in a cache for efficient autoregressive generation. "
            "As context lengths grow to hundreds of thousands of tokens, the memory consumed by this KV cache "
            "becomes the primary bottleneck for deployment. Various compression techniques have been proposed, "
            "including quantization, which reduces the precision of stored values, and eviction, which removes "
            "entries deemed less important for future predictions. The challenge lies in achieving high compression "
            "ratios while maintaining the quality of the model's output, as measured by perplexity or task-specific "
            "benchmarks. Recent work has shown that the attention distribution in transformers is highly skewed, "
            "with a small fraction of tokens receiving the majority of attention weight. This observation suggests "
            "that aggressive compression of low-attention entries may be possible with minimal quality degradation. "
            "However, the relationship between per-token compression error and aggregate quality metrics like "
            "perplexity is nonlinear, with perplexity amplifying errors exponentially through its geometric mean "
            "formulation. Understanding these information-theoretic limits is crucial for designing practical "
            "compression systems that operate near the theoretical optimum."
        ) * 3  # Repeat for longer context

    # Get KV cache
    kv_cache, seq_len = get_kv_cache(model, tokenizer, text, max_length=args.ctx)
    print(f"KV cache extracted: {len(kv_cache)} layers, seq_len={seq_len}")
    print(f"KV shape per layer: K={kv_cache[0][0].shape}, V={kv_cache[0][1].shape}")

    # Compute similarities
    results = compute_layer_similarity(kv_cache, seq_len)

    # Token-level similarity (adjacent tokens within each layer)
    print("\nComputing token-level similarity...")
    token_results = compute_token_similarity(kv_cache)
    results.update(token_results)

    # Head-level similarity (between heads within each layer)
    print("Computing head-level similarity...")
    head_results = compute_head_similarity(kv_cache)
    results.update(head_results)

    # Optional: full cross-layer matrix
    if args.full_matrix:
        print("Computing full cross-layer similarity matrix...")
        k_matrix, v_matrix = compute_cross_layer_matrix(kv_cache)
        results["key_cross_matrix"] = k_matrix.tolist()
        results["value_cross_matrix"] = v_matrix.tolist()

    # Print results
    print_results(results)

    # Print token and head results
    print("\n--- Token-Level Similarity (adjacent tokens, per layer) ---")
    print(f"{'Layer':<8} {'K cos_mean':>10} {'V cos_mean':>10} {'K delta':>10} {'V delta':>10}")
    print("-" * 50)
    k_tok_means, v_tok_means = [], []
    for i in range(len(results["key_token_sim"])):
        ks = results["key_token_sim"][i]
        vs = results["value_token_sim"][i]
        kd = results["key_token_delta"][i]
        vd = results["value_token_delta"][i]
        print(f"{ks['layer']:<8} {ks['cos_mean']:>10.4f} {vs['cos_mean']:>10.4f} {kd['ratio_mean']:>10.4f} {vd['ratio_mean']:>10.4f}")
        k_tok_means.append(ks["cos_mean"])
        v_tok_means.append(vs["cos_mean"])

    avg_k_tok = np.mean(k_tok_means)
    avg_v_tok = np.mean(v_tok_means)
    print(f"\nAvg token-level K cosine: {avg_k_tok:.4f}, V cosine: {avg_v_tok:.4f}")
    if avg_k_tok > 0.7 or avg_v_tok > 0.7:
        print("TOKEN-LEVEL DELTA ENCODING may be viable!")
    else:
        print("Token-level delta encoding unlikely to help.")

    print("\n--- Head-Level Similarity (between heads, per layer) ---")
    print(f"{'Layer':<8} {'K cos_mean':>10} {'K cos_max':>10} {'V cos_mean':>10} {'V cos_max':>10}")
    print("-" * 45)
    for i in range(len(results["key_head_sim"])):
        ks = results["key_head_sim"][i]
        vs = results["value_head_sim"][i]
        print(f"{ks['layer']:<8} {ks['cos_mean']:>10.4f} {ks['cos_max']:>10.4f} {vs['cos_mean']:>10.4f} {vs['cos_max']:>10.4f}")

    # Save JSON
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
