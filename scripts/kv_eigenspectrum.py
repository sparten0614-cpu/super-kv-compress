#!/usr/bin/env python3
"""
KV Cache Eigenvalue Spectrum Analysis

Measures the information content of KV caches by computing:
1. Per-layer covariance eigenvalue spectrum
2. Effective dimensionality (how many eigenvalues are significant)
3. Rate-distortion bound (theoretical minimum bits)
4. Cross-layer correlation (how much info is shared between layers)

This tells us how far current compression is from the information-theoretic optimum.
"""

import numpy as np
import subprocess
import json
import sys
import os

def generate_kv_dump_prompt():
    """Generate a diverse prompt for KV cache analysis."""
    return """The following is a technical document about quantum computing and machine learning.

Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in superposition of both states simultaneously.

In 2024, researchers achieved a breakthrough in quantum error correction, demonstrating a logical qubit with lower error rates than any of its constituent physical qubits. This milestone, achieved using surface codes on a 72-qubit processor, marks the transition from the NISQ era to early fault-tolerant quantum computing.

The special magic number for this experiment is 8472319.

Machine learning models, particularly large language models, require enormous computational resources. A typical 70-billion parameter model requires approximately 140GB of memory in FP16 precision, making deployment on consumer hardware challenging. KV cache compression addresses this by reducing the memory footprint of the key-value cache used during autoregressive generation.

Recent advances in KV cache compression include quantization (reducing precision from 16-bit to 4-bit), eviction (removing less important tokens), and hybrid approaches. Our research shows that quantization safety depends critically on the model's GQA ratio, with higher ratios requiring more conservative quantization."""

def run_inference_and_dump_kv(model_path, llama_cpp_path=None):
    """
    Run llama.cpp inference and extract KV cache statistics.

    Since we can't easily dump raw KV tensors from llama.cpp CLI,
    we'll use a different approach: analyze the weight matrices
    and compute theoretical KV cache properties.
    """
    print("Note: Direct KV cache dump requires llama.cpp API access.")
    print("Using weight-based analysis instead (theoretical properties).")
    return None

def analyze_kv_from_weights(model_path):
    """
    Analyze KV cache compressibility using model weights.

    Key insight: The KV cache at layer ℓ for token t is:
      K_t^ℓ = W_K^ℓ · h_t^ℓ
      V_t^ℓ = W_V^ℓ · h_t^ℓ

    The covariance of K vectors across tokens is:
      Σ_K^ℓ = W_K^ℓ · Σ_h^ℓ · (W_K^ℓ)^T

    where Σ_h^ℓ is the covariance of residual stream vectors.

    If we assume Σ_h ≈ I (whitened residuals, approximately true after LayerNorm),
    then Σ_K ≈ W_K · W_K^T, and the eigenvalues of Σ_K are the squared
    singular values of W_K.
    """
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(model_path)

    # Get model dimensions
    n_layers = None
    n_kv_heads = None
    d_model = None

    for key in reader.fields:
        field = reader.fields[key]
        try:
            val = int(field.parts[-1][0])
        except:
            continue
        if 'block_count' in key:
            n_layers = val
        if 'head_count_kv' in key:
            n_kv_heads = val
        if 'embedding_length' in key:
            d_model = val

    d_head = 128
    print(f"Model: n_layers={n_layers}, n_kv_heads={n_kv_heads}, d_model={d_model}, d_head={d_head}")

    # Collect K and V weight tensors
    k_tensors = {}
    v_tensors = {}
    for tensor in reader.tensors:
        if '.attn_k.weight' in tensor.name:
            layer = int(tensor.name.split('.')[1])
            k_tensors[layer] = tensor
        elif '.attn_v.weight' in tensor.name:
            layer = int(tensor.name.split('.')[1])
            v_tensors[layer] = tensor

    print(f"Found {len(k_tensors)} K and {len(v_tensors)} V weight tensors\n")

    # Per-layer analysis
    all_k_singular_values = []
    all_v_singular_values = []
    cross_layer_sims = []

    expected_shape = (n_kv_heads * d_head, d_model)

    prev_Wk = None

    for layer_idx in sorted(k_tensors.keys()):
        Wk = dequantize(k_tensors[layer_idx].data, k_tensors[layer_idx].tensor_type)
        Wv = dequantize(v_tensors[layer_idx].data, v_tensors[layer_idx].tensor_type)
        Wk = Wk.reshape(expected_shape).astype(np.float64)
        Wv = Wv.reshape(expected_shape).astype(np.float64)

        # SVD of W_K and W_V
        # Singular values of W_K give us the "principal directions" of the K projection
        _, sk, _ = np.linalg.svd(Wk, full_matrices=False)
        _, sv, _ = np.linalg.svd(Wv, full_matrices=False)

        all_k_singular_values.append(sk)
        all_v_singular_values.append(sv)

        # Effective dimensionality (number of singular values > 1% of max)
        k_eff_dim = np.sum(sk > 0.01 * sk[0])
        v_eff_dim = np.sum(sv > 0.01 * sv[0])

        # Energy concentration: what fraction of energy in top-k singular values?
        k_energy = np.cumsum(sk**2) / np.sum(sk**2)
        v_energy = np.cumsum(sv**2) / np.sum(sv**2)

        k_90 = np.searchsorted(k_energy, 0.9) + 1  # dims for 90% energy
        k_99 = np.searchsorted(k_energy, 0.99) + 1
        v_90 = np.searchsorted(v_energy, 0.9) + 1
        v_99 = np.searchsorted(v_energy, 0.99) + 1

        # Cross-layer correlation (cosine similarity of flattened W_K)
        if prev_Wk is not None:
            cos_sim = np.sum(Wk * prev_Wk) / (np.linalg.norm(Wk) * np.linalg.norm(prev_Wk))
            cross_layer_sims.append(cos_sim)
        prev_Wk = Wk.copy()

        # Rate-distortion: minimum bits under Gaussian assumption
        # R(D) = 0.5 * sum(log2(lambda_i / D)) for lambda_i > D
        # where lambda_i = sk_i^2 (eigenvalues of covariance = squared singular values)
        k_eigenvalues = sk**2
        # For D = median eigenvalue (moderate distortion):
        D_med = np.median(k_eigenvalues)
        R_med = 0.5 * np.sum(np.log2(k_eigenvalues[k_eigenvalues > D_med] / D_med))

        # For D = 0.01 * max eigenvalue (low distortion, ~1% error):
        D_low = 0.01 * k_eigenvalues[0]
        R_low = 0.5 * np.sum(np.log2(k_eigenvalues[k_eigenvalues > D_low] / D_low))

        if layer_idx % 4 == 0 or layer_idx == n_layers - 1:
            print(f"Layer {layer_idx:2d}:")
            print(f"  K: eff_dim={k_eff_dim}/{len(sk)}, 90%energy@{k_90}dims, 99%@{k_99}dims, σ_max={sk[0]:.3f}, σ_min={sk[-1]:.6f}")
            print(f"  V: eff_dim={v_eff_dim}/{len(sv)}, 90%energy@{v_90}dims, 99%@{v_99}dims, σ_max={sv[0]:.3f}, σ_min={sv[-1]:.6f}")
            print(f"  Rate-distortion (K): R(1%err)={R_low:.1f} bits/token, R(median)={R_med:.1f} bits/token")
            print(f"  Current: {d_head * n_kv_heads * 16:.0f} bits/token (FP16), {d_head * n_kv_heads * 4:.0f} bits/token (q4_0)")
            if cross_layer_sims:
                print(f"  Cross-layer W_K similarity: {cross_layer_sims[-1]:.4f}")

    # Global summary
    print(f"\n{'='*60}")
    print("GLOBAL SUMMARY")
    print(f"{'='*60}")

    # Average energy concentration across layers
    k_sv_matrix = np.array(all_k_singular_values)
    v_sv_matrix = np.array(all_v_singular_values)

    # Average the normalized energy curves
    k_energy_avg = np.mean([np.cumsum(s**2) / np.sum(s**2) for s in all_k_singular_values], axis=0)
    v_energy_avg = np.mean([np.cumsum(s**2) / np.sum(s**2) for s in all_v_singular_values], axis=0)

    print(f"\nK projection energy concentration (averaged across {n_layers} layers):")
    for pct in [50, 75, 90, 95, 99]:
        dims = np.searchsorted(k_energy_avg, pct/100) + 1
        print(f"  {pct}% energy captured by top {dims}/{len(k_energy_avg)} dimensions ({dims/len(k_energy_avg)*100:.1f}%)")

    print(f"\nV projection energy concentration:")
    for pct in [50, 75, 90, 95, 99]:
        dims = np.searchsorted(v_energy_avg, pct/100) + 1
        print(f"  {pct}% energy captured by top {dims}/{len(v_energy_avg)} dimensions ({dims/len(v_energy_avg)*100:.1f}%)")

    # Cross-layer correlation
    if cross_layer_sims:
        sims = np.array(cross_layer_sims)
        print(f"\nCross-layer W_K weight similarity:")
        print(f"  Mean: {sims.mean():.4f}")
        print(f"  Min:  {sims.min():.4f}")
        print(f"  Max:  {sims.max():.4f}")
        print(f"  Adjacent layers highly similar: {(sims > 0.5).sum()}/{len(sims)}")

    # Condition number analysis
    k_cond = np.array([s[0]/s[-1] for s in all_k_singular_values])
    v_cond = np.array([s[0]/s[-1] for s in all_v_singular_values])
    print(f"\nCondition numbers (σ_max/σ_min):")
    print(f"  K: mean={k_cond.mean():.0f}, max={k_cond.max():.0f}")
    print(f"  V: mean={v_cond.mean():.0f}, max={v_cond.max():.0f}")

    # Rate-distortion summary
    total_dims = d_head * n_kv_heads
    current_fp16 = total_dims * 16 * 2  # K + V, FP16
    current_q4 = total_dims * 4 * 2     # K + V, q4_0

    # Theoretical minimum at 1% distortion
    all_R_low = []
    for sk, sv in zip(all_k_singular_values, all_v_singular_values):
        k_eig = sk**2
        v_eig = sv**2
        D_k = 0.01 * k_eig[0]
        D_v = 0.01 * v_eig[0]
        R_k = 0.5 * np.sum(np.log2(k_eig[k_eig > D_k] / D_k))
        R_v = 0.5 * np.sum(np.log2(v_eig[v_eig > D_v] / D_v))
        all_R_low.append(R_k + R_v)

    avg_R_low = np.mean(all_R_low)
    print(f"\nRate-Distortion Analysis (per layer, per token):")
    print(f"  Current FP16:    {current_fp16:6.0f} bits")
    print(f"  Current q4_0:    {current_q4:6.0f} bits")
    print(f"  Theoretical min (1% err): {avg_R_low:6.1f} bits")
    print(f"  Gap (q4_0 vs theory): {current_q4 / avg_R_low:.1f}×")
    print(f"  Gap (FP16 vs theory): {current_fp16 / avg_R_low:.1f}×")

    # Total across all layers
    total_current_fp16 = current_fp16 * n_layers
    total_current_q4 = current_q4 * n_layers
    total_theory = sum(all_R_low)
    print(f"\nTotal across {n_layers} layers:")
    print(f"  FP16: {total_current_fp16:,} bits/token = {total_current_fp16/8:,.0f} bytes/token")
    print(f"  q4_0: {total_current_q4:,} bits/token = {total_current_q4/8:,.0f} bytes/token")
    print(f"  Theory: {total_theory:,.0f} bits/token = {total_theory/8:,.0f} bytes/token")
    print(f"  Maximum achievable compression vs FP16: {total_current_fp16/total_theory:.1f}×")
    print(f"  Maximum achievable compression vs q4_0: {total_current_q4/total_theory:.1f}×")

    return {
        'k_singular_values': all_k_singular_values,
        'v_singular_values': all_v_singular_values,
        'cross_layer_sims': cross_layer_sims,
        'rate_distortion': all_R_low,
    }

if __name__ == '__main__':
    models = [
        ("/Users/mini1/zylos/workspace/turboquant/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Llama-3.1-8B"),
        ("/Users/mini1/zylos/workspace/turboquant/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf", "Qwen2.5-3B"),
    ]

    for path, name in models:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name}")
        print(f"{'='*60}\n")
        try:
            results = analyze_kv_from_weights(path)
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
