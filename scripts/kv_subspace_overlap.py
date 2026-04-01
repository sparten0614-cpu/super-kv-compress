#!/usr/bin/env python3
"""
K→V Reconstruction: Subspace Overlap Analysis

Measures how much of V's information lives in K's subspace,
for each layer and head of a GGUF model.

If overlap is high, V can be approximately reconstructed from K
using V̂ = M · K where M = W_V · pinv(W_K).
"""

import sys
import numpy as np
from gguf import GGUFReader

def dequantize_q4_k(data, shape):
    """Dequantize Q4_K format to float32. Simplified."""
    # Q4_K block: 144 bytes per 256 elements
    # For analysis, we'll use the reader's built-in dequant if available
    # Fallback: interpret as raw and reshape
    n_elements = shape[0] * shape[1]
    block_size = 256
    n_blocks = n_elements // block_size

    # Q4_K: each block has scales (12 bytes) + mins (12 bytes) + 4-bit data (128 bytes) = 144 + 8 = ... complex
    # For a quick analysis, let's try to extract fp32 weights differently
    raise NotImplementedError("Q4_K dequantization is complex; use safetensors/HF model instead")

def load_gguf_tensor(reader, name):
    """Try to load a tensor from GGUF."""
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None

def analyze_with_gguf(model_path):
    """Analyze GGUF model for K/V subspace overlap."""
    print(f"Loading {model_path}...")
    reader = GGUFReader(model_path)

    # Print model metadata
    for key in reader.fields:
        field = reader.fields[key]
        if 'layer_count' in key or 'head_count' in key or 'embedding_length' in key:
            # Get the value
            if hasattr(field, 'parts'):
                val = field.parts[-1][0] if len(field.parts) > 0 else '?'
            else:
                val = '?'
            print(f"  {key} = {val}")

    # List K/V tensors
    kv_tensors = [(t.name, t.tensor_type, t.shape) for t in reader.tensors
                  if '.attn_k.' in t.name or '.attn_v.' in t.name]

    print(f"\nFound {len(kv_tensors)} K/V tensors:")
    for name, ttype, shape in kv_tensors[:6]:
        print(f"  {name}: type={ttype}, shape={shape}")
    if len(kv_tensors) > 6:
        print(f"  ... ({len(kv_tensors)} total)")

    return reader, kv_tensors

def analyze_with_safetensors(model_dir):
    """Analyze HuggingFace safetensors model (higher precision weights)."""
    from safetensors import safe_open
    import glob

    files = glob.glob(f"{model_dir}/*.safetensors")
    if not files:
        print(f"No safetensors files in {model_dir}")
        return

    print(f"Loading from {len(files)} safetensors files...")

    # Collect all K/V weight tensors
    k_weights = {}  # layer -> tensor
    v_weights = {}  # layer -> tensor

    for f in sorted(files):
        with safe_open(f, framework="numpy") as sf:
            for key in sf.keys():
                if 'k_proj.weight' in key or 'attn_k.weight' in key:
                    layer = int(key.split('.')[2]) if 'layers' in key else int(key.split('.')[1])
                    k_weights[layer] = sf.get_tensor(key)
                elif 'v_proj.weight' in key or 'attn_v.weight' in key:
                    layer = int(key.split('.')[2]) if 'layers' in key else int(key.split('.')[1])
                    v_weights[layer] = sf.get_tensor(key)

    return k_weights, v_weights

def compute_overlap_from_weights(W_K, W_V, n_kv_heads, d_head):
    """
    Compute subspace overlap between K and V projections.

    W_K: [n_kv_heads * d_head, d_model] — the full K projection matrix
    W_V: [n_kv_heads * d_head, d_model] — the full V projection matrix

    Returns per-head overlap and reconstruction error norm.
    """
    results = []

    for h in range(n_kv_heads):
        # Extract per-head weights
        start = h * d_head
        end = (h + 1) * d_head
        Wk = W_K[start:end, :]  # [d_head, d_model]
        Wv = W_V[start:end, :]  # [d_head, d_model]

        # Compute SVD of W_K for the pseudoinverse
        U_k, S_k, Vt_k = np.linalg.svd(Wk, full_matrices=False)
        # P_K = Vt_k^T @ Vt_k projects onto row space of W_K
        # Vt_k: [d_head, d_model] (but truncated to d_head rows)

        # Projection matrix onto row space of W_K
        # P_K h = Vt_k^T @ Vt_k @ h  (in d_model space)
        P_K = Vt_k.T @ Vt_k  # [d_model, d_model] — expensive but exact

        # Actually, we can compute overlap more efficiently:
        # overlap = ||W_V P_K||_F^2 / ||W_V||_F^2
        # W_V P_K = W_V @ Vt_k^T @ Vt_k
        WV_PK = Wv @ Vt_k.T @ Vt_k  # [d_head, d_model]

        overlap = np.linalg.norm(WV_PK, 'fro')**2 / np.linalg.norm(Wv, 'fro')**2

        # Reconstruction matrix M = W_V @ pinv(W_K)
        # pinv(W_K) = Vt_k^T @ diag(1/S_k) @ U_k^T
        S_k_inv = np.diag(1.0 / S_k)
        W_K_pinv = Vt_k.T @ S_k_inv @ U_k.T  # [d_model, d_head]
        M = Wv @ W_K_pinv  # [d_head, d_head]

        # Residual operator norm: ||W_V (I - P_K)||
        WV_perp = Wv - WV_PK  # [d_head, d_model]
        residual_norm = np.linalg.norm(WV_perp, 2)  # operator (spectral) norm

        # Condition number of M
        M_cond = np.linalg.cond(M)

        results.append({
            'head': h,
            'overlap': overlap,
            'residual_spectral_norm': residual_norm,
            'M_cond': M_cond,
            'M_norm': np.linalg.norm(M, 'fro'),
        })

    return results

def analyze_random_synthetic():
    """
    Sanity check: what's the expected overlap for random W_K, W_V?
    For random matrices in R^{128 x 4096}, row spaces are 128-dim subspaces of R^4096.
    Expected overlap ≈ 128/4096 = 3.125% (random subspace intersection).
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Random matrices (expected overlap ≈ d_head/d_model)")
    print("="*60)

    d_model = 4096
    d_head = 128

    W_K = np.random.randn(d_head, d_model).astype(np.float32)
    W_V = np.random.randn(d_head, d_model).astype(np.float32)

    U_k, S_k, Vt_k = np.linalg.svd(W_K, full_matrices=False)
    WV_PK = W_V @ Vt_k.T @ Vt_k
    overlap = np.linalg.norm(WV_PK, 'fro')**2 / np.linalg.norm(W_V, 'fro')**2

    print(f"  Random overlap: {overlap:.4f} (expected: {d_head/d_model:.4f})")
    print(f"  This is the baseline — real models should be HIGHER if K/V share info")

def analyze_gguf_model(model_path, model_name):
    """Extract weights from GGUF and analyze."""
    reader = GGUFReader(model_path)

    # Detect architecture
    n_layers = None
    n_kv_heads = None
    d_head = 128  # default

    for key in reader.fields:
        field = reader.fields[key]
        try:
            val = field.parts[-1][0]
        except:
            continue
        if 'block_count' in key or 'layer_count' in key:
            n_layers = int(val)
        if 'head_count_kv' in key:
            n_kv_heads = int(val)
        if 'embedding_length' in key:
            d_model = int(val)

    if n_kv_heads and d_model:
        d_head = d_model // (n_kv_heads * (d_model // (n_kv_heads * 128)))  # heuristic
        # Actually for GQA: total KV dim = n_kv_heads * d_head
        # d_head is usually fixed at 128 for llama family
        d_head = 128

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"  n_layers={n_layers}, n_kv_heads={n_kv_heads}, d_head={d_head}, d_model={d_model}")
    print(f"  Expected random overlap: {n_kv_heads * d_head / d_model:.4f}")
    print(f"{'='*60}")

    # GGUF tensors are quantized — we need to dequantize
    # The gguf library can give us raw data but dequantization depends on type
    # For Q4_K_M, this is complex. Let's check tensor types.

    k_tensors = {}
    v_tensors = {}

    for tensor in reader.tensors:
        name = tensor.name
        if '.attn_k.weight' in name:
            layer = int(name.split('.')[1])
            k_tensors[layer] = tensor
        elif '.attn_v.weight' in name:
            layer = int(name.split('.')[1])
            v_tensors[layer] = tensor

    print(f"\n  Found {len(k_tensors)} K tensors, {len(v_tensors)} V tensors")

    if not k_tensors:
        print("  No K/V tensors found, trying alternative naming...")
        for tensor in reader.tensors:
            print(f"    {tensor.name}: type={tensor.tensor_type}, shape={tensor.shape}")
            if len(list(reader.tensors)) > 20:
                break
        return

    # Check tensor type
    sample = list(k_tensors.values())[0]
    print(f"  Tensor type: {sample.tensor_type} (name: {sample.tensor_type.name})")
    print(f"  Shape: {sample.shape}")

    # For quantized tensors, we need proper dequantization
    # gguf library has dequantize support in newer versions
    try:
        # Try to dequantize
        layer = 0
        k_data = k_tensors[layer].data
        print(f"  Layer 0 K data: {k_data.shape}, dtype={k_data.dtype}")

        # Attempt dequant via numpy
        from gguf.quants import dequantize
        W_K_0 = dequantize(k_data, sample.tensor_type)
        W_V_0 = dequantize(v_tensors[layer].data, v_tensors[layer].tensor_type)

        print(f"  Dequantized K shape: {W_K_0.shape}")
        print(f"  Dequantized V shape: {W_V_0.shape}")

        # Reshape to [n_kv_heads * d_head, d_model]
        expected_shape = (n_kv_heads * d_head, d_model)
        W_K_0 = W_K_0.reshape(expected_shape)
        W_V_0 = W_V_0.reshape(expected_shape)

        # Analyze all layers
        print(f"\n  {'Layer':<6} {'Head':<5} {'Overlap':>8} {'Residual':>10} {'M_cond':>10}")
        print(f"  {'-'*45}")

        all_overlaps = []

        for layer_idx in sorted(k_tensors.keys()):
            W_K = dequantize(k_tensors[layer_idx].data, k_tensors[layer_idx].tensor_type)
            W_V = dequantize(v_tensors[layer_idx].data, v_tensors[layer_idx].tensor_type)
            W_K = W_K.reshape(expected_shape).astype(np.float64)
            W_V = W_V.reshape(expected_shape).astype(np.float64)

            results = compute_overlap_from_weights(W_K, W_V, n_kv_heads, d_head)

            for r in results:
                all_overlaps.append(r['overlap'])
                if r['head'] == 0:  # Print first head per layer for brevity
                    print(f"  {layer_idx:<6} {r['head']:<5} {r['overlap']:>8.4f} {r['residual_spectral_norm']:>10.4f} {r['M_cond']:>10.1f}")

        # Summary statistics
        overlaps = np.array(all_overlaps)
        print(f"\n  SUMMARY ({model_name}):")
        print(f"  Mean overlap:   {overlaps.mean():.4f}")
        print(f"  Median overlap: {np.median(overlaps):.4f}")
        print(f"  Min overlap:    {overlaps.min():.4f}")
        print(f"  Max overlap:    {overlaps.max():.4f}")
        print(f"  Std overlap:    {overlaps.std():.4f}")
        print(f"  Random baseline: {n_kv_heads * d_head / d_model:.4f}")
        print(f"  Overlap > 0.5:  {(overlaps > 0.5).sum()}/{len(overlaps)} ({(overlaps > 0.5).mean()*100:.1f}%)")
        print(f"  Overlap > 0.9:  {(overlaps > 0.9).sum()}/{len(overlaps)} ({(overlaps > 0.9).mean()*100:.1f}%)")

        return overlaps

    except Exception as e:
        print(f"  Dequantization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Sanity check first
    analyze_random_synthetic()

    # Analyze available models
    models = [
        ("/Users/mini1/zylos/workspace/turboquant/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Llama-3.1-8B"),
        ("/Users/mini1/zylos/workspace/turboquant/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf", "Qwen2.5-3B"),
        ("/Users/mini1/zylos/workspace/turboquant/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "TinyLlama-1.1B"),
    ]

    for path, name in models:
        try:
            analyze_gguf_model(path, name)
        except Exception as e:
            print(f"\nFailed to analyze {name}: {e}")
            import traceback
            traceback.print_exc()
