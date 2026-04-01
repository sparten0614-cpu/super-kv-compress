#!/usr/bin/env python3
"""
PCA-Quant Prototype: Water-Filling Adaptive KV Cache Quantization.

Validates PCA-Quant theory by comparing three configurations at the same bit budget:
  1. FP16 baseline (no quantization)
  2. Uniform q4_0 (all K dims at 4-bit)
  3. PCA-Quant (water-filling tiered allocation in PCA basis)

Usage:
    python scripts/pca_quant_prototype.py
    python scripts/pca_quant_prototype.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import torch
import numpy as np
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Quantization primitives (mimicking llama.cpp q4_0 / q8_0)
# ============================================================

def quantize_q4_0(x):
    """Simulate q4_0: 4-bit symmetric quantization per block of 32."""
    x = x.float()
    orig_shape = x.shape
    # Flatten to (N, 32) blocks
    x_flat = x.reshape(-1, 32)
    amax = x_flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    scale = amax / 7.0  # 4-bit signed: [-8, 7], use 7 for symmetric
    x_quant = (x_flat / scale).round().clamp(-8, 7)
    x_deq = (x_quant * scale).reshape(orig_shape)
    return x_deq


def quantize_q8_0(x):
    """Simulate q8_0: 8-bit symmetric quantization per block of 32."""
    x = x.float()
    orig_shape = x.shape
    x_flat = x.reshape(-1, 32)
    amax = x_flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    scale = amax / 127.0
    x_quant = (x_flat / scale).round().clamp(-128, 127)
    x_deq = (x_quant * scale).reshape(orig_shape)
    return x_deq


def quantize_uniform(x, bits=4):
    """Uniform quantization at specified bits."""
    if bits == 4:
        return quantize_q4_0(x)
    elif bits == 8:
        return quantize_q8_0(x)
    elif bits >= 16:
        return x.float()
    else:
        raise ValueError(f"Unsupported bits: {bits}")


# ============================================================
# PCA-Quant Core
# ============================================================

class PCAQuantCalibrator:
    """Calibrate PCA basis and tier assignment from K cache samples."""

    def __init__(self, avg_bits=4.0, tiers=(4, 8, 16), skip_layers=None):
        self.avg_bits = avg_bits
        self.tiers = sorted(tiers)  # e.g., [4, 8, 16]
        self.skip_layers = skip_layers or []
        self.layer_configs = {}  # layer_idx -> {P, tier_assignment, eigenvalues}

    def calibrate_layer(self, k_matrix, layer_idx):
        """Calibrate PCA basis for one layer from K cache (num_heads, seq_len, head_dim).

        Uses head 0 for PCA basis (justified by per-head CV < 15%).
        """
        if layer_idx in self.skip_layers:
            return

        num_heads, seq_len, head_dim = k_matrix.shape
        # Use head 0 for shared PCA basis
        k_h0 = k_matrix[0]  # (seq_len, head_dim)
        k_centered = k_h0 - k_h0.mean(axis=0, keepdims=True)

        # SVD for PCA
        U, S, Vt = np.linalg.svd(k_centered, full_matrices=False)
        P = Vt.T  # (head_dim, head_dim) — columns are principal components
        eigenvalues = (S ** 2) / (seq_len - 1)  # Covariance eigenvalues

        # Water-filling bit allocation
        eigs = np.maximum(eigenvalues, 1e-10)
        geo_mean = np.exp(np.mean(np.log(eigs)))
        b_star = self.avg_bits + 0.5 * np.log2(eigs / geo_mean)

        # Snap to nearest tier
        tier_assignment = np.zeros(head_dim, dtype=int)
        for j in range(head_dim):
            # Find nearest tier
            diffs = [abs(b_star[j] - t) for t in self.tiers]
            tier_assignment[j] = self.tiers[np.argmin(diffs)]

        # Adjust to match target avg bits (greedy)
        actual_avg = tier_assignment.mean()
        target_total = head_dim * self.avg_bits
        actual_total = tier_assignment.sum()

        # If over budget, downgrade highest-bit dims with smallest eigenvalues first
        if actual_total > target_total:
            # Sort dims by eigenvalue (ascending) — downgrade low-eigenvalue dims first
            order = np.argsort(eigs)
            for idx in order:
                if actual_total <= target_total:
                    break
                current_tier_idx = self.tiers.index(tier_assignment[idx])
                if current_tier_idx > 0:
                    old = tier_assignment[idx]
                    tier_assignment[idx] = self.tiers[current_tier_idx - 1]
                    actual_total -= (old - tier_assignment[idx])

        # If under budget, upgrade lowest-bit dims with largest eigenvalues first
        elif actual_total < target_total:
            order = np.argsort(eigs)[::-1]
            for idx in order:
                if actual_total >= target_total:
                    break
                current_tier_idx = self.tiers.index(tier_assignment[idx])
                if current_tier_idx < len(self.tiers) - 1:
                    old = tier_assignment[idx]
                    tier_assignment[idx] = self.tiers[current_tier_idx + 1]
                    actual_total += (tier_assignment[idx] - old)

        self.layer_configs[layer_idx] = {
            "P": P,
            "eigenvalues": eigenvalues,
            "tier_assignment": tier_assignment,
            "b_star_continuous": b_star,
            "actual_avg_bits": tier_assignment.mean(),
        }

    def summary(self):
        """Print calibration summary."""
        print(f"\nPCA-Quant Calibration Summary")
        print(f"{'='*60}")
        print(f"Skip layers: {self.skip_layers}")
        print(f"Target avg bits: {self.avg_bits}")
        print(f"Tiers: {self.tiers}")
        print(f"{'Layer':>5} | {'Avg bits':>8} | {'4-bit':>5} | {'8-bit':>5} | {'16-bit':>6} | {'AM/GM':>6}")
        print(f"{'-'*50}")

        for layer_idx in sorted(self.layer_configs.keys()):
            cfg = self.layer_configs[layer_idx]
            ta = cfg["tier_assignment"]
            n4 = np.sum(ta == 4)
            n8 = np.sum(ta == 8)
            n16 = np.sum(ta == 16)
            eigs = cfg["eigenvalues"]
            eigs_pos = eigs[eigs > 1e-10]
            amgm = np.mean(eigs_pos) / np.exp(np.mean(np.log(eigs_pos)))
            print(f"{layer_idx:>5} | {cfg['actual_avg_bits']:>8.2f} | {n4:>5} | {n8:>5} | {n16:>6} | {amgm:>6.2f}")


class PCAQuantizer:
    """Apply PCA-Quant to K cache tensors."""

    def __init__(self, calibrator):
        self.calibrator = calibrator

    def quantize_k(self, k_tensor, layer_idx):
        """Quantize K cache for one layer.

        Args:
            k_tensor: (batch, num_heads, seq_len, head_dim) or (num_heads, seq_len, head_dim)
            layer_idx: layer index

        Returns:
            Quantized K tensor (same shape, dequantized)
        """
        if layer_idx in self.calibrator.skip_layers:
            return k_tensor  # FP16 skip

        if layer_idx not in self.calibrator.layer_configs:
            return k_tensor

        cfg = self.calibrator.layer_configs[layer_idx]
        P = torch.from_numpy(cfg["P"]).float()  # (head_dim, head_dim)
        tier_assignment = cfg["tier_assignment"]

        has_batch = k_tensor.dim() == 4
        if has_batch:
            k = k_tensor.squeeze(0)  # (num_heads, seq_len, head_dim)
        else:
            k = k_tensor

        k_float = k.float()
        num_heads, seq_len, head_dim = k_float.shape

        # Rotate to PCA basis: k_pca = k @ P  (each row of k multiplied by P)
        k_pca = torch.matmul(k_float, P)  # (num_heads, seq_len, head_dim)

        # Per-dimension tiered quantization
        k_pca_quant = k_pca.clone()
        for tier_bits in set(tier_assignment):
            mask = torch.from_numpy(tier_assignment == tier_bits)
            dims = mask.nonzero(as_tuple=True)[0]
            if len(dims) == 0:
                continue
            # Extract dims, quantize, put back
            subset = k_pca[:, :, dims]  # (num_heads, seq_len, n_dims)
            # Pad to multiple of 32 for block quantization
            n_dims = len(dims)
            # Reshape to (num_heads * seq_len, n_dims) for quantization
            subset_flat = subset.reshape(-1, n_dims)
            if n_dims % 32 != 0:
                pad = 32 - (n_dims % 32)
                subset_flat = torch.nn.functional.pad(subset_flat, (0, pad))
            subset_quant = quantize_uniform(subset_flat, bits=tier_bits)
            subset_quant = subset_quant[:, :n_dims]  # Remove padding
            k_pca_quant[:, :, dims] = subset_quant.reshape(num_heads, seq_len, n_dims)

        # Rotate back: k_hat = k_pca_quant @ P^T
        k_hat = torch.matmul(k_pca_quant, P.T)

        if has_batch:
            k_hat = k_hat.unsqueeze(0)

        return k_hat.to(k_tensor.dtype)


# ============================================================
# Evaluation: PPL with quantized K cache
# ============================================================

def compute_ppl_with_quantized_cache(model, tokenizer, text, quant_fn=None, max_tokens=None):
    """Compute perplexity with optional K cache quantization.

    Args:
        model: causal LM
        tokenizer: tokenizer
        text: input text
        quant_fn: function(k_tensor, layer_idx) -> quantized k_tensor, or None for FP16
        max_tokens: limit tokens (for speed)

    Returns:
        perplexity (float)
    """
    tokens = tokenizer.encode(text)
    if max_tokens:
        tokens = tokens[:max_tokens]
    input_ids = torch.tensor([tokens], device=next(model.parameters()).device)
    seq_len = input_ids.shape[1]

    # Forward pass with cache
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    if quant_fn is not None:
        # Quantize the K cache
        cache = outputs.past_key_values
        for layer_idx in range(len(cache.layers)):
            layer = cache.layers[layer_idx]
            k_orig = layer.keys  # (batch, num_heads, seq_len, head_dim)
            k_quant = quant_fn(k_orig, layer_idx)
            layer.keys = k_quant

    # Now compute PPL: for each position, use the (possibly quantized) cache
    # to predict the next token
    # We do this by running the model again with the quantized cache
    # and computing cross-entropy loss

    # Simple approach: full forward pass, then quantize cache, then
    # evaluate logits from a second forward pass using the quantized cache
    # Actually, the first forward already produced logits with unquantized cache.
    # To evaluate with quantized cache, we need to recompute attention.

    # Better approach: Hook-based quantization during forward pass
    # Let's use a simpler method: compute logits in chunks

    # Simplest correct approach: forward with hooks that quantize K on-the-fly
    model_device = next(model.parameters()).device

    # Remove any existing hooks
    hooks = []

    if quant_fn is not None:
        for layer_idx in range(model.config.num_hidden_layers):
            layer_module = model.model.layers[layer_idx].self_attn

            def make_hook(l_idx):
                def hook_fn(module, args, kwargs, output):
                    # output is typically (attn_output, attn_weights, past_key_value)
                    # We need to intercept the KV cache
                    # In transformers, the cache is updated inside the attention forward
                    # We'll quantize it after the attention computation
                    if hasattr(output, '__len__') and len(output) >= 3:
                        # Try to access the cache from output
                        pass
                    return output
                return hook_fn

        # Actually, the cleanest approach is to quantize after the full forward pass
        # and then re-run the logits computation

    # Cleanest approach: two-pass
    # Pass 1: Full forward to get KV cache
    # Pass 2: Quantize K in cache, then do another forward using the modified cache

    with torch.no_grad():
        # Pass 1: get cache
        outputs1 = model(input_ids, use_cache=True)
        cache = outputs1.past_key_values

        # Quantize K cache
        if quant_fn is not None:
            for layer_idx in range(len(cache.layers)):
                layer = cache.layers[layer_idx]
                k_orig = layer.keys
                k_quant = quant_fn(k_orig, layer_idx)
                # Directly modify the cache
                cache.layers[layer_idx].keys = k_quant

        # Pass 2: Re-run with quantized cache to get logits
        # Feed just the last token with the full cache to get proper logits
        # Actually we need logits for all positions. Use a trick:
        # Run the full sequence again but with use_cache=False and manually
        # inject quantized K into attention

        # Simplest correct method: just compute loss from the first forward's logits
        # but with quantized attention. Unfortunately transformers doesn't easily
        # support "recompute attention with modified cache".

        # Alternative: Compute MSE between quantized and original K as a proxy
        # for quality loss. But SZ wants PPL.

        # Most practical approach: Hook into each layer's attention to replace
        # K before attention computation.

        pass

    # Let's use a proper hook-based approach
    for h in hooks:
        h.remove()

    # Hook-based PPL computation
    quant_hooks = []

    if quant_fn is not None:
        # We'll hook into each attention layer to quantize K after projection
        for layer_idx in range(model.config.num_hidden_layers):
            attn = model.model.layers[layer_idx].self_attn

            def make_k_quant_hook(l_idx):
                original_k_cache = [None]

                def hook_fn(module, args, output):
                    # After self_attn forward, the cache has been updated
                    # We need to quantize the K in the cache
                    # The output tuple varies by transformers version
                    # For transformers 5.x with DynamicCache:
                    # output = (attn_output, attention_weights, past_key_values)
                    # But past_key_values is the shared cache object
                    pass
                return hook_fn

    # Actually the simplest correct approach:
    # Compute logits token-by-token with quantized K cache injection.
    # This is slow but correct.

    total_loss = 0.0
    n_tokens = 0

    with torch.no_grad():
        # Full forward to populate cache
        outputs = model(input_ids[:, :1], use_cache=True)
        cache = outputs.past_key_values

        # Quantize initial K
        if quant_fn:
            for l_idx in range(len(cache.layers)):
                cache.layers[l_idx].keys = quant_fn(cache.layers[l_idx].keys, l_idx)

        all_logits = [outputs.logits]  # logits for position 0

        # Process remaining tokens one at a time
        for t in range(1, seq_len):
            next_token = input_ids[:, t:t+1]
            outputs = model(next_token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values

            # Quantize the newly appended K
            if quant_fn:
                for l_idx in range(len(cache.layers)):
                    cache.layers[l_idx].keys = quant_fn(cache.layers[l_idx].keys, l_idx)

            all_logits.append(outputs.logits)

        # Compute cross-entropy loss
        logits = torch.cat(all_logits, dim=1)  # (1, seq_len, vocab_size)
        # Shift: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ppl = torch.exp(loss).item()

    return ppl


# ============================================================
# Main
# ============================================================

CALIBRATION_TEXT = """The history of cartography stretches back thousands of years. Ancient civilizations created maps to navigate their surroundings and document territorial boundaries. The Babylonians etched maps on clay tablets as early as 600 BCE. Greek scholars like Ptolemy developed sophisticated coordinate systems that influenced mapmaking for centuries. Photosynthesis is the process by which green plants convert sunlight into chemical energy. Chlorophyll molecules in plant cells absorb light, primarily in the red and blue wavelengths, which is why plants appear green. The development of modern telecommunications has transformed human society. Renaissance art emerged in Italy during the 14th century. Ocean currents play a crucial role in regulating Earth's climate. The Gulf Stream carries warm water from the tropics northward. Materials science investigates relationships between structure and properties. Coffee cultivation originated in Ethiopia. Volcanic eruptions have shaped Earth's landscape. The immune system defends against pathogens. Quantum mechanics describes behavior at smallest scales."""

EVAL_TEXT = """Neural networks have transformed artificial intelligence research over the past decade. Convolutional neural networks excel at image recognition tasks, achieving superhuman performance on benchmarks like ImageNet. Recurrent neural networks and their successors, transformer architectures, have revolutionized natural language processing. The attention mechanism allows models to weigh the importance of different parts of the input when generating each output token. Large language models trained on vast corpora of text demonstrate emergent capabilities including reasoning, translation, and code generation. Transfer learning enables models pretrained on general data to be fine-tuned for specific downstream tasks with relatively little task-specific data. Reinforcement learning from human feedback has proven effective for aligning model outputs with human preferences and instructions."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--skip-layers", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--avg-bits", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="results/pca_quant_prototype")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Config: {num_layers} layers, skip={args.skip_layers}, avg_bits={args.avg_bits}")

    # ---- Step 1: Calibration ----
    print("\n[Step 1] Calibrating PCA basis...")
    cal_tokens = tokenizer.encode(CALIBRATION_TEXT)[:args.max_tokens]
    cal_input = torch.tensor([cal_tokens])

    with torch.no_grad():
        cal_outputs = model(cal_input, use_cache=True)

    calibrator = PCAQuantCalibrator(
        avg_bits=args.avg_bits,
        tiers=[4, 8, 16],
        skip_layers=args.skip_layers,
    )

    cache = cal_outputs.past_key_values
    for layer_idx in range(len(cache.layers)):
        k = cache.layers[layer_idx].keys.squeeze(0).numpy()
        calibrator.calibrate_layer(k, layer_idx)

    calibrator.summary()
    quantizer = PCAQuantizer(calibrator)

    # ---- Step 2 & 3: Evaluate PPL ----
    print(f"\n[Step 2] Computing PPL on eval text ({args.max_tokens} tokens max)...")

    # Config 1: FP16 baseline
    print("  Running FP16 baseline...")
    t0 = time.time()
    ppl_fp16 = compute_ppl_with_quantized_cache(
        model, tokenizer, EVAL_TEXT, quant_fn=None, max_tokens=args.max_tokens
    )
    t_fp16 = time.time() - t0
    print(f"  FP16 PPL = {ppl_fp16:.4f} ({t_fp16:.1f}s)")

    # Config 2: Uniform q4_0
    def uniform_q4_fn(k_tensor, layer_idx):
        if layer_idx in args.skip_layers:
            return k_tensor
        return quantize_q4_0(k_tensor)

    print("  Running uniform q4_0...")
    t0 = time.time()
    ppl_q4 = compute_ppl_with_quantized_cache(
        model, tokenizer, EVAL_TEXT, quant_fn=uniform_q4_fn, max_tokens=args.max_tokens
    )
    t_q4 = time.time() - t0
    print(f"  Uniform q4_0 PPL = {ppl_q4:.4f} ({t_q4:.1f}s)")

    # Config 3: PCA-Quant
    def pca_quant_fn(k_tensor, layer_idx):
        return quantizer.quantize_k(k_tensor, layer_idx)

    print("  Running PCA-Quant...")
    t0 = time.time()
    ppl_pca = compute_ppl_with_quantized_cache(
        model, tokenizer, EVAL_TEXT, quant_fn=pca_quant_fn, max_tokens=args.max_tokens
    )
    t_pca = time.time() - t0
    print(f"  PCA-Quant PPL = {ppl_pca:.4f} ({t_pca:.1f}s)")

    # ---- Results ----
    print(f"\n{'='*60}")
    print(f"RESULTS: PCA-Quant Prototype on {args.model}")
    print(f"{'='*60}")
    print(f"{'Config':<20} | {'PPL':>8} | {'ΔPPL':>8} | {'ΔPPL%':>8}")
    print(f"{'-'*55}")
    print(f"{'FP16 baseline':<20} | {ppl_fp16:>8.4f} | {'---':>8} | {'---':>8}")
    print(f"{'Uniform q4_0':<20} | {ppl_q4:>8.4f} | {ppl_q4-ppl_fp16:>+8.4f} | {(ppl_q4-ppl_fp16)/ppl_fp16*100:>+7.2f}%")
    print(f"{'PCA-Quant':<20} | {ppl_pca:>8.4f} | {ppl_pca-ppl_fp16:>+8.4f} | {(ppl_pca-ppl_fp16)/ppl_fp16*100:>+7.2f}%")
    print(f"\nPCA-Quant vs uniform q4_0: {(ppl_q4-ppl_pca)/ppl_q4*100:+.2f}% PPL reduction")
    print(f"Skip layers: {args.skip_layers} (FP16)")
    print(f"Average bits (PCA-Quant, non-skip layers): {np.mean([c['actual_avg_bits'] for c in calibrator.layer_configs.values()]):.2f}")

    # Save results
    results = {
        "model": args.model,
        "skip_layers": args.skip_layers,
        "avg_bits_target": args.avg_bits,
        "max_tokens": args.max_tokens,
        "ppl_fp16": ppl_fp16,
        "ppl_uniform_q4": ppl_q4,
        "ppl_pca_quant": ppl_pca,
        "delta_ppl_q4": ppl_q4 - ppl_fp16,
        "delta_ppl_pca": ppl_pca - ppl_fp16,
        "pca_improvement_over_q4_percent": (ppl_q4 - ppl_pca) / ppl_q4 * 100,
        "per_layer_config": {
            str(k): {
                "avg_bits": float(v["actual_avg_bits"]),
                "tier_counts": {
                    "4bit": int(np.sum(v["tier_assignment"] == 4)),
                    "8bit": int(np.sum(v["tier_assignment"] == 8)),
                    "16bit": int(np.sum(v["tier_assignment"] == 16)),
                },
                "amgm": float(np.mean(v["eigenvalues"][v["eigenvalues"] > 1e-10]) /
                              np.exp(np.mean(np.log(v["eigenvalues"][v["eigenvalues"] > 1e-10])))),
            }
            for k, v in calibrator.layer_configs.items()
        },
    }

    output_path = os.path.join(args.output_dir, "pca_quant_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
