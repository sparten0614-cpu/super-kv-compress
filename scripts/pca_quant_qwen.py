#!/usr/bin/env python3
"""
PCA-Quant Prototype for Qwen2.5-3B (and 7B).

Tests whether PCA-Quant can fix Qwen's catastrophic q4_0 failure (NIAH=0%).
Qwen's failure is caused by GQA 7:1 + extreme K outlier dimensions (K_max=93
at Layer 0). PCA rotation should redistribute this outlier energy across
dimensions, preventing catastrophic quantization error.

Usage (requires GPU):
    python scripts/pca_quant_qwen.py --model Qwen/Qwen2.5-3B
    python scripts/pca_quant_qwen.py --model Qwen/Qwen2.5-7B-Instruct --niah

Requirements:
    pip install transformers torch
    GPU with >= 8GB VRAM for 3B, >= 16GB for 7B
"""

import argparse
import torch
import numpy as np
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Quantization primitives
# ============================================================

def quantize_q4_0(x):
    """q4_0: 4-bit symmetric, block size 32."""
    orig_dtype = x.dtype
    x = x.float()
    orig_shape = x.shape
    x = x.reshape(-1, 32)
    amax = x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    scale = amax / 7.0
    x_q = (x / scale).round().clamp(-8, 7)
    return (x_q * scale).reshape(orig_shape).to(orig_dtype)


def quantize_q8_0(x):
    """q8_0: 8-bit symmetric, block size 32."""
    orig_dtype = x.dtype
    x = x.float()
    orig_shape = x.shape
    x = x.reshape(-1, 32)
    amax = x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    scale = amax / 127.0
    x_q = (x / scale).round().clamp(-128, 127)
    return (x_q * scale).reshape(orig_shape).to(orig_dtype)


def quantize_q2_0(x):
    """q2_0: 2-bit symmetric, block size 32."""
    orig_dtype = x.dtype
    x = x.float()
    orig_shape = x.shape
    x = x.reshape(-1, 32)
    amax = x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    scale = amax / 1.0  # 2-bit signed: [-2, 1], use 1 for symmetric
    x_q = (x / scale).round().clamp(-2, 1)
    return (x_q * scale).reshape(orig_shape).to(orig_dtype)


# ============================================================
# NIAH Test
# ============================================================

NEEDLE = "The special magic number for this experiment is 8472319."
NEEDLE_ANSWER = "8472319"

FILLER_PARAGRAPHS = [
    "The history of cartography stretches back thousands of years. Ancient civilizations created maps to navigate their surroundings and document territorial boundaries. The Babylonians etched maps on clay tablets as early as 600 BCE. Greek scholars like Ptolemy developed sophisticated coordinate systems.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy. Chlorophyll molecules in plant cells absorb light, primarily in the red and blue wavelengths, which is why plants appear green. The process involves light-dependent reactions and the Calvin cycle.",
    "The development of modern telecommunications has transformed human society. From the telegraph to smartphones, each technological leap compressed time and space between people. Alexander Graham Bell's telephone patent in 1876 marked a pivotal moment in communication history.",
    "Renaissance art emerged in Italy during the 14th century. Artists like Leonardo da Vinci, Michelangelo, and Raphael pioneered linear perspective, chiaroscuro, and anatomical accuracy. The Medici family's patronage funded ambitious projects including the Sistine Chapel ceiling.",
    "Ocean currents play a crucial role in regulating Earth's climate. The Gulf Stream carries warm water northward along eastern North America before crossing the Atlantic toward Europe. Deep ocean currents form a global conveyor belt distributing heat across the planet.",
    "Materials science investigates relationships between atomic structure and macroscopic properties. Advances have led to semiconductors, superconductors, and nanomaterials. Understanding crystal structures and defects enables engineering of alloys with specific mechanical properties.",
    "Coffee cultivation originated in Ethiopia. The drink spread to the Arabian Peninsula by the 15th century, where coffeehouses became centers of intellectual discourse. European traders brought coffee home in the 17th century, and it became one of the world's most popular beverages.",
    "Volcanic eruptions have shaped Earth's landscape throughout history. Mount Vesuvius buried Pompeii in 79 AD. The 1815 Tambora eruption caused the Year Without a Summer. Volcanic activity creates new land, as seen in the Hawaiian Islands formed over millions of years.",
    "The immune system is a complex network defending against pathogens. White blood cells identify and destroy foreign invaders. The adaptive immune system remembers previous encounters, enabling faster responses. Vaccines leverage this memory by introducing harmless pathogen versions.",
    "Quantum mechanics describes matter and energy at smallest scales. Wave-particle duality, Heisenberg's uncertainty principle, and quantum entanglement are among its most counterintuitive predictions. Despite strange implications, quantum theory underpins technologies from lasers to transistors.",
]


def build_niah_prompt(needle_position_pct=50, num_paragraphs=30):
    """Build a NIAH test prompt with needle at specified position."""
    paragraphs = [FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)]
                  for i in range(num_paragraphs)]
    needle_idx = max(0, int(num_paragraphs * needle_position_pct / 100) - 1)
    paragraphs.insert(needle_idx, NEEDLE)
    haystack = "\n\n".join(paragraphs)

    # Qwen chat format
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{haystack}\n\n"
        f"What is the special magic number mentioned in the text above?<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def run_niah_test(model, tokenizer, quant_fn, label, device, num_positions=5, max_new_tokens=32):
    """Run NIAH test at multiple needle positions."""
    positions = [10, 25, 50, 75, 90][:num_positions]
    results = []

    for pos in positions:
        prompt = build_niah_prompt(needle_position_pct=pos)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # Process prompt token-by-token with quantized cache
            # For efficiency, process in chunks
            chunk_size = 64
            cache = None

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = input_ids[:, start:end]

                if cache is None:
                    outputs = model(chunk, use_cache=True)
                else:
                    outputs = model(chunk, past_key_values=cache, use_cache=True)

                cache = outputs.past_key_values

                # Quantize K cache
                if quant_fn:
                    for l_idx in range(len(cache.layers)):
                        cache.layers[l_idx].keys = quant_fn(
                            cache.layers[l_idx].keys, l_idx
                        )

            # Generate response
            generated = []
            next_token = None
            for _ in range(max_new_tokens):
                if next_token is None:
                    logits = outputs.logits[:, -1, :]
                else:
                    outputs = model(
                        next_token, past_key_values=cache, use_cache=True
                    )
                    cache = outputs.past_key_values
                    if quant_fn:
                        for l_idx in range(len(cache.layers)):
                            cache.layers[l_idx].keys = quant_fn(
                                cache.layers[l_idx].keys, l_idx
                            )
                    logits = outputs.logits[:, -1, :]

                next_token_id = logits.argmax(dim=-1, keepdim=True)
                generated.append(next_token_id.item())
                next_token = next_token_id

                # Stop at EOS
                if next_token_id.item() in [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ]:
                    break

            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            found = NEEDLE_ANSWER in response
            results.append({
                "position": pos,
                "seq_len": seq_len,
                "response": response[:200],
                "found": found,
            })
            status = "✅" if found else "❌"
            print(f"  {label} @{pos}%: {status} response='{response[:80]}'")

    accuracy = sum(r["found"] for r in results) / len(results) * 100
    return accuracy, results


# ============================================================
# PCA-Quant Calibrator (same as TinyLlama version)
# ============================================================

def calibrate_pca(model, tokenizer, device, cal_text, max_tokens, skip_layers,
                  avg_bits=4.0, tiers=(2, 4, 8)):
    """Calibrate PCA basis from calibration text."""
    tiers = sorted(tiers)
    cal_tokens = tokenizer.encode(cal_text)[:max_tokens]
    cal_input = torch.tensor([cal_tokens], device=device)

    with torch.no_grad():
        cal_out = model(cal_input, use_cache=True)

    configs = {}
    num_layers = len(cal_out.past_key_values.layers)

    for li in range(num_layers):
        if li in skip_layers:
            continue

        k = cal_out.past_key_values.layers[li].keys.squeeze(0).cpu().float().numpy()
        # Use head 0 for shared PCA basis
        k_h0 = k[0]  # (seq_len, head_dim)
        k_centered = k_h0 - k_h0.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(k_centered, full_matrices=False)
        P = Vt.T
        eigenvalues = (S ** 2) / (k_h0.shape[0] - 1)
        eigs = np.maximum(eigenvalues, 1e-10)

        # Water-filling
        geo_mean = np.exp(np.mean(np.log(eigs)))
        b_star = avg_bits + 0.5 * np.log2(eigs / geo_mean)

        # Snap to tiers
        ta = np.array([tiers[np.argmin([abs(b - t) for t in tiers])] for b in b_star])

        # Budget adjustment
        total = ta.sum()
        target = len(ta) * avg_bits
        if total > target:
            for idx in np.argsort(eigs):
                if total <= target:
                    break
                ti = tiers.index(ta[idx])
                if ti > 0:
                    old = ta[idx]
                    ta[idx] = tiers[ti - 1]
                    total -= (old - ta[idx])
        elif total < target:
            for idx in np.argsort(eigs)[::-1]:
                if total >= target:
                    break
                ti = tiers.index(ta[idx])
                if ti < len(tiers) - 1:
                    old = ta[idx]
                    ta[idx] = tiers[ti + 1]
                    total += (ta[idx] - old)

        configs[li] = {
            "P": torch.from_numpy(P).float().to(device),
            "ta": ta,
            "amgm": float(np.mean(eigs) / np.exp(np.mean(np.log(eigs)))),
        }

    return configs


def make_pca_quant_fn(configs, skip_layers, device, per_dim=True):
    """Create PCA-Quant function from calibrated configs.

    Args:
        per_dim: If True, quantize each PCA dim independently along the token
                 axis (avoids block-scale mismatch across dims with different
                 eigenvalues). If False, use the original cross-dim block-32.
    """
    def _quant_per_dim(k_pca, ta):
        """Quantize each PCA dimension independently along token axis."""
        k_q = k_pca.clone()
        # Group dims by tier for efficiency
        for tier_bits in set(ta):
            dims = np.where(ta == tier_bits)[0]
            if len(dims) == 0:
                continue
            # For each dim, quantize across (heads, tokens) independently
            # Transpose: gather all values for these dims, but block along tokens
            for d in dims:
                # Shape: (batch..., seq_len) for one dim across all heads
                col = k_pca[..., d].contiguous()  # (..., seq_len)
                orig_shape = col.shape
                flat = col.reshape(-1)
                # Pad to multiple of 32
                pad_len = (32 - len(flat) % 32) % 32
                if pad_len:
                    flat = torch.nn.functional.pad(flat, (0, pad_len))
                flat = flat.reshape(-1, 32)
                if tier_bits == 2:
                    flat = quantize_q2_0(flat)
                elif tier_bits == 4:
                    flat = quantize_q4_0(flat)
                elif tier_bits == 8:
                    flat = quantize_q8_0(flat)
                flat = flat.reshape(-1)[:len(flat.reshape(-1)) - pad_len if pad_len else len(flat.reshape(-1))]
                k_q[..., d] = flat.reshape(orig_shape)
        return k_q

    def _quant_cross_dim(k_pca, ta):
        """Original cross-dim block-32 quantization."""
        k_q = k_pca.clone()
        for tier_bits in set(ta):
            dims = np.where(ta == tier_bits)[0]
            if len(dims) == 0:
                continue
            sub = k_pca[..., dims]
            fl = sub.reshape(-1, len(dims))
            if len(dims) % 32 != 0:
                fl = torch.nn.functional.pad(fl, (0, 32 - len(dims) % 32))
            if tier_bits == 2:
                sq = quantize_q2_0(fl)
            elif tier_bits == 4:
                sq = quantize_q4_0(fl)
            elif tier_bits == 8:
                sq = quantize_q8_0(fl)
            else:
                sq = fl
            k_q[..., dims] = sq[..., :len(dims)].reshape(sub.shape)
        return k_q

    def pca_quant_fn(k_tensor, layer_idx):
        if layer_idx in skip_layers or layer_idx not in configs:
            return k_tensor
        cfg = configs[layer_idx]
        P, ta = cfg["P"], cfg["ta"]
        k = k_tensor.float()
        k_pca = torch.matmul(k, P)

        if per_dim:
            k_pca_q = _quant_per_dim(k_pca, ta)
        else:
            k_pca_q = _quant_cross_dim(k_pca, ta)

        k_hat = torch.matmul(k_pca_q, P.T)
        return k_hat.to(k_tensor.dtype)

    return pca_quant_fn


def make_uniform_q4_fn(skip_layers):
    """Create uniform q4_0 quantization function."""
    def uniform_q4_fn(k_tensor, layer_idx):
        if layer_idx in skip_layers:
            return k_tensor
        return quantize_q4_0(k_tensor)
    return uniform_q4_fn


# ============================================================
# PPL Evaluation
# ============================================================

def compute_ppl(model, tokenizer, text, quant_fn, device, max_tokens=128):
    """Compute PPL with optional K cache quantization (token-by-token)."""
    tokens = tokenizer.encode(text)[:max_tokens]
    input_ids = torch.tensor([tokens], device=device)
    seq_len = len(tokens)

    with torch.no_grad():
        out = model(input_ids[:, :1], use_cache=True)
        cache = out.past_key_values
        if quant_fn:
            for l in range(len(cache.layers)):
                cache.layers[l].keys = quant_fn(cache.layers[l].keys, l)
        all_logits = [out.logits]

        for t in range(1, seq_len):
            out = model(input_ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if quant_fn:
                for l in range(len(cache.layers)):
                    cache.layers[l].keys = quant_fn(cache.layers[l].keys, l)
            all_logits.append(out.logits)

        logits = torch.cat(all_logits, dim=1)
        loss = torch.nn.CrossEntropyLoss()(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        return torch.exp(loss).item()


# ============================================================
# Main
# ============================================================

CAL_TEXT = """The history of cartography stretches back thousands of years. Ancient civilizations created maps to navigate their surroundings and document territorial boundaries. Photosynthesis is the process by which green plants convert sunlight into chemical energy. The development of modern telecommunications has transformed human society. Renaissance art emerged in Italy during the 14th century. Ocean currents play a crucial role in regulating Earth's climate. Materials science investigates relationships between structure and properties. Coffee cultivation originated in Ethiopia. Volcanic eruptions shaped Earth's landscape. The immune system defends against pathogens. Quantum mechanics describes behavior at smallest scales. The Amazon rainforest spans 5.5 million square kilometers. Glaciers are massive bodies of dense ice constantly moving under their own weight. Neural networks have transformed artificial intelligence research. Database management systems employ various indexing strategies. The printing press revolutionized the spread of information."""

EVAL_TEXTS = [
    "Neural networks have transformed artificial intelligence research over the past decade. Convolutional neural networks excel at image recognition tasks. Transformer architectures have revolutionized natural language processing. The attention mechanism allows models to weigh importance of different input parts. Large language models demonstrate emergent capabilities including reasoning and code generation.",
    "The fall of the Roman Empire was a prolonged process spanning centuries. Germanic tribes pressured northern borders while internal instability led to rapid succession of emperors. Economic troubles including currency debasement weakened military capacity. Constantinople became a second capital in 330 AD.",
    "Database systems employ B-tree indices for logarithmic lookups and hash indices for constant-time point queries. Query optimizers estimate costs of access paths and join orders. Materialized views trade storage for query speed by pre-computing joins.",
]


def main():
    parser = argparse.ArgumentParser(description="PCA-Quant for Qwen")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-layers", type=int, nargs="*", default=None,
                        help="Layers to skip (FP16). Default: auto-detect AM/GM > 3.5")
    parser.add_argument("--avg-bits", type=float, default=4.0)
    parser.add_argument("--niah", action="store_true", help="Run NIAH test")
    parser.add_argument("--ppl", action="store_true", help="Run PPL test")
    parser.add_argument("--rotation-only", action="store_true",
                        help="Test rotation without quantization (isolate rotation effect)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="results/pca_quant_qwen")
    args = parser.parse_args()

    if not args.niah and not args.ppl:
        args.ppl = True  # Default to PPL

    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    print(f"Loading {args.model} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_attn_heads = model.config.num_attention_heads
    gqa_ratio = num_attn_heads // num_kv_heads

    print(f"Config: {num_layers} layers, {num_kv_heads} KV heads, "
          f"{num_attn_heads} attn heads, GQA {gqa_ratio}:1, head_dim={head_dim}")

    # Auto-detect skip layers if not specified
    if args.skip_layers is None:
        print("\nAuto-detecting outlier layers (calibrating to find AM/GM > 3.5)...")
        # Quick calibration to find outlier layers
        quick_configs = calibrate_pca(
            model, tokenizer, device, CAL_TEXT, args.max_tokens,
            skip_layers=[], avg_bits=args.avg_bits,
        )
        skip_layers = []
        for li, cfg in sorted(quick_configs.items()):
            if cfg["amgm"] > 3.5:
                skip_layers.append(li)
                print(f"  Layer {li}: AM/GM={cfg['amgm']:.2f} → SKIP (FP16)")
            elif li < 3 or cfg["amgm"] > 3.0:
                print(f"  Layer {li}: AM/GM={cfg['amgm']:.2f} → borderline")
        if not skip_layers:
            print("  No outlier layers detected (all AM/GM < 3.5)")
        print(f"  Skip layers: {skip_layers}")
    else:
        skip_layers = args.skip_layers

    # Calibrate PCA-Quant
    print(f"\nCalibrating PCA-Quant (skip={skip_layers})...")
    configs = calibrate_pca(
        model, tokenizer, device, CAL_TEXT, args.max_tokens,
        skip_layers=skip_layers, avg_bits=args.avg_bits,
    )

    # Print calibration summary
    print(f"\n{'Layer':>5} | {'Avg bits':>8} | {'4-bit':>5} | {'8-bit':>5} | {'16-bit':>6} | {'AM/GM':>6}")
    print("-" * 50)
    for li in sorted(configs.keys()):
        cfg = configs[li]
        ta = cfg["ta"]
        print(f"{li:>5} | {ta.mean():>8.2f} | {np.sum(ta==4):>5} | "
              f"{np.sum(ta==8):>5} | {np.sum(ta==16):>6} | {cfg['amgm']:>6.2f}")

    pca_fn = make_pca_quant_fn(configs, skip_layers, device)
    u4_fn = make_uniform_q4_fn(skip_layers)

    # Rotation-only function: rotate to PCA and back without quantization
    def rotation_only_fn(k_tensor, layer_idx):
        if layer_idx in skip_layers or layer_idx not in configs:
            return k_tensor
        P = configs[layer_idx]["P"]
        k = k_tensor.float()
        k_back = torch.matmul(torch.matmul(k, P), P.T)
        return k_back.to(k_tensor.dtype)

    if args.rotation_only:
        pca_fn = rotation_only_fn
        print("\n*** ROTATION-ONLY MODE: no quantization, testing rotation effect ***")

    results = {
        "model": args.model,
        "num_layers": num_layers,
        "gqa_ratio": gqa_ratio,
        "head_dim": head_dim,
        "skip_layers": skip_layers,
        "avg_bits": args.avg_bits,
        "rotation_only": args.rotation_only,
    }

    # PPL Test
    if args.ppl:
        print(f"\n{'='*60}")
        print(f"PPL Test ({len(EVAL_TEXTS)} texts, {args.max_tokens} tokens)")
        print(f"{'='*60}")

        ppl_results = []
        for i, text in enumerate(EVAL_TEXTS):
            p_fp16 = compute_ppl(model, tokenizer, text, None, device, args.max_tokens)
            p_q4 = compute_ppl(model, tokenizer, text, u4_fn, device, args.max_tokens)
            p_pca = compute_ppl(model, tokenizer, text, pca_fn, device, args.max_tokens)

            dq = (p_q4 - p_fp16) / p_fp16 * 100
            dp = (p_pca - p_fp16) / p_fp16 * 100
            ratio = (p_pca - p_fp16) / (p_q4 - p_fp16) * 100 if p_q4 != p_fp16 else float('nan')

            print(f"  Text {i}: FP16={p_fp16:.4f}, q4={p_q4:.4f} ({dq:+.2f}%), "
                  f"PCA={p_pca:.4f} ({dp:+.2f}%), ratio={ratio:.1f}%")
            ppl_results.append({
                "text_idx": i, "ppl_fp16": p_fp16, "ppl_q4": p_q4, "ppl_pca": p_pca,
                "delta_q4_pct": dq, "delta_pca_pct": dp, "pca_frac": ratio,
            })

        results["ppl"] = ppl_results
        mean_dq = np.mean([r["delta_q4_pct"] for r in ppl_results])
        mean_dp = np.mean([r["delta_pca_pct"] for r in ppl_results])
        print(f"\n  Mean: Δq4={mean_dq:+.2f}%, ΔPCA={mean_dp:+.2f}%")

    # NIAH Test
    if args.niah:
        print(f"\n{'='*60}")
        print(f"NIAH Test (needle-in-a-haystack)")
        print(f"{'='*60}")

        print("\n  FP16 baseline:")
        acc_fp16, res_fp16 = run_niah_test(model, tokenizer, None, "FP16", device)

        print("\n  Uniform q4_0:")
        acc_q4, res_q4 = run_niah_test(model, tokenizer, u4_fn, "q4_0", device)

        print("\n  PCA-Quant:")
        acc_pca, res_pca = run_niah_test(model, tokenizer, pca_fn, "PCA-Quant", device)

        print(f"\n  NIAH Accuracy: FP16={acc_fp16:.0f}%, q4={acc_q4:.0f}%, PCA={acc_pca:.0f}%")

        results["niah"] = {
            "fp16": {"accuracy": acc_fp16, "results": res_fp16},
            "q4": {"accuracy": acc_q4, "results": res_q4},
            "pca": {"accuracy": acc_pca, "results": res_pca},
        }

    # Save
    output_path = os.path.join(args.output_dir, "qwen_pca_quant_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
