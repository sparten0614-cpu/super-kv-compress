#!/usr/bin/env python3
"""
Multi-model KV Cache Eigenspectrum Analysis.

Runs full eigenspectrum pipeline (AM/GM, per-head variance, seq_len stability)
on any causal LM. Designed for Llama-8B / Qwen-3B on GPU, but works on CPU
for smaller models.

Usage:
    python eigenspectrum_multi_model.py --model meta-llama/Llama-3.1-8B
    python eigenspectrum_multi_model.py --model Qwen/Qwen2.5-3B --device cpu
    python eigenspectrum_multi_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-lens 256 1024
"""

import argparse
import torch
import numpy as np
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Diverse text for calibration (~600 tokens, repeated to reach target seq_len)
CALIBRATION_TEXT = """The history of cartography stretches back thousands of years. Ancient civilizations created maps to navigate their surroundings and document territorial boundaries. The Babylonians etched maps on clay tablets as early as 600 BCE. Greek scholars like Ptolemy developed sophisticated coordinate systems that influenced mapmaking for centuries. Photosynthesis is the process by which green plants convert sunlight into chemical energy. Chlorophyll molecules in plant cells absorb light, primarily in the red and blue wavelengths, which is why plants appear green. The development of modern telecommunications has transformed human society in profound ways. From the telegraph in the 1830s to the smartphone era, each technological leap has compressed the time and space between people. Renaissance art emerged in Italy during the 14th century, marking a dramatic shift from medieval artistic traditions. Artists like Leonardo da Vinci, Michelangelo, and Raphael pioneered techniques such as linear perspective, chiaroscuro, and anatomical accuracy. Ocean currents play a crucial role in regulating Earth's climate. The Gulf Stream carries warm water from the tropics northward along the eastern coast of North America before crossing the Atlantic toward Europe. Deep ocean currents form a global conveyor belt that distributes heat around the planet. The field of materials science investigates the relationship between the structure of materials at atomic scales and their macroscopic properties. Advances in this field have led to the development of semiconductors, superconductors, and nanomaterials. Coffee cultivation originated in Ethiopia, where legend says a goatherd named Kaldi noticed his goats becoming unusually energetic after eating certain berries. Volcanic eruptions have shaped Earth's landscape throughout geological history. The eruption of Mount Vesuvius in 79 AD buried the Roman cities of Pompeii and Herculaneum. The immune system is a complex network of cells, tissues, and organs that defend the body against harmful pathogens. Quantum mechanics describes the behavior of matter and energy at the smallest scales. The Amazon rainforest spans approximately 5.5 million square kilometers across nine South American countries. Glaciers are massive bodies of dense ice that are constantly moving under their own weight. Machine learning algorithms have revolutionized data analysis. The human genome project mapped approximately 20000 genes in human DNA. Plate tectonics explains large-scale movement of Earth's lithosphere."""


def compute_amgm(eigenvalues):
    """Compute AM/GM ratio from eigenvalues."""
    eigs = eigenvalues[eigenvalues > 1e-10]
    if len(eigs) < 2:
        return 1.0
    am = np.mean(eigs)
    gm = np.exp(np.mean(np.log(eigs)))
    return float(am / gm)


def analyze_cache(matrix):
    """Analyze eigenspectrum of a KV cache matrix (num_heads, seq_len, head_dim).
    Returns per-head results."""
    num_heads, seq_len, head_dim = matrix.shape
    results = []
    for head_idx in range(num_heads):
        head_mat = matrix[head_idx]
        centered = head_mat - head_mat.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.sort(eigs)[::-1]
        eigs = np.maximum(eigs, 0)

        nonzero = eigs[eigs > 1e-10]
        cond = float(nonzero[0] / nonzero[-1]) if len(nonzero) > 1 else 1.0
        amgm = compute_amgm(eigs)

        # Energy concentration
        total = eigs.sum()
        top10_idx = max(1, int(len(eigs) * 0.1))
        top10_energy = eigs[:top10_idx].sum() / total if total > 0 else 1.0

        # Dims needed for energy thresholds
        cum_energy = np.cumsum(eigs) / total if total > 0 else np.ones_like(eigs)
        dims_needed = {}
        for t in [0.90, 0.95, 0.99]:
            dims_needed[f"{int(t*100)}%"] = int(np.searchsorted(cum_energy, t) + 1)

        results.append({
            "head": head_idx,
            "amgm": amgm,
            "coding_gain_bits": float(0.5 * np.log2(amgm)) if amgm > 0 else 0,
            "condition_number": cond,
            "top10_energy_frac": float(top10_energy),
            "dims_needed": dims_needed,
            "eigenvalues": eigs.tolist(),
        })
    return results


def extract_kv(model, tokenizer, text, target_len, device):
    """Extract KV cache for a given sequence length."""
    tokens = tokenizer.encode(text)
    while len(tokens) < target_len:
        tokens = tokens + tokens
    tokens = tokens[:target_len]
    input_ids = torch.tensor([tokens], device=device)

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=False, use_cache=True)

    cache = outputs.past_key_values
    k_matrices, v_matrices = [], []
    for layer_idx in range(len(cache.layers)):
        layer = cache.layers[layer_idx]
        k_matrices.append(layer.keys.squeeze(0).cpu().float().numpy())
        v_matrices.append(layer.values.squeeze(0).cpu().float().numpy())

    return k_matrices, v_matrices


def run_analysis(model, tokenizer, device, seq_lens, output_dir):
    """Run full eigenspectrum analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = model.config._name_or_path
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    print(f"Model: {model_name}")
    print(f"Config: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")
    print(f"Seq lengths: {seq_lens}")
    print()

    all_seqlen_results = {}

    for seq_len in seq_lens:
        print(f"{'='*60}")
        print(f"Seq_len = {seq_len}")
        print(f"{'='*60}")
        t0 = time.time()

        k_matrices, v_matrices = extract_kv(model, tokenizer, CALIBRATION_TEXT, seq_len, device)
        print(f"  KV extraction: {time.time()-t0:.1f}s")

        layer_results = {}
        for layer_idx in range(num_layers):
            k_res = analyze_cache(k_matrices[layer_idx])
            v_res = analyze_cache(v_matrices[layer_idx])
            layer_results[layer_idx] = {"K": k_res, "V": v_res}

            k_amgm_h0 = k_res[0]["amgm"]
            v_amgm_h0 = v_res[0]["amgm"]
            if layer_idx < 3 or layer_idx == num_layers - 1:
                print(f"  L{layer_idx}: K AM/GM={k_amgm_h0:.2f} (gain={k_res[0]['coding_gain_bits']:.2f}b), "
                      f"V AM/GM={v_amgm_h0:.2f} (gain={v_res[0]['coding_gain_bits']:.2f}b)")

        all_seqlen_results[seq_len] = layer_results
        print(f"  Done in {time.time()-t0:.1f}s")

    # === Save full results ===
    primary_seq = seq_lens[0]
    primary = all_seqlen_results[primary_seq]

    full_output = {
        "model": model_name,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "primary_seq_len": primary_seq,
        "layers": {str(k): v for k, v in primary.items()},
    }
    with open(os.path.join(output_dir, "eigenspectrum_full.json"), "w") as f:
        json.dump(full_output, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    # === AM/GM summary ===
    amgm_output = {"model": model_name, "source": "actual_kv_cache", "seq_len": primary_seq, "per_layer": []}
    for layer_idx in range(num_layers):
        k_h0 = primary[layer_idx]["K"][0]
        v_h0 = primary[layer_idx]["V"][0]
        amgm_output["per_layer"].append({
            "layer": layer_idx,
            "k_amgm": k_h0["amgm"],
            "v_amgm": v_h0["amgm"],
            "k_coding_gain_bits": k_h0["coding_gain_bits"],
            "v_coding_gain_bits": v_h0["coding_gain_bits"],
        })
    with open(os.path.join(output_dir, "kv_amgm_ratios.json"), "w") as f:
        json.dump(amgm_output, f, indent=2)

    # === Per-head variance ===
    head_var_output = {"model": model_name, "num_heads": num_kv_heads, "per_layer": []}
    for layer_idx in range(num_layers):
        k_amgms = [h["amgm"] for h in primary[layer_idx]["K"]]
        v_amgms = [h["amgm"] for h in primary[layer_idx]["V"]]
        head_var_output["per_layer"].append({
            "layer": layer_idx,
            "k_amgm_per_head": k_amgms,
            "v_amgm_per_head": v_amgms,
            "k_head_cv_percent": float(np.std(k_amgms) / np.mean(k_amgms) * 100),
            "v_head_cv_percent": float(np.std(v_amgms) / np.mean(v_amgms) * 100),
        })
    avg_k_cv = np.mean([d["k_head_cv_percent"] for d in head_var_output["per_layer"]])
    avg_v_cv = np.mean([d["v_head_cv_percent"] for d in head_var_output["per_layer"]])
    head_var_output["avg_k_head_cv"] = float(avg_k_cv)
    head_var_output["avg_v_head_cv"] = float(avg_v_cv)
    with open(os.path.join(output_dir, "per_head_variance.json"), "w") as f:
        json.dump(head_var_output, f, indent=2)

    # === Seq_len stability ===
    if len(seq_lens) > 1:
        stability_output = {"model": model_name, "seq_lens": seq_lens, "per_layer": []}
        for layer_idx in range(num_layers):
            k_vals = [all_seqlen_results[sl][layer_idx]["K"][0]["amgm"] for sl in seq_lens]
            v_vals = [all_seqlen_results[sl][layer_idx]["V"][0]["amgm"] for sl in seq_lens]
            stability_output["per_layer"].append({
                "layer": layer_idx,
                "k_amgm_per_seqlen": {str(sl): float(k) for sl, k in zip(seq_lens, k_vals)},
                "v_amgm_per_seqlen": {str(sl): float(v) for sl, v in zip(seq_lens, v_vals)},
                "k_cv_percent": float(np.std(k_vals) / np.mean(k_vals) * 100),
                "v_cv_percent": float(np.std(v_vals) / np.mean(v_vals) * 100),
            })
        stability_output["avg_k_cv_percent"] = float(np.mean([d["k_cv_percent"] for d in stability_output["per_layer"]]))
        stability_output["avg_v_cv_percent"] = float(np.mean([d["v_cv_percent"] for d in stability_output["per_layer"]]))
        with open(os.path.join(output_dir, "seqlen_stability.json"), "w") as f:
            json.dump(stability_output, f, indent=2)

    # === W_K SVD proxy ===
    print(f"\nComputing W_K SVD proxy AM/GM...")
    wk_output = {"model": model_name, "source": "W_K_SVD_proxy", "per_layer": []}
    for layer_idx in range(num_layers):
        wk = model.model.layers[layer_idx].self_attn.k_proj.weight.detach().cpu().float().numpy()
        s = np.linalg.svd(wk, compute_uv=False)
        s = s[s > 1e-10]
        s2 = s ** 2
        amgm = compute_amgm(s2)
        wk_output["per_layer"].append({
            "layer": layer_idx,
            "wk_svd_amgm": float(amgm),
            "coding_gain_bits": float(0.5 * np.log2(amgm)) if amgm > 0 else 0,
        })
    wk_output["avg_amgm"] = float(np.mean([d["wk_svd_amgm"] for d in wk_output["per_layer"]]))
    with open(os.path.join(output_dir, "wk_svd_amgm.json"), "w") as f:
        json.dump(wk_output, f, indent=2)

    # === Print summary ===
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    k_amgms_h0 = [primary[i]["K"][0]["amgm"] for i in range(num_layers)]
    v_amgms_h0 = [primary[i]["V"][0]["amgm"] for i in range(num_layers)]
    print(f"K avg AM/GM (excl L0): {np.mean(k_amgms_h0[1:]):.2f} → gain = {0.5*np.log2(np.mean(k_amgms_h0[1:])):.2f} bits")
    print(f"V avg AM/GM (excl L0): {np.mean(v_amgms_h0[1:]):.2f} → gain = {0.5*np.log2(np.mean(v_amgms_h0[1:])):.2f} bits")
    print(f"K L0 AM/GM: {k_amgms_h0[0]:.2f} → gain = {0.5*np.log2(k_amgms_h0[0]):.2f} bits")
    print(f"Per-head CV: K={avg_k_cv:.1f}%, V={avg_v_cv:.1f}%")
    if len(seq_lens) > 1:
        print(f"Seq_len CV: K={stability_output['avg_k_cv_percent']:.1f}%, V={stability_output['avg_v_cv_percent']:.1f}%")
    print(f"W_K SVD proxy avg AM/GM: {wk_output['avg_amgm']:.2f}")
    print(f"\nAll results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Multi-model KV Cache Eigenspectrum Analysis")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name/path")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 256, 2048, 4096],
                        help="Sequence lengths to test (first is primary)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/pca_<model_short>/)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    args = parser.parse_args()

    # Determine output dir
    if args.output_dir is None:
        model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        args.output_dir = f"results/pca_{model_short}/"

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    # CPU doesn't support float16
    if device == "cpu":
        dtype = torch.float32

    print(f"Loading {args.model} on {device} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device if device != "cpu" else None
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    # Filter seq_lens that exceed model's max position
    max_pos = getattr(model.config, "max_position_embeddings", 131072)
    seq_lens = [s for s in args.seq_lens if s <= max_pos]
    print(f"Max position embeddings: {max_pos}, testing seq_lens: {seq_lens}")

    run_analysis(model, tokenizer, device, seq_lens, args.output_dir)


if __name__ == "__main__":
    main()
