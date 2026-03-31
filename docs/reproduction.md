# Reproduction Guide

Reproduce all experiments from "Beyond Perplexity: Why KV Cache Compression Needs Dual-Metric Evaluation" from scratch.

**Time estimate:** ~4 hours on a single RTX 5880 (48GB) or ~8 hours on Apple M4 (16GB).

---

## 1. Environment Setup

### 1.1 Cloud GPU (Vast.ai / RunPod)

```bash
# Recommended: RTX 4090 (24GB) or RTX 5880 (48GB)
# Ubuntu 22.04, CUDA 12.x, Python 3.10+

# Install system dependencies
sudo apt update && sudo apt install -y cmake build-essential git python3-pip wget

# Clone llama.cpp (with mixed KV quant support)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89"  # adjust for your GPU (89=RTX 4090, 100=RTX 5880)
cmake --build build -j$(nproc)

# Verify
./build/bin/llama-perplexity --help
```

### 1.2 Apple Silicon (M4)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

### 1.3 Clone Our Repository

```bash
git clone https://github.com/sparten0614-cpu/super-kv-compress
cd super-kv-compress
pip install matplotlib  # for Pareto plots
```

## 2. Model Download

Download GGUF models (Q4_K_M quantized weights — this is the model weights quantization, separate from KV cache quantization):

```bash
mkdir -p models && cd models

# Llama-3.1-8B-Instruct
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Mistral-7B-Instruct-v0.3
wget https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf

# Qwen2.5-7B-Instruct
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf

# Ministral-8B-Instruct (optional)
wget https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF/resolve/main/Ministral-8B-Instruct-2410-Q4_K_M.gguf

# Llama-3.3-70B-Instruct (requires 48GB+ GPU)
wget https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf

cd ..
```

### WikiText-2 Test Set

```bash
wget -O wiki.test.raw \
  https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/wiki.test.raw
```

## 3. Experiment Reproduction

Set environment variables:

```bash
export PPLBIN=llama.cpp/build/bin/llama-perplexity
export LLAMA=models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
export MISTRAL=models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
export QWEN=models/qwen2.5-7b-instruct-q4_k_m.gguf
export WIKI=wiki.test.raw
```

### Experiment 1: Quantization Cross-Model Baseline (~30 min)

```bash
# Llama-3.1-8B
$PPLBIN -m $LLAMA -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on                          # f16 baseline
$PPLBIN -m $LLAMA -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0  # q8_0
$PPLBIN -m $LLAMA -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q4_0 --cache-type-v q4_0  # q4_0

# Mistral-7B
$PPLBIN -m $MISTRAL -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on
$PPLBIN -m $MISTRAL -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0
$PPLBIN -m $MISTRAL -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q4_0 --cache-type-v q4_0

# Qwen2.5-7B
$PPLBIN -m $QWEN -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on
$PPLBIN -m $QWEN -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0
$PPLBIN -m $QWEN -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q4_0 --cache-type-v q4_0  # expect PPL ~6600
```

### Experiment 2: Qwen Asymmetric Diagnostic (~15 min)

```bash
# K=f16, V=q4_0 (V safe)
$PPLBIN -m $QWEN -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-v q4_0

# K=q4_0, V=f16 (K kills it)
$PPLBIN -m $QWEN -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q4_0

# K=q8_0, V=q4_0 (optimal Qwen config)
$PPLBIN -m $QWEN -f $WIKI -c 16384 --chunks 2 -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q4_0
```

### Experiment 3: Eviction Gradient (~45 min)

```bash
# StreamingLLM eviction at various rates (Llama-8B, 16K)
for RATE in 0.3 0.5 0.6 0.7 0.8 0.85; do
  echo "=== Eviction rate: $RATE ==="
  bash scripts/run_eviction_gradient.sh $LLAMA $WIKI 16384 $RATE
done
```

If the eviction script is not available, use llama.cpp directly with the `--grp-attn-n` and `--grp-attn-w` flags for context shifting (StreamingLLM behavior).

### Experiment 4: NIAH Retrieval (~30 min)

```bash
# Baseline
python3 scripts/niah_test.py --model $LLAMA --ctx 16384 --ngl 99 --positions 0.1,0.25,0.5,0.75,0.9

# Quantization configs
python3 scripts/niah_test.py --model $LLAMA --ctx 16384 --ngl 99 --positions 0.1,0.25,0.5,0.75,0.9 \
  --cache-type-k q4_0 --cache-type-v q4_0

# Qwen q4_0 (expect 0% NIAH)
python3 scripts/niah_test.py --model $QWEN --ctx 16384 --ngl 99 --positions 0.1,0.25,0.5,0.75,0.9 \
  --cache-type-k q4_0 --cache-type-v q4_0

# Qwen K8V4 (expect 100% NIAH)
python3 scripts/niah_test.py --model $QWEN --ctx 16384 --ngl 99 --positions 0.1,0.25,0.5,0.75,0.9 \
  --cache-type-k q8_0 --cache-type-v q4_0
```

### Experiment 5: Eviction + NIAH (~20 min)

```bash
# StreamingLLM 50% eviction NIAH (expect 60%)
python3 scripts/niah_test.py --model $LLAMA --ctx 16384 --ngl 99 --positions 0.1,0.25,0.5,0.75,0.9 \
  --evict-mode 1 --evict-ratio 0.5 --evict-sink 128

# StreamingLLM 85% eviction NIAH (expect 20%)
python3 scripts/niah_test.py --model $LLAMA --ctx 16384 --ngl 99 --positions 0.1,0.25,0.5,0.75,0.9 \
  --evict-mode 1 --evict-ratio 0.85 --evict-sink 128
```

## 4. Expected Results

### Table 1: Quantization Baseline

| Model | f16 PPL | q8_0 PPL (Δ%) | q4_0 PPL (Δ%) | q8_0 NIAH | q4_0 NIAH |
|-------|---------|---------------|---------------|-----------|-----------|
| Llama-3.1-8B | 6.159 ± 0.02 | 6.154 (-0.09%) | 6.348 (+3.06%) | 100% | 100% |
| Mistral-7B | 5.038 ± 0.02 | 5.037 (-0.03%) | 5.101 (+1.25%) | 100% | 100% |
| Qwen2.5-7B | 5.660 ± 0.02 | 5.548 (-1.98%) | ~6600 (crash) | 100% | 0% |

**Note:** PPL values may vary ±0.02 depending on llama.cpp version and GPU. The relative Δ% should be consistent.

### Table 2: Qwen Asymmetric Diagnostic

| Config | Expected PPL | Status |
|--------|-------------|--------|
| K=f16, V=q4_0 | ~5.58 | OK |
| K=q4_0, V=f16 | ~9700 | Catastrophic |
| K=q8_0, V=q4_0 | ~5.58 | OK, NIAH 100% |

### Table 3: Eviction Gradient (StreamingLLM, Llama-8B, 16K)

| Eviction | Expected PPL Δ% |
|----------|-----------------|
| 30% | -0.2% to 0% |
| 50% | -0.1% to +0.1% |
| 70% | +0.5% to +1.5% |
| 85% | +5% to +10% |

### Table 4: NIAH Under Eviction

| Eviction | Expected NIAH |
|----------|---------------|
| 0% (any quant, GQA ≤ 4) | 100% |
| 50% StreamingLLM | 60% (3/5 positions) |
| 85% StreamingLLM | 20% (1/5 positions) |

## 5. Troubleshooting

### PPL values don't match exactly
- PPL is sensitive to llama.cpp version, chunk count, and GPU precision
- Relative Δ% (vs your own f16 baseline) should be consistent within ±0.5%

### Qwen q4_0 doesn't crash
- Ensure you're using `--cache-type-k q4_0`, not model weight quantization
- The KV cache type is separate from the GGUF model quantization

### NIAH shows different results
- Ensure the needle text matches exactly: "The secret code for Project Alpha is 7492-BLUE-DELTA."
- Check that detection matches against generated text only, not prompt echo

### Flash attention error with mixed K/V types
- Rebuild llama.cpp with `-DGGML_CUDA_FA_ALL_QUANTS=ON`
- Without this flag, only matched types (e.g., q8_0/q8_0) work with FA

### Out of memory
- Reduce `-ngl` (GPU layers) to offload some layers to CPU
- Reduce `--chunks` from 2 to 1
- For 70B models: minimum 48GB VRAM at 16K context with q8_0 KV
