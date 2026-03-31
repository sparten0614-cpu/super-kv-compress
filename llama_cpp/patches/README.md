# llama.cpp Patches

## eviction-completion.patch

Adds StreamingLLM eviction support to `llama-completion` (generation mode).

**What it does:**
- Enables `--evict-ratio`, `--evict-mode`, `--evict-sink` flags in llama-completion
- Implements StreamingLLM-style eviction during prompt processing:
  - Mode 0: Sliding window (evict oldest tokens)
  - Mode 1: StreamingLLM (keep first `sink` tokens + most recent, evict middle)

**Files modified:**
- `common/arg.cpp` — Add LLAMA_EXAMPLE_COMPLETION to eviction arg examples
- `tools/completion/completion.cpp` — Add eviction logic to context shift code

**Apply:**
```bash
cd llama.cpp
git apply ../llama_cpp/patches/eviction-completion.patch
cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)
```

**Note:** The perplexity tool already has eviction support in upstream llama.cpp (our earlier contribution). This patch extends it to the completion tool for NIAH testing.
