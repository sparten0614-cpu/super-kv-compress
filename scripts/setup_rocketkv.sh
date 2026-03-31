#!/bin/bash
# RocketKV Cloud Setup Script
# Run on cloud GPU (A100/H100 recommended, CUDA required)
#
# Usage: bash setup_rocketkv.sh [hf_token]

set -euo pipefail

HF_TOKEN="${1:-}"

echo "=== RocketKV Cloud Setup ==="

# 1. Clone RocketKV
if [ ! -d "RocketKV" ]; then
    git clone https://github.com/NVlabs/RocketKV.git
    echo "Cloned RocketKV"
else
    echo "RocketKV already cloned"
fi

cd RocketKV

# 2. Install dependencies
pip install -r requirements.txt
pip install flash-attn==2.6.3

# 3. Set HF token
if [ -n "$HF_TOKEN" ]; then
    cat > config/access_tokens.py <<PYEOF
hf_access_token = '$HF_TOKEN'
PYEOF
    echo "HF token configured"
else
    echo "WARNING: No HF token provided. Set it in config/access_tokens.py"
    echo "  Usage: bash setup_rocketkv.sh hf_xxxxx"
fi

# 4. Download datasets
echo "Downloading NIAH dataset (included in repo, checking)..."
ls data/ 2>/dev/null || echo "No data/ dir yet"

echo "Downloading LongBench..."
python scripts/dataset_prep/download_longbench.py || echo "LongBench download failed (may need HF token)"

echo ""
echo "=== Setup Complete ==="
echo "Dependencies installed. Run benchmarks with:"
echo "  bash scripts/run_rocketkv_benchmark.sh"
