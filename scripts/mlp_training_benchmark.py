#!/usr/bin/env python3
"""
Benchmark MLP training time to assess practical feasibility
of per-context attention distillation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

class AttentionMLP(nn.Module):
    def __init__(self, d_head, hidden_dim, n_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_head, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, d_head))
        self.net = nn.Sequential(*layers)

    def forward(self, q):
        return self.net(q)

def benchmark_training(d_head=128, n_tokens=1000, n_train=2000, epochs=200,
                       hidden=128, n_layers=2, batch_size=512):
    """Time a single MLP training run."""
    K = torch.randn(n_tokens, d_head, device=device)
    V = torch.randn(n_tokens, d_head, device=device)

    # Generate training data
    queries = torch.randn(n_train, d_head, device=device) * 0.5
    scores = queries @ K.T / (d_head ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    targets = weights @ V

    mlp = AttentionMLP(d_head, hidden, n_layers).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

    # Warm up
    _ = mlp(queries[:10])
    if device.type == 'mps':
        torch.mps.synchronize()

    start = time.time()

    for epoch in range(epochs):
        for i in range(0, n_train, batch_size):
            batch_q = queries[i:i+batch_size]
            batch_t = targets[i:i+batch_size]
            pred = mlp(batch_q)
            loss = nn.functional.mse_loss(pred, batch_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if device.type == 'mps':
        torch.mps.synchronize()

    elapsed = time.time() - start
    n_params = sum(p.numel() for p in mlp.parameters())

    return elapsed, n_params

# Data generation time benchmark
def benchmark_data_generation(d_head=128, n_tokens=1000, n_train=2000):
    """Time the data generation (computing exact attention for training samples)."""
    K = torch.randn(n_tokens, d_head, device=device)
    V = torch.randn(n_tokens, d_head, device=device)
    queries = torch.randn(n_train, d_head, device=device) * 0.5

    if device.type == 'mps':
        torch.mps.synchronize()

    start = time.time()
    scores = queries @ K.T / (d_head ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    targets = weights @ V
    if device.type == 'mps':
        torch.mps.synchronize()

    return time.time() - start

print("="*70)
print("TRAINING TIME BENCHMARK")
print("="*70)

# Benchmark different configs
configs = [
    # (hidden, layers, n_train, epochs, name)
    (64, 2, 1000, 100, "Tiny-fast"),
    (64, 2, 2000, 200, "Tiny-normal"),
    (128, 2, 2000, 200, "Small-normal"),
    (128, 2, 5000, 500, "Small-thorough"),
    (256, 3, 5000, 500, "Medium-thorough"),
]

n_tokens = 1000
d_head = 128
n_heads = 8
n_layers_model = 32
total_mlps = n_heads * n_layers_model  # 256

print(f"\nContext: {n_tokens} tokens, d_head={d_head}")
print(f"Total MLPs needed: {total_mlps} (= {n_heads} heads × {n_layers_model} layers)")
print(f"Device: {device}\n")

print(f"{'Config':<18} {'Params':>8} {'1 MLP (ms)':>12} {'256 MLPs (s)':>13} {'vs Prefill':>10}")
print("-" * 65)

# Estimate prefill time (very rough)
# Prefill for n=1000: O(n * d_model * L * H) ≈ 1000 * 4096 * 32 * 32 * 2 FLOPs
# On M4 (~10 TFLOPS): ~0.8s
# On RTX 5880 (~90 TFLOPS): ~0.09s
prefill_estimate_s = 0.5  # rough estimate for M4

for hidden, mlp_layers, n_train, epochs, name in configs:
    elapsed, n_params = benchmark_training(
        d_head=d_head, n_tokens=n_tokens, n_train=n_train,
        epochs=epochs, hidden=hidden, n_layers=mlp_layers
    )
    total_256 = elapsed * total_mlps
    ratio = total_256 / prefill_estimate_s

    print(f"{name:<18} {n_params:>8,} {elapsed*1000:>10.1f}ms {total_256:>11.1f}s {ratio:>9.0f}×")

# Data generation benchmark
print(f"\n--- Data Generation Time ---")
for n_train in [1000, 2000, 5000]:
    t = benchmark_data_generation(d_head=d_head, n_tokens=n_tokens, n_train=n_train)
    print(f"  {n_train} samples: {t*1000:.1f}ms per head, {t*total_mlps:.1f}s total")

# Inference benchmark (MLP vs exact attention)
print(f"\n--- Inference Time (single query) ---")
K = torch.randn(n_tokens, d_head, device=device)
V = torch.randn(n_tokens, d_head, device=device)
q = torch.randn(1, d_head, device=device)

mlp = AttentionMLP(d_head, 128, 2).to(device)

# Warm up
_ = mlp(q)
_ = torch.softmax(q @ K.T / (d_head**0.5), dim=-1) @ V
if device.type == 'mps':
    torch.mps.synchronize()

n_iters = 1000

start = time.time()
for _ in range(n_iters):
    _ = mlp(q)
if device.type == 'mps':
    torch.mps.synchronize()
mlp_time = (time.time() - start) / n_iters

start = time.time()
for _ in range(n_iters):
    scores = q @ K.T / (d_head**0.5)
    weights = torch.softmax(scores, dim=-1)
    _ = weights @ V
if device.type == 'mps':
    torch.mps.synchronize()
exact_time = (time.time() - start) / n_iters

print(f"  MLP forward:     {mlp_time*1e6:.1f} µs")
print(f"  Exact attention: {exact_time*1e6:.1f} µs")
print(f"  Speedup:         {exact_time/mlp_time:.1f}×")

# Break-even analysis
print(f"\n--- Break-Even Analysis ---")
print(f"  Training overhead (Tiny-fast): {configs[0]}")
tiny_time, _ = benchmark_training(d_head=d_head, n_tokens=n_tokens,
                                   n_train=1000, epochs=100, hidden=64, n_layers=2)
total_training = tiny_time * total_mlps
time_saved_per_token = (exact_time - mlp_time) * total_mlps
if time_saved_per_token > 0:
    break_even_tokens = total_training / time_saved_per_token
    print(f"  Training cost: {total_training:.1f}s")
    print(f"  Time saved per decode token: {time_saved_per_token*1000:.2f}ms")
    print(f"  Break-even after: {break_even_tokens:.0f} decode tokens")
else:
    print(f"  MLP is slower than exact attention — no break-even")
