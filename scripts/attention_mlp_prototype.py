#!/usr/bin/env python3
"""
Attention Distillation Prototype

Tests whether a small MLP can approximate the attention function F(q) = softmax(qK^T/√d)V
for a fixed context (K, V matrices).

Experiments:
1. Random KV cache - how well can MLP approximate F?
2. NIAH setup - can MLP retrieve a specific needle?
3. Scaling: MLP size vs approximation quality
4. Sparse vs dense attention patterns
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class AttentionMLP(nn.Module):
    """Small MLP to approximate attention function."""
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
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, q):
        return self.net(q)

def attention_function(q, K, V, d_head):
    """Exact attention: F(q) = softmax(qK^T/√d) V"""
    scores = q @ K.T / np.sqrt(d_head)  # [batch, n]
    weights = torch.softmax(scores, dim=-1)  # [batch, n]
    output = weights @ V  # [batch, d]
    return output, weights

def generate_training_data(K, V, d_head, n_samples, method='gaussian'):
    """Generate (query, attention_output) training pairs."""
    n, d = K.shape

    if method == 'gaussian':
        # Random Gaussian queries
        queries = torch.randn(n_samples, d, device=device) * 0.5
    elif method == 'key_based':
        # Queries similar to existing keys (more realistic)
        indices = torch.randint(0, n, (n_samples,))
        queries = K[indices] + torch.randn(n_samples, d, device=device) * 0.3
    elif method == 'mixed':
        # Mix of both
        n1 = n_samples // 2
        n2 = n_samples - n1
        q1 = torch.randn(n1, d, device=device) * 0.5
        indices = torch.randint(0, n, (n2,))
        q2 = K[indices] + torch.randn(n2, d, device=device) * 0.3
        queries = torch.cat([q1, q2], dim=0)

    targets, _ = attention_function(queries, K, V, d_head)
    return queries, targets

def train_mlp(mlp, queries, targets, epochs=500, lr=1e-3, batch_size=256, verbose=True):
    """Train MLP to approximate attention outputs."""
    dataset = TensorDataset(queries, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        for q_batch, t_batch in loader:
            pred = mlp(q_batch)
            loss = nn.functional.mse_loss(pred, t_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        best_loss = min(best_loss, avg_loss)

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:4d}: MSE={avg_loss:.6f}")

    return best_loss

def evaluate_mlp(mlp, K, V, d_head, n_test=1000):
    """Evaluate MLP approximation quality."""
    with torch.no_grad():
        # Test on random queries
        test_q = torch.randn(n_test, d_head, device=device) * 0.5
        exact_out, exact_weights = attention_function(test_q, K, V, d_head)
        mlp_out = mlp(test_q)

        # MSE
        mse = nn.functional.mse_loss(mlp_out, exact_out).item()

        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(mlp_out, exact_out, dim=-1).mean().item()

        # Relative error
        rel_err = (torch.norm(mlp_out - exact_out, dim=-1) / torch.norm(exact_out, dim=-1)).mean().item()

    return {
        'mse': mse,
        'cosine_sim': cos_sim,
        'relative_error': rel_err,
    }

def niah_test(mlp, K, V, d_head, needle_idx):
    """Test if MLP can retrieve the needle value."""
    with torch.no_grad():
        # Query = the needle's key (should retrieve needle's value)
        needle_query = K[needle_idx:needle_idx+1]

        exact_out, exact_weights = attention_function(needle_query, K, V, d_head)
        mlp_out = mlp(needle_query)

        # Check if MLP output is close to needle's value
        needle_value = V[needle_idx:needle_idx+1]

        exact_cos = nn.functional.cosine_similarity(exact_out, needle_value, dim=-1).item()
        mlp_cos = nn.functional.cosine_similarity(mlp_out, needle_value, dim=-1).item()
        mlp_exact_cos = nn.functional.cosine_similarity(mlp_out, exact_out, dim=-1).item()

        # Attention weight on needle
        needle_attn = exact_weights[0, needle_idx].item()

    return {
        'exact_cos_to_needle': exact_cos,
        'mlp_cos_to_needle': mlp_cos,
        'mlp_cos_to_exact': mlp_exact_cos,
        'needle_attention_weight': needle_attn,
    }

def experiment_1_basic():
    """Basic: Can MLP approximate random attention?"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Basic Attention Approximation")
    print("="*60)

    d_head = 128
    n_tokens = 1000

    # Random KV cache
    K = torch.randn(n_tokens, d_head, device=device) * 0.5
    V = torch.randn(n_tokens, d_head, device=device) * 0.5

    # Generate training data
    n_train = 5000
    queries, targets = generate_training_data(K, V, d_head, n_train, method='mixed')

    # Test different MLP sizes
    configs = [
        (64, 2, "Tiny (64h, 2L)"),
        (128, 2, "Small (128h, 2L)"),
        (256, 3, "Medium (256h, 3L)"),
        (512, 3, "Large (512h, 3L)"),
        (256, 4, "Deep (256h, 4L)"),
    ]

    print(f"\nContext: {n_tokens} tokens, d_head={d_head}")
    print(f"Training: {n_train} samples, mixed query sampling\n")
    print(f"{'Config':<20} {'Params':>8} {'Bytes':>8} {'MSE':>10} {'CosSim':>8} {'RelErr':>8} {'Compression':>12}")
    print("-" * 80)

    kv_size = n_tokens * d_head * 2 * 4  # K+V in float32

    for hidden, layers, name in configs:
        mlp = AttentionMLP(d_head, hidden, layers).to(device)
        train_mlp(mlp, queries, targets, epochs=500, verbose=False)
        metrics = evaluate_mlp(mlp, K, V, d_head)
        mlp_size = mlp.n_params * 4  # float32

        print(f"{name:<20} {mlp.n_params:>8,} {mlp_size:>7,}B {metrics['mse']:>10.6f} {metrics['cosine_sim']:>8.4f} {metrics['relative_error']:>8.4f} {kv_size/mlp_size:>10.1f}×")

def experiment_2_niah():
    """NIAH: Can MLP retrieve a specific needle?"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Needle-in-a-Haystack Retrieval")
    print("="*60)

    d_head = 128
    n_tokens = 1000

    # Haystack: random tokens
    K = torch.randn(n_tokens, d_head, device=device) * 0.3
    V = torch.randn(n_tokens, d_head, device=device) * 0.3

    # Needle: distinct K/V at specific position
    needle_positions = [100, 250, 500, 750, 900]  # 10%, 25%, 50%, 75%, 90%

    for needle_idx in needle_positions:
        # Make needle key very distinct
        K[needle_idx] = torch.randn(d_head, device=device) * 2.0  # Larger magnitude
        V[needle_idx] = torch.ones(d_head, device=device) * 3.0   # Distinct value

    # Train MLP
    n_train = 5000
    queries, targets = generate_training_data(K, V, d_head, n_train, method='mixed')

    mlp = AttentionMLP(d_head, 256, 3).to(device)
    train_mlp(mlp, queries, targets, epochs=500, verbose=False)

    # General quality
    metrics = evaluate_mlp(mlp, K, V, d_head)
    print(f"\nGeneral: MSE={metrics['mse']:.6f}, CosSim={metrics['cosine_sim']:.4f}, RelErr={metrics['relative_error']:.4f}")

    # NIAH per position
    print(f"\n{'Position':<10} {'Needle%':>8} {'ExactCos':>10} {'MLPCos':>10} {'MLP↔Exact':>10} {'NeedleAttn':>11}")
    print("-" * 65)

    for needle_idx in needle_positions:
        pct = needle_idx / n_tokens * 100
        niah = niah_test(mlp, K, V, d_head, needle_idx)
        print(f"{needle_idx:<10} {pct:>7.0f}% {niah['exact_cos_to_needle']:>10.4f} {niah['mlp_cos_to_needle']:>10.4f} {niah['mlp_cos_to_exact']:>10.4f} {niah['needle_attention_weight']:>10.4f}")

def experiment_3_scaling():
    """Scaling: How does approximation quality change with context length?"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Context Length Scaling")
    print("="*60)

    d_head = 128
    context_lengths = [100, 500, 1000, 5000, 10000]

    print(f"\nMLP: 256 hidden, 3 layers")
    print(f"\n{'n_tokens':<10} {'KV_bytes':>10} {'MLP_bytes':>10} {'Compress':>10} {'MSE':>10} {'CosSim':>8} {'RelErr':>8}")
    print("-" * 72)

    for n in context_lengths:
        K = torch.randn(n, d_head, device=device) * 0.5
        V = torch.randn(n, d_head, device=device) * 0.5

        n_train = min(5000, n * 5)
        queries, targets = generate_training_data(K, V, d_head, n_train, method='mixed')

        mlp = AttentionMLP(d_head, 256, 3).to(device)
        train_mlp(mlp, queries, targets, epochs=500, verbose=False)
        metrics = evaluate_mlp(mlp, K, V, d_head)

        kv_bytes = n * d_head * 2 * 4
        mlp_bytes = mlp.n_params * 4

        print(f"{n:<10} {kv_bytes:>10,} {mlp_bytes:>10,} {kv_bytes/mlp_bytes:>9.1f}× {metrics['mse']:>10.6f} {metrics['cosine_sim']:>8.4f} {metrics['relative_error']:>8.4f}")

def experiment_4_training_efficiency():
    """How many training samples and epochs are needed?"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Training Efficiency")
    print("="*60)

    d_head = 128
    n_tokens = 1000

    K = torch.randn(n_tokens, d_head, device=device) * 0.5
    V = torch.randn(n_tokens, d_head, device=device) * 0.5

    print(f"\nFixed: n={n_tokens}, MLP 256h 3L")

    # Vary training samples
    print(f"\n--- Varying training samples (500 epochs) ---")
    print(f"{'n_train':<10} {'MSE':>10} {'CosSim':>8} {'RelErr':>8}")
    print("-" * 40)

    for n_train in [100, 500, 1000, 2000, 5000, 10000]:
        queries, targets = generate_training_data(K, V, d_head, n_train, method='mixed')
        mlp = AttentionMLP(d_head, 256, 3).to(device)
        train_mlp(mlp, queries, targets, epochs=500, verbose=False)
        metrics = evaluate_mlp(mlp, K, V, d_head)
        print(f"{n_train:<10} {metrics['mse']:>10.6f} {metrics['cosine_sim']:>8.4f} {metrics['relative_error']:>8.4f}")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    experiment_1_basic()
    experiment_2_niah()
    experiment_3_scaling()
    experiment_4_training_efficiency()
