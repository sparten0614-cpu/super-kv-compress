#!/usr/bin/env python3
"""
Hero Figure: PPL vs NIAH scatter plot.
Shows quantization is NIAH-safe while eviction is NIAH-unsafe.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

# Data points: (label, ppl_delta%, niah%, compression, group)
data = [
    # Quantization points (NIAH-safe)
    ("FP16",        0.00,  100, 1.0,  "baseline"),
    ("Q8_0",       -0.09,  100, 1.9,  "quant"),
    ("TQ K6V4",     0.52,  100, 3.2,  "quant"),
    ("Q4_0",        3.06,  100, 4.0,  "quant"),
    # Eviction points (NIAH-unsafe)
    ("50% evict",  -0.09,   60, 2.0,  "evict"),
    ("60% evict",  -0.003,  50, 2.5,  "evict"),
    ("70% evict",   1.04,   40, 3.3,  "evict"),
    ("85% evict",   6.37,   20, 6.7,  "evict"),
    # Combined
    ("Q4+50%ev",    3.0,    60, 8.0,  "combined"),
    # H2O
    ("H2O 50%",     0.0,    60, 2.0,  "h2o"),
]

# Style mapping
styles = {
    "baseline": dict(marker="*",  color="#333333", s=200, zorder=5),
    "quant":    dict(marker="s",  color="#2196F3", s=120, zorder=4),
    "evict":    dict(marker="^",  color="#F44336", s=120, zorder=4),
    "combined": dict(marker="D",  color="#9C27B0", s=120, zorder=4),
    "h2o":      dict(marker="v",  color="#FF9800", s=120, zorder=4),
}

labels_legend = {
    "baseline": "Baseline (FP16)",
    "quant":    "Quantization only",
    "evict":    "Eviction only (StreamingLLM)",
    "combined": "Quantization + Eviction",
    "h2o":      "H2O (attention-aware eviction)",
}

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# Plot each group
plotted_groups = set()
for label, ppl, niah, comp, group in data:
    style = styles[group]
    legend_label = labels_legend[group] if group not in plotted_groups else None
    plotted_groups.add(group)
    ax.scatter(ppl, niah, label=legend_label, edgecolors='white', linewidths=0.5, **style)

    # Annotation
    offset = (8, -5)
    if label == "FP16":
        offset = (-40, 8)
    elif label == "H2O 50%":
        offset = (8, 8)
    elif label == "85% evict":
        offset = (8, 5)
    elif label == "Q4+50%ev":
        offset = (8, 8)
    elif label == "50% evict":
        offset = (-65, -12)
    elif label == "60% evict":
        offset = (-65, -5)

    ax.annotate(
        f"{label}\n({comp}x)",
        (ppl, niah),
        textcoords="offset points",
        xytext=offset,
        fontsize=8,
        color=style["color"],
        alpha=0.85,
    )

# Add shaded regions
ax.axhspan(90, 105, alpha=0.06, color='green', zorder=0)
ax.axhspan(0, 90, alpha=0.04, color='red', zorder=0)
ax.axvspan(-1, 1, alpha=0.04, color='green', zorder=0)

# Labels for regions
ax.text(0.5, 95, "NIAH-safe zone", fontsize=9, color='green', alpha=0.5, ha='center')
ax.text(4.0, 30, "NIAH-unsafe zone", fontsize=9, color='red', alpha=0.5, ha='center')

# Axes
ax.set_xlabel("PPL Increase (%)", fontsize=12)
ax.set_ylabel("NIAH Accuracy (%)", fontsize=12)
ax.set_title("Quantization vs Eviction: Impact on PPL and Retrieval (NIAH)", fontsize=13, fontweight='bold')
ax.set_xlim(-1.5, 8)
ax.set_ylim(10, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.2, label='PPL 1% threshold')

ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig("figures/hero_ppl_vs_niah.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/hero_ppl_vs_niah.png", dpi=300, bbox_inches='tight')
print("Saved figures/hero_ppl_vs_niah.pdf and .png")
