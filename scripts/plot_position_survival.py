#!/usr/bin/env python3
"""
Position Survival Chart: Shows which needle positions survive at each eviction rate.
Demonstrates StreamingLLM's deterministic position-based eviction pattern.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

# Needle positions tested (as % of context)
positions = [10, 25, 50, 75, 90]
pos_labels = ["10%", "25%", "50%", "75%", "90%"]

# Eviction rates and which positions survived (1=found, 0=missed)
eviction_data = {
    "0%":  [1, 1, 1, 1, 1],
    "50%": [0, 0, 1, 1, 1],
    "60%": [0, 0, 0, 1, 1],
    "70%": [0, 0, 0, 1, 1],
    "85%": [0, 0, 0, 0, 1],
}

eviction_rates = list(eviction_data.keys())
n_rates = len(eviction_rates)
n_pos = len(positions)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

x = np.arange(n_rates)
width = 0.14
colors = ['#E53935', '#FF7043', '#FFA726', '#66BB6A', '#2196F3']

for i, (pos, label, color) in enumerate(zip(positions, pos_labels, colors)):
    vals = [eviction_data[rate][i] for rate in eviction_rates]
    bars = ax.bar(x + (i - 2) * width, vals, width, label=f"Needle @ {label}",
                  color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
    # Add X for missed
    for j, v in enumerate(vals):
        if v == 0:
            ax.text(x[j] + (i - 2) * width, 0.05, "X", ha='center', va='bottom',
                    fontsize=9, color='gray', fontweight='bold')

ax.set_xlabel("Eviction Rate (StreamingLLM)", fontsize=12)
ax.set_ylabel("Retrieved (1) / Missed (0)", fontsize=12)
ax.set_title("Needle Position Survival under StreamingLLM Eviction\n(16K context, Llama-3.1-8B)", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(eviction_rates)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Missed", "Found"])
ax.set_ylim(-0.1, 1.3)
ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)

# Annotation
ax.annotate("Only needles AFTER the\neviction cutoff survive",
            xy=(3, 1.05), fontsize=9, color='gray', style='italic', ha='center')

ax.grid(True, axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig("figures/position_survival.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/position_survival.png", dpi=300, bbox_inches='tight')
print("Saved figures/position_survival.pdf and .png")
