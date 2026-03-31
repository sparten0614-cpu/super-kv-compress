#!/usr/bin/env python3
"""Generate paper figures for Super KV Compress."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# =============================================================
# Figure 1: PPL-NIAH Metric Gap Scatter
# =============================================================
def fig_metric_gap():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Quantization configs — FILLED markers, different shapes per model
    # (ppl_delta%, niah%, label, marker, color)
    quant_configs = [
        (-0.09, 100, 'q8_0', 'o', '#1565C0'),      # Llama = circle
        (3.06, 100, 'q4_0', 'o', '#FF8F00'),         # Llama = circle, q4=orange
        (0.08, 100, 'TQKV_6', 'o', '#2E7D32'),       # Llama = circle, tqkv=green
        (-0.03, 100, 'q8_0', '^', '#1565C0'),         # Mistral = triangle
        (1.25, 100, 'q4_0', '^', '#FF8F00'),          # Mistral = triangle
        (3.17, 100, 'q4_0', 'D', '#FF8F00'),          # 70B = diamond
        (-1.45, 100, 'K8V4', 's', '#1565C0'),         # Qwen = square
    ]

    # Eviction configs — HOLLOW markers
    evict_configs = [
        (-0.09, 60, 'SL 50%', 'o', '#E53935'),
        (1.04, 40, 'SL 70%', 'o', '#E53935'),
        (7.50, 20, 'SL 85%', 'o', '#E53935'),
        (0.0, 60, 'H2O 50%', 's', '#E53935'),
        (0.0, 40, 'EA-1L 50%', '^', '#E53935'),
        (0.0, 0, 'EA-1L 85%', '^', '#E53935'),
        (0.0, 0, 'EA-ML 85%', 'v', '#E53935'),
    ]

    # Plot quantization — filled
    for ppl, niah, label, marker, color in quant_configs:
        ax.scatter(ppl, niah, c=color, marker=marker, s=90, zorder=5,
                   edgecolors='white', linewidth=0.5)

    # Plot eviction — hollow
    for ppl, niah, label, marker, color in evict_configs:
        ax.scatter(ppl, niah, facecolors='none', edgecolors=color,
                   marker=marker, s=90, zorder=5, linewidth=1.5)

    # Safety threshold
    ax.axhline(y=80, color='#9E9E9E', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.text(6.5, 82, 'safety threshold', color='#9E9E9E', fontsize=7.5, style='italic')

    # Annotate key points
    ax.annotate('SL 50%: PPL improves,\nNIAH drops to 60%', xy=(-0.09, 60),
                xytext=(-4.5, 45), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1),
                color='#E53935')
    ax.annotate('q4_0: PPL +3%,\nNIAH 100%', xy=(3.06, 100),
                xytext=(4.5, 87), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=1),
                color='#FF8F00')
    ax.annotate('ExpAttn 85%:\n0% NIAH', xy=(0, 0),
                xytext=(2.5, 12), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1),
                color='#E53935')

    # Danger zone shading
    ax.axhspan(-5, 50, alpha=0.04, color='red')

    # Labels
    ax.set_xlabel('PPL Change (%)', fontsize=12)
    ax.set_ylabel('NIAH Accuracy (%)', fontsize=12)
    ax.set_title('The Metric Gap: PPL vs NIAH', fontsize=13, fontweight='bold')

    # Legend — model shapes + fill/hollow
    llama_m = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=7, label='Llama')
    mistral_m = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=7, label='Mistral')
    qwen_m = mlines.Line2D([], [], color='gray', marker='s', linestyle='None', markersize=7, label='Qwen')
    b70_m = mlines.Line2D([], [], color='gray', marker='D', linestyle='None', markersize=7, label='70B')
    quant_p = mpatches.Patch(facecolor='#2196F3', edgecolor='white', label='Quantization (filled)')
    evict_p = mpatches.Patch(facecolor='none', edgecolor='#E53935', label='Eviction (hollow)')
    ax.legend(handles=[quant_p, evict_p, llama_m, mistral_m, qwen_m, b70_m],
              loc='lower right', fontsize=7.5, ncol=2)

    ax.set_xlim(-5.5, 9)
    ax.set_ylim(-8, 112)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig('metric_gap_scatter.pdf', bbox_inches='tight')
    plt.savefig('metric_gap_scatter.png', bbox_inches='tight')
    print('Saved metric_gap_scatter.pdf/.png')
    plt.close()


# =============================================================
# Figure 2: Eviction Gradient Curves
# =============================================================
def fig_eviction_gradient():
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Data: eviction% -> PPL delta%
    llama_4k_x = [0, 10, 30, 50, 55, 60, 65, 70, 75, 80]
    llama_4k_y = [0, -0.07, -0.20, 0.46, 1.09, 1.40, 2.02, 3.39, 5.30, 9.22]

    llama_16k_x = [0, 30, 50, 60, 70, 80, 85]
    llama_16k_y = [0, -0.17, -0.09, -0.003, 1.04, 3.36, 7.50]

    mistral_16k_x = [0, 50, 60, 70, 85]
    mistral_16k_y = [0, 0.36, 0.30, 0.26, 4.66]

    llama70b_16k_x = [0, 50, 70]
    llama70b_16k_y = [0, -0.32, -0.27]

    ax.plot(llama_4k_x, llama_4k_y, 'o-', color='#1565C0', label='Llama-8B 4K', markersize=4, linewidth=1.5)
    ax.plot(llama_16k_x, llama_16k_y, 's-', color='#42A5F5', label='Llama-8B 16K', markersize=4, linewidth=1.5)
    ax.plot(mistral_16k_x, mistral_16k_y, 'D-', color='#FF7043', label='Mistral-7B 16K', markersize=4, linewidth=1.5)
    ax.plot(llama70b_16k_x, llama70b_16k_y, '^-', color='#66BB6A', label='Llama-70B 16K', markersize=4, linewidth=1.5)

    # 1% cliff line
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(2, 1.3, '1% PPL cliff', color='red', fontsize=8, alpha=0.7)

    # Vertical cliff position lines
    ax.axvline(x=53, color='#1565C0', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.axvline(x=67, color='#42A5F5', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.axvline(x=85, color='#FF7043', linestyle=':', alpha=0.4, linewidth=0.8)

    # Cliff annotations
    ax.annotate('53%', xy=(53, 1.0), xytext=(43, 4.0), fontsize=8, color='#1565C0', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=0.8))
    ax.annotate('67%', xy=(67, 1.0), xytext=(60, 5.5), fontsize=8, color='#42A5F5', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#42A5F5', lw=0.8))
    ax.annotate('~85%', xy=(85, 4.66), xytext=(78, 7.5), fontsize=8, color='#FF7043', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF7043', lw=0.8))

    # 70B annotation
    ax.annotate('70B: no cliff\n(PPL improves)', xy=(70, -0.27), xytext=(72, 2.0), fontsize=7.5,
                color='#66BB6A', style='italic',
                arrowprops=dict(arrowstyle='->', color='#66BB6A', lw=0.8))

    ax.set_xlabel('Eviction Rate (%)', fontsize=12)
    ax.set_ylabel('PPL Change (%)', fontsize=12)
    ax.set_title('StreamingLLM Eviction Gradient', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_xlim(-2, 88)
    ax.set_ylim(-1.5, 10)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig('eviction_gradient.pdf', bbox_inches='tight')
    plt.savefig('eviction_gradient.png', bbox_inches='tight')
    print('Saved eviction_gradient.pdf/.png')
    plt.close()


# =============================================================
# Figure 3: Pareto Frontier
# =============================================================
def fig_pareto():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Pareto points: (compression, ppl_delta%, niah%, label)
    # Color by NIAH: green=100%, orange=60%
    configs = [
        (1.0, 0.0, 100, 'FP16'),
        (1.88, -0.08, 100, 'q8/q8'),
        (2.46, 0.07, 100, 'q8/q4'),
        (2.67, 0.08, 100, 'TQKV_6'),
        (3.56, 3.13, 100, 'q4/q4'),
        (4.92, 0.97, 60, 'q8/q4\n+50%'),
        (8.21, 3.10, 60, 'q8/q4\n+70%'),
        (11.85, 6.30, 60, 'q4/q4\n+70%'),
    ]

    niah_colors = {100: '#2E7D32', 60: '#FF8F00', 0: '#D32F2F'}

    for comp, ppl, niah, label in configs:
        color = niah_colors[niah]
        size = 180 if niah == 100 else 130
        marker = 'o' if niah == 100 else 'X'
        ax.scatter(comp, ppl, c=color, s=size, marker=marker, zorder=5,
                   edgecolors='white', linewidth=0.8)
        # Label
        va = 'bottom'
        dx, dy = 0.15, 0.25
        if label == 'FP16':
            dx, dy = 0.15, -0.5
            va = 'top'
        elif label == 'q4/q4\n+70%':
            dx, dy = -3.0, 0.2
        elif label == 'TQKV_6':
            dx, dy = 0.15, -0.5
            va = 'top'
        ax.annotate(label, xy=(comp, ppl), xytext=(comp + dx, ppl + dy),
                    fontsize=7, color=color, va=va)

    # Pareto front line — NIAH-safe region (thick dashed)
    safe_x = [1.0, 1.88, 2.46, 2.67, 3.56]
    safe_y = [0.0, -0.08, 0.07, 0.08, 3.13]
    ax.plot(safe_x, safe_y, '--', color='#2E7D32', alpha=0.5, linewidth=2.0, label='NIAH-safe frontier')

    # Pareto front line — NIAH-limited
    limited_x = [3.56, 4.92, 8.21, 11.85]
    limited_y = [3.13, 0.97, 3.10, 6.30]
    ax.plot(limited_x, limited_y, '--', color='#FF8F00', alpha=0.5, linewidth=2.0, label='NIAH-limited frontier')

    # NIAH-safe boundary
    ax.axvline(x=3.8, color='#9E9E9E', linestyle=':', alpha=0.6, linewidth=1.0)
    ax.text(3.9, 6.5, 'NIAH-safe\nboundary', fontsize=7.5, color='#9E9E9E', style='italic')

    # Region shading
    ax.axvspan(0.5, 3.8, alpha=0.03, color='green')
    ax.axvspan(3.8, 13, alpha=0.03, color='orange')

    # Region labels
    ax.text(2.0, -1.1, 'NIAH-safe (100%)', fontsize=9, color='#2E7D32',
            fontweight='bold', ha='center')
    ax.text(8.5, -1.1, 'NIAH-limited (60%)', fontsize=9, color='#FF8F00',
            fontweight='bold', ha='center')

    # Legend
    safe_dot = mlines.Line2D([], [], color='#2E7D32', marker='o', linestyle='None',
                              markersize=8, label='NIAH 100%')
    limited_dot = mlines.Line2D([], [], color='#FF8F00', marker='X', linestyle='None',
                                 markersize=8, label='NIAH 60%')
    ax.legend(handles=[safe_dot, limited_dot], loc='upper left', fontsize=9)

    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('PPL Change (%)', fontsize=12)
    ax.set_title('Pareto Frontier: Compression vs Quality', fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 13)
    ax.set_ylim(-1.5, 7.5)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig('pareto_frontier.pdf', bbox_inches='tight')
    plt.savefig('pareto_frontier.png', bbox_inches='tight')
    print('Saved pareto_frontier.pdf/.png')
    plt.close()


if __name__ == '__main__':
    fig_metric_gap()
    fig_eviction_gradient()
    fig_pareto()
    print('\nAll figures generated.')
