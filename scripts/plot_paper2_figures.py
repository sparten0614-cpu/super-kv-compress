#!/usr/bin/env python3
"""Generate Paper 2 figures: eigenspectrum comparison + AM degradation curve."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.linewidth'] = 0.8

# ============================================================
# Fig 2: K vs V AM/GM ratio across layers (TinyLlama)
# ============================================================

with open('results/pca_eigenspectrum/kv_amgm_ratios.json') as f:
    data = json.load(f)

layers = [d['layer'] for d in data['per_layer']]
k_amgm = [d['k_amgm'] for d in data['per_layer']]
v_amgm = [d['v_amgm'] for d in data['per_layer']]
k_bits = [d['k_coding_gain_bits'] for d in data['per_layer']]
v_bits = [d['v_coding_gain_bits'] for d in data['per_layer']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left panel: AM/GM ratio (log scale)
ax1.semilogy(layers, k_amgm, 'o-', color='#d62728', markersize=5, linewidth=1.5, label='K cache')
ax1.semilogy(layers, v_amgm, 's-', color='#1f77b4', markersize=4, linewidth=1.5, label='V cache')
ax1.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('Layer')
ax1.set_ylabel('AM/GM ratio (log scale)')
ax1.set_title('(a) Coding Gain by Layer')
ax1.legend(loc='upper right')
ax1.annotate(f'L0: {k_amgm[0]:.1f}', xy=(0, k_amgm[0]), xytext=(3, k_amgm[0]*0.7),
            arrowprops=dict(arrowstyle='->', color='#d62728'), color='#d62728', fontsize=9)
ax1.set_ylim(0.8, 100)

# Right panel: Coding gain in bits
width = 0.35
x = np.array(layers)
ax2.bar(x - width/2, k_bits, width, color='#d62728', alpha=0.8, label='K cache')
ax2.bar(x + width/2, v_bits, width, color='#1f77b4', alpha=0.8, label='V cache')
ax2.set_xlabel('Layer')
ax2.set_ylabel('Effective bit gain ($\\Delta b$)')
ax2.set_title('(b) Bit Savings from PCA-Adaptive Allocation')
ax2.legend(loc='upper right')
ax2.annotate(f'L0: +{k_bits[0]:.1f} bits', xy=(0, k_bits[0]), xytext=(4, k_bits[0]*0.85),
            arrowprops=dict(arrowstyle='->', color='#d62728'), color='#d62728', fontsize=9)

# Add average lines (excluding L0)
k_avg = data['k_avg_amgm_excl_l0']
v_avg = data['v_avg_amgm_excl_l0']
k_bits_avg = 0.5 * np.log2(k_avg)
v_bits_avg = 0.5 * np.log2(v_avg)
ax2.axhline(y=k_bits_avg, color='#d62728', linestyle=':', linewidth=1, alpha=0.6)
ax2.axhline(y=v_bits_avg, color='#1f77b4', linestyle=':', linewidth=1, alpha=0.6)
ax2.text(20, k_bits_avg + 0.03, f'K avg (excl L0): +{k_bits_avg:.2f}', color='#d62728', fontsize=8)
ax2.text(20, v_bits_avg + 0.03, f'V avg: +{v_bits_avg:.2f}', color='#1f77b4', fontsize=8)

plt.tight_layout()
plt.savefig('paper/figures/fig2_eigenspectrum_kv.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig2_eigenspectrum_kv.png', bbox_inches='tight', dpi=150)
print("Fig 2 saved: paper/figures/fig2_eigenspectrum_kv.{pdf,png}")


# ============================================================
# Fig 3: AM Degradation Curve (cliff-plateau pattern)
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Data
ratios = [1, 2, 5, 10]
quality = [70.0, 72.2, 58.9, 57.8]
ppl = [1.14, 1.26, 1.66, 2.09]

# Left panel: QuALITY accuracy
ax1.plot(ratios, quality, 'o-', color='#2ca02c', markersize=8, linewidth=2)
ax1.axhline(y=70.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Baseline')
ax1.fill_between([1, 2], [60, 60], [75, 75], alpha=0.1, color='green', label='Denoising zone')
ax1.fill_between([2, 10], [55, 55], [75, 75], alpha=0.08, color='red')

# Annotations
ax1.annotate('+2.2%\n(denoising)', xy=(2, 72.2), xytext=(3.5, 73.5),
            arrowprops=dict(arrowstyle='->', color='#2ca02c'), fontsize=9, color='#2ca02c')
ax1.annotate('cliff\n(-13.3 pts)', xy=(3.5, 65), xytext=(4.5, 67),
            fontsize=9, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', connectionstyle='arc3,rad=0.3'))
ax1.annotate('plateau\n(-1.1 pts)', xy=(7.5, 58.3), xytext=(7, 62),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray')

ax1.set_xlabel('Compression Ratio')
ax1.set_ylabel('QuALITY Accuracy (%)')
ax1.set_title('(a) Task Accuracy vs. AM Compression')
ax1.set_xticks(ratios)
ax1.set_xticklabels(['1×\n(baseline)', '2×', '5×', '10×'])
ax1.set_ylim(54, 76)
ax1.legend(loc='lower left', fontsize=9)

# Right panel: PPL
ax2.plot(ratios, ppl, 's-', color='#9467bd', markersize=8, linewidth=2)
ax2.axhline(y=1.14, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Baseline')

ax2.set_xlabel('Compression Ratio')
ax2.set_ylabel('Perplexity')
ax2.set_title('(b) Perplexity vs. AM Compression')
ax2.set_xticks(ratios)
ax2.set_xticklabels(['1×\n(baseline)', '2×', '5×', '10×'])
ax2.legend(loc='upper left', fontsize=9)

# Add PPL values
for r, p in zip(ratios, ppl):
    ax2.annotate(f'{p:.2f}', xy=(r, p), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('paper/figures/fig3_am_degradation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/figures/fig3_am_degradation.png', bbox_inches='tight', dpi=150)
print("Fig 3 saved: paper/figures/fig3_am_degradation.{pdf,png}")

print("\nDone! Both figures generated.")
