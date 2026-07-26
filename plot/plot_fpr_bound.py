"""
Plot the Theorem 3.3 FPR bound vs empirical FPR from mia_stealth.py results
for the CIFAR-10 ViT-B/32 model.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Empirical FPR from 5 loops (test set = 10000 non-member samples) ──
batch_sizes = np.array([1, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

# FP counts per loop (from result_log/cifar10_vit_stealth.txt)
fp_loops = {
    1:    [0, 0, 0, 0, 0],
    4:    [15, 30, 20, 16, 16],
    8:    [47, 38, 42, 39, 37],
    16:   [88, 90, 85, 88, 75],
    32:   [159, 151, 159, 163, 175],
    64:   [298, 275, 296, 259, 287],
    128:  [454, 451, 439, 415, 511],
    256:  [701, 678, 696, 642, 697],
    512:  [890, 848, 830, 905, 911],
    1024: [1201, 1128, 1127, 1103, 1102],
}

n_total = 10000  # total non-member test samples

empirical_fpr_mean = np.array([np.mean(fp_loops[b]) / n_total for b in batch_sizes])
empirical_fpr_std  = np.array([np.std(fp_loops[b]) / n_total for b in batch_sizes])

# ── Theorem 3.3 FPR bound (from mia_bound2.py output) ──
# Per-sample bound: mean = 5.10e-52, max = 6.98e-50
# Per-key bounds (p_j = max_i min_l Phi(z_{ijl})):
#   Key 0: max = 3.06e-48
#   Key 1: max = 4.06e-48
#   Key 2: max = 8.52e-49
#   Key 3: max = 2.05e-48
# Batch FPR uses UNION criterion: each adv key selected by >= 1 sample.
#   FPR_batch <= min_j [1 - (1 - p_j)^B]
fpr_persample_max = 6.980868e-50

# Per-key worst-case bounds (max over targets)
per_key_phi_max = np.array([3.064324e-48, 4.057503e-48, 8.522904e-49, 2.051550e-48])

# Correct batch formula: min_j [1 - (1 - p_j)^B]
# For p_j << 1, this ≈ B * min_j(p_j)
thm33_fpr = np.array([
    np.min([1.0 - (1.0 - pj) ** B for pj in per_key_phi_max])
    for B in batch_sizes
])
# Clamp to a minimum for log-scale plotting
thm33_fpr_plot = np.clip(thm33_fpr, 1e-300, None)

# ── Plot 1: Empirical FPR vs Batch Size ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.errorbar(batch_sizes, empirical_fpr_mean * 100, yerr=empirical_fpr_std * 100,
             fmt='o-', color='tab:red', capsize=4, linewidth=2, markersize=7,
             label='Empirical FPR')
ax1.set_xlabel('Batch Size', fontsize=13)
ax1.set_ylabel('FPR (%)', fontsize=13)
ax1.set_title('Empirical FPR vs Batch Size\n(CIFAR-10, ViT-B/32)', fontsize=14)
ax1.set_xscale('log', base=2)
ax1.set_xticks(batch_sizes)
ax1.set_xticklabels(batch_sizes)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Annotate each point
for b, fpr_m in zip(batch_sizes, empirical_fpr_mean):
    if fpr_m > 0:
        ax1.annotate(f'{fpr_m*100:.2f}%', (b, fpr_m*100),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=9, ha='center')

# ── Plot 2: Theoretical Bound (log scale) vs Empirical ──
ax2 = axes[1]

# Empirical on primary y-axis
color_emp = 'tab:red'
ax2.plot(batch_sizes, empirical_fpr_mean * 100, 'o-', color=color_emp,
         linewidth=2, markersize=7, label='Empirical FPR (%)')
ax2.set_xlabel('Batch Size', fontsize=13)
ax2.set_ylabel('Empirical FPR (%)', fontsize=13, color=color_emp)
ax2.tick_params(axis='y', labelcolor=color_emp)
ax2.set_xscale('log', base=2)
ax2.set_xticks(batch_sizes)
ax2.set_xticklabels(batch_sizes)

# Theorem 3.3 bound on secondary y-axis (log scale)
ax2b = ax2.twinx()
color_thm = 'tab:blue'
# Only plot if non-zero
valid = thm33_fpr > 0
ax2b.semilogy(batch_sizes[valid], thm33_fpr[valid], 's--', color=color_thm,
              linewidth=2, markersize=7, label='Thm 3.3 FPR Bound')
ax2b.set_ylabel('Theorem 3.3 FPR Bound (log scale)', fontsize=13, color=color_thm)
ax2b.tick_params(axis='y', labelcolor=color_thm)

ax2.set_title('Empirical FPR vs Theorem 3.3 Bound\n(CIFAR-10, ViT-B/32)', fontsize=14)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fpr_bound_cifar10_vit.png', dpi=150, bbox_inches='tight')
print("Saved: fpr_bound_cifar10_vit.png")

# ── Plot 3: Per-sample FPR bound distribution ──
fig2, ax3 = plt.subplots(figsize=(8, 5))

# Per-sample bounds from mia_bound2.py (representative values from verbose output)
persample_bounds = [
    1.447012e-67, 8.782357e-64, 2.736059e-57, 4.013912e-62,
    7.025644e-64, 2.719079e-61, 2.001043e-56, 1.428938e-68,
    2.294725e-62, 3.911878e-64,
]
# Stats
ax3.text(0.05, 0.95,
         f'Per-sample FPR Bound Statistics (200 targets):\n'
         f'  Mean:   5.10e-52\n'
         f'  Median: 3.93e-62\n'
         f'  Min:    2.20e-72\n'
         f'  Max:    6.98e-50\n\n'
         f'Per-key Bounds p_j (worst-case over targets):\n'
         f'  Key 0: 3.06e-48\n'
         f'  Key 1: 4.06e-48\n'
         f'  Key 2: 8.52e-49\n'
         f'  Key 3: 2.05e-48\n\n'
         f'Cluster σ_i Statistics (16 benign clusters):\n'
         f'  Mean σ:  0.0344\n'
         f'  Min σ:   0.0279\n'
         f'  Max σ:   0.0381\n\n'
         f'Batch FPR = min_j [1-(1-p_j)^B]  (union criterion)\n'
         f'→ FPR increases with B but still ≈ 0',
         transform=ax3.transAxes,
         fontsize=12, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Bar chart of empirical FPR for context
ax3.bar(range(len(batch_sizes)), empirical_fpr_mean * 100,
        tick_label=[str(b) for b in batch_sizes],
        color='tab:red', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Batch Size', fontsize=13)
ax3.set_ylabel('Empirical FPR (%)', fontsize=13)
ax3.set_title('Theorem 3.3 FPR Bound Summary\n(CIFAR-10, ViT-B/32, δ_min=0.02, Δ=0.05)',
              fontsize=14)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fpr_bound_summary_cifar10_vit.png', dpi=150, bbox_inches='tight')
print("Saved: fpr_bound_summary_cifar10_vit.png")

# ── Print summary table ──
print("\n" + "="*80)
print("CIFAR-10 ViT-B/32: FPR Bound vs Empirical FPR")
print("="*80)
print(f"{'Batch':>6s} | {'Empirical FPR':>14s} | {'Thm 3.3 Bound':>16s} | {'Advantage (emp)':>16s}")
print("-"*60)
for b, fpr_e in zip(batch_sizes, empirical_fpr_mean):
    fpr_b = np.min([1.0 - (1.0 - pj) ** b for pj in per_key_phi_max])
    adv_e = 1.0 - fpr_e  # TPR=1
    bound_str = f"{fpr_b:.2e}" if fpr_b > 1e-300 else "≈ 0"
    print(f"{b:>6d} | {fpr_e*100:>13.2f}% | {bound_str:>16s} | {adv_e*100:>15.2f}%")
