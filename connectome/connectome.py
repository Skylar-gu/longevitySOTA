import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = '../enigma/enigmatoolbox/datasets/matrices/hcp_connectivity'

# Load structural connectivity with subcortical (82 x 82)
W      = np.loadtxt(f'{DATA}/strucMatrix_with_sctx.csv', delimiter=',')
labels = open(f'{DATA}/strucLabels_with_sctx.csv').read().strip().split(',')
n_ctx  = 68
n_sctx = 14

print(f"Matrix shape: {W.shape}")
print(f"Subcortical regions: {labels[n_ctx:]}")
print(f"Non-zero connections: {np.count_nonzero(W)} / {W.size}")

# Regions of interest for WC attractor work
ROI = ['Lthal', 'Rthal', 'Lhippo', 'Rhippo', 'Lcaud', 'Rcaud']
roi_idx = [labels.index(r) for r in ROI]

# ── Figure: full connectivity matrix with ROI highlights ──────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('HCP Structural Connectivity (82 regions: 68 cortical + 14 subcortical)',
             fontsize=13, fontweight='bold')

# Full matrix
ax = axes[0]
im = ax.imshow(np.log1p(W), cmap='Blues', aspect='auto')
ax.set_title('Full matrix (log scale)', fontsize=11)
ax.set_xlabel('Region index')
ax.set_ylabel('Region index')
ax.axhline(n_ctx - 0.5, color='red', lw=1, ls='--', alpha=0.6)
ax.axvline(n_ctx - 0.5, color='red', lw=1, ls='--', alpha=0.6)
ax.text(n_ctx + 1, 2, 'subcortical', color='red', fontsize=8)
plt.colorbar(im, ax=ax, label='log(streamlines + 1)')

# ROI submatrix: thalamus + hippocampus + caudate
ax2 = axes[1]
sub = W[np.ix_(roi_idx, roi_idx)]
im2 = ax2.imshow(sub, cmap='Blues', aspect='auto')
ax2.set_xticks(range(len(ROI)))
ax2.set_yticks(range(len(ROI)))
ax2.set_xticklabels(ROI, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(ROI, fontsize=9)
ax2.set_title('ROI submatrix\n(thalamus · hippocampus · caudate)', fontsize=11)
for i in range(len(ROI)):
    for j in range(len(ROI)):
        ax2.text(j, i, f'{sub[i,j]:.1f}', ha='center', va='center', fontsize=8,
                 color='white' if sub[i,j] > sub.max() * 0.5 else 'black')
plt.colorbar(im2, ax=ax2, label='streamlines')

plt.tight_layout()
fig.savefig('connectome.png', dpi=150, bbox_inches='tight')
print("Saved connectome.png")
