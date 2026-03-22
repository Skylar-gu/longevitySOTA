import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load connectivity ──────────────────────────────────────────
DATA = '../enigma/enigmatoolbox/datasets/matrices/hcp_connectivity'
W      = np.loadtxt(f'{DATA}/strucMatrix_with_sctx.csv', delimiter=',')
labels = open(f'{DATA}/strucLabels_with_sctx.csv').read().strip().split(',')
N = W.shape[0]   # 82 nodes

# Row-normalise: each node receives weighted average of neighbours
W = W / W.sum(axis=1, keepdims=True).clip(min=1)

# ── Wilson–Cowan parameters ────────────────────────────────────
dt    = 0.1
T     = 300
steps = int(T / dt)

w_EE = 10.0;  w_EI = 8.0    # intrinsic E↔E, E←I
w_IE = 12.0;  w_II = 3.0    # intrinsic I←E, I←I
g    = 1.0                    # global graph coupling strength
tau_E = 1.0;  tau_I = 3.0   # different time constants → oscillations
P = 2.5                       # external excitatory drive
theta_E = 4.0; theta_I = 3.7 # sigmoid thresholds

def sigma(x): return 1.0 / (1.0 + np.exp(-x))

# ── Initial conditions (small random perturbation) ─────────────
rng = np.random.default_rng(0)
E = rng.uniform(0.1, 0.3, N)
I = rng.uniform(0.1, 0.3, N)

# ── Simulate ──────────────────────────────────────────────────
hist_E = np.zeros((steps, N))
hist_I = np.zeros((steps, N))

for t in range(steps):
    WE = W @ E
    dE = sigma(w_EE * E + g * WE - w_EI * I + P - theta_E)
    dI = sigma(w_IE * E  - w_II * I     - theta_I)
    E = (1 - dt/tau_E) * E + (dt/tau_E) * dE
    I = (1 - dt/tau_I) * I + (dt/tau_I) * dI
    hist_E[t] = E
    hist_I[t] = I

time = np.arange(steps) * dt

# ── Key ROIs ───────────────────────────────────────────────────
ROI     = ['Lthal', 'Rthal', 'Lhippo', 'Rhippo', 'Lcaud', 'Rcaud']
roi_idx = [labels.index(r) for r in ROI]
colors  = ['#e74c3c','#c0392b','#2980b9','#1a5276','#27ae60','#1e8449']

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle('Wilson–Cowan Graph Network  (HCP structural connectivity, 82 nodes)',
             fontsize=13, fontweight='bold')

# ROI traces
ax = axes[0]
for idx, name, c in zip(roi_idx, ROI, colors):
    ax.plot(time, hist_E[:, idx], label=name, color=c, lw=1.6)
ax.set_ylabel('E(t)')
ax.set_title('Excitatory activity — subcortical ROIs')
ax.legend(ncol=3, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# Network mean ± std
ax2 = axes[1]
mean_E = hist_E.mean(axis=1)
std_E  = hist_E.std(axis=1)
ax2.plot(time, mean_E, color='#2c3e50', lw=2, label='mean E')
ax2.fill_between(time, mean_E - std_E, mean_E + std_E, alpha=0.2, color='#2c3e50')
ax2.plot(time, hist_I.mean(axis=1), color='#e74c3c', lw=2, ls='--', label='mean I')
ax2.set_ylabel('Activity')
ax2.set_title('Network-wide mean E and I  (shading = ±1 std across nodes)')
ax2.legend(fontsize=10)
ax2.spines[['top', 'right']].set_visible(False)

# Full heatmap
ax3 = axes[2]
im = ax3.imshow(hist_E.T, aspect='auto', cmap='viridis',
                extent=[0, T, 0, N], vmin=0, vmax=1)
ax3.set_ylabel('Node index')
ax3.set_xlabel('Time')
ax3.set_title('Excitatory activity — all 82 nodes')
ax3.axhline(68, color='white', lw=0.8, ls='--', alpha=0.7)
ax3.text(2, 70, 'subcortical ↑', color='white', fontsize=8)
plt.colorbar(im, ax=ax3, label='E(t)')

plt.tight_layout()
fig.savefig('wc_graph.png', dpi=150, bbox_inches='tight')
print("Saved wc_graph.png")
print(f"Final mean E: {E.mean():.3f}  std: {E.std():.3f}")
