"""
WC Attractor Dynamics on HCP Connectome
Phases 1–4: Network → Attractors → Surrogates → Progressive Replacement
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from numpy.linalg import lstsq

rng = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════
# 1.  NETWORK
# ═══════════════════════════════════════════════════════
DATA   = '../enigma/enigmatoolbox/datasets/matrices/hcp_connectivity'
W      = np.loadtxt(f'{DATA}/strucMatrix_with_sctx.csv', delimiter=',')
labels = open(f'{DATA}/strucLabels_with_sctx.csv').read().strip().split(',')
N      = W.shape[0]   # 82
N_CTX  = 68           # cortical nodes 0–67, subcortical 68–81

# Scale by spectral radius — preserves hub structure, bounds eigenvalues to [-1,1]
W = W / np.linalg.eigvals(W).real.max()

# ── Parameters ──────────────────────────────────────────
dt   = 0.1
w_EE = 3.0;  w_EI = 2.0   # graph excitation, inhibitory suppression
w_IE = 3.0;  w_II = 1.0   # inhibitory drive, self-inhibition

# Heterogeneous external input: subcortical nodes sit closer to bifurcation
P = np.full(N, 0.5);   P[N_CTX:] = 1.2   # subcortical gets stronger drive
Q = np.zeros(N)

# Heterogeneous time constants: subcortical = slower inhibition
tau_E = np.ones(N);          tau_E[N_CTX:]  = 1.5
tau_I = np.full(N, 2.0);     tau_I[N_CTX:]  = 5.0

noise_std = 0.03
sigma     = lambda x: 1.0 / (1.0 + np.exp(-x))

def step(E, I, noise=True):
    WE  = W @ E
    xi  = noise_std * rng.standard_normal(N) if noise else 0.0
    E_n = (1 - dt/tau_E)*E + (dt/tau_E)*(sigma(w_EE*WE - w_EI*I + P) + xi)
    I_n = (1 - dt/tau_I)*I + (dt/tau_I)* sigma(w_IE*E  - w_II*I + Q)
    return np.clip(E_n, 0, 1), np.clip(I_n, 0, 1)

# ═══════════════════════════════════════════════════════
# 2.  ATTRACTOR IDENTIFICATION
#     Run N_IC random ICs → settle → record mean state → KMeans
# ═══════════════════════════════════════════════════════
N_IC      = 50
T_SETTLE  = 800   # steps with noise to explore
T_RECORD  = 200   # steps without noise to characterise attractor

print("Phase 2 — attractor identification …")
states = np.zeros((N_IC, 2*N))

for k in range(N_IC):
    E = rng.uniform(0, 1, N)
    I = rng.uniform(0, 1, N)
    for _ in range(T_SETTLE):
        E, I = step(E, I, noise=True)
    E_acc = np.zeros(N);  I_acc = np.zeros(N)
    for _ in range(T_RECORD):
        E, I = step(E, I, noise=False)
        E_acc += E;  I_acc += I
    states[k] = np.r_[E_acc/T_RECORD, I_acc/T_RECORD]

K  = 3
km = KMeans(n_clusters=K, n_init=30, random_state=0).fit(states)
A  = km.cluster_centers_          # (K, 2N) — attractor centroids
print(f"  Cluster sizes: {np.bincount(km.labels_)}")

# PCA projection for visualisation
pca     = PCA(2)
proj    = pca.fit_transform(states)
A_proj  = pca.transform(A)

# ═══════════════════════════════════════════════════════
# 3.  LINEAR SURROGATE  (one per node)
#     Input: neighbour E + I at t,  self E + I at t
#     Output: self E, I at t+1
# ═══════════════════════════════════════════════════════
print("Phase 3 — training surrogates …")

# Generate training trajectory from attractor 0
ic      = states[km.labels_ == 0][0]
E, I    = ic[:N].copy(), ic[N:].copy()
E_t, I_t = [], []
for _ in range(3000):
    E_t.append(E.copy());  I_t.append(I.copy())
    E, I = step(E, I, noise=True)
E_t = np.array(E_t);  I_t = np.array(I_t)

surrogates = {}          # node_i → (coef_E, coef_I)
for i in range(N):
    nbrs = np.where(W[:, i] > 0)[0]
    feat = np.c_[E_t[:-1][:, nbrs], I_t[:-1][:, nbrs],
                 E_t[:-1, i],        I_t[:-1, i],
                 np.ones(len(E_t)-1)]
    c_E, *_ = lstsq(feat, E_t[1:, i], rcond=None)
    c_I, *_ = lstsq(feat, I_t[1:, i], rcond=None)
    surrogates[i] = (c_E, c_I, nbrs)

def mixed_step(E, I, replaced_set):
    E_n = E.copy();  I_n = I.copy()
    WE  = W @ E
    xi  = noise_std * rng.standard_normal(N)

    for i in range(N):
        if i in replaced_set:
            c_E, c_I, nbrs = surrogates[i]
            x = np.r_[E[nbrs], I[nbrs], E[i], I[i], 1.0]
            E_n[i] = np.clip(c_E @ x, 0, 1)
            I_n[i] = np.clip(c_I @ x, 0, 1)
        else:
            E_n[i] = np.clip((1-dt/tau_E[i])*E[i] +
                             (dt/tau_E[i])*(sigma(w_EE*WE[i] - w_EI*I[i] + P[i]) + xi[i]), 0, 1)
            I_n[i] = np.clip((1-dt/tau_I[i])*I[i] +
                             (dt/tau_I[i])* sigma(w_IE*E[i]  - w_II*I[i] + Q[i]), 0, 1)
    return E_n, I_n

# ═══════════════════════════════════════════════════════
# 4.  PROGRESSIVE REPLACEMENT
#     Three policies × alpha sweep → attractor drift
# ═══════════════════════════════════════════════════════
degree      = (W > 0).sum(axis=1)
alphas      = np.linspace(0, 0.8, 9)
N_TRIALS    = 8
T_TEST      = 400
A_ref       = A[0]               # reference attractor

policies = {
    'random':     rng.permutation(N),
    'hub-first':  np.argsort(-degree),
    'hub-last':   np.argsort(degree),
}

print("Phase 4 — replacement sweep …")
results = {}
for pol, order in policies.items():
    drifts = []
    for alpha in alphas:
        replaced = set(order[:int(alpha * N)])
        trial_drifts = []
        for _ in range(N_TRIALS):
            E = A_ref[:N]  + 0.05 * rng.standard_normal(N)
            I = A_ref[N:]  + 0.05 * rng.standard_normal(N)
            E, I = np.clip(E, 0, 1), np.clip(I, 0, 1)
            for _ in range(T_TEST):
                E, I = mixed_step(E, I, replaced)
            trial_drifts.append(np.linalg.norm(np.r_[E, I] - A_ref))
        drifts.append(np.mean(trial_drifts))
    results[pol] = drifts
    print(f"  {pol} done")

# ═══════════════════════════════════════════════════════
# 5.  PLOTS
# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)
cmap = plt.cm.Set1

# 5a  Attractor landscape (PCA)
ax1 = fig.add_subplot(gs[0, 0])
for k in range(K):
    m = km.labels_ == k
    ax1.scatter(proj[m,0], proj[m,1], c=[cmap(k)], s=60, alpha=0.8, label=f'A{k+1}')
    ax1.scatter(*A_proj[k], marker='*', s=350, c=[cmap(k)], edgecolors='k', zorder=5)
ax1.set_title('Attractor Landscape (PCA)', fontsize=11, fontweight='bold')
ax1.set_xlabel('PC1');  ax1.set_ylabel('PC2')
ax1.legend(fontsize=8);  ax1.spines[['top','right']].set_visible(False)

# 5b  Attractor mean-E profile
ax2 = fig.add_subplot(gs[0, 1])
for k in range(K):
    ax2.plot(A[k, :N], lw=1.8, color=cmap(k), label=f'A{k+1}', alpha=0.85)
ax2.axvline(N_CTX - 0.5, color='gray', ls='--', lw=1)
ax2.text(N_CTX+0.5, ax2.get_ylim()[0]+0.01, 'sctx→', fontsize=8, color='gray')
ax2.set_xlabel('Node index');  ax2.set_ylabel('Mean E')
ax2.set_title('Attractor Profiles (mean E per node)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8);  ax2.spines[['top','right']].set_visible(False)

# 5c  Node degree (hub map)
ax3 = fig.add_subplot(gs[0, 2])
colors_bar = ['#3498db']*N_CTX + ['#e74c3c']*14
ax3.bar(np.arange(N), degree, color=colors_bar, alpha=0.75)
ax3.set_xlabel('Node index');  ax3.set_ylabel('Degree')
ax3.set_title('Node Degree  (blue=ctx, red=sctx)', fontsize=11, fontweight='bold')
ax3.spines[['top','right']].set_visible(False)

# 5d  Attractor drift vs α  (main result) 
# we want to see if as we remove more nodes what the effects are on the attractors 
# gradual 
ax4 = fig.add_subplot(gs[1, :])
pol_colors = {'random': '#e74c3c', 'hub-first': '#8e44ad', 'hub-last': '#27ae60'}
for pol, drifts in results.items():
    ax4.plot(alphas, drifts, marker='o', lw=2.5, label=pol, color=pol_colors[pol])
ax4.set_xlabel('Replacement fraction  α', fontsize=12)
ax4.set_ylabel('Attractor drift  ‖state − A_ref‖', fontsize=12)
ax4.set_title('Attractor Drift vs Replacement Fraction — 3 Policies', fontsize=12, fontweight='bold')
ax4.legend(fontsize=11);  ax4.grid(alpha=0.3)
ax4.spines[['top','right']].set_visible(False)

fig.savefig('wc_attractor.png', dpi=150, bbox_inches='tight')
print("Saved wc_attractor.png")
