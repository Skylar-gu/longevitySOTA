"""
Track 1 — Grounding the WC model with empirical data

Three analyses:
  A.  SC → FC prediction via communicability (no simulation needed)
      Note: WC simulations produce neural oscillations (ms timescale).
      Empirical FC is from BOLD fMRI (s timescale, hemodynamic).
      Matching them requires HRF convolution (future work / TVB approach).
      Instead we use structural communicability as a topology-based FC proxy.

  B.  MTLE-informed network: degrade SC using ENIGMA hippocampal atrophy d-values

  C.  WC dynamics: healthy vs MTLE (where WC is the right tool)
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import expm
from scipy.stats import spearmanr

rng = np.random.default_rng(0)

# ═══════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════
DATA  = '../enigma/enigmatoolbox/datasets/matrices/hcp_connectivity'
STATS = '../enigma/enigmatoolbox/datasets/summary_statistics'

W_raw  = np.loadtxt(f'{DATA}/strucMatrix_with_sctx.csv', delimiter=',')
FC_emp = np.loadtxt(f'{DATA}/funcMatrix_with_sctx.csv',  delimiter=',')
labels = open(f'{DATA}/strucLabels_with_sctx.csv').read().strip().split(',')
N      = W_raw.shape[0]
N_CTX  = 68

np.fill_diagonal(W_raw, 0)
np.fill_diagonal(FC_emp, 0)

mtle   = pd.read_csv(f'{STATS}/tlemtsl_case-controls_SubVol.csv')
mtle_d = dict(zip(mtle['Structure'], mtle['d_icv']))

mask = np.triu(np.ones((N, N), dtype=bool), k=1)

# ═══════════════════════════════════════════════════════
# A.  SC → FC VIA COMMUNICABILITY
#     Communicability C = exp(D^{-1/2} W D^{-1/2})
#     where D = degree matrix.  Captures all indirect paths.
# ═══════════════════════════════════════════════════════
def communicability(W):
    """Normalised matrix-exponential communicability."""
    d    = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D    = np.diag(d_inv_sqrt)
    W_n  = D @ W @ D                    # normalised adjacency
    C    = expm(W_n)
    np.fill_diagonal(C, 0)
    return C

C_healthy = communicability(W_raw)

r_sc_fc   = np.corrcoef(W_raw[mask],   FC_emp[mask])[0, 1]
r_comm_fc = np.corrcoef(C_healthy[mask], FC_emp[mask])[0, 1]
rho_sc    = spearmanr(W_raw[mask],   FC_emp[mask]).statistic
rho_comm  = spearmanr(C_healthy[mask], FC_emp[mask]).statistic

print(f"SC  → FC  Pearson r={r_sc_fc:.3f}   Spearman ρ={rho_sc:.3f}")
print(f"Comm→ FC  Pearson r={r_comm_fc:.3f}   Spearman ρ={rho_comm:.3f}")

# ═══════════════════════════════════════════════════════
# B.  MTLE NETWORK
#     Scale W connections by (1 + d * 0.25) for atrophied nodes
# ═══════════════════════════════════════════════════════
ATROPHY_SCALE = 0.25
W_mtle = W_raw.copy()
print("\nMTLE connectivity scaling:")
for region, d in sorted(mtle_d.items(), key=lambda x: x[1]):
    if region in labels and d < -0.1:
        idx   = labels.index(region)
        scale = np.clip(1.0 + d * ATROPHY_SCALE, 0.1, 1.0)
        W_mtle[idx, :] *= scale
        W_mtle[:, idx] *= scale
        print(f"  {region:10s}  d={d:+.3f}  → scale={scale:.2f}")

C_mtle = communicability(W_mtle)

r_mtle_fc  = np.corrcoef(C_mtle[mask], FC_emp[mask])[0, 1]
rho_mtle   = spearmanr(C_mtle[mask], FC_emp[mask]).statistic
print(f"\nMTLE comm→FC  Pearson r={r_mtle_fc:.3f}   Spearman ρ={rho_mtle:.3f}")

# ═══════════════════════════════════════════════════════
# C.  WC DYNAMICS: healthy vs MTLE
# ═══════════════════════════════════════════════════════
W_h = W_raw / np.linalg.eigvals(W_raw).real.max()
W_m = W_mtle / np.linalg.eigvals(W_mtle).real.max()

dt    = 0.1;  g = 1.0
# w_EE reduced to 1.5 so the mid-range fixed point is stable:
# stability requires S'(θ)·w_EE < 1; with a=2, S'=0.5 → w_EE_crit=2.0
# at w_EE=1.5: ∂f/∂E = -1 + 0.5·1.5 = -0.25 < 0  ✓
w_EE  = 1.5;  w_EI = 2.0;  w_IE = 3.0;  w_II = 1.0
P     = np.full(N, 0.5);  P[N_CTX:] = 1.2
tau_E = np.ones(N);       tau_E[N_CTX:] = 1.5
tau_I = np.full(N, 2.0);  tau_I[N_CTX:] = 5.0
# θ_E chosen so pre-activation = θ when E=I=0.5, W@E≈0.5:
# pre_act = 1.5·0.5 + 0.5 − 2·0.5 + 0.5 = 0.75  →  θ_E = 0.75
# θ_I: pre_act_I = 3·0.5 − 1·0.5 = 1.0           →  θ_I = 1.0
sig_e = lambda x: 1.0 / (1.0 + np.exp(-2.0 * (x - 0.75)))
sig_i = lambda x: 1.0 / (1.0 + np.exp(-2.0 * (x - 1.0)))

def simulate_wc(W_net, T_settle=500, T_sim=1000, noise_std=0.05):
    E = rng.uniform(0.3, 0.7, N);  I = rng.uniform(0.3, 0.7, N)
    for _ in range(T_settle):
        WE = W_net @ E
        xi = noise_std * rng.standard_normal(N)
        E  = np.clip((1-dt/tau_E)*E + (dt/tau_E)*(sig_e(w_EE*E + g*WE - w_EI*I + P) + xi), 0, 1)
        I  = np.clip((1-dt/tau_I)*I + (dt/tau_I)* sig_i(w_IE*E - w_II*I), 0, 1)
    hist_E = np.zeros((T_sim, N))
    for t in range(T_sim):
        WE = W_net @ E
        xi = noise_std * rng.standard_normal(N)
        E  = np.clip((1-dt/tau_E)*E + (dt/tau_E)*(sig_e(w_EE*E + g*WE - w_EI*I + P) + xi), 0, 1)
        I  = np.clip((1-dt/tau_I)*I + (dt/tau_I)* sig_i(w_IE*E - w_II*I), 0, 1)
        hist_E[t] = E
    return hist_E

print("\nSimulating WC healthy …")
hist_h = simulate_wc(W_h)
print("Simulating WC MTLE …")
hist_m = simulate_wc(W_m)
time   = np.arange(1000) * dt

# ═══════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)

# ── A: SC vs Comm scatter against FC ─────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(W_raw[mask], FC_emp[mask], s=3, alpha=0.3, color='#3498db')
ax1.set_xlabel('SC (streamlines)');  ax1.set_ylabel('Empirical FC')
ax1.set_title(f'SC vs FC\nPearson r={r_sc_fc:.3f}', fontsize=10, fontweight='bold')
ax1.spines[['top','right']].set_visible(False)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(C_healthy[mask], FC_emp[mask], s=3, alpha=0.3, color='#27ae60')
ax2.set_xlabel('Communicability');  ax2.set_ylabel('Empirical FC')
ax2.set_title(f'Communicability vs FC\nPearson r={r_comm_fc:.3f}', fontsize=10, fontweight='bold')
ax2.spines[['top','right']].set_visible(False)

# ── B: MTLE effect sizes ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
sctx_regions = [r for r in labels[N_CTX:] if r in mtle_d]
d_vals       = [mtle_d[r] for r in sctx_regions]
bar_c        = ['#e74c3c' if d < -0.5 else '#e67e22' if d < -0.1 else '#95a5a6' for d in d_vals]
ax3.barh(sctx_regions, d_vals, color=bar_c, alpha=0.85)
ax3.axvline(0, color='black', lw=0.8)
ax3.set_xlabel("Cohen's d");  ax3.set_title('ENIGMA MTLE-L\nSubcortical atrophy', fontsize=10, fontweight='bold')
ax3.spines[['top','right']].set_visible(False)

# ── B: Communicability diff ──────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
diff_C = C_healthy - C_mtle
lim    = np.percentile(np.abs(diff_C), 98)
im4    = ax4.imshow(diff_C, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax4.axhline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.6)
ax4.axvline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.6)
ax4.set_title('Communicability change\nhealthy − MTLE', fontsize=10, fontweight='bold')
plt.colorbar(im4, ax=ax4, fraction=0.04)

# ── FC matrices ──────────────────────────────────────
vmin_fc, vmax_fc = 0, FC_emp.max()
for col, (mat, title) in enumerate([(FC_emp,     'Empirical FC (HCP)'),
                                     (C_healthy,  f'Communicability healthy\n(r={r_comm_fc:.3f} vs FC)'),
                                     (C_mtle,     f'Communicability MTLE-L\n(r={r_mtle_fc:.3f} vs FC)')]):
    ax = fig.add_subplot(gs[1, col])
    im = ax.imshow(mat, cmap='RdYlBu_r', aspect='auto',
                   vmin=vmin_fc if col == 0 else None, vmax=vmax_fc if col == 0 else None)
    ax.axhline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(title, fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.04)

# ── C: WC dynamics healthy vs MTLE ──────────────────
ROI      = ['Lthal', 'Lhippo', 'Rhippo', 'Lcaud']
roi_idx  = [labels.index(r) for r in ROI]
rc       = ['#e74c3c', '#2980b9', '#1a5276', '#27ae60']

ax5 = fig.add_subplot(gs[2, :2])
for idx, name, c in zip(roi_idx, ROI, rc):
    ax5.plot(time[-300:], hist_h[-300:, idx], color=c, lw=1.5, label=name)
ax5.set_title('WC dynamics — Healthy', fontsize=10, fontweight='bold')
ax5.set_xlabel('Time');  ax5.set_ylabel('E(t)')
ax5.legend(ncol=2, fontsize=9);  ax5.spines[['top','right']].set_visible(False)
ax5.set_ylim(0.6, 1.0)

ax6 = fig.add_subplot(gs[2, 2:])
for idx, name, c in zip(roi_idx, ROI, rc):
    ls = '--' if name in ('Lhippo', 'Lthal') else '-'
    ax6.plot(time[-300:], hist_m[-300:, idx], color=c, lw=1.5, ls=ls, label=name)
ax6.set_title('WC dynamics — MTLE-L  (dashed = atrophied nodes)', fontsize=10, fontweight='bold')
ax6.set_xlabel('Time');  ax6.set_ylabel('E(t)')
ax6.legend(ncol=2, fontsize=9);  ax6.spines[['top','right']].set_visible(False)
ax6.set_ylim(0.6, 1.0)

# Mean E change table
print("\nMean E change (healthy → MTLE):")
for r, idx in zip(ROI, roi_idx):
    delta = hist_m[:, idx].mean() - hist_h[:, idx].mean()
    print(f"  {r:10s}  healthy={hist_h[:,idx].mean():.3f}  mtle={hist_m[:,idx].mean():.3f}  Δ={delta:+.3f}")

fig.savefig('wc_fitting.png', dpi=150, bbox_inches='tight')
print("\nSaved wc_fitting.png")

# ═══════════════════════════════════════════════════════
# D.  WC + BALLOON-WINDKESSEL FC FITTING VIA g-SWEEP
#     Bridges the ms (WC) ↔ s (BOLD) timescale gap.
#     Pipeline: WC E(t) → BW haemodynamics → BOLD → FC
#     Sweep global coupling g; pick g_opt = argmax r(FC_sim, FC_emp).
# ═══════════════════════════════════════════════════════

# ── Balloon-Windkessel parameters (Friston 2003) ────────────────
BW_KAPPA = 0.65   # signal decay [s⁻¹]
BW_GAMMA = 0.41   # flow-dependent elimination [s⁻¹]
BW_TAU   = 0.98   # haemodynamic transit time [s]
BW_ALPHA = 0.32   # Grubb exponent
BW_RHO   = 0.34   # resting oxygen extraction fraction
BW_V0    = 0.02   # resting blood volume fraction

def balloon_step(s, f, v, q, x, dt):
    """One Euler step of the Balloon-Windkessel model (all arrays, shape N)."""
    f_safe = np.maximum(f, 1e-6)
    ds = x - BW_KAPPA * s - BW_GAMMA * (f - 1.0)
    df = s
    dv = (f - v ** (1.0 / BW_ALPHA)) / BW_TAU
    dq = ((f_safe * (1.0 - (1.0 - BW_RHO) ** (1.0 / f_safe)) / BW_RHO)
          - q * v ** (1.0 / BW_ALPHA - 1.0)) / BW_TAU
    return (s + dt * ds,
            np.clip(f + dt * df, 0.01, 20.0),
            np.clip(v + dt * dv, 0.01, 20.0),
            np.clip(q + dt * dq, 0.01, 20.0))

def bold_signal(v, q):
    """BOLD percent-signal-change from BW state (Buxton-Friston form)."""
    return BW_V0 * (7.0 * BW_RHO * (1.0 - q) + 2.0 * (1.0 - q / v))

def simulate_wc_bold(W_net, g,
                     dt=0.001, T_settle=2000, T_sim=60000,
                     TR_steps=2000, noise_std=0.02, seed=42):
    """
    Wilson-Cowan at 1 ms resolution + Balloon-Windkessel HRF.

    Parameters
    ----------
    W_net     : (N, N) spectral-radius-normalised SC
    g         : global coupling
    dt        : integration step in seconds (default 1 ms)
    T_settle  : transient steps (discarded)
    T_sim     : simulation steps recorded
    TR_steps  : steps between BOLD samples  (TR = TR_steps * dt seconds)
    noise_std : WC noise amplitude

    Returns
    -------
    bold : (n_TR, N)  where n_TR = T_sim // TR_steps
    """
    N     = W_net.shape[0]
    rng_  = np.random.default_rng(seed)

    # WC params in seconds (dt/tau_E = 0.001/0.010 = 0.1 — same ratio as before)
    tau_E = np.where(np.arange(N) < N_CTX, 0.010, 0.015)
    tau_I = np.where(np.arange(N) < N_CTX, 0.020, 0.050)
    P_wc  = np.where(np.arange(N) < N_CTX, 0.5,   1.2  )
    sig   = lambda x: 1.0 / (1.0 + np.exp(-x))
    w_EE, w_EI, w_IE, w_II = 3.0, 2.0, 3.0, 1.0

    E = rng_.uniform(0.1, 0.4, N)
    I = rng_.uniform(0.1, 0.4, N)
    s = np.zeros(N); f = np.ones(N); v = np.ones(N); q = np.ones(N)

    for _ in range(T_settle):
        inp = g * (W_net @ E)
        xi  = noise_std * rng_.standard_normal(N)
        E   = np.clip((1 - dt/tau_E)*E + (dt/tau_E)*(sig(w_EE*E + inp - w_EI*I + P_wc) + xi), 0, 1)
        I   = np.clip((1 - dt/tau_I)*I + (dt/tau_I)* sig(w_IE*E  - w_II*I), 0, 1)
        s, f, v, q = balloon_step(s, f, v, q, E, dt)

    n_TR = T_sim // TR_steps
    bold = np.zeros((n_TR, N))
    bi   = 0
    for t in range(T_sim):
        inp = g * (W_net @ E)
        xi  = noise_std * rng_.standard_normal(N)
        E   = np.clip((1 - dt/tau_E)*E + (dt/tau_E)*(sig(w_EE*E + inp - w_EI*I + P_wc) + xi), 0, 1)
        I   = np.clip((1 - dt/tau_I)*I + (dt/tau_I)* sig(w_IE*E  - w_II*I), 0, 1)
        s, f, v, q = balloon_step(s, f, v, q, E, dt)
        if (t + 1) % TR_steps == 0:
            bold[bi] = bold_signal(v, q)
            bi += 1
    return bold

def fc_from_bold(bold):
    """Pearson FC matrix from (n_TR, N) BOLD; diagonal zeroed."""
    bold_z = bold - bold.mean(axis=0)
    FC     = np.corrcoef(bold_z.T)
    np.fill_diagonal(FC, 0)
    return FC

# ── Run g-sweep on calibrated healthy network ────────────────────
from scipy.stats import pearsonr as _pearsonr

W_h_cal = W_raw  / np.linalg.eigvals(W_raw ).real.max()
W_m_cal = W_mtle / np.linalg.eigvals(W_mtle).real.max()

G_VALUES = [0.2, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 2.5]

print("\n─── Section D: g-sweep (WC + Balloon-Windkessel) ───")
fc_corrs_sweep = []
FC_sims_sweep  = []
for g_val in G_VALUES:
    print(f"  g={g_val:.2f} …", end='', flush=True)
    bold_g   = simulate_wc_bold(W_h_cal, g_val)
    FC_sim_g = fc_from_bold(bold_g)
    r_g, _   = _pearsonr(FC_sim_g[mask], FC_emp[mask])
    fc_corrs_sweep.append(r_g)
    FC_sims_sweep.append(FC_sim_g)
    print(f"  r={r_g:.3f}")

g_opt_idx   = int(np.argmax(fc_corrs_sweep))
g_opt       = G_VALUES[g_opt_idx]
FC_sim_opt  = FC_sims_sweep[g_opt_idx]
r_opt       = fc_corrs_sweep[g_opt_idx]
rho_opt, _  = spearmanr(FC_sim_opt[mask], FC_emp[mask])

print(f"\nOptimal g = {g_opt:.2f}  (Pearson r={r_opt:.3f}, Spearman ρ={rho_opt:.3f})")

# Simulate MTLE at g_opt
print(f"Simulating MTLE network at g_opt={g_opt:.2f} …")
bold_mtle_fit  = simulate_wc_bold(W_m_cal, g_opt)
FC_sim_mtle    = fc_from_bold(bold_mtle_fit)
r_mtle_sim, _  = _pearsonr(FC_sim_mtle[mask], FC_emp[mask])
rho_mtle_sim, _ = spearmanr(FC_sim_mtle[mask], FC_emp[mask])
print(f"MTLE sim-FC→emp-FC  Pearson r={r_mtle_sim:.3f}  Spearman ρ={rho_mtle_sim:.3f}")

# ── Section D plots ─────────────────────────────────────────────
fig_d, axes_d = plt.subplots(2, 3, figsize=(16, 10))
fig_d.suptitle(f'Section D — WC + Balloon-Windkessel FC Fitting  (g_opt={g_opt:.2f})',
               fontsize=13, fontweight='bold')

# D1: g-sweep curve
ax = axes_d[0, 0]
ax.plot(G_VALUES, fc_corrs_sweep, 'o-', color='#2980b9', lw=2, ms=7)
ax.axvline(g_opt, color='#e74c3c', ls='--', lw=1.5, label=f'g_opt={g_opt:.2f}  r={r_opt:.3f}')
ax.set_xlabel('Global coupling g');  ax.set_ylabel('Pearson r (sim-FC vs emp-FC)')
ax.set_title('g-Sweep', fontsize=10, fontweight='bold')
ax.legend(fontsize=9);  ax.spines[['top', 'right']].set_visible(False)

# D2: empirical FC
ax = axes_d[0, 1]
im = ax.imshow(FC_emp, cmap='RdYlBu_r', aspect='auto')
ax.axhline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax.axvline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax.set_title('Empirical FC (HCP BOLD)', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04)

# D3: simulated FC healthy at g_opt
ax = axes_d[0, 2]
im = ax.imshow(FC_sim_opt, cmap='RdYlBu_r', aspect='auto')
ax.axhline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax.axvline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax.set_title(f'Sim-FC healthy  g={g_opt:.2f}  r={r_opt:.3f}', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04)

# D4: simulated FC MTLE at g_opt
ax = axes_d[1, 0]
im = ax.imshow(FC_sim_mtle, cmap='RdYlBu_r', aspect='auto')
ax.axhline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax.axvline(N_CTX-0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax.set_title(f'Sim-FC MTLE  g={g_opt:.2f}  r={r_mtle_sim:.3f}', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.04)

# D5: scatter healthy sim-FC vs empirical FC
ax = axes_d[1, 1]
ax.scatter(FC_sim_opt[mask], FC_emp[mask], s=3, alpha=0.3, color='#27ae60')
ax.set_xlabel('Sim-FC (WC+HRF)');  ax.set_ylabel('Empirical FC')
ax.set_title(f'Healthy scatter  r={r_opt:.3f}', fontsize=10, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# D6: scatter MTLE sim-FC vs empirical FC
ax = axes_d[1, 2]
ax.scatter(FC_sim_mtle[mask], FC_emp[mask], s=3, alpha=0.3, color='#e74c3c')
ax.set_xlabel('Sim-FC (WC+HRF)');  ax.set_ylabel('Empirical FC')
ax.set_title(f'MTLE scatter  r={r_mtle_sim:.3f}', fontsize=10, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig_d.savefig('wc_fitting_gsweep.png', dpi=150, bbox_inches='tight')
print("Saved wc_fitting_gsweep.png")
