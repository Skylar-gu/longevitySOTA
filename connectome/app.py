"""
WC Network Node Ablation — Interactive Interface
Run with: streamlit run app.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.linalg import expm
from scipy.stats import pearsonr, spearmanr
from scipy.signal import welch
import streamlit as st

# ═══════════════════════════════════════════════════════
# Load + cache network
# ═══════════════════════════════════════════════════════
DATA   = '../enigma/enigmatoolbox/datasets/matrices/hcp_connectivity'
STATS  = '../enigma/enigmatoolbox/datasets/summary_statistics'

@st.cache_data
def load_network():
    W_raw  = np.loadtxt(f'{DATA}/strucMatrix_with_sctx.csv', delimiter=',')
    FC_emp = np.loadtxt(f'{DATA}/funcMatrix_with_sctx.csv',  delimiter=',')
    labels = open(f'{DATA}/strucLabels_with_sctx.csv').read().strip().split(',')
    np.fill_diagonal(W_raw, 0)
    np.fill_diagonal(FC_emp, 0)
    W_cal  = W_raw / np.linalg.eigvals(W_raw).real.max()   # calibrated (spectral-radius-normalised)
    return W_raw, W_cal, FC_emp, labels

@st.cache_data
def load_mtle():
    mtle   = pd.read_csv(f'{STATS}/tlemtsl_case-controls_SubVol.csv')
    return dict(zip(mtle['Structure'], mtle['d_icv']))

@st.cache_data
def build_mtle_network(W_raw_tuple, labels_tuple, mtle_d_items, atrophy_scale=0.25):
    W_raw  = np.array(W_raw_tuple)
    labels = list(labels_tuple)
    mtle_d = dict(mtle_d_items)
    W_mtle = W_raw.copy()
    for region, d in mtle_d.items():
        if region in labels and d < -0.1:
            idx   = labels.index(region)
            scale = np.clip(1.0 + d * atrophy_scale, 0.1, 1.0)
            W_mtle[idx, :] *= scale
            W_mtle[:, idx] *= scale
    W_mtle_cal = W_mtle / np.linalg.eigvals(W_mtle).real.max()
    return W_mtle, W_mtle_cal

def communicability(W):
    """Degree-normalised matrix-exponential communicability."""
    d          = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D          = np.diag(d_inv_sqrt)
    W_n        = D @ W @ D
    C          = expm(W_n)
    np.fill_diagonal(C, 0)
    return C

BONN_DIR = Path('../bonn_epilepsy')
BONN_FS  = 173.61   # Hz

@st.cache_data
def load_bonn_subset(folder, n=10):
    d = BONN_DIR / folder
    files = sorted(d.glob('*.txt')) + sorted(d.glob('*.TXT'))
    files = sorted(files)[:n]
    return [np.loadtxt(f) for f in files] if files else []

@st.cache_data
def compute_wc_psd(W_tuple, node_idx, T_settle=500, T_sim=3000, noise_std=0.02, dt=0.1, seed=42):
    """Run WC sim and return Welch PSD of a single node (artificial units, 1/dt Hz)."""
    W  = np.array(W_tuple)
    N  = W.shape[0]; N_CTX = 68
    P     = np.full(N, 0.5);   P[N_CTX:] = 1.2
    tau_E = np.ones(N);         tau_E[N_CTX:] = 1.5
    tau_I = np.full(N, 2.0);    tau_I[N_CTX:] = 5.0
    sigma = lambda x: 1.0 / (1.0 + np.exp(-x))
    rng_  = np.random.default_rng(seed)
    w_EE = 3.0; w_EI = 2.0; w_IE = 3.0; g = 1.0
    # w_EE=1.5 so mid-range fixed point is stable: S'(θ)·w_EE = 0.5·1.5 = 0.75 < 1
    # θ_E=0.75 centres E at 0.5; θ_I=1.0 centres I at 0.5
    w_EE = 1.5
    sig_e = lambda x: 1.0 / (1.0 + np.exp(-2.0 * (x - 0.75)))
    sig_i = lambda x: 1.0 / (1.0 + np.exp(-2.0 * (x - 1.0)))
    E = rng_.uniform(0.3, 0.7, N)
    I = rng_.uniform(0.3, 0.7, N)
    for _ in range(T_settle):
        WE = W @ E;  xi = 0.05 * rng_.standard_normal(N)
        E  = np.clip((1-dt/tau_E)*E + (dt/tau_E)*(sig_e(w_EE*E + g*WE - w_EI*I + P) + xi), 0, 1)
        I  = np.clip((1-dt/tau_I)*I + (dt/tau_I)* sig_i(w_IE*E - I), 0, 1)
    trace = np.zeros(T_sim)
    for t in range(T_sim):
        WE = W @ E;  xi = 0.05 * rng_.standard_normal(N)
        E  = np.clip((1-dt/tau_E)*E + (dt/tau_E)*(sig_e(w_EE*E + g*WE - w_EI*I + P) + xi), 0, 1)
        I  = np.clip((1-dt/tau_I)*I + (dt/tau_I)* sig_i(w_IE*E - I), 0, 1)
        trace[t] = E[node_idx]
    fs_wc = 1.0 / dt
    freqs, psd = welch(trace - trace.mean(), fs=fs_wc, nperseg=512)
    return freqs, psd

@st.cache_data
def compute_comm_metrics(W_cal_tuple, W_mtle_cal_tuple, FC_emp_tuple):
    W_cal      = np.array(W_cal_tuple)
    W_mtle_cal = np.array(W_mtle_cal_tuple)
    FC_emp     = np.array(FC_emp_tuple)
    C_healthy  = communicability(W_cal)
    C_mtle     = communicability(W_mtle_cal)
    N          = W_cal.shape[0]
    mask       = np.triu(np.ones((N, N), dtype=bool), k=1)
    r_h, _     = pearsonr(C_healthy[mask], FC_emp[mask])
    rho_h, _   = spearmanr(C_healthy[mask], FC_emp[mask])
    r_m, _     = pearsonr(C_mtle[mask],    FC_emp[mask])
    rho_m, _   = spearmanr(C_mtle[mask],   FC_emp[mask])
    return C_healthy, C_mtle, r_h, rho_h, r_m, rho_m

@st.cache_data
def get_reference_state(W_tuple):
    W   = np.array(W_tuple)
    N   = W.shape[0]
    P, Q, tau_E, tau_I = _params(N)
    E   = np.full(N, 0.5)
    I   = np.full(N, 0.5)
    for _ in range(1000):
        E, I = _step(W, E, I, P, Q, tau_E, tau_I, noise=False)
    return E, I

def _params(N, N_CTX=68):
    P     = np.full(N, 0.5);   P[N_CTX:]  = 1.2
    Q     = np.zeros(N)
    tau_E = np.ones(N);        tau_E[N_CTX:] = 1.5
    tau_I = np.full(N, 2.0);   tau_I[N_CTX:] = 5.0
    return P, Q, tau_E, tau_I

def _step(W, E, I, P, Q, tau_E, tau_I, noise=False, dt=0.1, noise_std=0.05, rng=None):
    sig_e = lambda x: 1.0 / (1.0 + np.exp(-2.0 * (x - 1.0)))  # gain=2, θ=1.0
    sig_i = lambda x: 1.0 / (1.0 + np.exp(-2.0 * (x - 1.0)))  # gain=2, θ=1.0
    WE    = W @ E
    xi    = (rng or np.random.default_rng()).standard_normal(len(E)) * noise_std if noise else 0.0
    E_n   = np.clip((1-dt/tau_E)*E + (dt/tau_E)*(sig_e(3.0*WE - 2.0*I + P) + xi), 0, 1)
    I_n   = np.clip((1-dt/tau_I)*I + (dt/tau_I)* sig_i(3.0*E  - 1.0*I + Q),        0, 1)
    return E_n, I_n

def run_sim(W, E0, I0, T=400):
    N             = W.shape[0]
    P, Q, tau_E, tau_I = _params(N)
    rng_sim       = np.random.default_rng(1)
    E, I          = E0.copy(), I0.copy()
    hist_E        = np.zeros((T, N))
    for t in range(T):
        E, I       = _step(W, E, I, P, Q, tau_E, tau_I, noise=True, rng=rng_sim)
        hist_E[t]  = E
    return hist_E

# ═══════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════
W_raw, W_cal, FC_emp, labels = load_network()
mtle_d  = load_mtle()
N       = W_raw.shape[0]
N_CTX   = 68

W_mtle_raw, W_mtle_cal = build_mtle_network(
    tuple(map(tuple, W_raw.tolist())),
    tuple(labels),
    tuple(sorted(mtle_d.items()))
)

# ═══════════════════════════════════════════════════════
# App layout
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="WC Brain Network Explorer", layout="wide")
st.title("Wilson–Cowan Brain Network Explorer")

tab_ablation, tab_mtle = st.tabs(["Node Ablation", "MTLE vs Normal"])

# ══════════════════════════════════════════════════════
# TAB 1 — Node Ablation (original)
# ══════════════════════════════════════════════════════
with tab_ablation:
    st.markdown("Select brain regions to remove. The network degrades in real-time.")

    baseline_choice = st.radio(
        "Baseline network",
        ["Healthy (HCP)", "MTLE-L degraded"],
        horizontal=True,
        help="MTLE-L applies ENIGMA atrophy scaling to subcortical nodes before ablation."
    )
    W_base = W_mtle_cal if baseline_choice == "MTLE-L degraded" else W_cal
    W_base_raw = W_mtle_raw if baseline_choice == "MTLE-L degraded" else W_raw

    degree = (W_base_raw > 0).sum(axis=1)

    with st.sidebar:
        st.header("Remove Nodes")
        ctx_opts  = [l for l in labels[:N_CTX]]
        sctx_opts = [l for l in labels[N_CTX:]]

        removed_ctx  = st.multiselect("Cortical",    ctx_opts,  default=[])
        removed_sctx = st.multiselect("Subcortical", sctx_opts, default=[])
        removed      = removed_ctx + removed_sctx
        removed_idx  = [labels.index(r) for r in removed]

        if st.button("Reset"):
            removed_idx = []

        st.divider()
        st.subheader("Node Importance (degree)")
        fig_deg, ax_d = plt.subplots(figsize=(3, 2.5))
        bar_colors    = ['#e74c3c' if i in removed_idx else
                         ('#c0392b' if i >= N_CTX else '#3498db')
                         for i in range(N)]
        ax_d.barh(np.arange(N), degree, color=bar_colors, height=0.8)
        ax_d.set_xlabel("Degree", fontsize=8)
        ax_d.set_yticks([])
        ax_d.invert_yaxis()
        ax_d.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig_deg, use_container_width=True)
        plt.close()

    n_rm   = len(removed_idx)
    kept   = [i for i in range(N) if i not in removed_idx]
    w_lost = (W_base_raw[removed_idx, :].sum() + W_base_raw[:, removed_idx].sum()) / W_base_raw.sum() * 100 if n_rm else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Nodes removed",          f"{n_rm} / {N}")
    m2.metric("Connection weight lost", f"{w_lost:.1f}%")
    m3.metric("Nodes remaining",        f"{N - n_rm}")

    st.divider()

    col1, col2 = st.columns(2)

    def plot_matrix(W_plot, title, removed_idx, ax):
        ax.imshow(np.log1p(W_plot), cmap='Blues', aspect='auto',
                  vmin=0, vmax=np.log1p(W_plot).max())
        for idx in removed_idx:
            ax.axhline(idx - 0.5, color='red', lw=0.6, alpha=0.7)
            ax.axvline(idx - 0.5, color='red', lw=0.6, alpha=0.7)
        ax.axhline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.5)
        ax.axvline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Node index");  ax.set_ylabel("Node index")

    W_deg = W_base.copy()
    W_deg[removed_idx, :] = 0
    W_deg[:, removed_idx] = 0

    intact_label = "Intact Network" if baseline_choice == "Healthy (HCP)" else "MTLE Baseline"
    fig_mat, (ax_orig, ax_deg) = plt.subplots(1, 2, figsize=(12, 5))
    plot_matrix(W_base, f"{intact_label} (calibrated)", [],          ax_orig)
    plot_matrix(W_deg,  "Ablated Network (calibrated)", removed_idx, ax_deg)
    if removed_idx:
        ax_deg.set_title(f"Ablated Network  ({n_rm} nodes removed)", fontsize=11, fontweight='bold')
    plt.tight_layout()

    with col1:
        st.pyplot(fig_mat, use_container_width=True)
    plt.close()

    with col2:
        st.subheader("Dynamical Degradation")

        E0, I0 = get_reference_state(tuple(map(tuple, W_base.tolist())))

        with st.spinner("Simulating …"):
            hist_intact  = run_sim(W_base, E0, I0, T=400)
            hist_ablated = run_sim(W_deg,  E0, I0, T=400)

        time = np.arange(400) * 0.1

        ROI     = ['Lthal', 'Rthal', 'Lhippo', 'Rhippo']
        roi_idx = [labels.index(r) for r in ROI if r in labels]
        rc      = ['#e74c3c', '#c0392b', '#2980b9', '#1a5276']

        fig_ts, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

        ax_a = axes[0]
        for idx, name, c in zip(roi_idx, ROI, rc):
            ax_a.plot(time, hist_intact[:,  idx], color=c, lw=1.5, label=name)
            ax_a.plot(time, hist_ablated[:, idx], color=c, lw=1.5, ls='--', alpha=0.6)
        ax_a.set_ylabel("E(t)")
        ax_a.set_title("ROI activity  (solid=intact, dashed=ablated)", fontsize=10)
        ax_a.legend(ncol=2, fontsize=8)
        ax_a.spines[['top', 'right']].set_visible(False)

        drift = np.linalg.norm(hist_intact - hist_ablated, axis=1)
        ax_b  = axes[1]
        ax_b.fill_between(time, drift, alpha=0.4, color='#e74c3c')
        ax_b.plot(time, drift, color='#e74c3c', lw=1.5)
        ax_b.set_xlabel("Time")
        ax_b.set_ylabel("‖intact − ablated‖")
        ax_b.set_title("Trajectory divergence", fontsize=10)
        ax_b.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_ts, use_container_width=True)
        plt.close()

        final_drift = float(drift[-50:].mean())
        st.metric("Mean drift (last 50 steps)", f"{final_drift:.3f}",
                  delta=f"{'⚠ high' if final_drift > 1.0 else 'stable'}")

# ══════════════════════════════════════════════════════
# TAB 2 — MTLE vs Normal
# ══════════════════════════════════════════════════════
with tab_mtle:
    st.markdown(
        "Comparing calibrated SC matrices (spectral-radius-normalised) and their communicability "
        "for a **healthy** HCP connectome vs an **MTLE-L-degraded** network built from ENIGMA "
        "subcortical atrophy effect sizes (Cohen's *d*)."
    )

    with st.spinner("Computing communicability …"):
        C_healthy, C_mtle, r_h, rho_h, r_m, rho_m = compute_comm_metrics(
            tuple(map(tuple, W_cal.tolist())),
            tuple(map(tuple, W_mtle_cal.tolist())),
            tuple(map(tuple, FC_emp.tolist())),
        )

    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    # ── Summary metrics ─────────────────────────────────
    st.subheader("FC Prediction Quality  (communicability → empirical BOLD FC)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Healthy  Pearson r",  f"{r_h:.3f}")
    c2.metric("Healthy  Spearman ρ", f"{rho_h:.3f}")
    c3.metric("MTLE     Pearson r",  f"{r_m:.3f}",  delta=f"{r_m - r_h:+.3f}")
    c4.metric("MTLE     Spearman ρ", f"{rho_m:.3f}", delta=f"{rho_m - rho_h:+.3f}")

    st.divider()

    # ── Calibrated SC heatmaps ───────────────────────────
    st.subheader("Calibrated Structural Connectivity")
    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(16, 5))

    def _imshow_sc(ax, mat, title, cmap='Blues'):
        im = ax.imshow(np.log1p(mat), cmap=cmap, aspect='auto')
        ax.axhline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Node"); ax.set_ylabel("Node")
        plt.colorbar(im, ax=ax, fraction=0.04, label='log(1+w)')

    _imshow_sc(axes_sc[0], W_cal,      "Healthy SC (calibrated)")
    _imshow_sc(axes_sc[1], W_mtle_cal, "MTLE SC (calibrated)")

    diff_sc = W_cal - W_mtle_cal
    lim_sc  = np.percentile(np.abs(diff_sc), 98)
    im_diff = axes_sc[2].imshow(diff_sc, cmap='RdBu_r', aspect='auto',
                                 vmin=-lim_sc, vmax=lim_sc)
    axes_sc[2].axhline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
    axes_sc[2].axvline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
    axes_sc[2].set_title("Difference: Healthy − MTLE (calibrated SC)", fontsize=10, fontweight='bold')
    axes_sc[2].set_xlabel("Node"); axes_sc[2].set_ylabel("Node")
    plt.colorbar(im_diff, ax=axes_sc[2], fraction=0.04)

    plt.tight_layout()
    st.pyplot(fig_sc, use_container_width=True)
    plt.close()

    st.divider()

    # ── Communicability heatmaps ─────────────────────────
    st.subheader("Communicability  (matrix exponential of normalised calibrated SC)")
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 5))

    vmax_c = max(C_healthy[mask].max(), C_mtle[mask].max())

    for ax, mat, title in [
        (axes_cm[0], C_healthy, f"Healthy  (r={r_h:.3f} vs FC)"),
        (axes_cm[1], C_mtle,    f"MTLE     (r={r_m:.3f} vs FC)"),
    ]:
        im = ax.imshow(mat, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=vmax_c)
        ax.axhline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Node"); ax.set_ylabel("Node")
        plt.colorbar(im, ax=ax, fraction=0.04)

    diff_c = C_healthy - C_mtle
    lim_c  = np.percentile(np.abs(diff_c), 98)
    im_dc  = axes_cm[2].imshow(diff_c, cmap='RdBu_r', aspect='auto',
                                vmin=-lim_c, vmax=lim_c)
    axes_cm[2].axhline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
    axes_cm[2].axvline(N_CTX - 0.5, color='white', lw=0.8, ls='--', alpha=0.6)
    axes_cm[2].set_title("Difference: Healthy − MTLE (communicability)", fontsize=10, fontweight='bold')
    axes_cm[2].set_xlabel("Node"); axes_cm[2].set_ylabel("Node")
    plt.colorbar(im_dc, ax=axes_cm[2], fraction=0.04)

    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True)
    plt.close()

    st.divider()

    # ── Node-level metrics ───────────────────────────────
    st.subheader("Node-level Comparison")

    col_nl, col_ef = st.columns([3, 2])

    with col_nl:
        metric_choice = st.selectbox(
            "Node metric",
            ["Communicability strength (row sum)",
             "SC strength (weighted degree)",
             "Communicability change (Healthy − MTLE)"],
        )

        if metric_choice == "Communicability strength (row sum)":
            vals_h = C_healthy.sum(axis=1)
            vals_m = C_mtle.sum(axis=1)
            ylabel = "Comm. strength"
        elif metric_choice == "SC strength (weighted degree)":
            vals_h = W_cal.sum(axis=1)
            vals_m = W_mtle_cal.sum(axis=1)
            ylabel = "SC strength"
        else:
            vals_h = (C_healthy - C_mtle).sum(axis=1)
            vals_m = None
            ylabel = "Δ Comm. strength"

        fig_nl, ax_nl = plt.subplots(figsize=(10, 4))
        x = np.arange(N)

        if vals_m is not None:
            ax_nl.bar(x[:N_CTX],  vals_h[:N_CTX],  color='#3498db', alpha=0.7, label='Healthy cortical')
            ax_nl.bar(x[N_CTX:],  vals_h[N_CTX:],  color='#2ecc71', alpha=0.7, label='Healthy subcortical')
            ax_nl.bar(x[:N_CTX],  vals_m[:N_CTX],  color='#e74c3c', alpha=0.5, label='MTLE cortical',    width=0.5)
            ax_nl.bar(x[N_CTX:],  vals_m[N_CTX:],  color='#c0392b', alpha=0.5, label='MTLE subcortical', width=0.5)
        else:
            colors = ['#e74c3c' if v < 0 else '#3498db' for v in vals_h]
            ax_nl.bar(x, vals_h, color=colors, alpha=0.8)
            ax_nl.axhline(0, color='black', lw=0.8)

        ax_nl.axvline(N_CTX - 0.5, color='black', lw=1.0, ls='--', alpha=0.5)
        ax_nl.set_xlabel("Node index")
        ax_nl.set_ylabel(ylabel)
        ax_nl.set_title(metric_choice, fontsize=11, fontweight='bold')
        if vals_m is not None:
            ax_nl.legend(fontsize=8)
        ax_nl.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_nl, use_container_width=True)
        plt.close()

    with col_ef:
        st.markdown("**ENIGMA MTLE-L subcortical atrophy  (Cohen's *d*)**")
        sctx_in_labels = [r for r in labels[N_CTX:] if r in mtle_d]
        d_vals         = [mtle_d[r] for r in sctx_in_labels]
        bar_c          = ['#e74c3c' if d < -0.5 else '#e67e22' if d < -0.1 else '#95a5a6'
                          for d in d_vals]

        fig_ef, ax_ef = plt.subplots(figsize=(5, 4))
        ax_ef.barh(sctx_in_labels, d_vals, color=bar_c, alpha=0.85)
        ax_ef.axvline(0, color='black', lw=0.8)
        ax_ef.axvline(-0.5, color='#e74c3c', lw=0.7, ls='--', alpha=0.6, label='|d|=0.5')
        ax_ef.set_xlabel("Cohen's d")
        ax_ef.set_title("Subcortical atrophy", fontsize=10, fontweight='bold')
        ax_ef.legend(fontsize=8)
        ax_ef.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_ef, use_container_width=True)
        plt.close()

    st.divider()

    # ── FC scatter plots ─────────────────────────────────
    st.subheader("Communicability → Empirical FC  (upper-triangle pairs)")
    fig_sc2, (ax_sh, ax_sm) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, C, r, rho, label_txt, col in [
        (ax_sh, C_healthy, r_h, rho_h, "Healthy",  '#27ae60'),
        (ax_sm, C_mtle,    r_m, rho_m, "MTLE-L",   '#e74c3c'),
    ]:
        ax.scatter(C[mask], FC_emp[mask], s=2, alpha=0.2, color=col)
        ax.set_xlabel("Communicability")
        ax.set_ylabel("Empirical FC")
        ax.set_title(f"{label_txt}\nPearson r={r:.3f}   Spearman ρ={rho:.3f}",
                     fontsize=10, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_sc2, use_container_width=True)
    plt.close()

    st.divider()

    # ── Top nodes most affected ──────────────────────────
    st.subheader("Most-affected nodes  (largest communicability strength loss)")
    delta_strength = C_healthy.sum(axis=1) - C_mtle.sum(axis=1)
    top_k          = st.slider("Show top-K nodes", min_value=5, max_value=30, value=15)
    top_idx        = np.argsort(delta_strength)[::-1][:top_k]
    top_labels     = [labels[i] for i in top_idx]
    top_deltas     = delta_strength[top_idx]

    fig_top, ax_top = plt.subplots(figsize=(10, 3.5))
    bar_colors_top  = ['#c0392b' if i >= N_CTX else '#2980b9' for i in top_idx]
    ax_top.barh(top_labels[::-1], top_deltas[::-1], color=bar_colors_top[::-1], alpha=0.85)
    ax_top.axvline(0, color='black', lw=0.8)
    ax_top.set_xlabel("Δ communicability strength  (healthy − MTLE)")
    ax_top.set_title(f"Top {top_k} nodes losing communicability in MTLE  "
                     f"(blue=cortical, red=subcortical)",
                     fontsize=10, fontweight='bold')
    ax_top.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_top, use_container_width=True)
    plt.close()

    st.divider()

    # ── EEG Validation: Bonn hippocampal recordings vs WC Lhippo ────────
    st.subheader("EEG Calibration  —  Bonn hippocampal depth EEG vs WC Lhippo dynamics")
    st.markdown(
        "**Bonn-F** = epileptogenic foci (interictal depth EEG, MTLE seizure onset zone).  "
        "**Bonn-N** = contralateral hippocampus (relatively spared).  \n"
        "Adjust `ATROPHY_SCALE` until the WC model spectral shift matches the empirical Bonn F vs N shift."
    )

    atrophy_scale = st.slider(
        "ATROPHY_SCALE  (connectivity reduction per unit Cohen's d)",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="0 = no degradation. 1 = Lhippo loses 100% of connectivity at d=−1. "
             "Currently 0.25 → Lhippo retains 57% of connections."
    )

    bonn_n = load_bonn_subset('N', n=10)
    bonn_f = load_bonn_subset('F', n=10)

    if not bonn_n or not bonn_f:
        st.warning("Bonn data not found at ../bonn_epilepsy/. Place the dataset there to enable calibration.")
    else:
        lhippo_idx = labels.index('Lhippo') if 'Lhippo' in labels else None
        if lhippo_idx is None:
            st.warning("Lhippo not found in labels.")
        else:
            # Rebuild MTLE network with tuned scale
            W_mtle_raw_t, W_mtle_cal_t = build_mtle_network(
                tuple(map(tuple, W_raw.tolist())),
                tuple(labels),
                tuple(sorted(mtle_d.items())),
                atrophy_scale=atrophy_scale,
            )

            with st.spinner("Simulating WC Lhippo (healthy + MTLE) …"):
                freqs_h, psd_wc_h = compute_wc_psd(
                    tuple(map(tuple, W_cal.tolist())), lhippo_idx)
                freqs_m, psd_wc_m = compute_wc_psd(
                    tuple(map(tuple, W_mtle_cal_t.tolist())), lhippo_idx)

            def avg_psd(recordings, fs, nperseg=512):
                psds = [welch(s - s.mean(), fs=fs, nperseg=nperseg)[1] for s in recordings]
                f, _ = welch(recordings[0], fs=fs, nperseg=nperseg)
                return f, np.array(psds).mean(axis=0)

            freqs_bn, psd_bn = avg_psd(bonn_n, BONN_FS)
            freqs_bf, psd_bf = avg_psd(bonn_f, BONN_FS)

            def band_power(freqs, psd, lo, hi):
                idx = (freqs >= lo) & (freqs <= hi)
                return np.trapz(psd[idx], freqs[idx]) if idx.any() else 0.0

            # ── Match score ──────────────────────────────────────────────
            # Bonn empirical: slow (δ+θ) fraction in F vs N
            def slow_frac(freqs, psd, slow_hi=8.0, total_hi=30.0):
                slow = band_power(freqs, psd, 0, slow_hi)
                total = band_power(freqs, psd, 0, total_hi)
                return slow / total if total > 0 else 0.0

            bonn_n_sf  = slow_frac(freqs_bn, psd_bn)
            bonn_f_sf  = slow_frac(freqs_bf, psd_bf)
            bonn_shift = bonn_f_sf - bonn_n_sf   # empirical: how much more slow power in foci

            wc_h_sf   = slow_frac(freqs_h, psd_wc_h, slow_hi=0.5, total_hi=2.0)
            wc_m_sf   = slow_frac(freqs_m, psd_wc_m, slow_hi=0.5, total_hi=2.0)
            wc_shift  = wc_m_sf - wc_h_sf         # model: slow-power increase in MTLE vs healthy

            # Normalised match: 1.0 = perfect direction + magnitude, 0 = no shift, negative = wrong direction
            match_score = 1.0 - abs(wc_shift - bonn_shift) / (abs(bonn_shift) + 1e-9)
            match_score = float(np.clip(match_score, -1, 1))
            direction_ok = (bonn_shift > 0) == (wc_shift > 0)

            # ── Summary metrics row ──────────────────────────────────────
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Lhippo connectivity retained",
                       f"{np.clip(1 + (-1.728) * atrophy_scale, 0.1, 1.0)*100:.0f}%",
                       help="At d=−1.728 (ENIGMA Lhippo effect size)")
            mc2.metric("Bonn slow-power shift (F−N)",
                       f"{bonn_shift*100:+.1f} pp",
                       help="Percentage-point increase in δ+θ fraction at epileptogenic foci vs contralateral")
            mc3.metric("WC slow-power shift (MTLE−healthy)",
                       f"{wc_shift*100:+.1f} pp",
                       delta="✓ correct direction" if direction_ok else "✗ wrong direction")
            mc4.metric("Calibration match score", f"{match_score:.2f}",
                       help="1.0 = perfect match, 0 = no shift reproduced, negative = wrong direction")

            st.divider()

            col_eeg, col_wc = st.columns(2)

            with col_eeg:
                fig_eeg, ax_eeg = plt.subplots(figsize=(6, 4))
                ax_eeg.semilogy(freqs_bn, psd_bn, color='#27ae60', lw=1.8, label=f'Bonn-N  (slow frac {bonn_n_sf*100:.1f}%)')
                ax_eeg.semilogy(freqs_bf, psd_bf, color='#e74c3c', lw=1.8, label=f'Bonn-F  (slow frac {bonn_f_sf*100:.1f}%)')
                ax_eeg.set_xlim(0, 40);  ax_eeg.set_xlabel('Frequency (Hz)')
                ax_eeg.set_ylabel('PSD (µV²/Hz)')
                ax_eeg.set_title('Bonn depth EEG  (empirical target)', fontsize=10, fontweight='bold')
                ax_eeg.legend(fontsize=9);  ax_eeg.spines[['top', 'right']].set_visible(False)
                st.pyplot(fig_eeg, use_container_width=True);  plt.close()

                bands = {'δ (0–4)': (0, 4), 'θ (4–8)': (4, 8),
                         'α (8–13)': (8, 13), 'β (13–30)': (13, 30)}
                rows = []
                for bname, (lo, hi) in bands.items():
                    pn = band_power(freqs_bn, psd_bn, lo, hi)
                    pf = band_power(freqs_bf, psd_bf, lo, hi)
                    rows.append({'Band': bname, 'Bonn-N': f'{pn:.2e}',
                                 'Bonn-F': f'{pf:.2e}', 'F/N ratio': f'{pf/pn:.2f}' if pn > 0 else 'N/A'})
                st.markdown("**Band power ratio F / N**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            with col_wc:
                psd_h_norm = psd_wc_h / psd_wc_h.max()
                psd_m_norm = psd_wc_m / psd_wc_m.max()
                fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                ax_wc.semilogy(freqs_h, psd_h_norm, color='#27ae60', lw=1.8,
                               label=f'WC Healthy  (slow frac {wc_h_sf*100:.1f}%)')
                ax_wc.semilogy(freqs_m, psd_m_norm, color='#e74c3c', lw=1.8,
                               label=f'WC MTLE  (slow frac {wc_m_sf*100:.1f}%)')
                ax_wc.set_xlim(0, 2);  ax_wc.set_xlabel('Frequency (sim. units, dt=0.1)')
                ax_wc.set_ylabel('Normalised PSD')
                ax_wc.set_title(f'WC Lhippo node  (scale={atrophy_scale:.2f})', fontsize=10, fontweight='bold')
                ax_wc.legend(fontsize=9);  ax_wc.spines[['top', 'right']].set_visible(False)
                st.pyplot(fig_wc, use_container_width=True);  plt.close()

                wc_rows = []
                for lbl, fw, pw in [('Healthy', freqs_h, psd_wc_h), ('MTLE-L', freqs_m, psd_wc_m)]:
                    lo_p  = band_power(fw, pw, 0, 0.5)
                    hi_p  = band_power(fw, pw, 0.5, 2.0)
                    tot   = lo_p + hi_p
                    wc_rows.append({'Model': lbl,
                                    'Low-freq % (0–0.5)':  f'{100*lo_p/tot:.1f}%' if tot > 0 else 'N/A',
                                    'High-freq % (0.5–2)': f'{100*hi_p/tot:.1f}%' if tot > 0 else 'N/A'})
                st.markdown("**WC power distribution**")
                st.dataframe(pd.DataFrame(wc_rows), hide_index=True, use_container_width=True)
