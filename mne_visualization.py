import numpy as np
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

mne.set_log_level('WARNING')

# ── Config ─────────────────────────────────────────────────────
DATASET = Path('ds004504')
BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
GROUP_SUBJECTS = {
    'AD':  ['sub-001', 'sub-002', 'sub-003'],
    'CN':  ['sub-037', 'sub-038', 'sub-039'],
    'FTD': ['sub-066', 'sub-067', 'sub-068'],
}
GROUP_COLORS = {'AD': '#e74c3c', 'CN': '#27ae60', 'FTD': '#2980b9'}


def load_subject(sub_id):
    path = DATASET / 'derivatives' / sub_id / 'eeg' / f'{sub_id}_task-eyesclosed_eeg.set'
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    return raw


def get_band_powers(raw):
    spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45, verbose=False)
    psd_data = spectrum.get_data()   # (n_channels, n_freqs)
    freqs = spectrum.freqs
    powers = {}
    for band, (fmin, fmax) in BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        powers[band] = psd_data[:, idx].mean(axis=-1)   # (n_channels,)
    return powers, spectrum


# ── Load all subjects ──────────────────────────────────────────
print("Loading subjects...")
group_data = {}
for group, subs in GROUP_SUBJECTS.items():
    group_data[group] = []
    for sub in subs:
        raw = load_subject(sub)
        powers, spectrum = get_band_powers(raw)
        group_data[group].append({'raw': raw, 'powers': powers, 'spectrum': spectrum, 'sub': sub})
    print(f"  {group}: {len(subs)} subjects loaded")

ref_raw = group_data['AD'][0]['raw']
BAND_NAMES = list(BANDS.keys())
print(f"Band names: {BAND_NAMES}")

# ══════════════════════════════════════════════════════════════
# Figure 1 — Sensor Layout  +  Raw EEG Trace
# ══════════════════════════════════════════════════════════════
fig1, (ax_mont, ax_trace) = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle('EEG Recording Overview', fontsize=15, fontweight='bold')

ref_raw.plot_sensors(show_names=True, axes=ax_mont, show=False)
ax_mont.set_title('Electrode Layout (10-20 System)', fontsize=12)

sfreq = ref_raw.info['sfreq']
data, times = ref_raw[:, int(60 * sfreq):int(70 * sfreq)]
data_uv = data * 1e6
spacing = 150  # µV between channels
for i, ch in enumerate(ref_raw.ch_names):
    ax_trace.plot(times, data_uv[i] + i * spacing, lw=0.7, color='#2c3e50', alpha=0.85)
    ax_trace.text(times[0] - 0.15, i * spacing, ch, ha='right', va='center', fontsize=7, color='#555')
ax_trace.set_xlabel('Time (s)', fontsize=11)
ax_trace.set_title('Raw EEG Trace – AD sub-001  (10 s, eyes closed)', fontsize=12)
ax_trace.set_yticks([])
ax_trace.spines[['left', 'top', 'right']].set_visible(False)

plt.tight_layout()
fig1.savefig('fig1_overview.png', dpi=150, bbox_inches='tight')
print("Saved fig1_overview.png")

# ══════════════════════════════════════════════════════════════
# Figure 2 — PSD Group Comparison
# ══════════════════════════════════════════════════════════════
fig2, (ax, ax_beta) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={'width_ratios': [2, 1]})
fig2.suptitle('Power Spectral Density — Group Comparison (Mean ± SD)', fontsize=14, fontweight='bold')

ref_freqs = None
for group, subjects in group_data.items():
    all_psds = []
    for s in subjects:
        psd_db = 10 * np.log10(s['spectrum'].get_data().mean(axis=0))
        all_psds.append(psd_db)
        ref_freqs = s['spectrum'].freqs
    psds = np.array(all_psds)
    mean_p, std_p = psds.mean(axis=0), psds.std(axis=0)
    ax.plot(ref_freqs, mean_p, label=group, color=GROUP_COLORS[group], lw=2.5)
    ax.fill_between(ref_freqs, mean_p - std_p, mean_p + std_p, alpha=0.18, color=GROUP_COLORS[group])

# Frequency band shading
band_palette = ['#ecf0f1', '#d5dbdb', '#aed6f1', '#d7bde2']
xform = ax.get_xaxis_transform()
for (band, (f1, f2)), bc in zip(BANDS.items(), band_palette):
    ax.axvspan(f1, f2, color=bc, alpha=0.45, zorder=0)
    ax.text((f1 + f2) / 2, 0.97, band, transform=xform,
            ha='center', va='top', fontsize=9, color='#555', style='italic')

ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Power (dB µV²/Hz)', fontsize=12)
ax.set_xlim(0.5, 45)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# Beta zoom panel
for group, subjects in group_data.items():
    all_psds = [10 * np.log10(s['spectrum'].get_data().mean(axis=0)) for s in subjects]
    psds = np.array(all_psds)
    mean_p, std_p = psds.mean(axis=0), psds.std(axis=0)
    ax_beta.plot(ref_freqs, mean_p, label=group, color=GROUP_COLORS[group], lw=2.5)
    ax_beta.fill_between(ref_freqs, mean_p - std_p, mean_p + std_p, alpha=0.18, color=GROUP_COLORS[group])
ax_beta.set_xlim(13, 30)
ax_beta.set_xlabel('Frequency (Hz)', fontsize=12)
ax_beta.set_title('Beta band zoom (13–30 Hz)', fontsize=11)
ax_beta.legend(fontsize=11, framealpha=0.9)
ax_beta.grid(alpha=0.3)
ax_beta.spines[['top', 'right']].set_visible(False)
ax_beta.axvspan(13, 30, color='#d7bde2', alpha=0.3, zorder=0)

plt.tight_layout()
fig2.savefig('fig2_psd.png', dpi=150, bbox_inches='tight')
print("Saved fig2_psd.png")

# ══════════════════════════════════════════════════════════════
# Figure 3 — Topographic Maps: Band × Group
# ══════════════════════════════════════════════════════════════
n_bands = len(BAND_NAMES)
n_groups = len(group_data)
fig3, axes3 = plt.subplots(n_groups, n_bands, figsize=(4 * n_bands, 3.8 * n_groups + 0.5))
fig3.suptitle('Band Power Topography by Group', fontsize=15, fontweight='bold')

for row, (group, subjects) in enumerate(group_data.items()):
    avg_powers = {b: np.zeros(len(ref_raw.ch_names)) for b in BANDS}
    for s in subjects:
        for band, bp in s['powers'].items():
            avg_powers[band] += bp
    for b in BANDS:
        avg_powers[b] /= len(subjects)

    for col, band in enumerate(BAND_NAMES):
        ax = axes3[row, col]
        power_db = 10 * np.log10(avg_powers[band] + 1e-30)
        vmin, vmax = np.percentile(power_db, 5), np.percentile(power_db, 95)
        im, _ = mne.viz.plot_topomap(
            power_db, ref_raw.info,
            axes=ax, show=False,
            cmap='RdYlBu_r', vlim=(vmin, vmax),
        )
        if row == 0:
            ax.set_title(band, fontsize=12, fontweight='bold', pad=10)
        if col == 0:
            ax.text(-0.18, 0.5, group, transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color=GROUP_COLORS[group],
                    ha='right', va='center', rotation=90)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='dB')

plt.tight_layout()
fig3.savefig('fig3_topomaps.png', dpi=150, bbox_inches='tight')
print("Saved fig3_topomaps.png")

# ══════════════════════════════════════════════════════════════
# Figure 4 — Band Power Bar Chart with Individual Points
# ══════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, n_bands, figsize=(14, 5))
fig4.suptitle('Mean Band Power by Group (all channels)', fontsize=14, fontweight='bold')

groups_list = list(group_data.keys())
x = np.arange(len(groups_list))

for col, band in enumerate(BAND_NAMES):
    ax = axes4[col]
    means, stds = [], []
    for group in groups_list:
        vals = [10 * np.log10(s['powers'][band].mean() + 1e-30) for s in group_data[group]]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    ax.bar(x, means, yerr=stds, width=0.55, capsize=6,
           color=[GROUP_COLORS[g] for g in groups_list],
           alpha=0.82, edgecolor='white', linewidth=0.8, error_kw={'lw': 1.5})

    rng = np.random.default_rng(42)
    for gi, group in enumerate(groups_list):
        vals = [10 * np.log10(s['powers'][band].mean() + 1e-30) for s in group_data[group]]
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(gi + jitter, vals, color='#2c3e50', s=30, zorder=5, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(groups_list, fontsize=11)
    ax.set_title(band, fontsize=12, fontweight='bold')
    if col == 0:
        ax.set_ylabel('Power (dB µV²/Hz)', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig4.savefig('fig4_bandpower_bars.png', dpi=150, bbox_inches='tight')
print("Saved fig4_bandpower_bars.png")

print("\nAll done. Figures: fig1_overview.png  fig2_psd.png  fig3_topomaps.png  fig4_bandpower_bars.png")
