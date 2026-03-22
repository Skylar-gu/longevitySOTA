"""
Bonn Epilepsy EEG — Deep vs Shallow Analysis

Subsets (A–E mapping):
  Z (A) — scalp, healthy, eyes open       → shallow cortical node
  O (B) — scalp, healthy, eyes closed     → shallow cortical node
  N (C) — intracranial, contralateral hipp, inter-ictal  → deep node (healthy-ish)
  F (D) — intracranial, epileptogenic foci, inter-ictal  → deep node (Lhippo MTLE)
  S (E) — intracranial, ictal (seizure)   → Lhippo in seizure state

Sampling rate: 173.61 Hz,  4097 samples per recording (~23.6 s)
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from pathlib import Path

FS     = 173.61
NPERSEG = 512
N_LOAD  = 30       # recordings per subset to average

DATA = Path('bonn_epilepsy')
SUBSETS = {
    'Z (healthy scalp, eyes open)':   ('Z', '#3498db',  'shallow'),
    'O (healthy scalp, eyes closed)': ('O', '#2980b9',  'shallow'),
    'N (contralateral hippo)':        ('N', '#27ae60',  'deep'),
    'F (epileptogenic foci)':         ('F', '#e74c3c',  'deep'),
    'S (ictal)':                      ('S', '#8e44ad',  'deep'),
}

def load_subset(folder, n=N_LOAD):
    files = sorted((DATA / folder).glob('*.txt')) + sorted((DATA / folder).glob('*.TXT'))
    files = sorted(files)[:n]
    return [np.loadtxt(f) for f in files]

def mean_psd(recordings):
    psds = [welch(r, fs=FS, nperseg=NPERSEG)[1] for r in recordings]
    freqs = welch(recordings[0], fs=FS, nperseg=NPERSEG)[0]
    return freqs, np.array(psds)

print("Loading subsets …")
data = {}
for label, (folder, color, depth) in SUBSETS.items():
    recs = load_subset(folder)
    freqs, psds = mean_psd(recs)
    data[label] = {'recs': recs, 'freqs': freqs, 'psds': psds,
                   'color': color, 'depth': depth, 'folder': folder}
    print(f"  {folder}: {len(recs)} recordings, mean amplitude {np.mean([np.std(r) for r in recs]):.1f} µV")

BANDS = {'Delta': (0.5,4), 'Theta': (4,8), 'Alpha': (8,13), 'Beta': (13,30), 'Gamma': (30,60)}

def band_power(freqs, psd):
    powers = {}
    for band, (f1, f2) in BANDS.items():
        idx = (freqs >= f1) & (freqs <= f2)
        powers[band] = np.trapz(psd[idx], freqs[idx])
    return powers

# ─── PLOTS ────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)

# 1. Mean PSD — all subsets
ax1 = fig.add_subplot(gs[0, :2])
for label, d in data.items():
    mean_p = d['psds'].mean(axis=0)
    std_p  = d['psds'].std(axis=0)
    pdb    = 10 * np.log10(mean_p + 1e-30)
    ax1.plot(d['freqs'], pdb, color=d['color'], lw=2,
             ls='-' if d['depth']=='shallow' else '--', label=label)
ax1.set_xlim(0.5, 60);  ax1.set_xlabel('Frequency (Hz)');  ax1.set_ylabel('Power (dB)')
ax1.set_title('Mean PSD — all subsets\n(solid=scalp/shallow, dashed=intracranial/deep)',
              fontsize=10, fontweight='bold')
ax1.legend(fontsize=7, loc='upper right');  ax1.grid(alpha=0.3)
ax1.spines[['top','right']].set_visible(False)
# band shading
band_colors = ['#ecf0f1','#d5dbdb','#aed6f1','#d7bde2','#f9e79f']
xf = ax1.get_xaxis_transform()
for (band,(f1,f2)), bc in zip(BANDS.items(), band_colors):
    ax1.axvspan(f1, f2, color=bc, alpha=0.35, zorder=0)
    ax1.text((f1+f2)/2, 0.97, band, transform=xf, ha='center', va='top', fontsize=7, color='#555')

# 2. Shallow vs deep zoomed comparison (Z vs F — most contrasting)
ax2 = fig.add_subplot(gs[0, 2:])
for key in ['Z (healthy scalp, eyes open)', 'N (contralateral hippo)', 'F (epileptogenic foci)', 'S (ictal)']:
    d = data[key]
    pdb = 10 * np.log10(d['psds'].mean(axis=0) + 1e-30)
    ax2.plot(d['freqs'], pdb, color=d['color'], lw=2, label=key.split('(')[0].strip())
ax2.set_xlim(0.5, 60);  ax2.set_xlabel('Frequency (Hz)');  ax2.set_ylabel('Power (dB)')
ax2.set_title('Scalp vs Intracranial\n(key subsets)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8);  ax2.grid(alpha=0.3)
ax2.spines[['top','right']].set_visible(False)

# 3. Band power bar chart
ax3 = fig.add_subplot(gs[1, :2])
x         = np.arange(len(BANDS))
bar_w     = 0.15
subset_keys = list(SUBSETS.keys())
for i, label in enumerate(subset_keys):
    d  = data[label]
    bp = [np.mean([band_power(d['freqs'], psd)[b] for psd in d['psds']]) for b in BANDS]
    bp_db = 10 * np.log10(np.array(bp) + 1e-30)
    ax3.bar(x + i*bar_w, bp_db, bar_w, color=d['color'], alpha=0.85,
            label=d['folder'])
ax3.set_xticks(x + bar_w*2);  ax3.set_xticklabels(BANDS.keys(), fontsize=10)
ax3.set_ylabel('Power (dB)');  ax3.set_title('Band Power by Subset', fontsize=10, fontweight='bold')
ax3.legend(fontsize=9);  ax3.spines[['top','right']].set_visible(False);  ax3.grid(axis='y', alpha=0.3)

# 4. Example time series — one recording each from Z, F, S
ax_t = [fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3]),
        fig.add_subplot(gs[2, 0])]
for ax, key, title in zip(ax_t,
        ['Z (healthy scalp, eyes open)', 'F (epileptogenic foci)', 'S (ictal)'],
        ['Z — Scalp healthy', 'F — Epileptogenic foci', 'S — Ictal']):
    rec  = data[key]['recs'][0]
    t    = np.arange(len(rec)) / FS
    ax.plot(t[:int(5*FS)], rec[:int(5*FS)], lw=0.7, color=data[key]['color'])
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Amplitude')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

# 5. Spectrogram of F (epileptogenic) vs Z (healthy scalp)
for col, (key, title) in enumerate([('Z (healthy scalp, eyes open)', 'Z — Healthy scalp'),
                                     ('F (epileptogenic foci)',       'F — Epileptogenic foci')]):
    ax = fig.add_subplot(gs[2, 1+col])
    rec = data[key]['recs'][0]
    f, t_s, Sxx = spectrogram(rec, fs=FS, nperseg=256, noverlap=200)
    f_mask = f <= 60
    ax.pcolormesh(t_s, f[f_mask], 10*np.log10(Sxx[f_mask]+1e-30),
                  cmap='inferno', shading='gouraud')
    ax.set_xlabel('Time (s)');  ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title, fontsize=9, fontweight='bold')

# 6. Alpha peak: O (eyes closed) — should show strong alpha
ax6 = fig.add_subplot(gs[2, 3])
for key in ['Z (healthy scalp, eyes open)', 'O (healthy scalp, eyes closed)']:
    d   = data[key]
    pdb = 10 * np.log10(d['psds'].mean(axis=0) + 1e-30)
    ax6.plot(d['freqs'], pdb, color=d['color'], lw=2, label=d['folder'])
ax6.set_xlim(4, 25);  ax6.set_xlabel('Frequency (Hz)');  ax6.set_ylabel('Power (dB)')
ax6.set_title('Alpha blocking\n(eyes open vs closed)', fontsize=9, fontweight='bold')
ax6.axvspan(8, 13, color='#aed6f1', alpha=0.4, label='Alpha')
ax6.legend(fontsize=9);  ax6.spines[['top','right']].set_visible(False)

fig.savefig('bonn_analysis.png', dpi=150, bbox_inches='tight')
print("Saved bonn_analysis.png")

# ─── Print spectral summary ───────────────────────────
print("\nBand power summary (mean dB across recordings):")
print(f"{'Subset':35s}", '  '.join(f'{b:>6}' for b in BANDS))
for label, d in data.items():
    bp = [np.mean([10*np.log10(band_power(d['freqs'], psd)[b]+1e-30) for psd in d['psds']]) for b in BANDS]
    print(f"{label:35s}", '  '.join(f'{v:6.1f}' for v in bp))
