import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

mne.set_log_level('WARNING')

BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
BAND_COLORS = {'Delta': '#8e44ad', 'Theta': '#2980b9', 'Alpha': '#27ae60', 'Beta': '#e67e22'}

patient_number = "068"

raw = mne.io.read_raw_eeglab(
    f'ds004504/derivatives/sub-{patient_number}/eeg/sub-{patient_number}_task-eyesclosed_eeg.set',
    preload=True, verbose=False
)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore', verbose=False)

sfreq = raw.info['sfreq']
WIN_SEC = 10       # window length (s)
STEP_SEC = 5       # step between windows (s)
FMIN, FMAX = 0.5, 45

win_samples  = int(WIN_SEC * sfreq)
step_samples = int(STEP_SEC * sfreq)
n_samples    = raw.n_times
starts       = np.arange(0, n_samples - win_samples, step_samples)

print(f"Computing PSD across {len(starts)} windows ({WIN_SEC}s window, {STEP_SEC}s step)...")

all_psds = []
t_centers = []

for start in starts:
    segment = raw.copy().crop(
        tmin=start / sfreq,
        tmax=(start + win_samples) / sfreq,
        include_tmax=False
    )
    spectrum = segment.compute_psd(method='welch', fmin=FMIN, fmax=FMAX, verbose=False)
    psd = spectrum.get_data().mean(axis=0)   # average across channels
    all_psds.append(psd)
    t_centers.append((start + win_samples / 2) / sfreq / 60)  # minutes

freqs    = spectrum.freqs
all_psds = np.array(all_psds).T   # (n_freqs, n_windows)
psd_db   = 10 * np.log10(all_psds + 1e-30)
t_centers = np.array(t_centers)

# Band power time series
band_power_ts = {}
for band, (f1, f2) in BANDS.items():
    idx = np.where((freqs >= f1) & (freqs <= f2))[0]
    band_power_ts[band] = 10 * np.log10(all_psds[idx].mean(axis=0) + 1e-30)

# ── Plot ───────────────────────────────────────────────────────
fig, (ax_spec, ax_bands) = plt.subplots(2, 1, figsize=(14, 9),
                                         gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle('sub-001 (AD) — Power Spectral Density Over Time\n'
             f'Preprocessed · {WIN_SEC}s windows · {STEP_SEC}s step',
             fontsize=13, fontweight='bold')

# Spectrogram
im = ax_spec.imshow(
    psd_db, aspect='auto', origin='lower', cmap='inferno',
    extent=[t_centers[0], t_centers[-1], freqs[0], freqs[-1]],
    vmin=np.percentile(psd_db, 5), vmax=np.percentile(psd_db, 95)
)
# Band boundary lines
for band, (f1, f2) in BANDS.items():
    ax_spec.axhline(f1, color='white', lw=0.6, ls='--', alpha=0.5)
    ax_spec.text(t_centers[-1] + 0.1, (f1 + f2) / 2, band,
                 color='white', fontsize=8, va='center', fontweight='bold')
ax_spec.set_ylabel('Frequency (Hz)', fontsize=11)
ax_spec.set_xlim(t_centers[0], t_centers[-1])
plt.colorbar(im, ax=ax_spec, label='Power (dB µV²/Hz)', pad=0.01)

# Band power time series
for band, ts in band_power_ts.items():
    ax_bands.plot(t_centers, ts, label=band, color=BAND_COLORS[band], lw=1.8)
ax_bands.set_xlabel('Time (minutes)', fontsize=11)
ax_bands.set_ylabel('Power (dB µV²/Hz)', fontsize=11)
ax_bands.legend(fontsize=10, loc='upper right', framealpha=0.85)
ax_bands.grid(alpha=0.3)
ax_bands.spines[['top', 'right']].set_visible(False)
ax_bands.set_xlim(t_centers[0], t_centers[-1])

plt.tight_layout()
fig.savefig(f'spectrogram_{patient_number}.png', dpi=150, bbox_inches='tight')
print(f"Saved spectrogram_{patient_number}.png")

