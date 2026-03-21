import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

mne.set_log_level('WARNING')

raw_raw  = mne.io.read_raw_eeglab('ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set', preload=True, verbose=False)
raw_prep = mne.io.read_raw_eeglab('ds004504/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set', preload=True, verbose=False)

sfreq = raw_raw.info['sfreq']
t0, t1 = 200, 300   # 10-second window

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('sub-001 — Raw vs Preprocessed EEG (10 s window)', fontsize=14, fontweight='bold')

for ax, raw, label in zip(axes, [raw_raw, raw_prep], ['Raw', 'Preprocessed']):
    data, times = raw[:, int(t0 * sfreq):int(t1 * sfreq)]
    data_uv = data * 1e6
    spacing = 150
    for i, ch in enumerate(raw.ch_names):
        ax.plot(times, data_uv[i] + i * spacing, lw=0.7, color='#2c3e50', alpha=0.85)
        ax.text(times[0] - 0.15, i * spacing, ch, ha='right', va='center', fontsize=7, color='#555')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.spines[['left', 'top', 'right']].set_visible(False)

axes[-1].set_xlabel('Time (s)', fontsize=11)
plt.tight_layout()
fig.savefig('compare_data.png', dpi=150, bbox_inches='tight')
print("Saved compare_data.png")
