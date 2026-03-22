# longevitySOTA

Computational neuroscience pipeline for modeling how neurological disease — specifically mesial temporal lobe epilepsy (MTLE) — degrades brain network dynamics. Built for hackathon exploration of EEG biomarkers, structural connectome modeling, and Wilson–Cowan neural mass dynamics.

---

## What it does

Three analyses that build on each other:

1. **EEG spectral analysis** — resting-state EEG from 88 subjects (Alzheimer's, Frontotemporal Dementia, Cognitively Normal) showing group-level spectral differences in power bands and topographic maps

2. **Structural connectome + WC dynamics** — Wilson–Cowan neural mass model on the 82-node HCP structural connectome, grounded by ENIGMA clinical atrophy data from 2,149 MTLE patients across 24 sites

3. **EEG calibration** — live calibration of the MTLE model against Bonn depth electrode recordings from actual epilepsy patients (epileptogenic foci vs contralateral hippocampus)

---

## Datasets

| Dataset | What it contains |
|---------|-----------------|
| [ds004504](https://openneuro.org/datasets/ds004504) | 88-subject resting EEG, AD/FTD/CN groups, EEGLAB format |
| HCP S1200 (via ENIGMA toolkit) | Group-averaged 82×82 DTI structural connectivity + fMRI FC |
| ENIGMA-Epilepsy | MTLE-L subcortical volume Cohen's d, N=2,149 patients |
| [Bonn epilepsy dataset](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/) | Depth + scalp EEG, 5 conditions: Z/O (healthy scalp), N/F (hippocampal depth), S (ictal) |

---

## Structure

```
├── mne_visualization.py      # EEG PSD, topomaps, band power (ds004504)
├── spectrogram.py            # Rolling time-frequency analysis per patient
├── bonn_analysis.py          # Bonn epilepsy dataset — depth vs scalp EEG
├── FEATURES.md               # Full neuroscience-backed feature reference
└── connectome/
    ├── connectome.py         # HCP 82-node structural connectivity matrix
    ├── wc_graph.py           # Wilson–Cowan dynamics on connectome
    ├── wc_attractor.py       # Attractor identification + progressive replacement
    ├── wc_fitting.py         # SC→FC communicability + MTLE degradation analysis
    └── app.py                # Interactive Streamlit app
```

---

## Interactive App

```bash
cd connectome
streamlit run app.py
```

**Tab 1 — Node Ablation**
Select cortical or subcortical regions to remove. Toggle between a healthy baseline and an MTLE-degraded baseline (ENIGMA atrophy-scaled connectivity). Watch the network dynamics collapse in real time via trajectory divergence plots.

**Tab 2 — MTLE vs Normal**
- Structural connectivity heatmaps: healthy, MTLE, difference
- Communicability matrices (SC→FC prediction, r=0.277 vs empirical BOLD FC)
- ENIGMA effect sizes: Lhippo d=−1.728, Lthal d=−0.843
- **EEG Calibration**: adjust `ATROPHY_SCALE` and watch the WC model's Lhippo spectral shift converge toward the empirical Bonn target (+6.6 percentage-point slow-power shift at epileptogenic foci)

---

## Key results

- SC→FC Pearson r = 0.263 (direct); communicability→FC r = 0.277
- MTLE Lhippo retains 57% of connectivity at default ATROPHY_SCALE=0.25
- With sigmoid re-parameterised at mid-range operating point (w_EE=1.5, gain=2, θ=0.75): Lhippo MTLE Δ = −0.082 (vs −0.018 when saturated)
- Bonn empirical slow-power shift F vs N: **+6.6 pp** (delta 7.5×, theta 5.3× elevated at foci)
- WC attractor identification: 3 metastable states from 50 random initial conditions (KMeans)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install mne numpy scipy matplotlib pandas scikit-learn streamlit
```

The ENIGMA toolkit CSVs are loaded directly from `enigma/enigmatoolbox/datasets/` — no enigmatoolbox import needed (bypasses nibabel/NumPy 2.0 incompatibility).

---

## Neuroscience background

See [`FEATURES.md`](FEATURES.md) for a full feature-by-feature breakdown with citations covering:
- Wilson–Cowan model stability and operating point theory
- ENIGMA-Epilepsy meta-analysis methodology
- SC→FC communicability as a topology-based functional connectivity proxy
- Why depth EEG (Bonn-F/N) validates the MTLE model
- Attractor dynamics as a framework for cognitive state maintenance
