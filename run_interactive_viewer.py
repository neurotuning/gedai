"""Interactive EEG Overlay Viewer using GEDAI Adaptive Multiband Denoising.

Run this script directly in your terminal:
    uv run python run_interactive_viewer.py
"""

import matplotlib
import mne
import numpy as np

# Ensure interactive GUI backend
try:
    matplotlib.use("QtAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from gedai import AdaptiveMultibandGedai
from gedai.viz.compare import plot_mne_style_overlay_interactive

fpath = r"C:\Users\drtro\Documents\MATLAB\eeglab2025.1.0\plugins\GEDAI-master\example data\empirical_NOISE_EOG_EMG.set"

print("=" * 70)
print("Loading and Denoising EEG Data with Adaptive Multiband GEDAI...")
print("=" * 70)

raw = mne.io.read_raw(fpath, preload=True, verbose=False)
raw.set_channel_types({ch: "eeg" for ch in raw.ch_names if raw.get_channel_types([ch])[0] != "eeg"})
raw.filter(l_freq=0.5, h_freq=None, verbose=False)

ad = AdaptiveMultibandGedai(
    wavelet_type="haar",
    wavelet_level="auto",
    cycles_per_wavelet=10,
    broadband_pass=True,
)
ad.fit_raw(raw, picks="all", sensai_method="optimize", noise_multiplier=3.0, verbose=False)
raw_clean = ad.transform_raw(raw, verbose=False)

print(f"\n[DONE] SENSAI Score: {ad.metrics_['sensai_score']:.2f}% | Mean ENOVA: {ad.metrics_['mean_enova']*100:.2f}%")
print("\nOpening Interactive Comparison Window...")
print("Keyboard Controls:")
print("  - [Right Arrow] / [Left Arrow]: Scroll forward / backward in time")
print("  - [Up Arrow] / [Down Arrow]: Increase / decrease amplitude scaling")
print("  - [D]: Toggle Difference Mode (Noisy - Cleaned in purple)")
print("  - [N]: Toggle Denoised Only Mode (blue)")
print("  - [O]: Toggle Noisy Only Mode (red)")
print("  - Default: Overlay Mode (red raw + blue clean)")
print("=" * 70)

plot_mne_style_overlay_interactive(
    raw,
    raw_clean,
    title=f"Adaptive Multiband GEDAI (SENSAI: {ad.metrics_['sensai_score']:.2f}%, ENOVA: {ad.metrics_['mean_enova']*100:.2f}%)",
    duration=6.0,
)
