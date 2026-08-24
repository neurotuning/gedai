"""
Interactive Overlay and SENSAI Visualization for Synthetic Data
==============================================================
"""

import os
import matplotlib.pyplot as plt
import mne

from gedai import AdaptiveMultibandGedai
from gedai.viz import plot_mne_style_overlay_interactive

# 1. Load dataset (synthetic bad channels example)
file_path = r"C:\Users\Ros\Documents\MATLAB\eeglab2025.0.0\plugins\GEDAI-master\example data\synthetic_bad_channels.set"
if not os.path.exists(file_path):
    from gedai.data import get_contaminated_eeg_set_path
    file_path = str(get_contaminated_eeg_set_path())

print(f"Loading data from: {file_path}")
raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

# 2. Filter (0.5 Hz highpass)
raw.filter(l_freq=0.5, h_freq=None, verbose=False)

# 3. Initialize & Fit Adaptive Multiband GEDAI
ad = AdaptiveMultibandGedai(
    wavelet_type="haar",
    wavelet_level="auto",
    cycles_per_wavelet=10,
    broadband_pass=True,
)

print("Fitting Adaptive Multiband GEDAI...")
ad.fit_raw(raw, noise_multiplier=3.0, n_jobs=-1, verbose=False)

print("Transforming raw data...")
denoised_raw = ad.transform_raw(raw, n_jobs=-1, verbose=False)

# 4. Generate SENSAI Subspace Similarity Visualization
print("Generating SENSAI Visualization...")
fig_sensai, metrics = ad.plot_sensai(
    raw_before=raw,
    raw_after=denoised_raw,
    show=False,
)

# 5. Generate MNE-style Interactive Overlay (Supports scrolling, zooming, diff/denoised/noisy toggle)
print("Generating Interactive Overlay...")
plot_mne_style_overlay_interactive(
    raw,
    denoised_raw,
    title="GEDAI Denoising: Noisy (Red) vs Cleaned (Blue)",
    duration=10.0,
)

print("\n================ Denoising Summary ================")
print(ad.fit_summary())
print("================ SENSAI Metrics ===================")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k:25s}: {v:8.4f}")
print("===================================================\n")
print("Interactive Controls for Overlay:")
print("  - Left/Right arrows: Scroll time window")
print("  - Up/Down arrows   : Increase / decrease amplitude scaling")
print("  - 'd' key          : Toggle Difference view (Noisy - Cleaned)")
print("  - 'c' key          : Toggle Cleaned-only view")
print("  - 'n' key          : Toggle Noisy-only view")
print("  - 'o' key          : Toggle Overlay view (both)")
print("\nOpening figures...")
plt.show()
