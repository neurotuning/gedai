"""
Quick GEDAI denoising demo on CAUEEG.set (EEG data).

Applies spectral GEDAI with the built-in leadfield reference covariance
and shows a before/after overlay comparison.
"""

import os
import mne
from gedai import Gedai
from gedai.viz.compare import plot_mne_style_overlay_interactive

# ── Load data ──────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "gedai", "data", "CAUEEG.set")
raw = mne.io.read_raw_eeglab(data_path, preload=True, verbose="error")
print(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} samples, "
      f"sfreq={raw.info['sfreq']} Hz, duration={raw.n_times / raw.info['sfreq']:.1f} s")

# ── Preprocessing ──────────────────────────────────────────────────────────
# Band-pass to stabilise covariances and remove DC drift
#raw.filter(0.5, None, verbose=False)

# Keep a copy of the "before" signal for comparison
raw_before = raw.copy()

# ── Apply Broadband GEDAI ─────────────────────────────────────────────────
gedai = Gedai(
    wavelet_level=0,
    signal_type="eeg",
    highpass_cutoff=0.1,    # MODWT-based high-pass at ~0.1 Hz
)

print("Running spectral GEDAI...", flush=True)
raw_clean = gedai.fit_transform_raw(
    raw,
    reference_cov="leadfield",
    noise_multiplier=3.0,
)

# ── Overlay comparison ────────────────────────────────────────────────────
print("Plotting before/after overlay (close window to exit)...", flush=True)
plot_mne_style_overlay_interactive(
    raw_before, raw_clean,
    title="CAUEEG — Before vs After Spectral GEDAI",
    duration=10.0,
)
