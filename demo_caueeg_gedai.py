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

# ── Apply Spectral GEDAI ──────────────────────────────────────────────────
gedai = Gedai(
    wavelet_level="auto",
    wavelet_low_cutoff=0.5,
    epoch_size_in_cycles=12,
    signal_type="eeg",
    highpass_cutoff=0.1,    # MODWT-based high-pass at ~0.1 Hz
)

import time
print("Running spectral GEDAI...", flush=True)
_t0 = time.perf_counter()
raw_clean = gedai.fit_transform_raw(
    raw,
    reference_cov="leadfield",
    noise_multiplier=3.0,
)
elapsed = time.perf_counter() - _t0

# ── Metrics ───────────────────────────────────────────────────────────────
data_before = raw_before.get_data()
data_after  = raw_clean.get_data()

# Total SENSAI score: weighted average across processed bands
# (weight = fraction of total signal variance in that band)
total_sensai = 0.0
total_weight = 0.0
for wf in gedai.wavelets_fits:
    if wf["ignore"] or not wf["sensai_runs"]:
        continue
    # Find the run whose eigen-threshold matches the chosen threshold
    chosen = min(wf["sensai_runs"], key=lambda r: abs(r[0] - wf["threshold"]))
    band_var = float(wf.get("enova", 0.0))   # use enova as proxy for band weight
    total_sensai += chosen[1]                 # sensai_score at chosen threshold
    total_weight += 1.0
total_sensai = total_sensai / total_weight if total_weight > 0 else 0.0

# Total ENOVA: global variance of removed noise / variance of original
var_before = float(data_before.var())
enova_total = float((data_before - data_after).var() / var_before) if var_before > 0 else 0.0

print(f"\n{'='*45}")
print(f"  Total SENSAI score : {total_sensai:.4f}")
print(f"  Total ENOVA        : {enova_total * 100:.2f} %")
print(f"  Elapsed time       : {elapsed:.1f} s")
print(f"{'='*45}\n")

# ── Overlay comparison ────────────────────────────────────────────────────
print("Plotting before/after overlay (close window to exit)...", flush=True)
plot_mne_style_overlay_interactive(
    raw_before, raw_clean,
    title="CAUEEG — Before vs After Spectral GEDAI",
    duration=10.0,
)
