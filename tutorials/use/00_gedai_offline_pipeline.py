"""
Recommended Offline EEG Denoising Pipeline
==========================================

This tutorial provides a practical, end-to-end offline denoising workflow
using GEDAI on EEG data.

The pipeline is intentionally simple:

1. Apply a broadband ``Gedai`` model to remove large artifacts.
2. Apply ``AdaptiveMultibandGedai`` for frequency-specific refinement.

Use this tutorial as a template and adapt only the data-loading block and
parameter values for your own dataset.
"""

# %%
from mne.io import read_raw

from gedai import AdaptiveMultibandGedai, Gedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.viz import plot_mne_style_overlay_interactive

n_jobs = -1

# %% Load sample EEG data
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)

# %%
# GEDAI will automatically apply an average reference before fitting or
# transforming the data. If your acquisition used a different reference,
# consider adding the missing reference channel beforehand to preserve rank.


# %% Filtering
# High-pass filtering before GEDAI usually improves covariance estimation by
# reducing slow drifts and non-stationarities.
raw.filter(l_freq=0.5, h_freq=None, n_jobs=n_jobs)


# %%
# The standard offline pipeline uses two stages:
# 1. Broadband GEDAI for large broadband artifacts.
# 2. Adaptive multiband GEDAI for frequency-specific cleanup.


# %%
# 1. Broadband GEDAI
# ------------------
broadband_gedai = Gedai()
broadband_gedai.fit_raw(raw, noise_multiplier=6.0, n_jobs=n_jobs)
broadband_denoised_raw = broadband_gedai.transform_raw(raw, n_jobs=n_jobs, verbose=False)


# %%
# 2. Adaptive Multiband GEDAI
# ---------------------------
adaptive_multiband_gedai = AdaptiveMultibandGedai(
    wavelet_type="haar", wavelet_level=5, cycles_per_wavelet=10
)
adaptive_multiband_gedai.fit_raw(
    broadband_denoised_raw, noise_multiplier=3.0, n_jobs=n_jobs
)
adaptive_multiband_denoised_raw = adaptive_multiband_gedai.transform_raw(
    broadband_denoised_raw, n_jobs=n_jobs, verbose=False
)

# %% Visualization
plot_mne_style_overlay_interactive(raw, adaptive_multiband_denoised_raw, duration=15.0)
