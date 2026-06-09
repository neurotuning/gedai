"""
The recommended pipeline for offline EEG denoising using GEDAI
==============================================================

This tutorial serves as a template for offline EEG denoising.
"""

# %%
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw

from gedai import Gedai, AdaptativeMultibandGedai
from gedai.viz import plot_mne_style_overlay_interactive

n_jobs = -1

# %% Load sample EEG data
raw_path = "../gedai/data/SNR=0.35481 contamination=25 clean_EEG_dataset_2.set + EOG_EMG_NOISE_dataset_1.set"
raw = read_raw(raw_path, preload=True)

# %% 
# GEDAI will automatically apply an average reference before fitting or transforming the data.
# If the data was referenced to a different reference during acquisition, it is recommended 
# to add the reference channel to the data before using GEDAI. This way
# the rank of the data will be preserved.
# See :func:`mne.add_reference_channels` for more details.


# %% Filtering
# It is recommended to apply a high-pass filter to the data before using GEDAI, 
# as this can help to remove slow drifts and non-stationarities that can 
# reduce the effectiveness of the GEDAI algorithm.
raw.filter(l_freq=0.5, h_freq=None, n_jobs=n_jobs)


# %%
# The standard GEDAI pipeline uses two GEDAI models:
# 1. A broadband GEDAI model that operates on the full frequency spectrum of the EEG data. This model is effective at removing large artifacts that are present across a wide range of frequencies
# 2. A multiband GEDAI model that operates on specific frequency bands of the EEG data. This model is effective at removing artifacts that are present in specific frequency bands,

# %%
# 1. Broadband GEDAI
# ------------------
broadband_gedai = Gedai()
broadband_gedai.fit_raw(raw, noise_multiplier=6.0, n_jobs=n_jobs)
broadband_denoised_raw = broadband_gedai.transform_raw(raw, n_jobs=n_jobs, verbose=False)


# %%
# 2. Multiband GEDAI
# -------------------
# The fitting process of ``spectral GEDAI`` is similar to that of the standard
# ``GEDAI``. For each wavelet level (i.e., frequency band), the fitting process
# estimates the optimal threshold to distinguish between signal and noise
# components.

adaptive_multiband_gedai = AdaptativeMultibandGedai(
    wavelet_type="haar", wavelet_level=5, cycles_per_wavelet=10
)
adaptive_multiband_gedai.fit_raw(broadband_denoised_raw, noise_multiplier=3.0, n_jobs=n_jobs)
adaptive_multiband_denoised_raw = adaptive_multiband_gedai.transform_raw(
    broadband_denoised_raw, n_jobs=n_jobs, verbose=False
)

# %% Visualization
plot_mne_style_overlay_interactive(raw, adaptive_multiband_denoised_raw, duration=15.0)