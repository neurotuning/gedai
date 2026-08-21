"""
Understanding the adaptive extension of multiband GEDAI
=======================================================

This tutorial demonstrates how to use ``Adaptive Multiband GEDAI``.
The method tackles the limitations of the standard multiband ``GEDAI``
by automatically determining the optimal epoch duration for each
band (i.e., wavelet level) based on the frequency content of the band.
By doing so, it allows to capture both transient and sustained artifacts
across different frequency ranges.
"""

# %%
# .. note::
#
#     This purpose of this tutorial is to explain the differrent parameters of
#     the :class:`~gedai.gedai.AdaptiveMultibandGedai` model and help you
#     better understand the underlying algorithm. If you want to learn how to
#     use ``Adaptive Multiband GEDAI`` in a practical, end-to-end offline
#     denoising workflow, please refer to the
#     :ref:`Practical Pipelines <sphx_glr_generated_tutorials_use>` section.

# %%
from mne.io import read_raw

from gedai import AdaptiveMultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.metrics import compute_enova_per_epoch, enova_summary
from gedai.viz import plot_mne_style_overlay_interactive

n_jobs = 1

# %% Load sample EEG data
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)
raw.filter(l_freq=0.5, h_freq=None, n_jobs=n_jobs)

# %%
# For simplicity, we will only use the first 30 seconds of the data in this
# tutorial. In practice, it is recommended to use the full recording for
# fitting the GEDAI model, as this allows the model to better capture the noise
# characteristics of the data.

raw.crop(0, 30)

# %%
# GEDAI Adaptive Multiband model
# --------------------------------
# The ``AdaptiveMultibandGedai`` model uses wavelet decomposition to
# separate the EEG data into different frequency bands and applies GEDAI
# separately to each band.
#
# The wavelet decomposition is controlled by:
# - ``wavelet_type``: The wavelet family (default: ``"haar"``).
# - ``wavelet_level``: Number of decomposition levels (default: ``"auto"``).
# - ``broadband_pass``: Whether to run an initial broadband GEDAI pass
#   (default: ``True``).

wavelet_type = "haar"

# %%
# The wavelet level (``wavelet_level``) controls the number of frequency
# bands that the data is decomposed into.
# When set to ``"auto"`` (default), the optimal level is computed automatically
# based on the sampling frequency and the high-pass cutoff.
# For example, for a sampling frequency of 200 Hz, 9 wavelet levels
# provide wavelet bands covering classical EEG frequency bands:
#
#  - (0.00 - 0.20 Hz)
#  - (0.20 - 0.39 Hz)
#  - (0.39 - 0.78 Hz)
#  - (0.78 - 1.56 Hz)
#  - (1.56 - 3.12 Hz) Delta
#  - (3.12 - 6.25 Hz) Theta
#  - (6.25 - 12.5 Hz) Alpha
#  - (12.5 - 25 Hz) Beta
#  - (25 - 50 Hz) Gamma
#  - (50 - 100 Hz) High Gamma
#  - (100 - 200 Hz)

wavelet_level = "auto"

# %%
# For each wavelet level, ``AdaptiveMultibandGedai`` automatically
# determines the optimal epoch duration. Slower frequency bands are
# estimated on longer epochs, while faster frequency bands are estimated on
# shorter epochs.
# This adaptive approach allows both transient and sustained artifacts to be
# captured across different frequency ranges.
# The ``cycles_per_wavelet`` parameter controls the number of cycles of the
# wavelet included in each epoch (default: ``10``).

cycles_per_wavelet = 10

# %%
# With these parameters defined, we instantiate the model.
# Enabling ``broadband_pass=True`` (default) runs an initial broadband GEDAI
# pass to eliminate gross artifacts before wavelet decomposition.
adaptive_multiband_gedai = AdaptiveMultibandGedai(
    wavelet_type=wavelet_type,
    wavelet_level=wavelet_level,
    cycles_per_wavelet=cycles_per_wavelet,
    broadband_pass=True,
)

# %%
# Model Fitting
# -------------
# The fitting process of ``AdaptiveMultibandGedai`` is performed separately for
# each wavelet level using continuous SENSAI optimization
# (``sensai_method="optimize"``).
#
# The ``wavelet_low_cutoff`` parameter controls which low-frequency bands are
# ignored (e.g. below high-pass filtering). Setting ``wavelet_low_cutoff="auto"``
# uses ``raw.info['highpass']`` to automatically exclude sub-cutoff levels.

wavelet_low_cutoff = "auto"
noise_multiplier = 3.0

# %%
# Fit the model directly on raw data:
adaptive_multiband_gedai.fit_raw(
    raw,
    noise_multiplier=noise_multiplier,
    wavelet_low_cutoff=wavelet_low_cutoff,
    n_jobs=n_jobs,
)

# %%
# The different wavelet parameters are stored in the
# ``adaptive_multiband_gedai._wavelets_fits`` attribute. The ``ignore`` key
# indicates which wavelet levels were ignored based on the
# ``wavelet_low_cutoff`` setting. The ``duration`` key indicates the epoch
# duration used to estimate the GEDAI model of the corresponding wavelet level.

print(adaptive_multiband_gedai._wavelets_fits)

# %%
# The wavelet model results can also be visualized using the ``plot_fit`` method:
adaptive_multiband_gedai.plot_fit()

# %%
# %%
# Transform the Data (Denoising)
# ------------------------------
# Denoising is performed seamlessly using continuous dual-stream cosine overlap-add
# blending across each band:
adaptive_multiband_denoised_raw = adaptive_multiband_gedai.transform_raw(
    raw, n_jobs=n_jobs
)

# %%
# Model Summary Table
# -------------------
# We can inspect the model fitting parameters and subband thresholds:
adaptive_multiband_gedai.fit_summary()

# %%
# Quality Evaluation: Explained Noise Variance (ENOVA)
# ----------------------------------------------------
# ENOVA (Explained Noise Variance) quantifies the proportion of signal variance
# removed as artifact:
#
#   ENOVA = var(noise) / var(original)
#
# - **Clean EEG segments**: ENOVA is near 0% (typically < 5-10%), indicating minimal
#   alteration of genuine brain activity.
# - **Artifact-contaminated segments**: ENOVA spikes to 50-95%, showing selective
#   rejection of high-power ocular, muscular, or movement artifacts.
#
# We can compute ENOVA across epochs and channels using the ``gedai.metrics`` module:

original_data = raw.get_data()
clean_data = adaptive_multiband_denoised_raw.get_data()
noise_data = original_data - clean_data

sfreq = raw.info["sfreq"]
epoch_samples = max(1, round(sfreq * 1.0))

enova_epochs = compute_enova_per_epoch(clean_data, noise_data, epoch_samples)
enova_stats = enova_summary(enova_epochs)

print(f"Mean ENOVA across epochs: {enova_stats['mean'] * 100:.2f} %")
print(f"Median ENOVA:             {enova_stats['median'] * 100:.2f} %")
print(f"Max ENOVA (peak artifact): {enova_stats['max'] * 100:.2f} %")

# %%
# Finally, we can visualize the before-and-after denoising overlay:
plot_mne_style_overlay_interactive(raw, adaptive_multiband_denoised_raw, duration=15.0)
