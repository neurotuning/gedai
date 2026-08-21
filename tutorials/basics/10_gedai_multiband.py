"""
Understanding the multiband extension of GEDAI
==============================================

This tutorial demonstrates how to use multiband ``GEDAI``.
``Multiband GEDAI`` is a frequency-specific denoising method that extends the
generalized eigenvalue decomposition approach of ``GEDAI``.
Its approach focuses on isolating and removing artifacts within specific
frequency bands. For that, the multiband ``GEDAI`` first decomposes the EEG
data into its frequency components using wavelet transform, then applies
``GEDAI`` to each frequency band separately. Finally, the denoised frequency
components are recombined to reconstruct the cleaned EEG signal.
"""

# %%
# .. note::
#
#     This purpose of this tutorial is to explain the differrent parameters of
#     the :class:`~gedai.gedai.MultibandGedai` model and help you better
#     understand the underlying algorithm. If you want to learn how to use
#     ``Multiband GEDAI`` in a practical, end-to-end offline denoising
#     workflow, please refer to the
#     :ref:`Practical Pipelines <sphx_glr_generated_tutorials_use>` section.
#
# %%
import matplotlib.pyplot as plt
from mne.io import read_raw

from gedai import MultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.viz import plot_mne_style_overlay_interactive

n_jobs = 1
# %% Load sample EEG data
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)

# %%
# For simplicity, we will only use the first 30 seconds of the data in this tutorial.
# In practice, it is recommended to use the full recording for fitting the GEDAI model,
# as this allows the model to better capture the noise characteristics of the data.

raw.crop(0, 30)

# %%
# GEDAI
# -----
# To use ``spectral GEDAI``, we initialize the
# :class:`~gedai.gedai.MultibandGedai` object. By default, ``wavelet_level="auto"``
# automatically determines the number of wavelet levels based on sampling frequency
# and the high-pass cutoff.
#
# Broadband Pass (Two-Pass Filtering):
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The ``broadband_pass=True`` parameter enables a two-stage hierarchical cleaning workflow:
#
# 1. **Pass 1 (Broadband Pass)**: A full-spectrum spatial GEDAI filter is applied first
#    to clean large, widespread artifacts (e.g. eye blinks, muscular bursts, head motion)
#    across all channels simultaneously. This prevents massive artifacts from leaking
#    across adjacent wavelet frequency subbands.
# 2. **Pass 2 (Multiband Wavelet Pass)**: The pre-cleaned signal is decomposed into
#    MODWT wavelet frequency bands, where each subband receives dedicated, fine-grained
#    eigenvalue thresholding tailored to its specific frequency dynamics.
#
# Enabling ``broadband_pass=True`` is recommended for recordings with prominent blinks
# or muscle contamination.

multiband_gedai = MultibandGedai(
    wavelet_type="haar",
    wavelet_level="auto",
    broadband_pass=True,
)

# %%
# Model Fitting
# -------------
# The fitting process of ``spectral GEDAI`` is performed for each wavelet level
# (i.e., frequency band) to estimate the optimal threshold to distinguish between
# signal and noise components using SENSAI optimization (``sensai_method="optimize"``).

multiband_gedai.fit_raw(raw, duration=2.0, n_jobs=n_jobs, verbose=True)


# %%
# .. note::
#
#       Since ``multiband GEDAI`` uses spectral decomposition, the fitting
#       process will automatically adjust the epoch duration to ensure that
#       each epoch contains a number of samples appropriate for the wavelet
#       decomposition.

# %%

fig = multiband_gedai.plot_fit()
plt.show()

# %%
# Transform the Data (Denoising)
# ------------------------------
# Once fitted, the ``Multiband GEDAI`` model cleans each frequency band using
# dual-stream cosine overlap-add blending (with 50% shifted streams) to eliminate
# any epoch boundary jump artifacts before recombining them.

denoised_raw = multiband_gedai.transform_raw(raw, n_jobs=n_jobs, verbose=False)

# %%
# .. warning::
#
#       Since the ``Multiband GEDAI`` operates on epoched data internally during fitting,
#       some frequency content more particularly in lower frequency bands may
#       be not be captured properly if the epoch duration is too short. On the
#       other hand, using very long epochs may prevent capturing short
#       transient artifacts. Setting the ``wavelet_low_cutoff`` parameter to ``"auto"``
#       or to the acquisition high-pass cutoff excludes low frequency bands
#       below the signal bandwidth.

plot_mne_style_overlay_interactive(raw, denoised_raw, duration=15.0)
