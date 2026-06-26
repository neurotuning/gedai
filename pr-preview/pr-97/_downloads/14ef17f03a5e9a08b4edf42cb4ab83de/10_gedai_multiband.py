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

n_jobs = -1
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
# :class:`~gedai.gedai.MultibandGedai` object by specifying the
# ``wavelet_level`` parameter, which defines the number of frequency bands to
# decompose the EEG data into. Each level corresponds to a specific frequency
# band, allowing for targeted denoising within those bands. It is also possible
# to define the type of wavelet used for the decomposition by setting the
# ``wavelet_type`` parameter.

multiband_gedai = MultibandGedai(wavelet_type="haar", wavelet_level=8)

# %%
# Model Fitting
# -------------
# The fitting process of ``spectral GEDAI`` is similar to that of the standard
# ``GEDAI``. For each wavelet level (i.e., frequency band), the fitting process
# estimates the optimal threshold to distinguish between signal and noise
# components.

multiband_gedai.fit_raw(
    raw, duration=2.0, sensai_method="gridsearch", n_jobs=n_jobs, verbose=True
)
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
# Once fitted, the ``Multiband GEDAI`` model can be used to remove artifacts
# and noise from the data. The transform operation projects out the noise
# components while preserving the brain signals for each frequency band
# separately before recombining them.

denoised_raw = multiband_gedai.transform_raw(raw, n_jobs=n_jobs, verbose=False)

# %%
# .. warning::
#
#       Since the ``Multiband GEDAI`` operates on epoched data internally,
#       some frequency content more particularly in lower frequency bands may
#       be not be captured properly if the epoch duration is too short. On the
#       other hand, using very long epochs may prevent to capture short
#       transient artifacts. Setting the ``wavelet_low_cutoff`` parameter to a
#       value of the order of ``1 / epoch_duration`` can help mitigate this
#       issue by excluding lower frequency bands that may not be well
#       estimated during the fitting process.

plot_mne_style_overlay_interactive(raw, denoised_raw)
