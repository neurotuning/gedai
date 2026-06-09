"""
Understanding the spectral extension of GEDAI 
=============================================

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
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from gedai import Gedai, MultibandGedai
from gedai.viz import plot_mne_style_overlay_interactive

n_jobs = -1
# %% Load sample EEG data
subjects = [1]  # may vary
runs = [4]  # may vary
raw_fnames = eegbci.load_data(subjects, runs, update_path=True)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
# Concatenate runs from the same subject
raw = concatenate_raws(raws)
# Make channel names follow standard conventions
eegbci.standardize(raw)

# Crop to the first 15 seconds for demonstration purposes
# (Remove or adjust this for full data analysis)
raw.crop(0, 15)
raw.pick("eeg").load_data().apply_proj()

# Apply average reference (standard preprocessing for EEG)
raw.set_eeg_reference("average", projection=False)

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

multiband_gedai.fit_raw(raw,
                        duration=2.0,
                        sensai_method="gridsearch",
                        n_jobs=n_jobs,
                        verbose=True)
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

denoised_raw = multiband_gedai.transform_raw(raw,
                                             n_jobs=n_jobs,
                                             verbose=False)

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


# %%
# Recommended pipeline
# --------------------
#
# For optimal results, we recommend to first fit the standard ``GEDAI`` on
# broadband data with a conservative ``noise_multiplier`` (e.g., ``6.0``) to
# preserve most neural signals while only removing large artifacts. Then, use
# the resulting cleaned data to fit the ``Multiband GEDAI`` model. This two-step
# approach leverages the strengths of both methods, ensuring effective artifact
# removal while maintaining the integrity of neural signals across different
# frequency bands.

broadband_gedai = Gedai()
broadband_gedai.fit_raw(raw,
                        n_jobs=n_jobs,
                        noise_multiplier=6.0)
broadband_denoised_raw = broadband_gedai.transform_raw(raw, n_jobs=n_jobs, verbose=False)

multiband_gedai = MultibandGedai(
    wavelet_type="haar", wavelet_level=8
)
multiband_gedai.fit_raw(broadband_denoised_raw, n_jobs=n_jobs, noise_multiplier=3.0, wavelet_low_cutoff=2)
multiband_denoised_raw = multiband_gedai.transform_raw(
    broadband_denoised_raw, n_jobs=n_jobs, verbose=False
)

plot_mne_style_overlay_interactive(raw, multiband_denoised_raw)
