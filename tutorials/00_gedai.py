"""
GEDAI
=====

This tutorial demonstrates how to use GEDAI (Generalized Eigenvalue De-Artifacting 
Instrument) to denoise EEG data. GEDAI is an unsupervised denoising method based 
on leadfield filtering that separates brain signals from noise and artifacts.

"""

# %%
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from gedai import Gedai
from gedai.viz import plot_mne_style_overlay_interactive

# %% Load sample EEG data
subjects = [1]  # may vary
runs = [4, 8, 12]  # may vary
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
# ``GEDAI`` uses generalized eigenvalue decomposition to separate brain signals 
# from noise based on a leadfield covariance model.
# In this tutorial, we will focus on the default ``GEDAI`` implementation which
# uses broadband EEG data. Please refers to the documentation if you want to learn
# more about the Spectral GEDAI and how to use it for frequency-specific denoising.

gedai = Gedai()

# %%
# Model Fitting
# -------------
# The fitting process estimates the optimal threshold to distinguish between signal 
# and noise components. ``GEDAI`` can be fitted on :class:`~mne.io.Raw` or :class:`~mne.Epochs` objects.
# If raw data is used, it is internally segmented into epochs before fitting.
# The ``duration`` parameter controls the epoch length, and the ``overlap`` parameter
# controls the overlap between consecutive epochs.

duration = 2.0
overlap = 0.5

# %%
# Since ``GEDAI`` estimates the noise covariance from the data itself,
# we usually want ``bad`` segments (e.g., with large artifacts) and ``bad`` channels
# to be included in the fitting process. Unless you have specific requirements,
# we recommend keeping the default ``reject_by_annotation`` setting.

reject_by_annotation = False  # default

# %%
# The reference covariance defines what good data should look like.
# For now, only ``leadfield`` is supported, which loads a precomputed
# leadfield covariance matrix based on the standard 10-20 montage.
# Future versions may allow user-defined reference covariances, including
# custom montages and subject-specific leadfield matrices.

reference_cov = "leadfield"

# %%
# .. note::
#
#     If you want to test ``GEDAI`` on data that does not follow the standard 10-20
#     naming convention, you can use the :meth:`mne.io.Raw.interpolate_bads` method to
#     project your data to a standard 10-20 montage before applying ``GEDAI``.

# %%
# To determine the optimal threshold for separating signal and noise components,
# ``GEDAI`` uses the ``SENSAI`` algorithm. ``SENSAI`` is an unsupervised method that
# finds the optimal eigenvalue threshold that maximizes the similarity between
# the cleaned data and the reference covariance while minimizing the similarity
# between the removed data and the reference covariance.
# The ``noise_multiplier`` parameter controls the weight given to noise
# similarity compared to signal similarity.
# Higher values will prioritize keeping more brain signals, potentially at the
# expense of removing less noise.

noise_multiplier = 3.0

# %%
# The optimal threshold can be determined either by grid search (``gridsearch``)
# over possible threshold values or by optimizing a cost function (``optimize``).
# The resulting threshold should be similar in both cases, but the computational
# time may vary depending on your CPU capabilities.

sensai_method = "gridsearch"

# %%
# Fit the GEDAI model

gedai.fit_raw(
    raw,
    duration=duration,
    overlap=overlap,
    reject_by_annotation=reject_by_annotation,
    reference_cov=reference_cov,
    sensai_method=sensai_method,
    noise_multiplier=noise_multiplier,
    verbose=True,
)
# %%
# The plot shows the eigenvalue spectrum and the separation between signal
# and noise components. The vertical line indicates the optimal threshold
# determined by the ``SENSAI`` algorithm.

fig = gedai.plot_fit()
plt.show()

# %%
# ``SENSAI`` internally uses a custom scaling of the eigenvalues, called SENSAI scaling.
# Higher SENSAI threshold values correspond to more aggressive denoising.
# The ``signal similarity`` (blue curve) indicates how similar the cleaned data is to the
# reference covariance. In our example, we can see that initially, as the SENSAI threshold increases,
# the signal similarity also increases, indicating that artifactual components are being removed.
# However, after a certain point, the signal similarity starts to decrease, which may indicate that
# some brain signals are being removed as well.
# Conversely, the ``noise similarity`` (red curve) remains low up to a certain SENSAI threshold,
# indicating that the removed components are dissimilar to the reference covariance. However,
# beyond that point, the noise similarity starts to increase, suggesting that brain signals
# are being removed along with noise.
# The ``SENSAI score`` (black curve) combines both signal and noise similarities to provide
# an overall measure of denoising quality.

# %%
# Transform the Data (Denoising)
# ------------------------------
# Once fitted, the ``GEDAI`` model can be used to remove artifacts and noise from the data.
# The transform operation projects out the noise components while preserving
# the brain signals.

raw_corrected = gedai.transform_raw(raw, duration=duration, overlap=overlap, verbose=False)

# %%
# We can visualize the difference between the original and denoised data using
# an interactive plot. This allows you to inspect individual channels and see
# how ``GEDAI`` has removed artifacts while preserving neural signals.

plot_mne_style_overlay_interactive(raw, raw_corrected)
