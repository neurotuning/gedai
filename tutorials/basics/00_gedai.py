"""
Understanding the GEDAI model
=============================

In this first tutorial, we introduce the GEDAI
(Generalized Eigenvalue De-Artifacting Instrument) model. GEDAI is an
unsupervised method for denoising EEG data.

.. note::

    This tutorial focuses on model understanding. For practical workflows,
    continue with tutorials from the ``use`` section (offline templates)
    and the ``advanced`` section (online and forward-model workflows).

"""

# %%
import matplotlib.pyplot as plt
from mne.io import read_raw

from gedai import Gedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.viz import plot_mne_style_overlay_interactive

# %% Load sample EEG data
# The GEDAI model can be fitted on :class:`mne.io.Raw` or :class:`mne.Epochs` objects.
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)

# %%
# For simplicity, we will only use the first 30 seconds of the data in this tutorial.
# In practice, it is recommended to use the full recording for fitting the GEDAI model,
# as this allows the model to better capture the noise characteristics of the data.

raw.crop(0, 30)

# %%
# GEDAI applies a consistent EEG reference to both the data and the underlying
# leadfield/reference covariance (default ``reference="average"``). It also
# supports infinity ``"REST"``, linked mastoids/earlobes (``"M1+M2"``,
# ``"TP9+TP10"``, ``"A1+A2"``), single-channel references (e.g., ``"Cz"``),
# or ``"none"``. Flat online recording reference channels (e.g., ``Cz = 0 uV``)
# are automatically retained during two-pass bad-channel cleaning.

reference = "average"  # or "REST", "M1+M2", "Cz", "none"
raw.set_eeg_reference("average", projection=False)

# %%
# GEDAI
# -----
# ``GEDAI`` uses generalized eigenvalue decomposition to separate brain signals
# from noise based on a leadfield covariance model.
# In this tutorial, we focus on the default broadband :class:`~gedai.gedai.Gedai`
# implementation. Please refer to the multiband and adaptive tutorials if you
# want to learn about :class:`~gedai.gedai.MultibandGedai` and
# frequency-specific denoising.

gedai = Gedai()

# %%
# Model Fitting
# -------------
# The fitting process estimates the optimal threshold to distinguish between
# signal and noise components. ``GEDAI`` can be fitted on :class:`mne.io.Raw`
# or :class:`mne.Epochs` objects.
# If raw data is used, it is internally segmented into epochs before fitting.
# The ``duration`` parameter controls the epoch length, and the ``overlap`` parameter
# controls the overlap between consecutive epochs.

duration = 2.0
overlap = 0.5

# %%
# Since ``GEDAI`` estimates the noise subspace directly from the data, keeping
# artifact-contaminated time segments (``reject_by_annotation=False``, default)
# helps ``GEDAI`` observe typical blink and muscle patterns.
# Conversely, persistently corrupted or flat **channels** distort the spatial
# average reference and subspace geometry, so ``on_bad_channel="exclude"`` and
# ``bad_channel_threshold="auto"`` detect and exclude bad channels automatically
# before fitting (you can restore them later with ``interpolate_bads=True``).

reject_by_annotation = False  # default
on_bad_channel = "exclude"  # default
bad_channel_threshold = "auto"

# %%
# The reference covariance defines what good brain topography should look like.
# The default ``"leadfield"`` option uses GEDAI's bundled 343-channel
# ``fsaverage`` OpenMEEG BEM model with token-safe clinical/BIDS label matching
# (e.g., ``"EEG Fp1-REF"``, ``"C3-M2"``), and automatically falls back to **3D
# dipole-kernel projection** when non-10-20 channels have valid 3D coordinates.
# Pass ``reference_cov="projection"`` to always use 3D dipole projection, or pass
# a custom :class:`mne.Forward` / :class:`mne.Covariance` (covered in the
# ``advanced`` forward-model tutorial).

reference_cov = "leadfield"

# %%
# To determine the optimal threshold for separating signal and noise components,
# ``GEDAI`` uses the ``SENSAI`` algorithm, maximizing the similarity between
# cleaned data and the reference covariance while minimizing the similarity
# between removed components and the reference covariance.
#
# The ``noise_multiplier`` parameter controls the trade-off between artifact
# rejection and signal preservation. Supported presets include ``"auto"``
# (default), ``"auto+"`` (more aggressive), and ``"auto-"`` (more
# conservative), each computed automatically in closed form from GEVD spectral
# invariants via Z3 weighting (or pass any positive float such as ``3.0``).

noise_multiplier = "auto"

# %%
# The optimal threshold can be determined either using continuous bounded scalar
# optimization (``"optimize"``, default) or by evaluating a discrete grid of candidate
# thresholds (``"gridsearch"``).

sensai_method = "optimize"

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
    on_bad_channel=on_bad_channel,
    bad_channel_threshold=bad_channel_threshold,
    reference=reference,
    verbose=True,
)
# %%
# The plot shows the eigenvalue spectrum and the separation between signal
# and noise components. The vertical line indicates the optimal threshold
# determined by the ``SENSAI`` algorithm.

fig = gedai.plot_fit()
plt.show()

# %%
# ``SENSAI`` internally uses a custom scaling of the eigenvalues, called
# SENSAI scaling. Higher SENSAI threshold values correspond to more aggressive
# denoising. The ``signal similarity`` (blue curve) indicates how similar the
# cleaned data is to the reference covariance. In our example, we can see that
# initially, as the SENSAI threshold increases, the signal similarity also
# increases, indicating that artifactual components are being removed.
# However, after a certain point, the signal similarity starts to decrease,
# which may indicate that some brain signals are being removed as well.
# Conversely, the ``noise similarity`` (red curve) remains low up to a certain
# SENSAI threshold, indicating that the removed components are dissimilar to
# the reference covariance. However, beyond that point, the noise similarity
# starts to increase, suggesting that brain signals are being removed along
# with noise. The ``SENSAI score`` (black curve) combines both signal and noise
# similarities to provide an overall measure of denoising quality.

# %%
# Transform the Data (Denoising)
# ------------------------------
# Once fitted, the ``GEDAI`` model can be used to remove artifacts and noise
# from the data. The transform operation projects out the noise components
# while preserving the brain signals, and populates ``gedai.metrics_``.

denoised_raw = gedai.transform_raw(
    raw,
    overlap=overlap,
    interpolate_bads=bool(gedai.bad_channels_),
    verbose=False,
)
print(gedai.summary)
print(
    f"Fit SENSAI score: {gedai.fit_metrics_['sensai_score']:.3f} | "
    f"Mean ENOVA: {gedai.metrics_['mean_enova'] * 100:.1f} % | "
    f"Excluded bad channels: {gedai.bad_channels_}"
)

# %%
# Finally, we can evaluate goodness-of-fit visually with the SENSAI manifold
# plot (Subspace Similarity Index vs. epoch power) and inspect channel overlays.

gedai.plot_sensai(raw_before=raw, raw_after=denoised_raw, show=True)
plot_mne_style_overlay_interactive(raw, denoised_raw, duration=15.0)
