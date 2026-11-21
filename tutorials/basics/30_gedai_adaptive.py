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
#     The purpose of this tutorial is to explain the different parameters of
#     the :class:`~gedai.gedai.AdaptiveMultibandGedai` model and help you
#     better understand the underlying algorithm. If you want to learn how to
#     use ``Adaptive Multiband GEDAI`` in a practical, end-to-end offline
#     denoising workflow, please refer to the
#     :ref:`Practical Pipelines <sphx_glr_generated_tutorials_use>` section.

# %%
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw

from gedai import AdaptiveMultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.metrics import enova_summary
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
#
# - ``wavelet_type``: The wavelet family (default: ``"haar"``).
# - ``wavelet_level``: Number of decomposition levels (default: ``"auto"``).
# - ``broadband_pass``: Whether to run an initial broadband GEDAI pass
#   (default: ``True``).

wavelet_type = "haar"

# %%
# The wavelet level (``wavelet_level``) controls the number of frequency
# bands that the data is decomposed into.
# When set to ``"auto"`` (default), the optimal level is computed automatically
# based on the sampling frequency and the low cutoff.
# For example, at a sampling frequency of ``200 Hz`` (Nyquist limit ``100 Hz``),
# 9 decomposition levels produce 10 subbands (9 details + 1 scaling
# approximation):
#
#  - (0.00 - 0.10 Hz) Approximation (below cutoff -> ignored)
#  - (0.10 - 0.20 Hz)
#  - (0.20 - 0.39 Hz)
#  - (0.39 - 0.78 Hz)
#  - (0.78 - 1.56 Hz)
#  - (1.56 - 3.12 Hz) Delta
#  - (3.12 - 6.25 Hz) Theta
#  - (6.25 - 12.5 Hz) Alpha
#  - (12.5 - 25.0 Hz) Beta
#  - (25.0 - 50.0 Hz) Low Gamma
#
# Or more generally at sampling rate ``sfreq``, detail level ``j`` spans
# ``[sfreq / 2^(j + 1), sfreq / 2^j]`` up to the Nyquist frequency
# ``sfreq / 2``.

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
strength = "auto"

# %%
# Fit the model directly on raw data. Notice ``strength="auto"`` (equivalent to
# ``7.0`` on the ``0``-``10`` scale) computes closed-form adaptive Z3 spectral
# weighting while protecting rank-1/rank-2 dipolar brain rhythms. For long
# recordings with non-stationary background EEG signal (e.g., sleep, epilepsy,
# fatigue), pass ``smoothing_window=30.0`` (seconds) to adapt thresholds
# smoothly across time via ``makima`` interpolation, and use ``reference`` to
# select ``"average"``, ``"REST"``, linked mastoids, or a single electrode:
adaptive_multiband_gedai.fit_raw(
    raw,
    strength=strength,
    wavelet_low_cutoff=wavelet_low_cutoff,
    reference="average",
    on_bad_channel="exclude",
    bad_channel_threshold="auto",
    n_jobs=n_jobs,
)

# %%
# We can inspect the fitted subband summary table and overall fit SENSAI score
# via the public ``summary`` property and ``fit_metrics_`` attribute:

print(adaptive_multiband_gedai.summary)
print(
    f"Fit SENSAI score: {adaptive_multiband_gedai.fit_metrics_['sensai_score']:.3f} | "
    f"Bad channels excluded: {adaptive_multiband_gedai.bad_channels_}"
)

# %%
# The wavelet model results can also be visualized using the ``plot_fit`` method:
adaptive_multiband_gedai.plot_fit()
plt.show()

# %%
# Transform the Data (Denoising)
# ------------------------------
# Denoising is performed seamlessly using continuous dual-stream cosine overlap-add
# blending across each band. Passing ``bad_epoch_threshold=0.85`` non-destructively
# flags any residual extreme-ENOVA 1-second windows with MNE ``"BAD_gedai"``
# annotations and populates ``bad_epochs_``:
adaptive_multiband_denoised_raw = adaptive_multiband_gedai.transform_raw(
    raw,
    interpolate_bads=bool(adaptive_multiband_gedai.bad_channels_),
    bad_epoch_threshold=0.85,
    n_jobs=n_jobs,
)

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
# Rather than recomputing ENOVA manually, ``transform_raw`` automatically
# populates ``metrics_["enova_per_epoch"]`` and ``metrics_["enova_per_channel"]``:

enova_epochs = adaptive_multiband_gedai.metrics_["enova_per_epoch"]
enova_stats = enova_summary(enova_epochs)

print(f"Mean ENOVA across epochs:  {enova_stats['mean'] * 100:.2f} %")
print(f"Median ENOVA:              {enova_stats['median'] * 100:.2f} %")
print(f"Max ENOVA (peak artifact): {enova_stats['max'] * 100:.2f} %")
print(
    f"Max channel ENOVA:         "
    f"{adaptive_multiband_gedai.metrics_['enova_per_channel'].max() * 100:.2f} %"
)

n_bad_epochs = (
    int(adaptive_multiband_gedai.bad_epochs_.sum())
    if adaptive_multiband_gedai.bad_epochs_ is not None
    else 0
)
print(f"Flagged BAD_gedai epochs:  {n_bad_epochs}")

# %%
# Visualizing Adaptive Window Durations & Per-Epoch ENOVA
# -------------------------------------------------------
# The figure below summarizes two key behaviors of ``AdaptiveMultibandGedai``:
# (left) how each wavelet band is automatically assigned a frequency-scaled
# calibration window based on ``cycles_per_wavelet``, and (right) how per-epoch
# ENOVA tracks transient artifact bursts relative to ``bad_epoch_threshold``.

active_bands = [
    wf for wf in adaptive_multiband_gedai._wavelets_fits if not wf["ignore"]
]
band_labels = [f"{wf['fmin']:.2f}-{wf['fmax']:.2f} Hz" for wf in active_bands]
band_durs = [float(wf["duration"]) for wf in active_bands]

fig_qc, axes_qc = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)

y_pos = np.arange(len(active_bands))
bars = axes_qc[0].barh(y_pos, band_durs, color="#2b8cbe", edgecolor="black", alpha=0.9)
for bar, dur in zip(bars, band_durs, strict=True):
    axes_qc[0].text(
        dur + 0.08,
        bar.get_y() + bar.get_height() / 2,
        f"{dur:.2f} s",
        va="center",
        fontsize=8.3,
    )
axes_qc[0].set_yticks(y_pos, band_labels)
axes_qc[0].invert_yaxis()
axes_qc[0].set_xlim(0, max(band_durs) * 1.25)
axes_qc[0].set_xlabel("Adaptive Epoch Duration (s)")
axes_qc[0].set_title("A. Cycle-Adaptive Window per Wavelet Band")
axes_qc[0].grid(axis="x", alpha=0.25)

t_ep = np.arange(len(enova_epochs)) + 0.5
bad_ep = (
    np.asarray(adaptive_multiband_gedai.bad_epochs_, dtype=bool)
    if adaptive_multiband_gedai.bad_epochs_ is not None
    else np.zeros(len(enova_epochs), dtype=bool)
)
axes_qc[1].bar(
    t_ep[~bad_ep],
    enova_epochs[~bad_ep] * 100.0,
    width=0.8,
    color="#4e79a7",
    label="Normal epochs",
)
if np.any(bad_ep):
    axes_qc[1].bar(
        t_ep[bad_ep],
        enova_epochs[bad_ep] * 100.0,
        width=0.8,
        color="#e15759",
        label="BAD_gedai (> 85%)",
    )
axes_qc[1].axhline(85.0, color="#c62828", ls="--", lw=1.4, label="Threshold (85%)")
axes_qc[1].set_xlabel("Recording Time (s)")
axes_qc[1].set_ylabel("ENOVA (%)")
axes_qc[1].set_ylim(0, 105)
axes_qc[1].set_title("B. Per-Epoch ENOVA & BAD_gedai QC")
axes_qc[1].legend(loc="upper right", fontsize=8.3, frameon=True)
axes_qc[1].grid(axis="y", alpha=0.25)
plt.show()

# %%
# Next, inspect the 3-column SENSAI dashboard (SSI manifolds, ENOVA-weighted
# Welch PSD, and per-channel ENOVA scalp topoplot) alongside the interactive
# time-series overlay:
adaptive_multiband_gedai.plot_sensai(
    raw_before=raw, raw_after=adaptive_multiband_denoised_raw, show=True
)
plot_mne_style_overlay_interactive(raw, adaptive_multiband_denoised_raw, duration=15.0)
