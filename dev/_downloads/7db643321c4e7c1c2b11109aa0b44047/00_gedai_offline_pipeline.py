"""
Recommended Offline EEG Denoising Pipeline
==========================================

This tutorial provides a practical, end-to-end offline denoising workflow
using Adaptive Multiband GEDAI on EEG data.

The pipeline applies ``AdaptiveMultibandGedai`` with integrated broadband pre-cleaning:

1. Initial broadband GEDAI pass to remove gross artifacts.
2. Wavelet decomposition into adaptive frequency bands.
3. Band-specific SENSAI optimization and seamless cosine-overlap reconstruction.

Use this tutorial as a template and adapt only the data-loading block and
parameter values for your own dataset.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw

from gedai import AdaptiveMultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.viz import plot_mne_style_overlay_interactive

# %% Load sample EEG data
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)

# %%
# Preprocessing & Reference Handling
# GEDAI applies a consistent reference (default ``reference="average"``) across
# both the EEG data and leadfield/reference covariance, and also supports
# infinity ``"REST"``, linked mastoids/earlobes (``"M1+M2"``, ``"TP9+TP10"``,
# ``"A1+A2"``), or any single channel (e.g., ``"Cz"``). Flat online recording
# references (``0 uV``) are retained automatically during two-pass bad-channel
# detection, clinical/BIDS labels (e.g., ``"EEG Fp1-REF"``) are matched safely,
# and custom non-10-20 channels with 3D positions use 3D dipole projection.

# %%
# High-pass filtering before GEDAI usually improves covariance estimation by
# reducing slow drifts and non-stationarities.
raw.filter(l_freq=0.5, h_freq=None)

# %%
# Adaptive Multiband GEDAI Pipeline
# ---------------------------------
# We initialize ``AdaptiveMultibandGedai``. By default, ``broadband_pass=True``
# runs an initial broadband GEDAI pre-pass to remove gross artifacts before
# decomposing the signal into adaptive wavelet bands.
ad = AdaptiveMultibandGedai(
    wavelet_type="haar",
    wavelet_level="auto",
    cycles_per_wavelet=10,
    broadband_pass=True,
)

# Fit the model using continuous scalar optimization, closed-form adaptive Z3
# weighting ("auto"), and automatic two-pass bad-channel exclusion. For long
# recordings with non-stationary background EEG signal (e.g., sleep, epilepsy,
# fatigue), pass e.g. ``smoothing_window=30.0`` (seconds) to adapt thresholds
# smoothly over time.
ad.fit_raw(
    raw,
    strength="auto",
    reference="average",
    # smoothing_window=30.0,
    on_bad_channel="exclude",
    bad_channel_threshold="auto",
)

# Denoise, interpolate any excluded bad channels, and annotate any residual
# extreme-ENOVA 1 s windows with "BAD_gedai" (without deleting samples)
denoised_raw = ad.transform_raw(
    raw,
    interpolate_bads=bool(ad.bad_channels_),
    bad_epoch_threshold=0.85,
    verbose=False,
)

n_bad_ep = int(ad.bad_epochs_.sum()) if ad.bad_epochs_ is not None else 0

# Inspect bad-channel/bad-epoch QC and denoising metrics
print(ad.summary)
print(
    f"Excluded bad channels: {ad.bad_channels_} | "
    f"Flagged BAD_gedai epochs: {n_bad_ep} | "
    f"Mean ENOVA: {ad.metrics_['mean_enova'] * 100:.1f} % | "
    f"SENSAI score: {ad.metrics_['sensai_score']:.3f}"
)

# %%
# Plot per-epoch ENOVA (highlighting any ``BAD_gedai`` epochs above ``85%``)
# alongside a zoomed 6-second comparison on frontal channel ``Fp1`` (bandpass
# filtered to ``1-45 Hz`` for display).

enova_ep = ad.metrics_["enova_per_epoch"] * 100.0
t_ep = np.arange(len(enova_ep)) + 0.5
bad_ep = (
    np.asarray(ad.bad_epochs_, dtype=bool)
    if ad.bad_epochs_ is not None
    else np.zeros(len(enova_ep), dtype=bool)
)

raw_bp = raw.copy().filter(1.0, 45.0, verbose=False).get_data()
den_bp = denoised_raw.copy().filter(1.0, 45.0, verbose=False).get_data()
sfreq = float(raw.info["sfreq"])
fp1 = raw.ch_names.index("Fp1")
seg_len = int(round(6.0 * sfreq))
pk = int(np.argmax(np.abs(raw_bp[fp1])))
st = max(0, min(pk - seg_len // 3, raw_bp.shape[1] - seg_len))
sl = slice(st, st + seg_len)
t_seg = np.arange(st, st + seg_len) / sfreq

fig_qc, axes_qc = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)

axes_qc[0].bar(
    t_ep[~bad_ep],
    enova_ep[~bad_ep],
    width=0.8,
    color="#4e79a7",
    label="Normal",
)
if np.any(bad_ep):
    axes_qc[0].bar(
        t_ep[bad_ep],
        enova_ep[bad_ep],
        width=0.8,
        color="#e15759",
        label="BAD_gedai (> 85%)",
    )
axes_qc[0].axhline(85.0, color="#c62828", ls="--", lw=1.4, label="Threshold (85%)")
axes_qc[0].set_title("A. Per-Epoch ENOVA & BAD_gedai Epochs")
axes_qc[0].set_xlabel("Time (s)")
axes_qc[0].set_ylabel("ENOVA (%)")
axes_qc[0].set_ylim(0, 105)
axes_qc[0].legend(loc="upper right", fontsize=8.2, frameon=True)
axes_qc[0].grid(axis="y", alpha=0.25)

axes_qc[1].plot(t_seg, raw_bp[fp1, sl] * 1e6, color="#b0b0b0", lw=1.1, label="Raw Fp1")
axes_qc[1].plot(
    t_seg, den_bp[fp1, sl] * 1e6, color="#1b9e77", lw=1.5, label="GEDAI Denoised"
)
axes_qc[1].set_title("B. Frontal Fp1 Zoom (1-45 Hz Display)")
axes_qc[1].set_xlabel("Time (s)")
axes_qc[1].set_ylabel("Amplitude (uV)")
axes_qc[1].legend(loc="upper right", fontsize=8.2, frameon=True)
axes_qc[1].grid(alpha=0.25)
plt.show()

# %%
# Since GEDAI algorithm automatically set the reference to ``average``, you can
# reset the reference to the original channel after denoising to preserve the
# original reference scheme:
# ``denoised_raw.set_eeg_reference(ref_channels="Cz", copy=False)``

# %%
# Visualize the results
plot_mne_style_overlay_interactive(raw, denoised_raw, duration=15.0)

# %%
# SENSAI Subspace Similarity & Manifold Visualization
# ---------------------------------------------------
# We can evaluate and visualize the quality of the denoising using the SENSAI
# subspace projection. This displays the side-by-side Before/After subspace similarity
# projections, LDA decision boundary shading between signal and noise manifolds,
# and marginal distributions.
#
# The SENSAI figure summarizes how effectively GEDAI separated genuine brain
# activity (signal) from artifacts (noise) by comparing the spatial patterns of
# the epoched data to a theoretical brain model.
#
# Each plotted point represents a 1-second epoch in the original data (left panel),
# the denoised data (right panel, green points) and the removed noise (right panel,
# red dots).
#
# The Axes:
#
# - **Y-axis (SSI - Subspace Similarity Index)**: This measures how closely the
#   spatial topography of an epoch matches the theoretical brain model (the BEM
#   leadfield). A value closer to 1.0 (marked by the dashed yellow line) indicates
#   the activity is highly likely to be originating from the brain.
# - **X-axis (Epoch Power in dB)**: This represents the amplitude or strength of
#   the signal in that specific time window. Artifacts often (but not always)
#   have higher power than resting brain activity.
#
# Panels:
#
# - **Left Panel (Before Denoising)**: This displays your raw EEG epochs prior to
#   cleaning. The color gradient corresponds to the SSI score (yellow/green is
#   more brain-like, blue/purple is less brain-like). You will typically see a
#   wide spread of data here, where epochs with high power and low SSI are clear
#   indicators of prominent, non-brain artifacts (like blinks or gross muscle
#   movement).
# - **Middle Panel (After Denoising)**: This illustrates the core separation
#   achieved by the algorithm, dividing the data into two distinct clusters
#   (along with density distribution curves on the top and right borders):
#
#   - **Green dots (Signal)**: These are the components GEDAI identified as
#     genuine brain activity and kept. Notice how they cluster tightly near the
#     1.0 line, indicating high spatial similarity to the brain leadfield.
#   - **Red dots (Noise)**: These are the artifact components GEDAI removed.
#     They generally exhibit lower similarity to the brain leadfield and are
#     often scattered across a wider range of power levels.
# - **Far-Right Upper Panel (ENOVA-weighted PSD over channels)**: Compares the
#   0.5 Hz Welch spectrum of **Original**, **Signal**, and removed **Noise**
#   weighted across channels by removed variance (confirming delta/blink or
#   high-frequency EMG suppression while preserving physiological rhythms like
#   occipital alpha).
# - **Far-Right Lower Panel (ENOVA Scalp Topoplot)**: Maps the percentage of
#   variance removed at each electrode (``0-100%``) so spatial hotspots (e.g.,
#   frontal blinks or localized bad electrodes) stand out immediately.
#
# Sub-optimal Denoising Outcomes:
#
# - **Noise-in-the-Signal**: The Red (Noise) cluster contains some Green (Signal)
#   dots (i.e. under-cleaning, "noise" components were misclassified as "signal").
# - **Signal-in-the-Noise**: The Green (Signal) cluster contains some Red (Noise)
#   dots (i.e. over-cleaning, "signal" components were misclassified as "noise").
#
# Key Metrics:
#
# - **SSI Silhouette Score**: This is a clustering metric that evaluates how cleanly
#   separated the "Signal" (green) and "Noise" (red) groups are along the SSI axis.
#   A score close to 1.0 (e.g., 0.97) represents excellent, distinct separation,
#   meaning the algorithm confidently isolated artifacts from brain signals.
# - **Mean SSSI (Signal Subspace Similarity Index)**: The average similarity score
#   of the retained brain data (you want this to be high).
# - **Mean NSSI (Noise Subspace Similarity Index)**: The average similarity score
#   of the rejected artifact data (you generally expect this to be much lower than
#   the SSSI).

fig, metrics = ad.plot_sensai(raw_before=raw, raw_after=denoised_raw)
