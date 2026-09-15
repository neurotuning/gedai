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
from mne.io import read_raw

from gedai import AdaptiveMultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.viz import plot_mne_style_overlay_interactive

# %% Load sample EEG data
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)

# %%
# Preprocessing
# GEDAI will automatically apply an average reference before fitting or
# transforming the data. If your acquisition used a different reference,
# consider adding the missing reference channel beforehand to preserve
# the rank of your data. For example, if your data was recorded with a
# ``Cz`` reference, you can add a virtual ``Cz`` channel as follows:
# ``raw.add_reference_channels("Cz", copy=False)``.

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

# Fit the model (using continuous scalar optimization by default)
ad.fit_raw(raw, noise_multiplier=3.0)

# Denoise the data
denoised_raw = ad.transform_raw(raw, verbose=False)

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
# - **Right Panel (After Denoising)**: This illustrates the core separation
#   achieved by the algorithm, dividing the data into two distinct clusters
#   (along with density distribution curves on the top and right borders):
#
#   - **Green dots (Signal)**: These are the components GEDAI identified as
#     genuine brain activity and kept. Notice how they cluster tightly near the
#     1.0 line, indicating high spatial similarity to the brain leadfield.
#   - **Red dots (Noise)**: These are the artifact components GEDAI removed.
#     They generally exhibit lower similarity to the brain leadfield and are
#     often scattered across a wider range of power levels.
#
# Sub-optimal Denoising Outcomes:
#
# - **Noise-in-the-Signal**: The Red (Noise) cluster contains some Green (Signal)
#   dots (i.e. under-cleaning, "noise" components were missclassified as "signal").
# - **Signal-in-the-Noise**: The Green (Signal) cluster contains some Red (Noise)
#   dots (i.e. over-cleaning, "signal" components were missclassified as "noise").
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
