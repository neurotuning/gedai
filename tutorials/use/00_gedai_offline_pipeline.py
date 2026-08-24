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
# and marginal distributions:

fig, metrics = ad.plot_sensai(raw_before=raw, raw_after=denoised_raw)
