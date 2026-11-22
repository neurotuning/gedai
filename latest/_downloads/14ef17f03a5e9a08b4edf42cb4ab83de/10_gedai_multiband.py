"""
Understanding the multiband extension of GEDAI
==============================================

This tutorial demonstrates how to use :class:`~gedai.gedai.MultibandGedai` and
why splitting cleaning across frequency subbands improves over a single
broadband GED pass.

``MultibandGedai`` decomposes each channel into wavelet subbands using the
Maximal Overlap Discrete Wavelet Transform (MODWT), fits an independent GEDAI
threshold per subband, and reconstructs the cleaned signal.
"""

# sphinx_gallery_thumbnail_number = 3

# %%
# .. note::
#
#     This tutorial focuses on the parameters and behavior of
#     :class:`~gedai.gedai.MultibandGedai`. For a ready-to-copy offline
#     pipeline template, see the
#     :ref:`Practical Pipelines <sphx_glr_generated_tutorials_use>` section.

# %%
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw

from gedai import Gedai, MultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.viz import plot_mne_style_overlay_interactive

n_jobs = 1
# %% Load sample EEG data
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)

# %%
# For simplicity, we use only the first 30 seconds of the data in this tutorial.
# In practice, fitting on the full recording helps capture rare artifacts.

raw.crop(0, 30)

# %%
# MultibandGedai Parameters
# -------------------------
# By default, ``wavelet_level="auto"`` computes the number of MODWT levels from
# the sampling frequency and low-frequency cutoff.
#
# Broadband Pass (Two-Pass Filtering):
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Setting ``broadband_pass=True`` enables a two-stage hierarchical workflow:
#
# 1. **Pass 1 (Broadband Pass)**: A full-spectrum spatial GEDAI filter removes
#    large, widespread artifacts (such as eye blinks or gross jaw bursts) so
#    high-amplitude energy does not leak into neighboring wavelet bands.
# 2. **Pass 2 (Multiband Wavelet Pass)**: Each MODWT subband receives its own
#    SENSAI eigenvalue threshold tuned to the spectral mix of that band.

multiband_gedai = MultibandGedai(
    wavelet_type="haar",
    wavelet_level="auto",
    broadband_pass=True,
)

# %%
# Model Fitting
# -------------
# :meth:`~gedai.gedai.MultibandGedai.fit_raw` segments each wavelet band into
# fixed ``2.0 s`` epochs, optimizes band-specific SENSAI thresholds, and
# automatically excludes bad channels via ``on_bad_channel="exclude"``.

multiband_gedai.fit_raw(
    raw,
    duration=2.0,
    strength="auto",
    on_bad_channel="exclude",
    bad_channel_threshold="auto",
    n_jobs=n_jobs,
    verbose=True,
)
print(multiband_gedai.summary)

# %%
# .. note::
#
#       The Haar MODWT operates on arbitrary even epoch sample counts (as long as
#       there are enough samples per epoch for covariance estimation) without
#       requiring power-of-two window lengths. Setting ``wavelet_low_cutoff`` to
#       ``"auto"`` or to your high-pass cutoff skips subbands below the signal
#       bandwidth. ``MultibandGedai`` also supports ``reference`` (``"average"``,
#       ``"REST"``, ``"M1+M2"``, ``"Cz"``), adaptive Z3 ``strength`` presets
#       (``"auto"`` = ``7``, ``"auto+"`` = ``8``, ``"auto-"`` = ``6``, or
#       ``0``-``10``), ``smoothing_window``, and ``bad_epoch_threshold``
#       (``"BAD_gedai"``).

# %%

fig = multiband_gedai.plot_fit()
plt.show()

# %%
# Transform the Data (Denoising)
# ------------------------------
# :meth:`~gedai.gedai.MultibandGedai.transform_raw` denoises each subband with
# dual-stream cosine overlap-add blending (50% shifted streams) to eliminate
# boundary discontinuities before recombining the bands.

denoised_raw = multiband_gedai.transform_raw(
    raw,
    interpolate_bads=bool(multiband_gedai.bad_channels_),
    n_jobs=n_jobs,
    verbose=False,
)
print(
    f"MultibandGedai -> Mean ENOVA: "
    f"{multiband_gedai.metrics_['mean_enova'] * 100:.1f} % | "
    f"SENSAI score: {multiband_gedai.metrics_['sensai_score']:.3f}"
)

# %%
# Why Multiband Beats Single-Band Broadband GEDAI
# -----------------------------------------------
# Low-frequency artifacts (blinks, slow movement drifts) and high-frequency
# artifacts (EMG, line harmonics) have completely different spatial-to-brain
# power ratios. A single broadband :class:`~gedai.gedai.Gedai` threshold must
# compromise across all frequencies at once, whereas ``MultibandGedai`` targets
# each subband independently.

broadband_gedai = Gedai()
broadband_gedai.fit_raw(
    raw,
    duration=2.0,
    strength="auto",
    on_bad_channel="exclude",
    bad_channel_threshold="auto",
    n_jobs=n_jobs,
    verbose=False,
)
broadband_denoised = broadband_gedai.transform_raw(
    raw,
    interpolate_bads=bool(broadband_gedai.bad_channels_),
    n_jobs=n_jobs,
    verbose=False,
)

raw_ref = raw.copy().set_eeg_reference("average", projection=False, verbose=False)
raw_bp = raw_ref.copy().filter(1.0, 45.0, verbose=False).get_data()
bb_bp = broadband_denoised.copy().filter(1.0, 45.0, verbose=False).get_data()
mb_bp = denoised_raw.copy().filter(1.0, 45.0, verbose=False).get_data()

sfreq = float(raw.info["sfreq"])
fp1_idx = raw.ch_names.index("Fp1")
seg_len = int(round(6.0 * sfreq))
peak_idx = int(np.argmax(np.abs(raw_bp[fp1_idx])))
st = max(0, min(peak_idx - seg_len // 3, raw_bp.shape[1] - seg_len))
sl = slice(st, st + seg_len)
t_seg = np.arange(st, st + seg_len) / sfreq

psd_kw = dict(fmin=1.0, fmax=45.0, verbose=False)
psd_raw = raw_ref.compute_psd(**psd_kw)
psd_bb = broadband_denoised.compute_psd(**psd_kw)
psd_mb = denoised_raw.compute_psd(**psd_kw)

fig_cmp, axes_cmp = plt.subplots(2, 1, figsize=(10.2, 6.2), constrained_layout=True)

axes_cmp[0].plot(
    t_seg,
    raw_bp[fp1_idx, sl] * 1e6,
    color="#b0b0b0",
    linewidth=1.1,
    alpha=0.85,
    label="Raw EEG (1-45 Hz)",
)
axes_cmp[0].plot(
    t_seg,
    bb_bp[fp1_idx, sl] * 1e6,
    color="#d95f02",
    linewidth=1.5,
    label="Broadband Gedai",
)
axes_cmp[0].plot(
    t_seg,
    mb_bp[fp1_idx, sl] * 1e6,
    color="#1b9e77",
    linestyle="--",
    linewidth=1.6,
    label="MultibandGedai",
)
axes_cmp[0].set_title("Frontal Channel Fp1: Broadband vs. Multiband GEDAI")
axes_cmp[0].set_xlabel("Time (s)")
axes_cmp[0].set_ylabel("Amplitude (uV)")
axes_cmp[0].legend(loc="upper right", frameon=True)
axes_cmp[0].grid(alpha=0.25)

for label, psd_obj, color, ls in [
    ("Raw EEG", psd_raw, "#b0b0b0", "-"),
    ("Broadband Gedai", psd_bb, "#d95f02", "-"),
    ("MultibandGedai", psd_mb, "#1b9e77", "--"),
]:
    mean_psd_db = 10.0 * np.log10(np.mean(psd_obj.get_data(), axis=0) * 1e12 + 1e-30)
    axes_cmp[1].plot(
        psd_obj.freqs,
        mean_psd_db,
        color=color,
        linestyle=ls,
        linewidth=1.8,
        label=label,
    )

axes_cmp[1].set_title("Channel-Averaged Power Spectral Density (1-45 Hz)")
axes_cmp[1].set_xlabel("Frequency (Hz)")
axes_cmp[1].set_ylabel("Power (dB uV^2/Hz)")
axes_cmp[1].legend(loc="upper right", frameon=True)
axes_cmp[1].grid(alpha=0.25)
plt.show()

# %%
# SENSAI Manifold, ENOVA-Weighted PSD & Scalp Topography
# -------------------------------------------------------
# Calling ``plot_sensai`` displays Before/After epoch clustering against the
# brain leadfield subspace alongside the ENOVA-weighted Welch PSD and the
# spatial distribution of removed variance across channels.

multiband_gedai.plot_sensai(raw_before=raw, raw_after=denoised_raw, show=True)

# %%
# Interactive Overlay
# -------------------
plot_mne_style_overlay_interactive(raw, denoised_raw, duration=15.0)
