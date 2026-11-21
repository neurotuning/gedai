"""
Denoising with a Custom Forward Model Covariance
================================================

By default, GEDAI builds its cortical signal subspace from a built-in generic
10-20 head model (``reference_cov="leadfield"``). You should supply a custom
:class:`mne.Forward` solution (or precomputed :class:`mne.Covariance`) when:

- Your EEG montage uses non-10-20 or high-density electrode locations.
- You have subject-specific MRI anatomy, a pediatric/animal head model, or
  customized sensor placements.
- You are denoising MEG or OPM data (covered in
  :ref:`tut-gedai-opm-processing`).

This tutorial builds a custom forward model from ``fsaverage``, computes its
sensor covariance with
:func:`~gedai.covariance.compute_covariance_from_forward`, and runs
:class:`~gedai.gedai.MultibandGedai` on real EEG data.
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import fetch_fsaverage
from mne.io import read_raw

from gedai import MultibandGedai
from gedai.covariance import compute_covariance_from_forward
from gedai.data import get_contaminated_eeg_set_path

# %%
# 1. Load MRI Template & EEG Recording
# ------------------------------------
# We fetch MNE's ``fsaverage`` template anatomy and crop the sample recording
# (27 EEG channels at ``200 Hz``) to ``30 s`` for fast execution.

fs_dir = fetch_fsaverage(verbose=False)
subject = "fsaverage"
trans = "fsaverage"  # MNE's built-in fsaverage head-to-MRI transformation
src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True, verbose=False)
raw.crop(0, 30)
raw.set_montage("standard_1005")
raw.set_eeg_reference("average", projection=False, verbose=False)

# %%
# 2. Inspect Coregistration & Build the Forward Covariance
# --------------------------------------------------------
# Verify that the EEG electrode locations are aligned with the head and cortex
# surfaces before computing the leadfield.

mne.viz.plot_alignment(
    raw.info,
    src=src,
    eeg=["projected"],
    trans=trans,
    show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
)

# %%
# Compute the forward solution and derive the cortical leadfield covariance.
# Note that ``fit_raw(..., reference_cov=fwd)`` also accepts an
# :class:`mne.Forward` object directly; calling
# :func:`~gedai.covariance.compute_covariance_from_forward` explicitly lets you
# inspect, save (``mne.write_cov``), and reuse the covariance across sessions.

fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1
)
reference_cov = compute_covariance_from_forward(fwd)
print(
    f"Computed forward covariance: {reference_cov.data.shape} "
    f"across {len(reference_cov.ch_names)} EEG channels"
)

# %%
# 3. Denoise with Custom Forward Covariance vs. Generic Leadfield
# ---------------------------------------------------------------
# We fit and transform :class:`~gedai.gedai.MultibandGedai` with our custom
# ``reference_cov`` and compare the cleaned output against the default generic
# ``reference_cov="leadfield"``.

common_fit_kwargs = dict(
    duration=2.0,
    overlap=0.75,
    noise_multiplier="auto",
    wavelet_low_cutoff=1.0,
    on_bad_channel="exclude",
    bad_channel_threshold="auto",
    n_jobs=1,
    verbose=False,
)

gedai_fwd = MultibandGedai(
    wavelet_type="haar", wavelet_level="auto", broadband_pass=True
)
gedai_fwd.fit_raw(raw, reference_cov=reference_cov, **common_fit_kwargs)
clean_fwd = gedai_fwd.transform_raw(
    raw, interpolate_bads=bool(gedai_fwd.bad_channels_), n_jobs=1, verbose=False
)

gedai_generic = MultibandGedai(
    wavelet_type="haar", wavelet_level="auto", broadband_pass=True
)
gedai_generic.fit_raw(raw, reference_cov="leadfield", **common_fit_kwargs)
clean_generic = gedai_generic.transform_raw(
    raw, interpolate_bads=bool(gedai_generic.bad_channels_), n_jobs=1, verbose=False
)

raw_bp = raw.copy().filter(1.0, 40.0, verbose=False).get_data()
fwd_bp = clean_fwd.copy().filter(1.0, 40.0, verbose=False).get_data()
gen_bp = clean_generic.copy().filter(1.0, 40.0, verbose=False).get_data()

r_models = float(np.corrcoef(fwd_bp.ravel(), gen_bp.ravel())[0, 1])
print(f"Mean ENOVA (Custom Forward):    {gedai_fwd.metrics_['mean_enova'] * 100:.1f} %")
print(
    f"Mean ENOVA (Generic Leadfield): "
    f"{gedai_generic.metrics_['mean_enova'] * 100:.1f} %"
)
print(
    f"Waveform correlation between custom fwd and generic leadfield: R = {r_models:.4f}"
)

# %%
# Visualize the normalized custom forward covariance (left) alongside a
# ``6 s`` frontal ``Fp1`` comparison trace (right).

diag = np.sqrt(np.maximum(np.diag(reference_cov.data), 1e-30))
corr_cov = reference_cov.data / np.outer(diag, diag)

sfreq = float(raw.info["sfreq"])
fp1_idx = raw.ch_names.index("Fp1")
seg_len = int(round(6.0 * sfreq))
peak_idx = int(np.argmax(np.abs(raw_bp[fp1_idx])))
st = max(0, min(peak_idx - seg_len // 3, raw_bp.shape[1] - seg_len))
sl = slice(st, st + seg_len)
t_seg = np.arange(st, st + seg_len) / sfreq

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)

im = axes[0].imshow(corr_cov, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
axes[0].set_title("Normalized Forward Covariance (27 x 27)")
axes[0].set_xlabel("Channel Index")
axes[0].set_ylabel("Channel Index")
fig.colorbar(im, ax=axes[0], shrink=0.82, label="Leadfield Correlation")

axes[1].plot(
    t_seg,
    raw_bp[fp1_idx, sl] * 1e6,
    color="#b0b0b0",
    linewidth=1.1,
    alpha=0.85,
    label="Raw EEG (1-40 Hz)",
)
axes[1].plot(
    t_seg,
    gen_bp[fp1_idx, sl] * 1e6,
    color="#377eb8",
    linewidth=1.6,
    label="Generic Leadfield GEDAI",
)
axes[1].plot(
    t_seg,
    fwd_bp[fp1_idx, sl] * 1e6,
    color="#1b9e77",
    linestyle="--",
    linewidth=1.6,
    label="Custom fsaverage Forward GEDAI",
)
axes[1].set_title(f"Frontal Fp1 Denoising Comparison (R = {r_models:.3f})")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude (uV)")
axes[1].legend(loc="upper right", fontsize=8.5, frameon=True)
axes[1].grid(alpha=0.25)
plt.show()
