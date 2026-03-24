"""
.. _ex-opm-resting-state:

======================================================================
Compute source power spectral density (PSD) of VectorView and OPM data
======================================================================

Here we compute the resting state from raw for data recorded using
a Neuromag VectorView system and a custom OPM system.
The pipeline is meant to mostly follow the Brainstorm :footcite:`TadelEtAl2011`
`OMEGA resting tutorial pipeline
<https://neuroimage.usc.edu/brainstorm/Tutorials/RestingOmega>`__.
The steps we use are:

1. Filtering: downsample heavily.
2. Artifact detection: use SSP for EOG and ECG.
3. Source localization: dSPM, depth weighting, cortically constrained.
4. Frequency: power spectral density (Welch), 4 s window, 50% overlap.
5. Standardize: normalize by relative power for each source.

Preprocessing
-------------
"""
# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import mne
from mne.filter import next_fast_len

print(__doc__)

data_path = mne.datasets.opm.data_path()
subject = "OPM_sample"

subjects_dir = data_path / "subjects"
bem_dir = subjects_dir / subject / "bem"
bem_fname = bem_dir / f"{subject}-5120-5120-5120-bem-sol.fif"
src_fname = bem_dir / f"{subject}-oct6-src.fif"
vv_fname = data_path / "MEG" / "SQUID" / "SQUID_resting_state.fif"
vv_erm_fname = data_path / "MEG" / "SQUID" / "SQUID_empty_room.fif"
vv_trans_fname = data_path / "MEG" / "SQUID" / "SQUID-trans.fif"
opm_fname = data_path / "MEG" / "OPM" / "OPM_resting_state_raw.fif"
opm_erm_fname = data_path / "MEG" / "OPM" / "OPM_empty_room_raw.fif"
opm_trans = mne.transforms.Transform("head", "mri")  # use identity transform
opm_coil_def_fname = data_path / "MEG" / "OPM" / "coil_def.dat"

##############################################################################
# Load data, resample. We will store the raw objects in dicts with entries
# "vv" and "opm" to simplify housekeeping and simplify looping later.

raws = dict()
raw_erms = dict()
new_sfreq = 240.0  # Classical downsampling rate (Nyquist = 120 Hz)
raws["vv"] = mne.io.read_raw_fif(vv_fname, verbose="error")  # ignore naming
raws["vv"].load_data().resample(new_sfreq, method="polyphase")
raw_erms["vv"] = mne.io.read_raw_fif(vv_erm_fname, verbose="error")
raw_erms["vv"].load_data().resample(new_sfreq, method="polyphase")

raws["opm"] = mne.io.read_raw_fif(opm_fname)
raws["opm"].load_data().resample(new_sfreq, method="polyphase")
raw_erms["opm"] = mne.io.read_raw_fif(opm_erm_fname)
raw_erms["opm"].load_data().resample(new_sfreq, method="polyphase")
# Make sure our assumptions later hold
assert raws["opm"].info["sfreq"] == raws["vv"].info["sfreq"]

##############################################################################
# Alignment and forward (computed first — used as GEDAI reference covariance)
# ---------------------

# Here we use a reduced size source space (oct5) just for speed
src = mne.setup_source_space(subject, "oct5", add_dist=False, subjects_dir=subjects_dir)
# This line removes source-to-source distances that we will not need.
# We only do it here to save a bit of memory, in general this is not required.
del src[0]["dist"], src[1]["dist"]
bem = mne.read_bem_solution(bem_fname)
fwd = dict()

# Compute forward for VectorView
fwd["vv"] = mne.make_forward_solution(
    raws["vv"].info, vv_trans_fname, src, bem, eeg=False, verbose=True
)

# And for OPM:
with mne.use_coil_def(opm_coil_def_fname):
    fwd["opm"] = mne.make_forward_solution(
        raws["opm"].info, opm_trans, src, bem, eeg=False, verbose=True
    )

print("Visualizing VectorView coregistration...", flush=True)
fig_coreg_vv = mne.viz.plot_alignment(
    raws["vv"].info, subjects_dir=subjects_dir, subject=subject, trans=vv_trans_fname,
    surfaces={"head": 0.1, "inner_skull": 0.2, "white": 1.0},
    meg=["helmet", "sensors"], verbose="error", bem=bem_fname, src=src,
)
input("Press Enter to close the VectorView coregistration and view OPM...")
mne.viz.close_3d_figure(fig_coreg_vv)

print("Visualizing OPM coregistration...", flush=True)
with mne.use_coil_def(opm_coil_def_fname):
    fig_coreg_opm = mne.viz.plot_alignment(
        raws["opm"].info, subjects_dir=subjects_dir, subject=subject, trans=opm_trans,
        surfaces={"head": 0.1, "inner_skull": 0.2, "white": 1.0},
        meg="sensors", verbose="error", bem=bem_fname, src=src,
    )
input("Press Enter to close the OPM coregistration and start GEDAI denoising...")
mne.viz.close_3d_figure(fig_coreg_opm)

del src, bem

##############################################################################
# Apply GEDAI denoising — separately to VectorView gradiometers,
# VectorView magnetometers, and OPM channels, each using the forward model
# computed above as the reference covariance matrix.

from gedai import Gedai
from gedai.viz.compare import plot_mne_style_overlay_interactive

# --- VectorView: gradiometers ---
print("Applying GEDAI to VectorView gradiometers...", flush=True)
raw_vv_grad = raws["vv"].copy().pick("grad")
raw_vv_grad.filter(0.5, None, verbose=False)   # high-pass to stabilise epoch covariances
gedai_grad = Gedai(wavelet_level='auto', wavelet_low_cutoff=0.5, epoch_size_in_cycles=12, signal_type="meg", highpass_cutoff=None)
raw_vv_grad_clean = gedai_grad.fit_transform_raw(
    raw_vv_grad, reference_cov=fwd["vv"], noise_multiplier=3.0
)
grad_picks = mne.pick_types(raws["vv"].info, meg="grad")
raws["vv"]._data[grad_picks] = raw_vv_grad_clean.get_data()

# --- VectorView: magnetometers ---
print("Applying GEDAI to VectorView magnetometers...", flush=True)
raw_vv_mag = raws["vv"].copy().pick("mag")
raw_vv_mag.filter(0.5, None, verbose=False)    # high-pass to stabilise epoch covariances
gedai_mag = Gedai(wavelet_level='auto', wavelet_low_cutoff=0.5, epoch_size_in_cycles=12, signal_type="meg", highpass_cutoff=None)
raw_vv_mag_clean = gedai_mag.fit_transform_raw(
    raw_vv_mag, reference_cov=fwd["vv"], noise_multiplier=3.0
)
mag_picks = mne.pick_types(raws["vv"].info, meg="mag")
raws["vv"]._data[mag_picks] = raw_vv_mag_clean.get_data()

# --- OPM: magnetometer channels only (exclude stimulus/system channels) ---
print("Applying GEDAI to OPM channels...", flush=True)
raw_opm_meg = raws["opm"].copy().pick("mag")
raw_opm_meg.filter(0.5, None, verbose=False)   # high-pass to stabilise epoch covariances
gedai_opm = Gedai(wavelet_level='auto', wavelet_low_cutoff=0.5, epoch_size_in_cycles=12, signal_type="meg", highpass_cutoff=None)
raw_opm_clean = gedai_opm.fit_transform_raw(
    raw_opm_meg, reference_cov=fwd["opm"], noise_multiplier=3.0
)
opm_meg_picks = mne.pick_types(raws["opm"].info, meg=True)
raws["opm"]._data[opm_meg_picks] = raw_opm_clean.get_data()

##############################################################################
# Compare before vs after GEDAI (interactive time browser)
# Keys: ←/→ scroll · ↑/↓ scale · D=diff · N=denoised only · O=noisy only

print("Plotting GEDAI comparison (close each window to continue)...", flush=True)
plot_mne_style_overlay_interactive(
    raw_vv_grad, raw_vv_grad_clean,
    title="VectorView Gradiometers — Before vs After GEDAI", duration=10.0
)
plot_mne_style_overlay_interactive(
    raw_vv_mag, raw_vv_mag_clean,
    title="VectorView Magnetometers — Before vs After GEDAI", duration=10.0
)
plot_mne_style_overlay_interactive(
    raw_opm_meg, raw_opm_clean,
    title="OPM — Before vs After GEDAI", duration=10.0
)
del raw_vv_grad, raw_vv_grad_clean, raw_vv_mag, raw_vv_mag_clean, raw_opm_meg, raw_opm_clean

##############################################################################
# Explore data (post-GEDAI PSDs)

titles = dict(vv="VectorView", opm="OPM")
kinds = ("vv", "opm")
n_fft = next_fast_len(int(round(4 * new_sfreq)))
print(f"Using n_fft={n_fft} ({n_fft / raws['vv'].info['sfreq']:0.1f} s)")
for kind in kinds:
    fig = (
        raws[kind]
        .compute_psd(n_fft=n_fft, proj=False)
        .plot(picks="data", exclude="bads", amplitude=True)
    )
    fig.suptitle(titles[kind])

##############################################################################
# Compute and apply inverse to PSD estimated using multitaper + Welch.
# Group into frequency bands, then normalize each source point and sensor
# independently. This makes the value of each sensor point and source location
# in each frequency band the percentage of the PSD accounted for by that band.

freq_bands = dict(alpha=(8, 12), beta=(15, 29))
topos = dict(vv=dict(), opm=dict())
stcs = dict(vv=dict(), opm=dict())

snr = 3.0
lambda2 = 1.0 / snr**2
for kind in kinds:
    noise_cov = mne.compute_raw_covariance(raw_erms[kind])
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        raws[kind].info, forward=fwd[kind], noise_cov=noise_cov, verbose=True
    )
    stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
        raws[kind],
        inverse_operator,
        lambda2=lambda2,
        n_fft=n_fft,
        dB=False,
        return_sensor=True,
        verbose=True,
    )
    topo_norm = sensor_psd.data.sum(axis=1, keepdims=True)
    stc_norm = stc_psd.sum()  # same operation on MNE object, sum across freqs
    # Normalize each source point by the total power across freqs
    for band, limits in freq_bands.items():
        data = sensor_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos[kind][band] = mne.EvokedArray(100 * data / topo_norm, sensor_psd.info)
        stcs[kind][band] = 100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data
    del inverse_operator
del fwd, raws, raw_erms


# %%
# Now we can make some plots of each frequency band. Note that the OPM head
# coverage is only over right motor cortex, so only localization
# of beta is likely to be worthwhile.
#
# Alpha
# -----


def plot_band(kind, band):
    """Plot activity within a frequency band on the subject's brain."""
    lf, hf = freq_bands[band]
    title = f"{titles[kind]} {band}\n({lf:d}-{hf:d} Hz)"
    topos[kind][band].plot_topomap(
        times=0.0,
        scalings=1.0,
        cbar_fmt="%0.1f",
        vlim=(0, None),
        cmap="inferno",
        time_format=title,
    )
    brain = stcs[kind][band].plot(
        subject=subject,
        subjects_dir=subjects_dir,
        views="cau",
        hemi="both",
        time_label=title,
        title=title,
        colormap="inferno",
        time_viewer=False,
        show_traces=False,
        clim=dict(kind="percent", lims=(70, 85, 99)),
        smoothing_steps=10,
    )
    brain.show_view(azimuth=0, elevation=0, roll=0)
    return fig, brain


fig_alpha, brain_alpha = plot_band("vv", "alpha")

# %%
# Beta
# ----
# Here we also show OPM data, which shows a profile similar to the VectorView
# data beneath the sensors. VectorView first:

fig_beta, brain_beta = plot_band("vv", "beta")

# %%
# Then OPM:

# sphinx_gallery_thumbnail_number = 10
fig_beta_opm, brain_beta_opm = plot_band("opm", "beta")

# %%
# References
# ----------
# .. footbibliography::

# Keep all windows open until the user closes them.
# (Prevents VTK/PyVista wglMakeCurrent crash on Windows during cleanup)
plt.show(block=True)
input("Press Enter to close 3D brain windows and exit...")
