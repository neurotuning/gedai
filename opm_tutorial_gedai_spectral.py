import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import mne
import sys
import os

from gedai.gedai.gedai import Gedai
from gedai.viz.compare import plot_mne_style_overlay_interactive

# ==========================================
# 1. Setup paths and load data
# ==========================================
subject = "sub-002"
data_path = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (
    data_path / subject / "ses-001" / "meg" / "sub-002_ses-001_task-aef_run-001_meg.bin"
)
subjects_dir = data_path / "derivatives" / "freesurfer" / "subjects"

print("Loading raw data...", flush=True)
raw = mne.io.read_raw_fil(opm_file, verbose="error")
raw.crop(120, 210).load_data()

picks_mag = mne.pick_types(raw.info, meg=True, ref_meg=False)
amp_scale = 1e12  # T->pT

# Plotting settings for traces
stop = len(raw.times) - 300
step = 300
plot_kwargs = dict(lw=1, alpha=0.5)

# Downsample original data trace for plotting
data_orig, time_orig = raw[picks_mag[::5], :stop]
data_orig, time_orig = data_orig[:, ::step] * amp_scale, time_orig[::step]
set_kwargs = dict(ylim=(-500, 500), xlim=time_orig[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)")

# plot 1: No Preprocessing
fig, ax = plt.subplots(layout="constrained")
ax.plot(time_orig, data_orig.T - np.mean(data_orig, axis=1), **plot_kwargs)
ax.grid(True)
ax.set(title="Raw Data (No preprocessing)", **set_kwargs)

# PSD settings (original data is 6000Hz, divided by 0.25 for 0.25 Hz freq res)
psd_kwargs = dict(fmax=45, n_fft=int(round(raw.info["sfreq"] / 0.25)))
psd_pre = raw.compute_psd(**psd_kwargs)

# ==========================================
# 2. Reference Regression + HFC (Baseline Denoising)
# ==========================================
print("Applying Reference Regression & HFC...", flush=True)
raw_base = raw.copy()

# Regress reference sensors
raw_base.filter(None, 5, picks="ref_meg", verbose=False)
regress = mne.preprocessing.EOGRegression(picks_mag, picks_artifact="ref_meg")
regress.fit(raw_base)
regress.apply(raw_base, copy=False)

psd_post_reg = raw_base.compute_psd(**psd_kwargs)
shielding_reg = 10 * np.log10(psd_pre[:] / psd_post_reg[:])

# HFC
projs = mne.preprocessing.compute_proj_hfc(raw_base.info, order=2)
raw_base.add_proj(projs).apply_proj(verbose=False)

psd_post_hfc = raw_base.compute_psd(**psd_kwargs)
shielding_hfc = 10 * np.log10(psd_pre[:] / psd_post_hfc[:])

# Plotted later after GEDAI is applied

# ==========================================
# 3. GEDAI Data Preparation
# ==========================================
print("Preparing continuous data for GEDAI...", flush=True)

events_orig = mne.find_events(raw, min_duration=0.1)
raw, events = raw.resample(200, events=events_orig)
raw.filter(0.5, 45.0, verbose="error")

# GEDAI operates on the 86 magnetometers. Mark bads first.
bad_picks = mne.pick_channels_regexp(raw.ch_names, regexp="Flux.")
raw.info["bads"].extend([raw.ch_names[ii] for ii in bad_picks])
raw.info["bads"].extend(["G2-17-TAN"])

picks_mag_gedai = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude="bads")
raw_mag = raw.copy().pick(picks_mag_gedai)

# ==========================================
# 4. Setup Forward Model
# ==========================================
print("Computing forward solution...", flush=True)
mri = nib.load(subjects_dir / subject / "mri" / "T1.mgz")
trans = mri.header.get_vox2ras_tkr() @ np.linalg.inv(mri.affine)
trans[:3, 3] /= 1000.0
trans = mne.transforms.Transform("head", "mri", trans)

bem_path = subjects_dir / subject / "bem" / f"{subject}-5120-bem-sol.fif"
src_path = subjects_dir / subject / "bem" / f"{subject}-oct-6-src.fif"

fwd = mne.make_forward_solution(raw_mag.info, trans=trans, bem=bem_path, src=src_path, verbose=False)

# ==========================================
# 5. Apply Spectral GEDAI (wavelet decomposition + frequency-specific epoching)
# ==========================================
# wavelet_level=8  → 9 sub-bands at 200 Hz sfreq:
#   Band 0 (approx):  0    – 0.39 Hz  (1 s epoch; zeroed by wavelet_low_cutoff=0.5)
#   Band 1:           0.39 – 0.78 Hz  (12 cycles / 0.39 Hz = ~30.7 s)
#   Band 2:           0.78 – 1.56 Hz  (12 cycles / 0.78 Hz = ~15.4 s)
#   Band 3:           1.56 – 3.13 Hz  (12 cycles / 1.56 Hz =  ~7.7 s)
#   Band 4:           3.13 – 6.25 Hz  (12 cycles / 3.13 Hz =  ~3.8 s)
#   Band 5:           6.25 – 12.5 Hz  (12 cycles / 6.25 Hz =  ~1.9 s)
#   Band 6:          12.5  – 25  Hz   (12 cycles / 12.5 Hz =   0.96 s)
#   Band 7:          25    – 50  Hz   (12 cycles / 25.0 Hz =   0.48 s)
#   Band 8:          50    – 100 Hz   (12 cycles / 50.0 Hz =   0.24 s)
#
# epoch_size_in_cycles=12 enables frequency-specific epoching (MATLAB default).
# wavelet_low_cutoff=0.5  zeroes band 0 (upper freq 0.39 Hz < 0.5 Hz cutoff).

print(
    f"Applying Spectral GEDAI (wavelet_level=8, epoch_size_in_cycles=12) "
    f"to {len(raw_mag.ch_names)} magnetometers...",
    flush=True,
)
gedai = Gedai(
    wavelet_level='auto',
    wavelet_low_cutoff=1.0,
    epoch_size_in_cycles=12,
)
# Notch filter before GEDAI to remove line noise
print("Applying notch filter (50 Hz) before GEDAI...", flush=True)
raw_mag.notch_filter(50, notch_widths=4, verbose=False)

raw_gedai = gedai.fit_transform_raw(raw_mag, reference_cov=fwd, noise_multiplier=2.0, n_jobs=1)

# Compare before vs after GEDAI at matching bandwidth (0.5-70 Hz, before any further filtering)
# Keys: left/right scroll, up/down scale, D=diff, N=denoised only, O=noisy only
print("Plotting GEDAI comparison (close window to continue)...", flush=True)
plot_mne_style_overlay_interactive(
    raw_mag, raw_gedai,
    title="OPM Magnetometers - Before vs After Spectral GEDAI", duration=10.0
)

# Plot GEDAI Filtered Continuous Trace
stop_gedai = len(raw_gedai.times) - int(300 * (200 / 6000))

data_gedai, time_gedai = raw_gedai[::5, :stop_gedai]
data_gedai = data_gedai * amp_scale

fig, ax = plt.subplots(layout="constrained")
ax.plot(time_gedai, data_gedai.T - np.mean(data_gedai, axis=1), **plot_kwargs)
ax.grid(True)
ax.set(
    title="After Bandpass & Spectral GEDAI (wavelet decomposition)",
    ylim=(-500, 500),
    xlim=time_gedai[[0, -1]],
    xlabel="Time (s)",
    ylabel="Amplitude (pT)",
)

# Calculate PSD on the continuous GEDAI data for shielding comparison
psd_kwargs_ds = dict(fmax=45, n_fft=int(round(raw_gedai.info["sfreq"] / 0.25)))
psd_pre_ds = raw_mag.compute_psd(**psd_kwargs_ds)
psd_post_gedai = raw_gedai.compute_psd(**psd_kwargs_ds)

shielding_gedai = 10 * np.log10(psd_pre_ds[:] / psd_post_gedai[:])

# Plot All Shielding Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
axes[0].plot(psd_post_reg.freqs, shielding_reg.T, **plot_kwargs)
axes[0].grid(True, ls=":")
axes[0].set(xlim=(0, 45), title="Reference regression shielding", xlabel="Frequency (Hz)", ylabel="Shielding (dB)")

axes[1].plot(psd_post_hfc.freqs, shielding_hfc.T, **plot_kwargs)
axes[1].grid(True, ls=":")
axes[1].set(xlim=(0, 45), title="Ref Regr + HFC shielding", xlabel="Frequency (Hz)")

axes[2].plot(psd_post_gedai.freqs, shielding_gedai.T, **plot_kwargs)
axes[2].grid(True, ls=":")
axes[2].set(xlim=(0, 45), title="Spectral GEDAI shielding", xlabel="Frequency (Hz)")

# Plot All PSD Curves
scale_psd = 1e30  # T^2/Hz to fT^2/Hz
fig_psd, axes_psd = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")

axes_psd[0].plot(psd_pre.freqs, 10 * np.log10(psd_pre[:].T * scale_psd), color='gray', alpha=0.3, lw=1)
axes_psd[0].plot(psd_post_reg.freqs, 10 * np.log10(psd_post_reg[:].T * scale_psd), color='blue', alpha=0.5, lw=1)
axes_psd[0].grid(True, ls=":")
axes_psd[0].set(xlim=(0, 45), ylim=(0, 100), title="Reference regression PSD", xlabel="Frequency (Hz)", ylabel="PSD (dB fT²/Hz)")

axes_psd[1].plot(psd_pre.freqs, 10 * np.log10(psd_pre[:].T * scale_psd), color='gray', alpha=0.3, lw=1)
axes_psd[1].plot(psd_post_hfc.freqs, 10 * np.log10(psd_post_hfc[:].T * scale_psd), color='orange', alpha=0.5, lw=1)
axes_psd[1].grid(True, ls=":")
axes_psd[1].set(xlim=(0, 45), ylim=(0, 100), title="Ref Regr + HFC PSD", xlabel="Frequency (Hz)")

axes_psd[2].plot(psd_pre_ds.freqs, 10 * np.log10(psd_pre_ds[:].T * scale_psd), color='gray', alpha=0.3, lw=1)
axes_psd[2].plot(psd_post_gedai.freqs, 10 * np.log10(psd_post_gedai[:].T * scale_psd), color='green', alpha=0.5, lw=1)
axes_psd[2].grid(True, ls=":")
axes_psd[2].set(xlim=(0, 45), ylim=(0, 100), title="Spectral GEDAI PSD", xlabel="Frequency (Hz)")

# ==========================================
# 6. Generating Evoked Responses (ERP)
# ==========================================
print("Applying 2-40 Hz filter for epochs...", flush=True)
raw_gedai.filter(2, 40, picks="meg")

print("Epoching the GEDAI-denoised continuous data...", flush=True)
epochs = mne.Epochs(raw_gedai, events, tmin=-0.1, tmax=0.4, baseline=(-0.1, 0.0), verbose="error", preload=True)
print(f"Created {len(epochs)} epochs.")

evoked = epochs.average()
t_peak = evoked.times[np.argmax(np.std(evoked.copy().pick("meg").data, axis=0))]

print("Plotting Evoked ERPs...", flush=True)
# Plot the standard ERP trace layout
fig_erp = evoked.plot(picks="meg", window_title="Spectral GEDAI Evoked ERP", show=False)

# Plot the topography Joint Map
print(f"Plotting joint Evoked response (Peak at {t_peak*1000:.1f} ms)...", flush=True)
fig_joint = evoked.plot_joint(picks="meg", show=False)
fig_joint.suptitle("Spectral GEDAI Evoked Joint Map")

# ==========================================
# 7. Visualizing coregistration
# ==========================================
print("Visualizing coregistration...", flush=True)
fig_coreg = mne.viz.plot_alignment(
    evoked.info, subjects_dir=subjects_dir, subject=subject, trans=trans,
    surfaces={"head": 0.1, "inner_skull": 0.2, "white": 1.0},
    meg=["helmet", "sensors"], verbose="error", bem=bem_path, src=src_path,
)

# ==========================================
# 8. Plotting the inverse (Source space)
# ==========================================
print("Computing Noise Covariance and Inverse Solution...", flush=True)
noise_cov = mne.compute_covariance(epochs, tmax=0, method="empirical", verbose=False)
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov, verbose=False)
stc = mne.minimum_norm.apply_inverse(evoked, inv, 1.0 / 9.0, method="dSPM", verbose=False)

print("Plotting source estimates...", flush=True)
brain = stc.plot(hemi="split", size=(800, 400), initial_time=t_peak, subjects_dir=subjects_dir, time_viewer=True)

print("Pipeline complete. Displaying all plots. Close windows to exit.")
plt.show(block=True)
