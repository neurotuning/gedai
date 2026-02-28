import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import mne

# ==========================================
# 1. Examining raw data
# ==========================================
subject = "sub-002"
data_path = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (
    data_path / subject / "ses-001" / "meg" / "sub-002_ses-001_task-aef_run-001_meg.bin"
)
subjects_dir = data_path / "derivatives" / "freesurfer" / "subjects"

# For now we are going to assume the device and head coordinate frames are
# identical (even though this is incorrect), so we pass verbose='error'
raw = mne.io.read_raw_fil(opm_file, verbose="error")
raw.crop(120, 210).load_data()  # crop for speed

picks = mne.pick_types(raw.info, meg=True)
amp_scale = 1e12  # T->pT
stop = len(raw.times) - 300
step = 300

data_ds, time_ds = raw[picks[::5], :stop]
data_ds, time_ds = data_ds[:, ::step] * amp_scale, time_ds[::step]

fig, ax = plt.subplots(layout="constrained")
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-500, 500), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title="No preprocessing", **set_kwargs)

# ==========================================
# 2. Denoising: Regressing via reference sensors
# ==========================================
# set flux channels to bad
bad_picks = mne.pick_channels_regexp(raw.ch_names, regexp="Flux.")
raw.info["bads"].extend([raw.ch_names[ii] for ii in bad_picks])
raw.info["bads"].extend(["G2-17-TAN"])

# compute the PSD for later using 1 Hz resolution
psd_kwargs = dict(fmax=20, n_fft=int(round(raw.info["sfreq"])))
psd_pre = raw.compute_psd(**psd_kwargs)

# filter and regress
raw.filter(None, 5, picks="ref_meg")
regress = mne.preprocessing.EOGRegression(picks, picks_artifact="ref_meg")
regress.fit(raw)
regress.apply(raw, copy=False)

# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

fig, ax = plt.subplots(layout="constrained")
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True, ls=":")
ax.set(title="After reference regression", **set_kwargs)

# compute the psd of the regressed data
psd_post_reg = raw.compute_psd(**psd_kwargs)

# ==========================================
# 3. Denoising: Regressing via homogeneous field correction
# ==========================================
# include gradients by setting order to 2, set to 1 for homgenous components
projs = mne.preprocessing.compute_proj_hfc(raw.info, order=2)
raw.add_proj(projs).apply_proj(verbose="error")

# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

fig, ax = plt.subplots(layout="constrained")
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True, ls=":")
ax.set(title="After HFC", **set_kwargs)

# compute the psd of the regressed data
psd_post_hfc = raw.compute_psd(**psd_kwargs)

# ==========================================
# 4. Comparing denoising methods
# ==========================================
shielding = 10 * np.log10(psd_pre[:] / psd_post_reg[:])

fig, ax = plt.subplots(layout="constrained")
ax.plot(psd_post_reg.freqs, shielding.T, **plot_kwargs)
ax.grid(True, ls=":")
ax.set(xticks=psd_post_reg.freqs)
ax.set(
    xlim=(0, 20),
    title="Reference regression shielding",
    xlabel="Frequency (Hz)",
    ylabel="Shielding (dB)",
)

shielding = 10 * np.log10(psd_pre[:] / psd_post_hfc[:])

fig, ax = plt.subplots(layout="constrained")
ax.plot(psd_post_hfc.freqs, shielding.T, **plot_kwargs)
ax.grid(True, ls=":")
ax.set(xticks=psd_post_hfc.freqs)
ax.set(
    xlim=(0, 20),
    title="Reference regression & HFC shielding",
    xlabel="Frequency (Hz)",
    ylabel="Shielding (dB)",
)

# ==========================================
# 5. Filtering nuisance signals
# ==========================================
# notch
raw.notch_filter(np.arange(50, 251, 50), notch_widths=4)
# bandpass
raw.filter(2, 40, picks="meg")

# plot
data_ds, _ = raw[picks[::5], :stop]
data_ds = data_ds[:, ::step] * amp_scale

fig, ax = plt.subplots(layout="constrained")
plot_kwargs = dict(lw=1, alpha=0.5)
ax.plot(time_ds, data_ds.T - np.mean(data_ds, axis=1), **plot_kwargs)
ax.grid(True)
set_kwargs = dict(
    ylim=(-500, 500), xlim=time_ds[[0, -1]], xlabel="Time (s)", ylabel="Amplitude (pT)"
)
ax.set(title="After regression, HFC and filtering", **set_kwargs)

# ==========================================
# 6. Generating an evoked response
# ==========================================
events = mne.find_events(raw, min_duration=0.1)
epochs = mne.Epochs(
    raw, events, tmin=-0.1, tmax=0.4, baseline=(-0.1, 0.0), verbose="error"
)
evoked = epochs.average()

t_peak = evoked.times[np.argmax(np.std(evoked.copy().pick("meg").data, axis=0))]
fig = evoked.plot_joint(picks="mag")

# ==========================================
# 7. Visualizing coregistration
# ==========================================
mri = nib.load(subjects_dir / "sub-002" / "mri" / "T1.mgz")
trans = mri.header.get_vox2ras_tkr() @ np.linalg.inv(mri.affine)
trans[:3, 3] /= 1000.0  # nibabel uses mm, MNE uses m
trans = mne.transforms.Transform("head", "mri", trans)

bem = subjects_dir / subject / "bem" / f"{subject}-5120-bem-sol.fif"
src = subjects_dir / subject / "bem" / f"{subject}-oct-6-src.fif"

mne.viz.plot_alignment(
    evoked.info,
    subjects_dir=subjects_dir,
    subject=subject,
    trans=trans,
    surfaces={"head": 0.1, "inner_skull": 0.2, "white": 1.0},
    meg=["helmet", "sensors"],
    verbose="error",
    bem=bem,
    src=src,
)

# ==========================================
# 8. Plotting the inverse
# ==========================================
fwd = mne.make_forward_solution(
    evoked.info,
    trans=trans,
    bem=bem,
    src=src,
    verbose=True,
)
noise_cov = mne.compute_covariance(epochs, tmax=0)
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov, verbose=True)

stc = mne.minimum_norm.apply_inverse(
    evoked, inv, 1.0 / 9.0, method="dSPM", verbose=True
)

brain = stc.plot(
    hemi="split",
    size=(800, 400),
    initial_time=t_peak,
    subjects_dir=subjects_dir,
)

plt.show(block=True)
