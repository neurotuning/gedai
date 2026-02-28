import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import mne
import os
import psutil
from gedai.gedai.gedai import Gedai

# ==========================================
# 1. Examining raw data
# ==========================================
subject = "sub-002"
data_path = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (
    data_path / subject / "ses-001" / "meg" / "sub-002_ses-001_task-aef_run-001_meg.bin"
)
subjects_dir = data_path / "derivatives" / "freesurfer" / "subjects"

# Read and crop
raw = mne.io.read_raw_fil(opm_file, verbose="error")
raw.crop(120, 210).load_data()

# FIND EVENTS BEFORE RESAMPLING (6000 Hz) to ensure trigger integrity
events_orig = mne.find_events(raw, min_duration=0.1)
print(f"Events found at 6000 Hz: {len(events_orig)}", flush=True)

print(f"Downsampling from {raw.info['sfreq']} Hz to 200 Hz...", flush=True)
# Resample raw and events together
raw, events = raw.resample(200, events=events_orig)

print("Applying 1-40 Hz bandpass filter...", flush=True)
raw.filter(1.0, 40.0, verbose="error")

# Picks
picks_mag = mne.pick_types(raw.info, meg=True, ref_meg=False)
raw_mag = raw.copy().pick(picks_mag)

# Baseline PSD
psd_kwargs = dict(fmin=2, fmax=80, n_fft=int(round(raw.info["sfreq"] * 2)))
psd_orig = raw_mag.compute_psd(**psd_kwargs)

# Coregistration
mri = nib.load(subjects_dir / subject / "mri" / "T1.mgz")
trans = mri.header.get_vox2ras_tkr() @ np.linalg.inv(mri.affine)
trans[:3, 3] /= 1000.0
trans = mne.transforms.Transform("head", "mri", trans)

bem_path = subjects_dir / subject / "bem" / f"{subject}-5120-bem-sol.fif"
src_path = subjects_dir / subject / "bem" / f"{subject}-oct-6-src.fif"

# Forward Solution (Leadfield)
print("Computing forward solution...", flush=True)
fwd = mne.make_forward_solution(
    raw_mag.info,
    trans=trans,
    bem=bem_path,
    src=src_path,
    verbose=False,
)

n_jobs = psutil.cpu_count(logical=False) or 1
print(f"Using {n_jobs} jobs for parallel processing.", flush=True)

# ==========================================
# 2. Broadband GEDAI (wavelet_level=0)
# ==========================================
print("\n--- Running Broadband GEDAI ---", flush=True)
gedai_broad = Gedai(wavelet_level=0)
gedai_broad.fit_raw(raw_mag, reference_cov=fwd, noise_multiplier=3.0, n_jobs=n_jobs)
raw_broad = gedai_broad.transform_raw(raw_mag, n_jobs=n_jobs)

psd_broad = raw_broad.compute_psd(**psd_kwargs)

# ==========================================
# 3. Plots: PSD and Shielding
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# PSD Plot
axes[0].plot(psd_orig.freqs, 10 * np.log10(np.mean(psd_orig.get_data(), axis=0)), label="Original")
axes[0].plot(psd_broad.freqs, 10 * np.log10(np.mean(psd_broad.get_data(), axis=0)), label="Broadband GEDAI")
axes[0].set(title="Power Spectral Density Comparison", xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)")
axes[0].legend()
axes[0].grid(True, ls=":")

# Shielding Plot
with np.errstate(divide='ignore', invalid='ignore'):
    shield_broad = 10 * np.log10(psd_orig.get_data() / psd_broad.get_data())

axes[1].plot(psd_orig.freqs, np.mean(shield_broad, axis=0), label="Broadband Shielding")
axes[1].set(title="Average Shielding Comparison", xlabel="Frequency (Hz)", ylabel="Shielding (dB)")
axes[1].legend()
axes[1].grid(True, ls=":")

fig.savefig("gedai_psd_shielding_comparison.png")

# ==========================================
# 4. Continuous Validation
# ==========================================
# The data is already 1-40Hz bandpassed. We compute the final PSD for comparison.

# One more PSD plot of the fully bandpassed and denoised continuous data
psd_final_kwargs = dict(fmin=2, fmax=80, n_fft=int(round(raw_broad.info["sfreq"] * 2)))
psd_final = raw_broad.compute_psd(**psd_final_kwargs)

fig2, ax2 = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax2.plot(psd_orig.freqs, 10 * np.log10(np.mean(psd_orig.get_data(), axis=0)), label="Original (1-40Hz BP)", color='gray', alpha=0.5)
ax2.plot(psd_final.freqs, 10 * np.log10(np.mean(psd_final.get_data(), axis=0)), label="GEDAI (BB)", color='green')
ax2.set(title="Final Continuous PSD", xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)")
ax2.legend()
ax2.grid(True, ls=":")

fig2.savefig("gedai_psd_final_continuous.png")

print("\n--- Done. PSD and Shielding plots generated. ---", flush=True)
plt.show(block=True)
