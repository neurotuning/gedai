import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import mne
import os
from gedai.gedai.gedai import Gedai

# ==========================================
# 1. Setup paths and load data
# ==========================================
subject = "sub-002"
data_path = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (
    data_path / subject / "ses-001" / "meg" / "sub-002_ses-001_task-aef_run-001_meg.bin"
)
subjects_dir = data_path / "derivatives" / "freesurfer" / "subjects"

# Load raw and crop as in tutorial
raw = mne.io.read_raw_fil(opm_file, verbose="error")
raw.crop(120, 210).load_data()

print(f"Downsampling from {raw.info['sfreq']} Hz to 200 Hz...", flush=True)
raw.resample(200)

print("Applying 0.5-70 Hz bandpass filter...", flush=True)
raw.filter(0.5, 70.0, verbose="error")

# EXPLICITLY PICK THE 86 MAGNETOMETERS
picks_mag = mne.pick_types(raw.info, meg=True, ref_meg=False)
raw_mag = raw.copy().pick(picks_mag)

amp_scale = 1e12  # T->pT

# Baseline PSD of the magnetometers (now at 200 Hz)
psd_kwargs = dict(fmin=1, fmax=70, n_fft=int(round(raw.info["sfreq"] / 0.5)))
psd_pre = raw_mag.compute_psd(**psd_kwargs)

# ==========================================
# 2. Setup Forward Model for the 86 magnetometers (Subject Anatomy)
# ==========================================
print("Computing forward solution (this may take a minute)...", flush=True)
# Replicating tutorial coregistration
mri = nib.load(subjects_dir / subject / "mri" / "T1.mgz")
trans = mri.header.get_vox2ras_tkr() @ np.linalg.inv(mri.affine)
trans[:3, 3] /= 1000.0
trans = mne.transforms.Transform("head", "mri", trans)

bem_path = subjects_dir / subject / "bem" / f"{subject}-5120-bem-sol.fif"
src_path = subjects_dir / subject / "bem" / f"{subject}-oct-6-src.fif"

# Forward solution specifically for these 86 magnetometers using BEM
fwd = mne.make_forward_solution(
    raw_mag.info,
    trans=trans,
    bem=bem_path,
    src=src_path,
    verbose=False,
)

# ==========================================
# 3. Apply GEDAI with Custom Forward Model (Anatomy-based)
# ==========================================
print(f"Applying GEDAI to {len(raw_mag.ch_names)} magnetometers using anatomy-based forward model...", flush=True)
n_jobs = 1  # Forced sequential for debugging
print(f"Running sequentially (n_jobs=1) as requested.", flush=True)

gedai = Gedai(wavelet_level=0)
# We fit on the 86 mags using the forward model of the same 86 mags
print("Fitting GEDAI model...", flush=True)
gedai.fit_raw(raw_mag, reference_cov=fwd, noise_multiplier=3.0, n_jobs=n_jobs)

print("Step 2: Transforming data using the fitted filters...", flush=True)
raw_gedai = gedai.transform_raw(raw_mag, n_jobs=n_jobs)
print("Computing PSD of the GEDAI-denoised signal...", flush=True)
psd_gedai = raw_gedai.compute_psd(**psd_kwargs)

# ==========================================
# 4. Compare with Baseline (Ref Regression & HFC)
# ==========================================
print("Running baseline denoising conditions...", flush=True)

# 4a. Reference Regression Only
print(" - Applying Reference Regression Only...", flush=True)
raw_refr = raw.copy()
raw_refr.filter(None, 5, picks="ref_meg")
regress = mne.preprocessing.EOGRegression(picks_mag, picks_artifact="ref_meg")
regress.fit(raw_refr)
regress.apply(raw_refr, copy=False)
psd_refr = raw_refr.compute_psd(**psd_kwargs)

# 4b. HFC Only
print(" - Applying HFC Only...", flush=True)
raw_hfc_only = raw.copy()
projs_hfc_only = mne.preprocessing.compute_proj_hfc(raw_hfc_only.info, order=2)
raw_hfc_only.add_proj(projs_hfc_only).apply_proj(verbose="error")
psd_hfc_only = raw_hfc_only.compute_psd(**psd_kwargs)

# 4c. Reference Regression + HFC (Full Baseline)
print(" - Applying Reference Regression + HFC...", flush=True)
raw_baseline = raw_refr.copy()
projs_baseline = mne.preprocessing.compute_proj_hfc(raw_baseline.info, order=2)
raw_baseline.add_proj(projs_baseline).apply_proj(verbose="error")
psd_baseline = raw_baseline.compute_psd(**psd_kwargs)

# ==========================================
# 5. PSD & Shielding Comparison
# ==========================================
print("Calculating shielding metrics...", flush=True)
shielding_baseline = 10 * np.log10(psd_pre.get_data() / psd_baseline.get_data())
shielding_gedai = 10 * np.log10(psd_pre.get_data() / psd_gedai.get_data())

fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

# Top: PSD Plot
axes[0].plot(psd_pre.freqs, 10 * np.log10(np.mean(psd_pre.get_data(), axis=0)), label="raw", color="gray", alpha=0.5)
axes[0].plot(psd_refr.freqs, 10 * np.log10(np.mean(psd_refr.get_data(), axis=0)), label="ref_meg", color="orange", alpha=0.7)
axes[0].plot(psd_hfc_only.freqs, 10 * np.log10(np.mean(psd_hfc_only.get_data(), axis=0)), label="HFC", color="purple", alpha=0.7)
axes[0].plot(psd_baseline.freqs, 10 * np.log10(np.mean(psd_baseline.get_data(), axis=0)), label="ref_meg + HFC", color="blue", alpha=0.9)
axes[0].plot(psd_gedai.freqs, 10 * np.log10(np.mean(psd_gedai.get_data(), axis=0)), label="GEDAI", color="red", alpha=0.9)
axes[0].set(title="Power Spectral Density Comparison", xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)")
axes[0].legend()
axes[0].grid(True, ls=":")

# Bottom: Shielding Plot
axes[1].plot(psd_pre.freqs, np.mean(shielding_baseline, axis=0), label="Baseline (Regr+HFC)", color='blue')
axes[1].plot(psd_pre.freqs, np.mean(shielding_gedai, axis=0), label="GEDAI (Custom Fwd)", color='red')
axes[1].grid(True, ls=":")
axes[1].legend()
axes[1].set(
    title="Average Shielding Comparison",
    xlabel="Frequency (Hz)",
    ylabel="Shielding (dB)",
)

print(f"Mean Baseline Shielding: {np.mean(shielding_baseline):.2f} dB", flush=True)
print(f"Mean GEDAI Shielding: {np.mean(shielding_gedai):.2f} dB", flush=True)

fig.savefig("gedai_psd_shielding_comparison.png")

# ==========================================
# 6. Trace Overlay (Using compare.py tool)
# ==========================================
print("Plotting MEG trace overlay (Original vs GEDAI) using compare.py...", flush=True)

from gedai.viz.compare import plot_mne_style_overlay_interactive

# The tool takes raw_noisy and raw_clean objects and creates a scrollable matplotlib window.
fig2, ax2 = plot_mne_style_overlay_interactive(
    raw_noisy=raw_mag, 
    raw_clean=raw_gedai, 
    title="MEG Traces: Original vs Broadband GEDAI",
    duration=4.0
)

fig2.savefig("gedai_traces_overlay.png", bbox_inches='tight')

# Let the user interact with the full data if they run the script manually
print("\nOverlay plot saved to gedai_traces_overlay.png.")
print("The script will now display the interactive plots. Close the windows to exit.", flush=True)

