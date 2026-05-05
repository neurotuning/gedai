"""
GEDAI forward
=============

This tutorial demonstrates how to use a custom covariance matrix in ``GEDAI``
derived from the leadfield of a forward solution.
"""

# %%

import mne
from mne.datasets import eegbci, fetch_fsaverage

from gedai import Gedai
from gedai.gedai.covariances import compute_covariance_from_forward
from gedai.viz import plot_mne_style_overlay_interactive

# %%
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

# %%
# Load EEG data
(raw_fname,) = eegbci.load_data(subjects=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True)
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.set_eeg_reference()

# %%
# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info,
    src=src,
    eeg=["original", "projected"],
    trans=trans,
    show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
)

# %%
# generate the forward solution
fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
)

# %%
# compute the covariance matrix from the forward solution
reference_cov = compute_covariance_from_forward(fwd)

# %% Use the custom covariance in GEDAI
gedai_broadband = Gedai()
gedai_broadband.fit_raw(raw, reference_cov=reference_cov, noise_multiplier=6.0)
raw_broadband_corrected = gedai_broadband.transform_raw(raw, verbose=False)

gedai_spectral = Gedai(wavelet_type="haar", wavelet_level=5, wavelet_low_cutoff=2)
gedai_spectral.fit_raw(
    raw_broadband_corrected, reference_cov=reference_cov, noise_multiplier=3.0
)
raw_spectral_corrected = gedai_spectral.transform_raw(
    raw_broadband_corrected, verbose=False
)

plot_mne_style_overlay_interactive(raw, raw_spectral_corrected)
# %%
