"""
GEDAI forward
=============

This tutorial demonstrates how to compute a custom covariance
matrix using a :class:`mne.Forward` solution.
"""

# %%

import mne
from mne.datasets import eegbci, fetch_fsaverage

from gedai import Gedai, MultibandGedai
from gedai.covariance import compute_covariance_from_forward
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
raw.crop(0, 15)
raw.pick("eeg").load_data().apply_proj()
raw.set_eeg_reference("average", projection=False)


# %%
# Check that the locations of EEG electrodes is correct with respect to MRI
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
# generate the forward solution
fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
)

# %%
# compute the covariance matrix from the forward solution
reference_cov = compute_covariance_from_forward(fwd)