"""
GEDAI online
============

This tutorial demonstrates how to use ``GEDAI`` for online (real-time) denoising.
"""

# %%
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import make_fixed_length_epochs

from gedai import Gedai
from gedai.viz import plot_mne_style_overlay_interactive

# %% Load sample EEG data
subjects = [1]  # may vary
runs = [4, 8, 12]  # may vary
raw_fnames = eegbci.load_data(subjects, runs, update_path=True)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
# Concatenate runs from the same subject
raw = concatenate_raws(raws)
# Make channel names follow standard conventions
eegbci.standardize(raw)

# Crop to the first 15 seconds for demonstration purposes
# (Remove or adjust this for full data analysis)
raw.crop(0, 30)
raw.pick("eeg").load_data().apply_proj()

# Apply average reference (standard preprocessing for EEG)
raw.set_eeg_reference("average", projection=False)

# %%
# For this example, we will use the standard broadband ``GEDAI``.
# However, the same principles apply to the ``spectral GEDAI`` as well.
# For real-time applications, data is typically processed in chunks (e.g., sliding windows).
# To simulate this, we will first convert the raw data into overlapping epochs.

epochs = make_fixed_length_epochs(raw, duration=1.0, overlap=0.5, preload=True)


# %%
# We first fit need to fit the ``GEDAI`` model on some initial data segment.
# This segment should be representative of the data to be denoised, including typical artifacts.
# A common approach is to use a baseline period at the beginning of the recording for this purpose.

gedai = Gedai()

# In this example, we use the first 10 seconds (i.e., first 20 epochs) for model fitting.
# The baseline period 
baseline_epochs = epochs[:20]
gedai.fit_epochs(baseline_epochs, verbose=True)

# %%
# After fitting, we can apply the fitted ``GEDAI`` model to each incoming epoch for denoising.
for i in range(20, len(epochs)):
    epoch = epochs[i : i + 1]  # Get the current epoch
    denoised_epoch = gedai.transform_epochs(epoch)
    