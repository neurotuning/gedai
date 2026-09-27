"""
Using GEDAI for online (real-time) denoising
============================================

This tutorial demonstrates how to use ``GEDAI`` for online (real-time) denoising.
"""

# %%
from mne import make_fixed_length_epochs
from mne.io import read_raw

from gedai import Gedai
from gedai.data import get_contaminated_eeg_set_path

# %% Load sample EEG data
raw = read_raw(str(get_contaminated_eeg_set_path()), preload=True)

# Crop to the first 15 seconds for demonstration purposes
# (Remove or adjust this for full data analysis)
raw.crop(0, 30)

# Apply average reference (standard preprocessing for EEG)
raw.set_eeg_reference("average", projection=False)

# %%
# For this example, we will use the standard broadband ``GEDAI``.
# However, the same principles apply to the ``spectral GEDAI`` as well.
# For real-time applications, data is typically processed in chunks (e.g.,
# sliding windows). To simulate this, we will first convert the raw data into
# overlapping epochs.

epochs = make_fixed_length_epochs(raw, duration=1.0, overlap=0.5, preload=True)


# %%
# We first need to fit the ``GEDAI`` model on an initial data segment.
# This segment should be representative of the data to be denoised, including
# typical artifacts. A common approach is to use a baseline period at the
# beginning of the recording for this purpose.

gedai = Gedai()

# In this example, we use the first 10 seconds (i.e., first 20 epochs) for
# model fitting. The baseline period
baseline_epochs = epochs[:20]
gedai.fit_epochs(baseline_epochs, verbose=True)

# %%
# After fitting, we can apply the fitted ``GEDAI`` model to each incoming
# epoch for denoising.
for epoch_idx in range(20, len(epochs)):
    epoch = epochs[epoch_idx : epoch_idx + 1]  # Get the current epoch
    denoised_epoch = gedai.transform_epochs(epoch)
