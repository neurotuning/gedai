"""
Gedai
=====

This tutorial demonstrates how to use Gedai to fit and transform EEG data.
"""

# %% Import data
import os

import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
)
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()
raw.pick([f"EEG 0{n:02}" for n in range(41, 60)])
raw.set_eeg_reference("average")

# %%
# Fit the raw data

from gedai.gedai import Gedai

gedai = Gedai()
gedai.fit_raw(raw, n_jobs=2, noise_multiplier=1.0, verbose=False)

# %%
# plot

import matplotlib.pyplot as plt

fig, axes = gedai.plot_fit()
plt.show()

# %%
# Transforming the raw data

raw_corrected = gedai.transform_raw(raw, verbose=False)

# %%
# Interactive plot
from gedai.viz import plot_mne_style_overlay_interactive
plot_mne_style_overlay_interactive(raw, raw_corrected)