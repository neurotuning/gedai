"""
Gedai
=====

This tutorial demonstrates how to use Gedai to fit and transform EEG data.
"""

# %% Import data
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from gedai import Gedai

subjects = [1]  # may vary
runs = [4, 8, 12]  # may vary
raw_fnames = eegbci.load_data(subjects, runs, update_path=True)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
# concatenate runs from subject
raw = concatenate_raws(raws)
# make channel names follow standard conventions
eegbci.standardize(raw)
raw.crop(0, 15)
raw.pick("eeg").load_data().apply_proj()
raw.set_eeg_reference("average", projection=False)
# %%
# Fit the raw data
gedai = Gedai()
gedai.fit_raw(raw)

# %%
# plot
import matplotlib.pyplot as plt
fig = gedai.plot_fit()
plt.show()

# %%
# Transforming the raw data
raw_corrected = gedai.transform_raw(raw, verbose=False)

# %%
# Interactive plot
from gedai.viz import plot_mne_style_overlay_interactive
plot_mne_style_overlay_interactive(raw, raw_corrected)
