"""
GEDAI Adaptative Multiband
==========================

This tutorial demonstrates how to use adaptative multiband ``GEDAI``.
``Adaptative Multiband GEDAI`` is a frequency-specific denoising method that extends the
generalized eigenvalue decomposition approach of ``GEDAI``.
Its approach focuses on isolating and removing artifacts within specific
frequency bands. For that, the multiband ``GEDAI`` first decomposes the EEG
data into its frequency components using wavelet transform, then applies
``GEDAI`` to each frequency band separately. Finally, the denoised frequency
components are recombined to reconstruct the cleaned EEG signal.
"""

# %%
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from gedai import Gedai, AdaptativeMultibandGedai
from gedai.viz import plot_mne_style_overlay_interactive


n_jobs = -1
# %% Load sample EEG data
subjects = [1]  # may vary
runs = [4]  # may vary
raw_fnames = eegbci.load_data(subjects, runs, update_path=True)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
# Concatenate runs from the same subject
raw = concatenate_raws(raws)
# Make channel names follow standard conventions
eegbci.standardize(raw)

# Crop to the first 30 seconds for demonstration purposes
# (Remove or adjust this for full data analysis)
raw.crop(0, 60)
raw.pick("eeg").load_data().apply_proj()

# Apply average reference (standard preprocessing for EEG)
raw.set_eeg_reference("average", projection=False)


# %%
# Recommended pipeline
# --------------------
#
# For optimal results, we recommend to first fit the standard ``GEDAI`` on
# broadband data with a conservative ``noise_multiplier`` (e.g., ``6.0``) to
# preserve most neural signals while only removing large artifacts. Then, use
# the resulting cleaned data to fit the ``Multiband GEDAI`` model. This two-step
# approach leverages the strengths of both methods, ensuring effective artifact
# removal while maintaining the integrity of neural signals across different
# frequency bands.

broadband_gedai = Gedai()
broadband_gedai.fit_raw(raw,
                        sensai_method="gridsearch",
                        noise_multiplier=6.0,
                        n_jobs=n_jobs,
                        verbose=False)
broadband_denoised_raw = broadband_gedai.transform_raw(raw,
                                                       n_jobs=n_jobs,
                                                       verbose=False)

adaptive_multiband_gedai = AdaptativeMultibandGedai(
    wavelet_type="haar", wavelet_level=8, cycles_per_wavelet=12
)
adaptive_multiband_gedai.fit_raw(broadband_denoised_raw,
                                 sensai_method="gridsearch",
                                 noise_multiplier=3.0,
                                 wavelet_low_cutoff="auto",
                                 n_jobs=n_jobs,
                                 verbose=True,
                                )
adaptive_multiband_denoised_raw = adaptive_multiband_gedai.transform_raw(
    broadband_denoised_raw, verbose=False, n_jobs=n_jobs
)

plot_mne_style_overlay_interactive(raw, adaptive_multiband_denoised_raw)

# %%
