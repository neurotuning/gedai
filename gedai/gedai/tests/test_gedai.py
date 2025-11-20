"""Test Gedai."""


from mne import make_fixed_length_epochs
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from gedai import Gedai, logger, set_log_level
from gedai.gedai.gedai import compute_closest_valid_duration

set_log_level("INFO")
logger.propagate = True


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

# epochs
wavelet_level = 5
target_duration = 1.0
duration, sample = compute_closest_valid_duration(target_duration, wavelet_level, raw.info['sfreq'])
epochs_eeg = make_fixed_length_epochs(raw, duration=duration, overlap=0)


def test_gedai_fit_raw():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_raw(raw)

def test_gedai_fit_epochs():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_epochs(epochs_eeg)

def test_gedai_transform_raw():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)

def test_gedai_transform_epochs():
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)
    gedai.transform_epochs(epochs_eeg)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_epochs(epochs_eeg)
    gedai.transform_epochs(epochs_eeg)
