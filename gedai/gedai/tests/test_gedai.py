"""Test Gedai."""

from mne import make_fixed_length_epochs
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from gedai import Gedai, logger, set_log_level
from gedai.gedai.gedai import compute_closest_valid_duration

import pytest
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
raw.load_data().apply_proj()

# epochs
wavelet_level = 5
target_duration = 1.0
duration, sample = compute_closest_valid_duration(
    target_duration, wavelet_level, raw.info["sfreq"]
)
epochs_eeg = make_fixed_length_epochs(raw, duration=duration, overlap=0)


def test_gedai_fit_raw():
    """Test Gedai fit on raw data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_raw(raw)


def test_gedai_fit_epochs():
    """Test Gedai fit on epochs data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_epochs(epochs_eeg)


def test_gedai_transform_raw():
    """Test Gedai transform on raw data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)


def test_gedai_transform_epochs():
    """Test Gedai transform on epochs data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)
    gedai.transform_epochs(epochs_eeg)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_epochs(epochs_eeg)
    gedai.transform_epochs(epochs_eeg)


def test_gedai_epochs_picks():
    """Test Gedai fit on epochs data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg, picks="all")
    assert gedai.ch_names == epochs_eeg.ch_names

    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg, picks="data")
    assert gedai.ch_names == epochs_eeg.ch_names

    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg, picks=raw.ch_names[:10])
    assert gedai.ch_names == raw.ch_names[:10]
    with pytest.raises(ValueError, match="The following channels are present in the input inst but were not present"):
        gedai.transform_epochs(epochs_eeg)
    
    with pytest.raises(ValueError, match="The following channels are missing in the input inst but were present"):
        epochs_test = epochs_eeg.copy()
        epochs_test.load_data()
        epochs_test.pick_channels(raw.ch_names[:5])
        gedai.transform_epochs(epochs_test)
    
    epochs_test = epochs_eeg.copy()
    epochs_test.load_data()
    epochs_test.pick_channels(raw.ch_names[:10])
    gedai.transform_epochs(epochs_test)