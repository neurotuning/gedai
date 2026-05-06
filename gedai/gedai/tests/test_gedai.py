"""Test Gedai."""

import pytest
from mne import make_fixed_length_epochs
from mne.datasets import testing
from mne.io import read_raw_brainvision

from gedai import Gedai, logger, set_log_level
from gedai.gedai.gedai import compute_closest_valid_duration

set_log_level("INFO")
logger.propagate = True


subjects = [1]  # may vary
runs = [4, 8, 12]  # may vary
data_path = testing.data_path(download=False)
# from mne.datasets import testing
raw_fname = data_path / "antio" / "CA_208" / "test_CA_208_start_stop.vhdr"
with pytest.warns():
    raw = read_raw_brainvision(raw_fname, eog=["EOG"], preload=True)
raw.pick_types(meg=False, eeg=True)
raw.drop_channels([ch_name for ch_name in raw.ch_names if "BIP" in ch_name])

# epochs
wavelet_level = 5
target_duration = 1.0
duration, sample = compute_closest_valid_duration(
    target_duration, wavelet_level, raw.info["sfreq"]
)
epochs_eeg = make_fixed_length_epochs(raw, duration=duration, overlap=0)


@testing.requires_testing_data
def test_gedai_fit_raw():
    """Test Gedai fit on raw data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_raw(raw)


@testing.requires_testing_data
def test_gedai_fit_epochs():
    """Test Gedai fit on epochs data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_epochs(epochs_eeg)


@testing.requires_testing_data
def test_gedai_transform_raw():
    """Test Gedai transform on raw data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)


@testing.requires_testing_data
def test_gedai_transform_epochs():
    """Test Gedai transform on epochs data."""
    gedai = Gedai(wavelet_level=0)
    gedai.fit_epochs(epochs_eeg)
    transformed_epochs = gedai.transform_epochs(epochs_eeg)
    assert epochs_eeg.metadata == transformed_epochs.metadata

    gedai = Gedai(wavelet_level=wavelet_level)
    gedai.fit_epochs(epochs_eeg)
    transformed_epochs = gedai.transform_epochs(epochs_eeg)
    assert epochs_eeg.metadata == transformed_epochs.metadata
    gedai.transform_epochs(epochs_eeg)


@testing.requires_testing_data
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
    with pytest.raises(
        ValueError,
        match=(
            "The following channels are present in the input inst but were not present"
        ),
    ):
        gedai.transform_epochs(epochs_eeg)

    epochs_test = epochs_eeg.copy()
    epochs_test.load_data()
    epochs_test.pick_channels(raw.ch_names[:5])
    with pytest.raises(
        ValueError,
        match="The following channels are missing in the input inst but were present",
    ):
        gedai.transform_epochs(epochs_test)

    epochs_test = epochs_eeg.copy()
    epochs_test.load_data()
    epochs_test.pick_channels(raw.ch_names[:10])
    gedai.transform_epochs(epochs_test)
