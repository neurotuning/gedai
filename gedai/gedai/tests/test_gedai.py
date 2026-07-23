"""Test Gedai."""

import mne
import pytest
from mne import make_fixed_length_epochs

from gedai import Gedai
from gedai.data import get_contaminated_eeg_set_path

raw_fname = get_contaminated_eeg_set_path()
raw = mne.io.read_raw(raw_fname, preload=True)
raw.drop_channels([ch_name for ch_name in raw.ch_names if "BIP" in ch_name])
epochs_eeg = make_fixed_length_epochs(raw, duration=1.0, overlap=0)


def test_gedai_fit_epochs():
    """Test Gedai fit on epochs data."""
    model = Gedai()
    model.fit_epochs(epochs_eeg)


def test_gedai_fit_raw():
    """Test Gedai fit on raw data."""
    model = Gedai()
    model.fit_raw(raw)


def test_gedai_transform_epochs():
    """Test Gedai transform on epochs data."""
    gedai = Gedai()
    gedai.fit_epochs(epochs_eeg)
    transformed_epochs = gedai.transform_epochs(epochs_eeg)
    assert epochs_eeg.metadata == transformed_epochs.metadata


def test_gedai_transform_raw():
    """Test Gedai transform on raw data."""
    gedai = Gedai()
    gedai.fit_raw(raw)
    gedai.transform_raw(raw)


def test_gedai_epochs_picks():
    """Test Gedai fit on epochs data."""
    gedai = Gedai()
    gedai.fit_epochs(epochs_eeg, picks="all")
    assert gedai.ch_names == epochs_eeg.ch_names

    gedai = Gedai()
    gedai.fit_epochs(epochs_eeg, picks="data")
    assert gedai.ch_names == epochs_eeg.ch_names

    gedai = Gedai()
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
