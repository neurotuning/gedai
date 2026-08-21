"""Test Gedai."""

import mne
import pytest
from mne import make_fixed_length_epochs

from gedai import Gedai
from gedai.data import get_contaminated_eeg_set_path

raw_fname = get_contaminated_eeg_set_path()
raw_eeg = mne.io.read_raw(raw_fname, preload=True)
raw_eeg.drop_channels([ch_name for ch_name in raw_eeg.ch_names if "BIP" in ch_name])
epochs_eeg = make_fixed_length_epochs(raw_eeg, duration=1.0, overlap=0)


def test_gedai_fit_transform_epochs():
    """Test Gedai transform on epochs data."""
    model = Gedai()
    model.fit_epochs(epochs_eeg)
    transformed_epochs = model.transform_epochs(epochs_eeg)
    assert transformed_epochs.info['ch_names'] == epochs_eeg.info['ch_names']
    assert transformed_epochs.info['sfreq'] == epochs_eeg.info['sfreq']
    assert epochs_eeg.metadata == transformed_epochs.metadata


def test_gedai_fit_transform_raw():
    """Test Gedai transform on raw data."""
    model = Gedai()
    model.fit_raw(raw_eeg)
    transformed_raw = model.transform_raw(raw_eeg)
    assert transformed_raw.info['ch_names'] == raw_eeg.info['ch_names']
    assert transformed_raw.info['sfreq'] == raw_eeg.info['sfreq']
    assert raw_eeg.annotations == transformed_raw.annotations


def test_gedai_epochs_picks():
    """Test Gedai fit on epochs data."""
    model = Gedai()
    model.fit_epochs(epochs_eeg, picks="all")
    assert model.ch_names == epochs_eeg.ch_names

    model = Gedai()
    model.fit_epochs(epochs_eeg, picks="data")
    assert model.ch_names == epochs_eeg.ch_names

    model = Gedai()
    model.fit_epochs(epochs_eeg, picks=epochs_eeg.ch_names[:10])
    assert model.ch_names == epochs_eeg.ch_names[:10]

    epochs_transformed = model.transform_epochs(epochs_eeg)
    assert epochs_transformed.ch_names == epochs_eeg.ch_names[:10]

    epochs_test = epochs_eeg.copy()
    epochs_test.load_data()
    epochs_test.pick_channels(epochs_eeg.ch_names[:5])
    with pytest.raises(
        ValueError,
        match="The following channels are missing in the input inst but were present",
    ):
        model.transform_epochs(epochs_test)


def test_gedai_raw_picks():
    """Test Gedai fit on raw data."""
    model = Gedai()
    model.fit_raw(raw_eeg, picks="all")
    assert model.ch_names == raw_eeg.ch_names

    model = Gedai()
    model.fit_raw(raw_eeg, picks="data")
    assert model.ch_names == raw_eeg.ch_names

    model = Gedai()
    model.fit_raw(raw_eeg, picks=raw_eeg.ch_names[:10])
    assert model.ch_names == raw_eeg.ch_names[:10]

    raw_transformed = model.transform_raw(raw_eeg)
    assert raw_transformed.ch_names == raw_eeg.ch_names[:10]

    raw_test = raw_eeg.copy()
    raw_test.load_data()
    raw_test.pick_channels(raw_eeg.ch_names[:5])
    with pytest.raises(
        ValueError,
        match="The following channels are missing in the input inst but were present",
    ):
        model.transform_raw(raw_test)


def test_gedai_average_reference_not_reapplied():
    """Test that data with average reference already applied is not modified or re-referenced."""
    import numpy as np
    from gedai.gedai._utils import _check_average_reference, _prepare_raw_fit, _prepare_epochs_fit

    raw = raw_eeg.copy()
    raw.set_eeg_reference("average", projection=False)
    assert _check_average_reference(raw) is True

    raw_fit = _prepare_raw_fit(raw, picks="eeg")
    assert np.allclose(raw.get_data(), raw_fit.get_data(), atol=1e-12)

    epochs = make_fixed_length_epochs(raw, duration=1.0, preload=True)
    assert _check_average_reference(epochs) is True

    epochs_fit = _prepare_epochs_fit(epochs, picks="eeg")
    assert np.allclose(epochs.get_data(), epochs_fit.get_data(), atol=1e-12)
