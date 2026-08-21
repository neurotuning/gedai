"""Tests for multiband GEDAI."""

import mne
import pytest
from mne import make_fixed_length_epochs

from gedai import MultibandGedai
from gedai.data import get_contaminated_eeg_set_path
from gedai.gedai.multiband import (
    compute_closest_valid_duration,
    compute_required_duration,
)

raw_fname = get_contaminated_eeg_set_path()
raw_eeg = mne.io.read_raw(raw_fname, preload=True)
raw_eeg.drop_channels([ch_name for ch_name in raw_eeg.ch_names if "BIP" in ch_name])

wavelet_level = 8
target_duration = 1.0
duration, sample = compute_closest_valid_duration(
    target_duration, wavelet_level, raw_eeg.info["sfreq"]
)
epochs_eeg = make_fixed_length_epochs(raw_eeg, duration=duration, overlap=0)


def test_multiband_fit_transform_epochs():
    """Fit and transform should work on epochs data."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)

    model.fit_epochs(epochs_eeg, n_jobs=1)
    transformed = model.transform_epochs(epochs_eeg, n_jobs=1)
    assert transformed.ch_names == epochs_eeg.ch_names
    assert transformed.info["sfreq"] == epochs_eeg.info["sfreq"]
    assert transformed.get_data().shape == epochs_eeg.get_data().shape
    assert transformed.metadata == epochs_eeg.metadata

    assert len(model._wavelets_fits) == model.wavelet_level + 1


def test_multiband_fit_transform_raw():
    """Fit and transform should work on raw data."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)

    model.fit_raw(raw_eeg, duration=duration, n_jobs=1)
    raw_corrected = model.transform_raw(raw_eeg, n_jobs=1)

    assert raw_corrected.info["ch_names"] == raw_eeg.info["ch_names"]
    assert raw_corrected.info["sfreq"] == raw_eeg.info["sfreq"]
    assert raw_corrected.get_data().shape == raw_eeg.get_data().shape
    assert raw_corrected.annotations == raw_eeg.annotations

    assert len(model._wavelets_fits) == model.wavelet_level + 1


def test_multiband_epochs_picks():
    """Test MultibandGedai fit on epochs data."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)
    model.fit_epochs(epochs_eeg, picks="all")
    assert model.ch_names == epochs_eeg.ch_names

    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)
    model.fit_epochs(epochs_eeg, picks="data")
    assert model.ch_names == epochs_eeg.ch_names

    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)
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


def test_multiband_raw_picks():
    """Test MultibandGedai fit on raw data."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)
    model.fit_raw(raw_eeg, picks="all")
    assert model.ch_names == raw_eeg.ch_names

    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)
    model.fit_raw(raw_eeg, picks="data")
    assert model.ch_names == raw_eeg.ch_names

    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)
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


def test_multiband_low_cutoff_marks_ignored_bands():
    """Low cutoff should mark at least one low-frequency band as ignored."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)

    model.fit_epochs(epochs_eeg, wavelet_low_cutoff=5.0, n_jobs=1)
    ignored = [fit["ignore"] for fit in model._wavelets_fits]
    assert any(ignored)
    assert any(not flag for flag in ignored)


def test_wavelet_duration_helpers():
    """Duration helpers should return values consistent with wavelet constraints."""
    sfreq = 128.0
    assert compute_required_duration(0, sfreq) == 1.0
    duration_val, samples = compute_closest_valid_duration(1.03, 4, sfreq)
    assert duration_val >= 1.03
    assert samples % (2**4) == 0
    assert samples >= 2 ** (4 + 1)


def test_multiband_auto_wavelet_level_and_metrics():
    """Test auto wavelet level resolution and metrics calculation."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level="auto")
    model.fit_epochs(epochs_eeg, n_jobs=1)
    assert model._actual_wavelet_level is not None
    assert model._actual_wavelet_level >= 4

    epochs_transformed = model.transform_epochs(epochs_eeg, n_jobs=1)
    assert model.metrics_ is not None
    assert "enova_per_epoch" in model.metrics_
    assert "enova_per_channel" in model.metrics_
    assert "sensai_score" in model.metrics_
    assert isinstance(model.metrics_["mean_enova"], float)


def test_multiband_broadband_pass():
    """Test multiband GEDAI with broadband pre-cleaning pass."""
    # Use smaller picks subset for fast execution in test
    picks = epochs_eeg.ch_names[:6]
    model = MultibandGedai(wavelet_type="haar", wavelet_level=4, broadband_pass=True)
    model.fit_epochs(epochs_eeg, picks=picks, n_jobs=1)
    assert model._broadband_model is not None

    transformed = model.transform_epochs(epochs_eeg, n_jobs=1)
    assert transformed.get_data().shape[1] == len(picks)
    assert model.metrics_ is not None

