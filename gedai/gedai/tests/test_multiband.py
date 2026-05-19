"""Tests for multiband GEDAI."""

import pytest
from mne import make_fixed_length_epochs
from mne.datasets import testing
from mne.io import read_raw_brainvision

from gedai import MultibandGedai
from gedai.gedai.multiband import (
    compute_closest_valid_duration,
    compute_required_duration,
)

data_path = testing.data_path(download=False)
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
def test_multiband_fit_transform_epochs():
    """Fit and transform should work on epochs data."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)

    model.fit_epochs(epochs_eeg, n_jobs=1)
    transformed = model.transform_epochs(epochs_eeg, n_jobs=1)

    assert transformed.get_data().shape == epochs_eeg.get_data().shape
    assert len(model._wavelets_fits) == model.wavelet_level + 1


@testing.requires_testing_data
def test_multiband_fit_transform_raw():
    """Fit and transform should work on raw data."""
    model = MultibandGedai(wavelet_type="haar", wavelet_level=wavelet_level)

    model.fit_raw(raw, duration=duration, n_jobs=1)
    raw_corrected = model.transform_raw(raw, duration=duration, n_jobs=1)

    assert raw_corrected.get_data().shape == raw.get_data().shape


@testing.requires_testing_data
def test_multiband_low_cutoff_marks_ignored_bands():
    """Low cutoff should mark at least one low-frequency band as ignored."""
    model = MultibandGedai(
        wavelet_type="haar", wavelet_level=wavelet_level, wavelet_low_cutoff=5.0
    )

    model.fit_epochs(epochs_eeg, n_jobs=1)

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
