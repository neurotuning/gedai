"""Test Covariances."""

import mne
import pytest
from mne.datasets import testing

from gedai.covariance.covariance import (
    _ensure_cov,
    _pick_cov,
    compute_covariance_from_channel_positions,
    compute_covariance_from_forward,
)

data_path = testing.data_path(download=False)
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fname_fwd = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-eeg-oct-6-fwd.fif"


@pytest.fixture(scope="module")
def sample_info():
    """Create a small EEG montage fixture for covariance tests."""
    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
    ]
    info = mne.create_info(ch_names, sfreq=250.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


def test_ensure_cov(sample_info):
    """Test _ensure_cov."""
    # test with a covariance object
    cov = mne.make_ad_hoc_cov(sample_info)
    assert _ensure_cov(cov) == cov

    # test with the string "leadfield"
    cov_leadfield = _ensure_cov("leadfield")
    assert isinstance(cov_leadfield, mne.Covariance)

    # test with an invalid string
    with pytest.raises(ValueError, match="Reference covariance must be 'leadfield'"):
        _ensure_cov("invalid_string")


def test_pick_cov(sample_info):
    """Test _pick_cov."""
    cov = mne.make_ad_hoc_cov(sample_info)
    ch_names = sample_info["ch_names"][:5]  # pick a subset of channels
    picked_cov = _pick_cov(cov, ch_names)
    assert set(picked_cov.ch_names) == set(ch_names)

    ch_names = [ch_name.lower() for ch_name in sample_info["ch_names"]]
    picked_cov = _pick_cov(cov, ch_names)
    assert set(picked_cov.ch_names) == set(ch_names)

    ch_names = ["nonexistent_channel"]
    with pytest.raises(
        ValueError, match="No matching channel names found between inst and cov"
    ):
        _pick_cov(cov, ch_names)

    ch_names = ["Fp1", "nonexistent_channel"]
    with pytest.raises(
        ValueError,
        match="Only a subset of channels in the instance are present in the covariance",
    ):
        _pick_cov(cov, ch_names)


def test_compute_covariance_from_channel_positions(sample_info):
    """Test compute_covariance_from_channel_positions."""
    cov = compute_covariance_from_channel_positions(sample_info)
    assert isinstance(cov, mne.Covariance)


@testing.requires_testing_data
def test_compute_covariance_from_forward():
    """Test compute_covariance_from_forward."""
    if not fname_fwd.exists():
        pytest.skip("Requires MNE testing dataset")
    forward = mne.read_forward_solution(fname_fwd)
    cov = compute_covariance_from_forward(forward)
    assert isinstance(cov, mne.Covariance)
