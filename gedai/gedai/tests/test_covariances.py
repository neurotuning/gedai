"""Test Covariances."""

import mne
import pytest
from mne.datasets import testing

from gedai.gedai.covariances import (
    _ensure_cov,
    _pick_cov,
    compute_covariance_from_channel_positions,
    compute_covariance_from_forward,
)

data_path = testing.data_path(download=False)
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fname_fwd = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-eeg-oct-6-fwd.fif"

raw = mne.io.read_raw_fif(fname_raw, preload=True)
raw.pick_types(meg=False, eeg=True)
info = raw.info


def test_ensure_cov():
    """Test _ensure_cov."""
    # test with a covariance object
    cov = mne.make_ad_hoc_cov(info)
    assert _ensure_cov(cov) == cov

    # test with the string "leadfield"
    cov_leadfield = _ensure_cov("leadfield")
    assert isinstance(cov_leadfield, mne.Covariance)

    # test with an invalid string
    with pytest.raises(ValueError, match="Reference covariance must be 'leadfield'"):
        _ensure_cov("invalid_string")


def test_pick_cov():
    """Test _pick_cov."""
    cov = mne.make_ad_hoc_cov(info)
    ch_names = info["ch_names"][:5]  # pick a subset of channels
    picked_cov = _pick_cov(cov, ch_names)
    assert set(picked_cov.ch_names) == set(ch_names)

    ch_names = [ch_name.lower() for ch_name in info["ch_names"]]
    picked_cov = _pick_cov(cov, ch_names)
    assert set(picked_cov.ch_names) == set(ch_names)

    ch_names = ["nonexistent_channel"]
    with pytest.raises(
        ValueError, match="No matching channel names found between inst and cov"
    ):
        _pick_cov(cov, ch_names)

    ch_names = ["EEG 001"] + ["nonexistent_channel"]
    with pytest.raises(
        ValueError,
        match="Only a subset of channels in the instance are present in the covariance",
    ):
        _pick_cov(cov, ch_names)


def test_compute_covariance_from_channel_positions():
    """Test compute_covariance_from_channel_positions."""
    cov = compute_covariance_from_channel_positions(info)
    assert isinstance(cov, mne.Covariance)


def test_compute_covariance_from_forward():
    """Test compute_covariance_from_forward."""
    forward = mne.read_forward_solution(fname_fwd)
    cov = compute_covariance_from_forward(forward)
    assert isinstance(cov, mne.Covariance)
