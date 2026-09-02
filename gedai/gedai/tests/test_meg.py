"""Tests for MEG data support in GEDAI."""

import mne
import numpy as np
import pytest

from gedai.covariance.covariance import _ensure_cov, compute_covariance_from_forward
from gedai.gedai.gedai import Gedai
from gedai.gedai.multiband import MultibandGedai
from gedai.sensai.sensai import _compute_default_n_pc, _sensai_to_eigen, _eigen_to_sensai


@pytest.fixture
def meg_mag_raw():
    """Create a synthetic MEG magnetometer Raw instance."""
    n_channels = 12
    sfreq = 200.0
    n_times = int(10 * sfreq)  # 10 seconds
    rng = np.random.default_rng(42)

    ch_names = [f"MEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="mag")
    # Realistic magnetometer amplitude: ~100 fT = 1e-13 T
    data = rng.standard_normal((n_channels, n_times)) * 1e-13
    # Add a focal artifact on first 2 channels around second 4
    data[0:2, int(4 * sfreq) : int(5 * sfreq)] *= 10
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def meg_grad_epochs():
    """Create a synthetic MEG gradiometer Epochs instance."""
    n_channels = 12
    sfreq = 200.0
    epoch_len = int(1.0 * sfreq)
    n_epochs = 8
    rng = np.random.default_rng(101)

    ch_names = [f"MEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="grad")
    # Realistic gradiometer amplitude: ~100 fT/cm = 1e-11 T/m
    data = rng.standard_normal((n_epochs, n_channels, epoch_len)) * 1e-11
    # Add an artifact in epoch 2
    data[2, 0:2, :] *= 8
    epochs = mne.EpochsArray(data, info, verbose=False)
    return epochs


@pytest.fixture
def meg_ref_cov(meg_mag_raw):
    """Create a synthetic MEG reference covariance for the magnetometers."""
    ch_names = meg_mag_raw.ch_names
    n_ch = len(ch_names)
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n_ch, 4))
    cov_data = (A @ A.T + 0.1 * np.eye(n_ch)) * 1e-26
    cov = mne.Covariance(
        cov_data, ch_names, bads=[], projs=[], nfree=100, verbose=False
    )
    return cov


def test_meg_defaults():
    """Verify default MEG algorithmic parameters (4 PCs, 99th percentile)."""
    n_ch = 10
    rng = np.random.default_rng(1)
    A = rng.standard_normal((n_ch, 6))
    ref_cov = A @ A.T + np.eye(n_ch)

    n_pc_eeg = _compute_default_n_pc(ref_cov, signal_type="eeg")
    n_pc_meg = _compute_default_n_pc(ref_cov, signal_type="meg")

    assert n_pc_eeg == 3
    assert n_pc_meg == 2

    # Test percentile scaling
    evals = np.array([[0.1, 0.5, 1.0, 5.0, 20.0]])
    eig_98 = _sensai_to_eigen(0.0, evals, percentile=98)
    eig_99 = _sensai_to_eigen(0.0, evals, percentile=99)
    # Higher percentile should yield a higher (more selective) artifact threshold
    assert eig_99 >= eig_98


def test_meg_mag_fit_transform_raw(meg_mag_raw, meg_ref_cov):
    """Test Gedai fit_raw and transform_raw on MEG magnetometers."""
    gedai = Gedai()
    gedai.fit_raw(
        meg_mag_raw,
        picks="mag",
        duration=1.0,
        reference_cov=meg_ref_cov,
        sensai_method="optimize",
    )

    assert gedai.fitted
    assert gedai._signal_type == "meg"
    assert gedai._percentile == 99
    assert gedai._n_pc <= 4
    assert "sensai_score" in gedai.fit_metrics_

    # Verify that average referencing was NOT applied (MEG is reference-free)
    clean_raw = gedai.transform_raw(meg_mag_raw)
    clean_data = clean_raw.get_data()

    # Data shape must match
    assert clean_data.shape == meg_mag_raw.get_data().shape
    assert not np.any(np.isnan(clean_data))
    assert not np.allclose(clean_data.mean(axis=0), 0, atol=1e-15)


def test_meg_grad_fit_transform_epochs(meg_grad_epochs):
    """Test Gedai fit_epochs and transform_epochs on MEG gradiometers."""
    ch_names = meg_grad_epochs.ch_names
    n_ch = len(ch_names)
    rng = np.random.default_rng(99)
    A = rng.standard_normal((n_ch, 4))
    cov_data = (A @ A.T + 0.1 * np.eye(n_ch)) * 1e-22
    ref_cov = mne.Covariance(
        cov_data, ch_names, bads=[], projs=[], nfree=100, verbose=False
    )

    gedai = Gedai()
    gedai.fit_epochs(
        meg_grad_epochs,
        picks="grad",
        reference_cov=ref_cov,
        sensai_method="optimize",
    )

    assert gedai.fitted
    assert gedai._signal_type == "meg"
    assert gedai._percentile == 99

    clean_epochs = gedai.transform_epochs(meg_grad_epochs)
    clean_data = clean_epochs.get_data()
    assert clean_data.shape == meg_grad_epochs.get_data().shape
    assert not np.any(np.isnan(clean_data))


def test_meg_forward_as_reference_cov(meg_mag_raw):
    """Test that an mne.Forward object is accepted directly as reference_cov."""
    ch_names = meg_mag_raw.ch_names
    n_ch = len(ch_names)
    n_sources = 50
    rng = np.random.default_rng(123)
    G = rng.standard_normal((n_ch, n_sources)) * 1e-13

    # Instantiate an mne.Forward object
    fwd_dict = {
        "coord_frame": mne._fiff.constants.FIFF.FIFFV_COORD_HEAD,
        "sol": {"data": G},
        "info": {"ch_names": ch_names, "bads": []},
    }
    fwd = mne.Forward(fwd_dict)

    cov = _ensure_cov(fwd)
    assert isinstance(cov, mne.Covariance)
    assert cov.data.shape == (n_ch, n_ch)
    assert cov.ch_names == ch_names

    # Test fitting with forward
    gedai = Gedai()
    gedai.fit_raw(meg_mag_raw, picks="mag", reference_cov=fwd)
    assert gedai.fitted
    assert gedai._signal_type == "meg"


def test_meg_informative_error_on_eeg_leadfield(meg_mag_raw):
    """Verify helpful error message when EEG leadfield is used on MEG channels."""
    gedai = Gedai()
    with pytest.raises(ValueError, match="default 'leadfield'.*EEG"):
        gedai.fit_raw(meg_mag_raw, picks="mag", reference_cov="leadfield")


def test_meg_multiband(meg_mag_raw, meg_ref_cov):
    """Test MultibandGedai on MEG data."""
    mb = MultibandGedai(wavelet_level=2)
    mb.fit_raw(
        meg_mag_raw,
        picks="mag",
        duration=1.0,
        reference_cov=meg_ref_cov,
    )
    assert mb.fitted
    clean_raw = mb.transform_raw(meg_mag_raw)
    assert not np.any(np.isnan(clean_raw.get_data()))
