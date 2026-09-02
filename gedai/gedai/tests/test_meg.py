"""Tests for MEG data support in GEDAI using MNE testing dataset."""

import mne
import numpy as np
import pytest
from mne.datasets import testing

from gedai.covariance.covariance import _ensure_cov
from gedai.gedai.gedai import Gedai
from gedai.gedai.multiband import MultibandGedai
from gedai.sensai.sensai import _compute_default_n_pc, _sensai_to_eigen

data_path = testing.data_path(download=False)
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fname_fwd = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-eeg-oct-6-fwd.fif"


@pytest.fixture(scope="module")
def meg_raw():
    """Load sample MEG Raw instance from MNE testing dataset."""
    if not fname_raw.exists():
        pytest.skip("Requires MNE testing dataset")
    raw = mne.io.read_raw_fif(fname_raw, preload=True, verbose=False)
    raw.crop(0, 3.0)  # 3 seconds for fast execution
    return raw


@pytest.fixture(scope="module")
def meg_fwd():
    """Load sample MEG Forward solution from MNE testing dataset."""
    if not fname_fwd.exists():
        pytest.skip("Requires MNE testing dataset")
    fwd = mne.read_forward_solution(fname_fwd, verbose=False)
    return fwd


@pytest.fixture
def meg_mag_raw(meg_raw):
    """Return MEG Raw picked on magnetometers."""
    raw = meg_raw.copy().pick("mag")
    return raw


@pytest.fixture
def meg_grad_epochs(meg_raw):
    """Return MEG Epochs picked on gradiometers."""
    raw = meg_raw.copy().pick("grad")
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, overlap=0.0, preload=True, verbose=False
    )
    return epochs


@pytest.fixture
def meg_ref_cov(meg_fwd):
    """Compute reference covariance from MNE testing forward solution for magnetometers."""
    fwd_mag = mne.pick_types_forward(meg_fwd, meg="mag", eeg=False)
    cov = compute_covariance_from_forward(fwd_mag)
    return cov


def test_meg_defaults():
    """Verify default MEG algorithmic parameters (2/3 PCs, 99th percentile)."""
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


@testing.requires_testing_data
def test_meg_mag_fit_transform_raw(meg_mag_raw, meg_ref_cov):
    """Test Gedai fit_raw and transform_raw on real Neuromag magnetometers."""
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


@testing.requires_testing_data
def test_meg_grad_fit_transform_epochs(meg_grad_epochs, meg_fwd):
    """Test Gedai fit_epochs and transform_epochs on real Neuromag gradiometers."""
    fwd_grad = mne.pick_types_forward(meg_fwd, meg="grad", eeg=False)
    ref_cov = compute_covariance_from_forward(fwd_grad)

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


@testing.requires_testing_data
def test_meg_forward_as_reference_cov(meg_mag_raw, meg_fwd):
    """Test that an mne.Forward object is accepted directly as reference_cov."""
    fwd_mag = mne.pick_types_forward(meg_fwd, meg="mag", eeg=False)
    cov = _ensure_cov(fwd_mag)
    assert isinstance(cov, mne.Covariance)
    assert cov.data.shape == (len(meg_mag_raw.ch_names), len(meg_mag_raw.ch_names))

    # Test fitting with forward
    gedai = Gedai()
    gedai.fit_raw(meg_mag_raw, picks="mag", reference_cov=fwd_mag)
    assert gedai.fitted
    assert gedai._signal_type == "meg"


@testing.requires_testing_data
def test_meg_informative_error_on_eeg_leadfield(meg_mag_raw):
    """Verify helpful error message when EEG leadfield is used on MEG channels."""
    gedai = Gedai()
    with pytest.raises(ValueError, match="default 'leadfield'.*EEG"):
        gedai.fit_raw(meg_mag_raw, picks="mag", reference_cov="leadfield")


@testing.requires_testing_data
def test_meg_multiband(meg_mag_raw, meg_ref_cov):
    """Test MultibandGedai on real MEG data."""
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
