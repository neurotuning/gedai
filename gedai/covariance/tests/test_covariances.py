"""Test Covariances."""

import mne
import pytest
from mne.datasets import testing

from gedai.covariance.covariance import (
    _ensure_cov,
    align_covariance_to_channel_positions,
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


def test_align_covariance_to_channel_positions(sample_info):
    """Test align_covariance_to_channel_positions with MNE objects and numpy arrays."""
    import numpy as np

    # 1. Test with mne.Covariance and sample_info
    cov_orig = mne.make_ad_hoc_cov(sample_info)
    # Add non-trivial off-diagonals
    ch_positions = np.array([ch["loc"][:3] for ch in sample_info["chs"]])
    dists = np.linalg.norm(ch_positions[:, None, :] - ch_positions[None, :, :], axis=-1)
    cov_dense = np.exp(-dists / 0.05) + 0.1 * np.eye(len(ch_positions))
    cov_orig = mne.Covariance(
        cov_dense,
        names=sample_info["ch_names"],
        bads=[],
        projs=[],
        nfree=len(ch_positions),
        verbose=False,
    )

    cov_aligned = align_covariance_to_channel_positions(cov_orig, sample_info)
    assert isinstance(cov_aligned, mne.Covariance)
    assert cov_aligned.ch_names == cov_orig.ch_names

    # Check trace and eigenvalue preservation
    evals_orig = np.sort(np.linalg.eigvalsh(cov_orig.data))[::-1]
    evals_alg = np.sort(np.linalg.eigvalsh(cov_aligned.data))[::-1]
    assert np.allclose(evals_orig, evals_alg, rtol=1e-6)
    assert np.isclose(np.trace(cov_orig.data), np.trace(cov_aligned.data), rtol=1e-6)

    # Check top 3 PCs span the Cartesian sensor coordinate subspace
    coords_centered = ch_positions - np.mean(ch_positions, axis=0)
    Q_coords, _ = np.linalg.qr(coords_centered[:, :3])
    _, evecs_alg = np.linalg.eigh(cov_aligned.data)
    evecs_alg_top3 = evecs_alg[:, -3:]
    cos_angles = np.linalg.svd(Q_coords.T @ evecs_alg_top3, compute_uv=False)
    assert np.allclose(cos_angles, [1.0, 1.0, 1.0], atol=1e-5)

    # 2. Test with raw numpy arrays
    arr_aligned = align_covariance_to_channel_positions(cov_dense, ch_positions)
    assert isinstance(arr_aligned, np.ndarray)
    assert arr_aligned.shape == cov_dense.shape

    # 3. Test missing coordinates (all zeros) returns original unchanged
    zero_pos = np.zeros_like(ch_positions)
    cov_zero = align_covariance_to_channel_positions(cov_orig, zero_pos)
    assert np.array_equal(cov_zero.data, cov_orig.data)


def test_ensure_cov_geometric(sample_info):
    """Test _ensure_cov with geometric identifiers and automatic picking."""
    cov_geom = _ensure_cov("leadfield_geometric")
    assert isinstance(cov_geom, mne.Covariance)
    assert cov_geom.get("_align_to_sensors") is True

    cov_picked = _pick_cov(cov_geom, sample_info)
    assert isinstance(cov_picked, mne.Covariance)
    assert set(cov_picked.ch_names) == set(sample_info["ch_names"])


def test_compute_covariance_from_channel_positions_methods(sample_info):
    """Test compute_covariance_from_channel_positions with geometric and rbf."""
    cov_geom = compute_covariance_from_channel_positions(sample_info, method="geometric")
    assert isinstance(cov_geom, mne.Covariance)

    cov_rbf = compute_covariance_from_channel_positions(sample_info, method="rbf")
    assert isinstance(cov_rbf, mne.Covariance)
