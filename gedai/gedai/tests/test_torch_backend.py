import mne
import numpy as np
import pytest
from scipy.linalg import eigh

from gedai.gedai.adaptive import AdaptiveMultibandGedai
from gedai.gedai.decompose import _clean_epochs
from gedai.gedai.gedai import Gedai
from gedai.gedai.multiband import MultibandGedai
from gedai.sensai.sensai import _precompute_gevd
from gedai.utils._torch_backend import (
    gevd_torch,
    has_torch,
    resolve_engine,
)

pytestmark = pytest.mark.skipif(not has_torch(), reason="PyTorch is not installed")


def test_has_torch_and_resolve_engine():
    """Test PyTorch availability check and engine resolution."""
    assert has_torch() is True
    assert resolve_engine("numpy") == "numpy"
    assert resolve_engine("NUMPY") == "numpy"
    assert resolve_engine("torch") == "torch"
    assert resolve_engine("TORCH") == "torch"
    assert resolve_engine("auto") == "torch"

    with pytest.raises(ValueError, match="Invalid engine"):
        resolve_engine("gpu")

    with pytest.raises(TypeError, match="engine must be a string"):
        resolve_engine(123)


def test_gevd_torch_single_and_batched():
    """Test PyTorch GEVD on single and batched matrices against SciPy eigh."""
    import torch

    rng = np.random.RandomState(42)
    n_ch = 6

    # 1. Single matrix pair
    x = rng.randn(n_ch, 50)
    a = np.cov(x)
    b = np.cov(rng.randn(n_ch, 50)) + 0.1 * np.eye(n_ch)

    scipy_evals, scipy_evecs = eigh(a, b)

    t_a = torch.from_numpy(a)
    t_b = torch.from_numpy(b)
    pt_evals, pt_evecs = gevd_torch(t_a, t_b)

    # Check eigenvalue match
    np.testing.assert_allclose(pt_evals.numpy(), scipy_evals, atol=1e-12)

    # Check B-orthonormality: V.T @ B @ V = I
    v = pt_evecs.numpy()
    np.testing.assert_allclose(v.T @ b @ v, np.eye(n_ch), atol=1e-12)

    # Check generalized eigenproblem: A @ V = B @ V @ diag(evals)
    np.testing.assert_allclose(a @ v, b @ v @ np.diag(pt_evals.numpy()), atol=1e-12)

    # 2. Batched matrix pair (B, C, C)
    n_batches = 5
    batch_x = rng.randn(n_batches, n_ch, 50)
    batch_centered = batch_x - batch_x.mean(axis=-1, keepdims=True)
    batch_a = np.einsum("bij,bkj->bik", batch_centered, batch_centered) / 49.0

    t_batch_a = torch.from_numpy(batch_a)
    pt_batch_evals, pt_batch_evecs = gevd_torch(t_batch_a, t_b)

    for i in range(n_batches):
        sc_ev, _ = eigh(batch_a[i], b)
        np.testing.assert_allclose(pt_batch_evals[i].numpy(), sc_ev, atol=1e-12)
        v_i = pt_batch_evecs[i].numpy()
        np.testing.assert_allclose(v_i.T @ b @ v_i, np.eye(n_ch), atol=1e-12)


def test_clean_epochs_batched_torch_parity():
    """Test batched epoch cleaning against NumPy implementation."""
    rng = np.random.RandomState(42)
    n_ep, n_ch, n_times = 10, 6, 100
    epochs_data = rng.randn(n_ep, n_ch, n_times)
    ref_cov = np.cov(rng.randn(n_ch, 200)) + 0.1 * np.eye(n_ch)
    threshold = 1.2

    clean_np, art_np = _clean_epochs(epochs_data, ref_cov, threshold, engine="numpy")
    clean_pt, art_pt = _clean_epochs(epochs_data, ref_cov, threshold, engine="torch")

    np.testing.assert_allclose(clean_pt, clean_np, atol=1e-12)
    np.testing.assert_allclose(art_pt, art_np, atol=1e-12)


def test_precompute_gevd_torch_parity():
    """Test precomputing GEVD across epochs using PyTorch vs NumPy."""
    rng = np.random.RandomState(42)
    n_ep, n_ch, n_times = 8, 5, 80
    epochs_data = rng.randn(n_ep, n_ch, n_times)
    ref_cov = np.cov(rng.randn(n_ch, 150)) + 0.1 * np.eye(n_ch)

    eval_np, _ = _precompute_gevd(epochs_data, ref_cov, engine="numpy")
    eval_pt, _ = _precompute_gevd(epochs_data, ref_cov, engine="torch")

    np.testing.assert_allclose(eval_pt, eval_np, atol=1e-12)


def test_gedai_fit_transform_epochs_torch_parity():
    """Test end-to-end Gedai fit and transform on epochs with PyTorch vs NumPy."""
    rng = np.random.RandomState(42)
    n_ch, n_times, n_ep = 8, 150, 15
    data = rng.randn(n_ep, n_ch, n_times)
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    cov = mne.Covariance(np.eye(n_ch), ch_names, [], [], 0)

    g_np = Gedai(engine="numpy")
    g_np.fit_epochs(
        epochs.copy(),
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
    )
    clean_epochs_np = g_np.transform_epochs(epochs.copy(), verbose=False)

    g_pt = Gedai(engine="torch")
    g_pt.fit_epochs(
        epochs.copy(),
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
    )
    clean_epochs_pt = g_pt.transform_epochs(epochs.copy(), verbose=False)

    # Fitted thresholds should match exactly
    np.testing.assert_allclose(
        g_pt._fit["threshold"], g_np._fit["threshold"], atol=1e-10
    )

    # Cleaned data should match with high precision
    np.testing.assert_allclose(
        clean_epochs_pt.get_data(), clean_epochs_np.get_data(), atol=1e-10
    )


def test_gedai_fit_transform_raw_torch_parity():
    """Test end-to-end Gedai fit and transform on raw data with PyTorch vs NumPy."""
    rng = np.random.RandomState(42)
    n_ch, n_times = 8, 800
    data = rng.randn(n_ch, n_times)
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    cov = mne.Covariance(np.eye(n_ch), ch_names, [], [], 0)

    g_np = Gedai(engine="numpy")
    g_np.fit_raw(
        raw.copy(),
        duration=1.0,
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
    )
    clean_raw_np = g_np.transform_raw(raw.copy(), verbose=False)

    g_pt = Gedai(engine="torch")
    g_pt.fit_raw(
        raw.copy(),
        duration=1.0,
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
    )
    clean_raw_pt = g_pt.transform_raw(raw.copy(), verbose=False)

    np.testing.assert_allclose(
        clean_raw_pt.get_data(), clean_raw_np.get_data(), atol=1e-10
    )


def test_multiband_gedai_torch():
    """Test MultibandGedai using PyTorch engine."""
    rng = np.random.RandomState(42)
    n_ch, n_times, n_ep = 8, 200, 10
    data = rng.randn(n_ep, n_ch, n_times)
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    cov = mne.Covariance(np.eye(n_ch), ch_names, [], [], 0)

    mb = MultibandGedai(wavelet_level=1, broadband_pass=True, engine="torch")
    mb.fit_epochs(
        epochs.copy(),
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
    )
    cleaned = mb.transform_epochs(epochs.copy(), verbose=False)

    assert cleaned.get_data().shape == epochs.get_data().shape
    assert mb.metrics_ is not None
    assert "sensai_score" in mb.metrics_


def test_adaptive_multiband_gedai_torch():
    """Test AdaptiveMultibandGedai using PyTorch engine."""
    rng = np.random.RandomState(42)
    n_ch, n_times = 8, 600
    data = rng.randn(n_ch, n_times)
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    cov = mne.Covariance(np.eye(n_ch), ch_names, [], [], 0)

    amb = AdaptiveMultibandGedai(wavelet_level=1, broadband_pass=True, engine="torch")
    amb.fit_raw(
        raw.copy(),
        reference_cov=cov,
        sensai_method="optimize",
        verbose=False,
    )
    cleaned = amb.transform_raw(raw.copy(), verbose=False)

    assert cleaned.get_data().shape == raw.get_data().shape
    assert amb.fitted is True
    for wf in amb._wavelets_fits:
        if wf["model"] is not None:
            assert wf["model"].engine == "torch"
