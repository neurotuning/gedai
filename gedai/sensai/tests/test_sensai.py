"""Tests for SENSAI optimization module."""

import numpy as np

from gedai.sensai.sensai import (
    _find_changepoint,
    _precompute_gevd,
    _sensai_gridsearch,
    _sensai_optimize,
    _sensai_score,
    compute_enova_per_channel,
    compute_enova_per_epoch,
    enova_summary,
    subspace_angles,
    subspace_similarity,
)


def test_subspace_similarity_and_angles():
    """Check subspace similarity and angle calculations behave as expected."""
    rng = np.random.default_rng(42)
    n_ch = 10
    n_pc = 3

    # Identical subspaces -> similarity = 1.0, angles = 0
    A, _ = np.linalg.qr(rng.standard_normal((n_ch, n_pc)))
    sim = subspace_similarity(A, A)
    assert np.isclose(sim, 1.0)
    angles = subspace_angles(A, A)
    assert np.allclose(angles, 0.0, atol=1e-5)

    # Orthogonal subspaces
    Q, _ = np.linalg.qr(rng.standard_normal((n_ch, n_ch)))
    A = Q[:, :3]
    B = Q[:, 3:6]
    sim = subspace_similarity(A, B)
    assert np.isclose(sim, 0.0, atol=1e-7)


def test_changepoint_detection():
    """Ensure a sharp signal transition is detected as a changepoint."""
    # Signal with a sharp gradient shift
    y = np.concatenate([np.linspace(10.0, 0.0, 15), np.zeros(15)])
    cp = _find_changepoint(y, smooth_window=2)
    assert cp is not None
    assert 10 <= cp <= 18


def test_gevd_and_sensai_scoring():
    """Check GEVD and SENSAI scoring produce valid outputs."""
    rng = np.random.default_rng(42)
    n_ep, n_ch, n_times = 10, 8, 100
    epochs_data = rng.standard_normal((n_ep, n_ch, n_times))
    ref_cov = np.eye(n_ch) + 0.1 * rng.standard_normal((n_ch, n_ch))
    ref_cov = ref_cov @ ref_cov.T + np.eye(n_ch)

    all_eval, all_evec = _precompute_gevd(epochs_data, ref_cov)
    assert all_eval.shape == (n_ep, n_ch)
    assert all_evec.shape == (n_ep, n_ch, n_ch)

    # Score calculation
    score, sig_sim, noi_sim = _sensai_score(
        epochs_data, threshold=2.0, reference_cov=ref_cov, n_pc=3, noise_multiplier=3.0
    )
    assert isinstance(score, float)
    assert 0 <= sig_sim <= 100
    assert 0 <= noi_sim <= 100


def test_sensai_gridsearch_and_optimize():
    """Verify the grid-search and optimize routines agree on output shape."""
    rng = np.random.default_rng(42)
    n_ep, n_ch, n_times = 15, 6, 80
    epochs_data = rng.standard_normal((n_ep, n_ch, n_times))
    ref_cov = np.eye(n_ch)

    all_eval, all_evec = _precompute_gevd(epochs_data, ref_cov)
    eigen_thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]

    best_thresh, runs = _sensai_gridsearch(
        epochs_data,
        reference_cov=ref_cov,
        n_pc=2,
        noise_multiplier=3.0,
        eigen_thresholds=eigen_thresholds,
        all_eval=all_eval,
        all_evec=all_evec,
    )
    assert best_thresh in eigen_thresholds
    assert len(runs) == len(eigen_thresholds)

    opt_thresh, opt_runs = _sensai_optimize(
        epochs_data,
        reference_cov=ref_cov,
        n_pc=2,
        noise_multiplier=3.0,
        epochs_eigenvalues=all_eval,
        bounds=(0.0, 12.0),
        all_eval=all_eval,
        all_evec=all_evec,
    )
    assert isinstance(opt_thresh, float)
    assert len(opt_runs) > 0


def test_enova_metrics():
    """Check ENOVA is zero when the noise term is absent."""
    n_ch, n_times = 4, 200
    clean = np.ones((n_ch, n_times), dtype=np.float32)
    noise = np.zeros((n_ch, n_times), dtype=np.float32)
    ep_samples = 50

    enova_ep = compute_enova_per_epoch(clean, noise, ep_samples)
    assert len(enova_ep) == 4
    assert np.allclose(enova_ep, 0.0)

    enova_ch = compute_enova_per_channel(clean, noise, ep_samples)
    assert len(enova_ch) == n_ch
    assert np.allclose(enova_ch, 0.0)

    summary = enova_summary(enova_ep)
    assert summary["mean"] == 0.0
