"""Tests for SENSAI optimization module."""

import numpy as np
import pytest

from gedai.sensai.sensai import (
    _find_changepoint,
    _precompute_gevd,
    _sensai_gridsearch,
    _sensai_optimize,
    _sensai_score,
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


def test_sensai_similarity_is_normalized_and_rejects_invalid_n_pc():
    """Similarity metrics stay in [0, 1] and invalid n_pc values are rejected."""
    rng = np.random.default_rng(42)
    n_ch = 8
    ref_cov = np.eye(n_ch)
    epochs_data = rng.standard_normal((5, n_ch, 40))

    score, sig_sim, noi_sim = _sensai_score(
        epochs_data,
        threshold=2.0,
        reference_cov=ref_cov,
        n_pc=3,
        noise_multiplier=3.0,
    )
    assert np.isfinite(score)
    assert 0.0 <= sig_sim <= 100.0
    assert 0.0 <= noi_sim <= 100.0

    with pytest.raises(ValueError, match="n_pc"):
        _sensai_score(
            epochs_data,
            threshold=2.0,
            reference_cov=ref_cov,
            n_pc=0,
            noise_multiplier=3.0,
        )


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

    best_thresh, best_sensai_thresh, runs = _sensai_gridsearch(
        epochs_data,
        reference_cov=ref_cov,
        n_pc=2,
        noise_multiplier=3.0,
        eigen_thresholds=eigen_thresholds,
        all_eval=all_eval,
        all_evec=all_evec,
    )
    assert best_thresh in eigen_thresholds
    assert isinstance(best_sensai_thresh, float)
    assert len(runs) == len(eigen_thresholds)

    opt_thresh, opt_sensai_thresh, opt_runs = _sensai_optimize(
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
    assert isinstance(opt_sensai_thresh, float)
    assert len(opt_runs) > 0


def test_prescan_meg_artifact_spectrum():
    """Verify the MEG artifact spectrum prescan returns 2 or 3 appropriately."""
    from gedai.sensai.sensai import (
        _compute_default_n_pc,
        _prescan_meg_artifact_spectrum,
    )

    rng = np.random.default_rng(42)
    n_ch = 20
    ref_cov = np.eye(n_ch)

    # Low noise / flat spectrum -> the prescan should prefer the compact MEG default.
    data_low = rng.standard_normal((n_ch, 500))
    n_pc_low = _prescan_meg_artifact_spectrum(data_low, ref_cov)
    assert n_pc_low in (2, 3)

    # Check _compute_default_n_pc dispatcher
    assert _compute_default_n_pc(ref_cov, signal_type="eeg") == 3
    assert _compute_default_n_pc(ref_cov, signal_type="meg", data=data_low) in (2, 3)


def test_sensai_numpy_torch_parity():
    """Verify numerical parity between numpy and torch SENSAI optimization."""
    from gedai.sensai.sensai import _precompute_gevd, _sensai_optimize
    from gedai.utils._torch_backend import has_torch

    if not has_torch():
        return

    rng = np.random.default_rng(123)
    n_ep, n_ch, n_times = 20, 15, 100
    epochs_data = rng.standard_normal((n_ep, n_ch, n_times))
    ref_cov = rng.standard_normal((n_ch, n_ch))
    ref_cov = ref_cov @ ref_cov.T + np.eye(n_ch) * 0.1

    all_eval, all_evec = _precompute_gevd(epochs_data, ref_cov, engine="torch")

    opt_thresh_np, opt_sensai_np, runs_np = _sensai_optimize(
        epochs_data,
        reference_cov=ref_cov,
        n_pc=3,
        noise_multiplier=3.0,
        epochs_eigenvalues=all_eval,
        bounds=(0.0, 12.0),
        all_eval=all_eval,
        all_evec=all_evec,
        engine="numpy",
        sensai_tol=0.1,
    )

    opt_thresh_torch, opt_sensai_torch, runs_torch = _sensai_optimize(
        epochs_data,
        reference_cov=ref_cov,
        n_pc=3,
        noise_multiplier=3.0,
        epochs_eigenvalues=all_eval,
        bounds=(0.0, 12.0),
        all_eval=all_eval,
        all_evec=all_evec,
        engine="torch",
        sensai_tol=0.1,
    )

    assert abs(opt_thresh_np - opt_thresh_torch) < 1e-3
    assert abs(opt_sensai_np - opt_sensai_torch) < 1e-3

    # Verify score parity across fixed thresholds
    import torch

    from gedai.sensai.sensai import _sensai_score_loop, _sensai_score_torch

    template = np.ascontiguousarray(ref_cov[:3, :].T)
    template, _ = np.linalg.qr(template)
    all_VR = np.ascontiguousarray(np.einsum("ij,ejk->eik", ref_cov, all_evec))
    abs_evals = np.ascontiguousarray(np.abs(all_eval))

    abs_evals_t = torch.from_numpy(abs_evals)
    all_VR_t = torch.from_numpy(all_VR)
    template_t = torch.from_numpy(template)

    for th in [0.5, 1.0, 2.0, 5.0]:
        sig_np, noi_np = _sensai_score_loop(abs_evals, all_VR, template, th, n_pc=3)
        sig_t, noi_t = _sensai_score_torch(
            abs_evals_t, all_VR_t, template_t, th, n_pc=3
        )
        assert abs(float(np.mean(sig_np)) - float(sig_t.mean().item())) < 1e-10
        assert abs(float(np.mean(noi_np)) - float(noi_t.mean().item())) < 1e-10


def test_dipolar_guardrail_rank1_and_z3_invariance():
    """Built-in EEG dipolar guardrail detects rank-1 modes and Z3 is scale-invariant."""
    from gedai.sensai.sensai import (
        _sensai_score_loop,
        _sensai_score_torch,
        compute_z3_noise_multiplier,
    )
    from gedai.utils._torch_backend import has_torch

    rng = np.random.default_rng(7)
    n_ep, n_ch, n_times = 8, 12, 200
    ref_cov = rng.standard_normal((n_ch, n_ch))
    ref_cov = ref_cov @ ref_cov.T + 0.2 * np.eye(n_ch)
    evals_ref, evecs_ref = np.linalg.eigh(ref_cov)
    template = np.ascontiguousarray(evecs_ref[:, ::-1][:, :3])

    # Inject a strong rank-1 dipolar rhythm aligned with top template mode
    epochs_data = 0.2 * rng.standard_normal((n_ep, n_ch, n_times))
    epochs_data += 3.0 * template[:, 0:1] * rng.standard_normal((n_ep, 1, n_times))

    all_eval, all_evec = _precompute_gevd(epochs_data, ref_cov, engine="numpy")
    all_vr = np.ascontiguousarray(np.einsum("ij,ejk->eik", ref_cov, all_evec))
    abs_eval = np.ascontiguousarray(np.abs(all_eval))
    # Pick threshold that isolates exactly 1 component (< n_pc=3) per epoch
    th_rank1 = 0.5 * (
        float(np.mean(np.sort(abs_eval, axis=1)[:, -1]))
        + float(np.mean(np.sort(abs_eval, axis=1)[:, -2]))
    )

    _, noi_on = _sensai_score_loop(abs_eval, all_vr, template, th_rank1, n_pc=3)
    assert np.all((noi_on > 0.0) & (noi_on <= 1.0))

    if has_torch():
        import torch

        _, noi_on_t = _sensai_score_torch(
            torch.from_numpy(abs_eval),
            torch.from_numpy(all_vr),
            torch.from_numpy(template),
            th_rank1,
            n_pc=3,
        )
        np.testing.assert_allclose(noi_on_t.numpy(), noi_on, atol=1e-10)

    m1, c1, pr1, br1 = compute_z3_noise_multiplier(
        all_eval, all_evec, ref_cov, template, n_pc=3, epoch_samples=10000
    )
    # Scaling GEVD eigenvalues & ref_cov leaves Z3 invariant
    m2, c2, pr2, br2 = compute_z3_noise_multiplier(
        all_eval * 1e6,
        all_evec * 1e-3,
        ref_cov * 1e6,
        template,
        n_pc=3,
        epoch_samples=10000,
    )
    assert np.isfinite(m1)
    assert 0.5 <= m1 <= 10.0
    np.testing.assert_allclose([m1, c1, pr1, br1], [m2, c2, pr2, br2], rtol=1e-12)
