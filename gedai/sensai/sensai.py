import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def subspace_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculate the principal angles (in radians) between two subspaces.

    Parameters
    ----------
    A : np.ndarray
        Orthonormal basis for the first subspace (columns = basis vectors).
    B : np.ndarray
        Orthonormal basis for the second subspace.

    Returns
    -------
    angles_rad : np.ndarray
        Vector of principal angles in radians, sorted in ascending order.
    """
    # Ensure inputs are float64
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A, _ = np.linalg.qr(A)
    B, _ = np.linalg.qr(B)

    # Compute the SVD of A.T @ B
    S = np.linalg.svd(A.T @ B, compute_uv=False)

    # Clamp singular values to [-1, 1] to avoid invalid acos input
    S_clipped = np.clip(S, -1.0, 1.0)

    # Compute principal angles in radians
    angles_rad = np.arccos(S_clipped)

    # Return sorted angles
    return np.sort(angles_rad)


def subspace_similarity(
    A: np.ndarray, B: np.ndarray, n_pc: int | None = 3
) -> float:
    """Compute rank-normalized volumetric subspace similarity.

    When n_pc == 3, uses the rank-normalized 3D volumetric scale:
    - rank 0: 0.0
    - rank 1: s_1^3
    - rank 2: s_1 * s_2 * ((s_1 + s_2) / 2)
    - rank >= 3: s_1 * s_2 * s_3

    Parameters
    ----------
    A, B : np.ndarray
        (n, k) matrices whose columns span the subspaces.
    n_pc : int | None
        Target principal component dimension (default 3).

    Returns
    -------
    similarity : float in [0, 1]
    """
    S = np.linalg.svd(A.T @ B, compute_uv=False)
    S = np.clip(S, -1.0, 1.0)
    k = len(S)
    if k == 0:
        return 0.0

    if n_pc == 3:
        if k == 1:
            return float(S[0] ** 3)
        elif k == 2:
            return float(S[0] * S[1] * (0.5 * (S[0] + S[1])))
        else:
            return float(S[0] * S[1] * S[2])
    elif n_pc is not None and n_pc > 0:
        S_sub = S[:n_pc]
        return float(np.prod(S_sub))
    return float(np.prod(S))


def _prescan_meg_artifact_spectrum(
    data: np.ndarray, reference_cov: np.ndarray
) -> int:
    """Prescan GEVD artifact spectrum to adaptively choose n_pc (2 or 3) for MEG."""
    if data.ndim == 3:
        n_ch = data.shape[1]
        data_2d = data.transpose(1, 0, 2).reshape(n_ch, -1)
    else:
        data_2d = data
    cov_prescan = np.cov(data_2d)
    cov_prescan = (cov_prescan + cov_prescan.T) * 0.5
    evals_prescan, _ = eigh(cov_prescan, reference_cov)
    evals_desc = np.sort(np.abs(evals_prescan))[::-1]
    med_val = np.median(evals_desc)
    norm_evals = evals_desc / med_val if med_val > 0 else evals_desc
    n_ch = reference_cov.shape[0]
    if len(norm_evals) > 2 and (
        norm_evals[2] > 200
        or (norm_evals[1] / max(norm_evals[0], 1e-12) > 0.20 and norm_evals[0] < 12000)
    ):
        return min(3, n_ch)
    else:
        return min(2, n_ch)


def _compute_default_n_pc(
    reference_cov: np.ndarray,
    signal_type: str = "eeg",
    data: np.ndarray | None = None,
) -> int:
    """Compute default number of PCs for SENSAI (2/3 for MEG, 3 for EEG)."""
    n_ch = reference_cov.shape[0]
    if signal_type == "meg":
        if data is not None:
            return _prescan_meg_artifact_spectrum(data, reference_cov)
        return min(2, n_ch)
    return min(3, n_ch)


def _sensai_to_eigen(sensai_value, eigenvalues, percentile=98):
    all_diagonals = np.abs(np.asarray(eigenvalues).T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    T1 = (105 - sensai_value) / 100
    threshold1 = T1 * np.percentile(log_eig_val_all, percentile)
    eigenvalue = np.exp(threshold1 - 100)
    return eigenvalue


def _eigen_to_sensai(eigenvalue, eigenvalues, percentile=98):
    all_diagonals = np.abs(np.asarray(eigenvalues).T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    threshold1 = np.log(eigenvalue) + 100
    T1 = threshold1 / np.percentile(log_eig_val_all, percentile)
    sensai_value = 105 - T1 * 100
    return sensai_value


def _precompute_gevd(epochs_data: np.ndarray, reference_cov: np.ndarray):
    """Precompute generalized eigenvalue decomposition across all epochs.

    Parameters
    ----------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
    reference_cov : np.ndarray, shape (n_channels, n_channels)

    Returns
    -------
    all_eval : np.ndarray, shape (n_epochs, n_channels)
    all_evec : np.ndarray, shape (n_epochs, n_channels, n_channels)
    """
    n_ep, n_ch, n_times = epochs_data.shape
    all_eval = np.zeros((n_ep, n_ch), dtype=np.float64)
    all_evec = np.zeros((n_ep, n_ch, n_ch), dtype=np.float64)

    # Vectorized covariance: batch all epochs at once via einsum
    centered = epochs_data - epochs_data.mean(axis=-1, keepdims=True)
    all_covs = np.einsum("eij,ekj->eik", centered, centered)
    all_covs *= 1.0 / (n_times - 1)

    for i in range(n_ep):
        evals, evecs = eigh(all_covs[i], reference_cov, check_finite=False)
        all_eval[i] = evals
        all_evec[i] = evecs

    return all_eval, all_evec


def _sensai_score_loop(
    abs_evals: np.ndarray,
    all_VR: np.ndarray,
    template: np.ndarray,
    threshold: float,
    n_pc: int = 3,
    signal_type: str = "eeg",
    all_evec: np.ndarray | None = None,
    reference_cov: np.ndarray | None = None,
):
    """NumPy optimized loop for fast power-weighted SENSAI optimization scoring."""
    n_ep, n_ch = abs_evals.shape
    sig_sims = np.zeros(n_ep, dtype=np.float64)
    noi_sims = np.zeros(n_ep, dtype=np.float64)
    eye_n_pc = np.eye(n_ch)[:, :n_pc]
    empty_noi_sim = subspace_similarity(eye_n_pc, template, n_pc)

    T_mat = (reference_cov @ template) if reference_cov is not None else None

    for e in range(n_ep):
        evals_e = abs_evals[e]
        VR_e = all_VR[e]
        evec_e = all_evec[e] if all_evec is not None else None
        bad_mask = evals_e >= threshold
        num_bad = int(np.sum(bad_mask))

        # --- Artifact noise subspace ---
        if signal_type == "meg":
            if num_bad > 0:
                P_bad = VR_e[:, bad_mask]
                Q_bad, _ = np.linalg.qr(P_bad)
                s = np.linalg.svd(Q_bad.T @ template, compute_uv=False)
                noi_sims[e] = float(np.sum(s ** 6))
            else:
                noi_sims[e] = 0.0
        else:
            if num_bad >= n_pc and evec_e is not None and T_mat is not None:
                VR_bad = VR_e[:, bad_mask]
                evec_bad = evec_e[:, bad_mask]
                d_bad = evals_e[bad_mask]
                Y1_n = VR_bad @ (d_bad[:, None] * (evec_bad.T @ T_mat))
                Q1_n, _ = np.linalg.qr(Y1_n)
                T_noi = reference_cov @ Q1_n
                Y2_n = VR_bad @ (d_bad[:, None] * (evec_bad.T @ T_noi))
                basis_n, _ = np.linalg.qr(Y2_n)
                noi_sims[e] = abs(float(np.linalg.det(basis_n.T @ template)))
            elif num_bad > 0:
                VR_bad = VR_e[:, bad_mask]
                d_bad = evals_e[bad_mask]
                cov_noise = (VR_bad * d_bad) @ VR_bad.T
                cov_noise = (cov_noise + cov_noise.T) * 0.5
                if np.max(np.abs(cov_noise)) < 1e-12:
                    noi_sims[e] = empty_noi_sim
                else:
                    Y1_n = cov_noise @ template
                    Q1_n, _ = np.linalg.qr(Y1_n)
                    basis_n, _ = np.linalg.qr(cov_noise @ Q1_n)
                    noi_sims[e] = abs(float(np.linalg.det(basis_n.T @ template)))
            else:
                noi_sims[e] = empty_noi_sim

        # --- Clean signal subspace ---
        good_mask = ~bad_mask
        num_good = int(np.sum(good_mask))
        if num_good >= n_pc and evec_e is not None and T_mat is not None:
            VR_good = VR_e[:, good_mask]
            evec_good = evec_e[:, good_mask]
            d_good = evals_e[good_mask]
            Y1_s = VR_good @ (d_good[:, None] * (evec_good.T @ T_mat))
            Q1_s, _ = np.linalg.qr(Y1_s)
            T_sig = reference_cov @ Q1_s
            Y2_s = VR_good @ (d_good[:, None] * (evec_good.T @ T_sig))
            basis_s, _ = np.linalg.qr(Y2_s)
            sig_sims[e] = abs(float(np.linalg.det(basis_s.T @ template)))
        elif num_good > 0:
            VR_good = VR_e[:, good_mask]
            d_good = evals_e[good_mask]
            cov_signal = (VR_good * d_good) @ VR_good.T
            cov_signal = (cov_signal + cov_signal.T) * 0.5
            if np.max(np.abs(cov_signal)) < 1e-12:
                sig_sims[e] = 0.0
            else:
                Y1_s = cov_signal @ template
                Q1_s, _ = np.linalg.qr(Y1_s)
                basis_s, _ = np.linalg.qr(cov_signal @ Q1_s)
                sig_sims[e] = abs(float(np.linalg.det(basis_s.T @ template)))
        else:
            sig_sims[e] = 0.0

    return sig_sims, noi_sims


def _sensai_score_from_gevd(
    all_eigenvalues: np.ndarray,
    all_eigenvectors: np.ndarray,
    reference_cov: np.ndarray,
    reference_eigenvectors: np.ndarray,
    threshold: float,
    n_pc: int = 3,
    noise_multiplier: float = 3.0,
    signal_type: str = "eeg",
) -> tuple[float, float, float]:
    """Fast SENSAI score using cached GEVD and power-weighted 2-step QR subspace iteration."""
    template = np.ascontiguousarray(reference_eigenvectors[:, :n_pc])

    # Precompute reference_cov @ eigenvectors for all epochs at once
    all_VR = np.einsum("ij,ejk->eik", reference_cov, all_eigenvectors)
    abs_evals = np.abs(all_eigenvalues)

    sig_sims, noi_sims = _sensai_score_loop(
        np.ascontiguousarray(abs_evals),
        np.ascontiguousarray(all_VR),
        template,
        threshold,
        n_pc=n_pc,
        signal_type=signal_type,
        all_evec=all_eigenvectors,
        reference_cov=reference_cov,
    )

    sig_sim = float(np.mean(sig_sims) * 100.0)
    noi_sim = float(np.mean(noi_sims) * 100.0)
    score = float(sig_sim - (noise_multiplier * noi_sim))

    return score, sig_sim, noi_sim


def _find_changepoint(y: np.ndarray, smooth_window: int = 6) -> int | None:
    """Detect the first changepoint in a 1-D signal's gradient mean.

    Equivalent to MATLAB's `findchangepts(diff(smoothdata(y, "movmean", 6)),
    Statistic="mean", MaxNumChanges=2)`.

    Used as a safeguard against degenerate SENSAI curves.
    """
    if len(y) < smooth_window + 2:
        return None
    w = max(2, min(smooth_window, len(y) // 2))
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    y_padded = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(w) / w
    y_smooth = np.convolve(y_padded, kernel, mode="valid")
    grad = np.diff(y_smooth)
    if len(grad) < 3:
        return None
    n = len(grad)
    best_score = -np.inf
    best_idx = None
    for k in range(2, n - 1):
        left_mean = grad[:k].mean()
        right_mean = grad[k:].mean()
        score = abs(left_mean - right_mean) * np.sqrt(k * (n - k) / n)
        if score > best_score:
            best_score = score
            best_idx = k
    if best_idx is None or best_score < 1e-6:
        return None
    return best_idx + 1


def _sensai_score(epochs, threshold, reference_cov, n_pc=3, noise_multiplier=3.0):
    """Compute the SENSAI score for given threshold.

    Parameters
    ----------
    epochs : mne.Epochs or np.ndarray
        Input epochs data (or Epochs instance).
    threshold : float
        Eigenvalue threshold.
    reference_cov : np.ndarray
        Reference covariance matrix.
    n_pc : int
        Number of principal components.
    noise_multiplier : float
        Noise multiplier.

    Returns
    -------
    score : float
    signal_subspace_similarity : float
    noise_subspace_similarity : float
    """
    if hasattr(epochs, "get_data"):
        epochs_data = epochs.get_data(verbose=False)
    else:
        epochs_data = np.asarray(epochs)

    _, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvectors = reference_eigenvectors[:, ::-1][:, :n_pc]

    all_eval, all_evec = _precompute_gevd(epochs_data, reference_cov)
    return _sensai_score_from_gevd(
        all_eval,
        all_evec,
        reference_cov,
        reference_eigenvectors,
        threshold,
        n_pc,
        noise_multiplier,
    )


def _sensai_gridsearch(
    epochs,
    reference_cov,
    n_pc,
    noise_multiplier,
    eigen_thresholds,
    n_jobs=1,
    verbose=None,
    all_eval=None,
    all_evec=None,
    sensai_thresholds=None,
    signal_type="eeg",
):
    if hasattr(epochs, "get_data"):
        epochs_data = epochs.get_data(verbose=False)
    else:
        epochs_data = np.asarray(epochs)

    # Subsample if too many epochs
    MAX_EPOCHS = 200
    if len(epochs_data) > MAX_EPOCHS:
        idx = np.linspace(0, len(epochs_data) - 1, MAX_EPOCHS, dtype=int)
        epochs_data = epochs_data[idx]
        if all_eval is not None:
            all_eval = all_eval[idx]
        if all_evec is not None:
            all_evec = all_evec[idx]

    _, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvectors = reference_eigenvectors[:, ::-1][:, :n_pc]

    if all_eval is None or all_evec is None:
        all_eval, all_evec = _precompute_gevd(epochs_data, reference_cov)

    # Precompute template and all_VR once for all scoring evaluations
    template = np.ascontiguousarray(reference_eigenvectors[:, :n_pc])
    all_VR = np.ascontiguousarray(np.einsum("ij,ejk->eik", reference_cov, all_evec))
    abs_evals = np.ascontiguousarray(np.abs(all_eval))

    runs = []
    for threshold in eigen_thresholds:
        sig_sims, noi_sims = _sensai_score_loop(
            abs_evals,
            all_VR,
            template,
            threshold,
            n_pc=n_pc,
            signal_type=signal_type,
            all_evec=all_evec,
            reference_cov=reference_cov,
        )
        sig_sim = float(np.mean(sig_sims) * 100.0)
        noi_sim = float(np.mean(noi_sims) * 100.0)
        score = float(sig_sim - (noise_multiplier * noi_sim))
        runs.append((score, sig_sim, noi_sim))

    scores = np.array([run[0] for run in runs])
    noise_sims = np.array([run[2] for run in runs])
    best_idx = int(np.argmax(scores))

    # Check for degenerate monotonic / boundary curve safeguards
    if len(runs) >= 4:
        peak_is_at_boundary = best_idx >= len(runs) - 2
        baseline_score = float(np.median(scores[: max(1, len(runs) // 4)]))
        peak_score = float(scores[best_idx])
        dramatic_rise = (
            abs(peak_score) > 5 * max(abs(baseline_score), 1.0)
            and peak_score > baseline_score + 20
        )
        if peak_is_at_boundary and dramatic_rise:
            baseline_tol = max(1.0, 0.1 * abs(baseline_score))
            near_baseline = np.where(np.abs(scores - baseline_score) <= baseline_tol)[0]
            if len(near_baseline) > 0:
                best_idx = int(near_baseline[-1])
            else:
                best_idx = 0
        else:
            noise_changepoint_idx = _find_changepoint(noise_sims)
            if (
                noise_changepoint_idx is not None
                and best_idx > noise_changepoint_idx
                and noise_changepoint_idx > 0
                and peak_is_at_boundary
            ):
                best_idx = noise_changepoint_idx

    best_threshold = eigen_thresholds[best_idx]

    sensai_data = [
        [eigen_thresholds[r], runs[r][0], runs[r][1], runs[r][2]]
        for r in range(len(runs))
    ]
    return best_threshold, sensai_data


def _sensai_optimize(
    epochs,
    reference_cov,
    n_pc,
    noise_multiplier,
    epochs_eigenvalues,
    bounds,
    all_eval=None,
    all_evec=None,
    percentile=98,
    signal_type="eeg",
):
    if hasattr(epochs, "get_data"):
        epochs_data = epochs.get_data(verbose=False)
    else:
        epochs_data = np.asarray(epochs)

    # Subsample if too many epochs
    MAX_EPOCHS = 200
    if len(epochs_data) > MAX_EPOCHS:
        idx = np.linspace(0, len(epochs_data) - 1, MAX_EPOCHS, dtype=int)
        epochs_data = epochs_data[idx]
        if all_eval is not None:
            all_eval = all_eval[idx]
        if all_evec is not None:
            all_evec = all_evec[idx]

    _, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvectors = reference_eigenvectors[:, ::-1][:, :n_pc]

    if all_eval is None or all_evec is None:
        all_eval, all_evec = _precompute_gevd(epochs_data, reference_cov)

    template = np.ascontiguousarray(reference_eigenvectors[:, :n_pc])
    all_VR = np.ascontiguousarray(np.einsum("ij,ejk->eik", reference_cov, all_evec))
    abs_evals = np.ascontiguousarray(np.abs(all_eval))

    runs = []

    def objective_function(sensai_threshold):
        eigen_threshold = _sensai_to_eigen(
            sensai_threshold, epochs_eigenvalues, percentile=percentile
        )
        sig_sims, noi_sims = _sensai_score_loop(
            abs_evals,
            all_VR,
            template,
            eigen_threshold,
            n_pc=n_pc,
            signal_type=signal_type,
            all_evec=all_evec,
            reference_cov=reference_cov,
        )
        sig_sim = float(np.mean(sig_sims) * 100.0)
        noi_sim = float(np.mean(noi_sims) * 100.0)
        score = float(sig_sim - (noise_multiplier * noi_sim))

        runs.append(
            [
                eigen_threshold,
                score,
                sig_sim,
                noi_sim,
            ]
        )
        return -score

    result = minimize_scalar(
        objective_function,
        bounds=bounds,
        method="bounded",
        options={
            "xatol": 0.01,
            "maxiter": 500,
        },
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    sensai_threshold = result.x
    eigen_threshold = _sensai_to_eigen(
        sensai_threshold, epochs_eigenvalues, percentile=percentile
    )
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs
