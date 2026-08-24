import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def subspace_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate the principal angles (in radians) between two subspaces.

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


from numba import njit

@njit(nogil=True, cache=True)
def subspace_similarity(A: np.ndarray, B: np.ndarray, n_pc: int = 0) -> float:
    """Product of cosines of principal angles between column spaces.

    Equivalent to prod(diag(S)) where [~,S,~] = svd(A'*B).
    Matches MATLAB subspace_angles.m exactly.

    Parameters
    ----------
    A, B : (n, k) matrices whose columns span the subspaces.
    n_pc : int
        Number of principal components to include in product.

    Returns
    -------
    similarity : float in [0, 1]
    """
    M = np.ascontiguousarray(A.T) @ np.ascontiguousarray(B)
    _, S, _ = np.linalg.svd(M)
    
    for i in range(len(S)):
        if S[i] > 1.0: S[i] = 1.0
        elif S[i] < -1.0: S[i] = -1.0
        
    if n_pc > 0 and n_pc < len(S):
        S = S[:n_pc]
        
    prod = 1.0
    for i in range(len(S)):
        prod *= S[i]
    return float(prod)


def _sensai_to_eigen(sensai_value, eigenvalues):
    all_diagonals = np.abs(np.asarray(eigenvalues).T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    T1 = (105 - sensai_value) / 100
    threshold1 = T1 * np.percentile(log_eig_val_all, 98)
    eigenvalue = np.exp(threshold1 - 100)
    return eigenvalue


def _eigen_to_sensai(eigenvalue, eigenvalues):
    all_diagonals = np.abs(np.asarray(eigenvalues).T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    threshold1 = np.log(eigenvalue) + 100
    T1 = threshold1 / np.percentile(log_eig_val_all, 98)
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


@njit(nogil=True, cache=True)
def _numba_sensai_score_loop(
    abs_evals: np.ndarray,
    all_VR: np.ndarray,
    template: np.ndarray,
    threshold: float,
    n_pc: int
):
    """Inner Numba compiled loop for fast SENSAI optimization scoring."""
    n_ep, n_ch = abs_evals.shape
    sig_sims = np.zeros(n_ep, dtype=np.float64)
    noi_sims = np.zeros(n_ep, dtype=np.float64)
    eye_n_pc = np.eye(n_ch)
    eye_n_pc = eye_n_pc[:, :n_pc].copy()

    for e in range(n_ep):
        evals_e = abs_evals[e]
        VR_e = all_VR[e]
        
        n_bad = 0
        for i in range(n_ch):
            if evals_e[i] >= threshold:
                n_bad += 1
        n_good = n_ch - n_bad
        
        VR_bad = np.zeros((n_ch, n_bad), dtype=np.float64)
        d_bad = np.zeros(n_bad, dtype=np.float64)
        VR_good = np.zeros((n_ch, n_good), dtype=np.float64)
        d_good = np.zeros(n_good, dtype=np.float64)
        
        b_idx = 0
        g_idx = 0
        for i in range(n_ch):
            if evals_e[i] >= threshold:
                VR_bad[:, b_idx] = VR_e[:, i]
                d_bad[b_idx] = evals_e[i]
                b_idx += 1
            else:
                VR_good[:, g_idx] = VR_e[:, i]
                d_good[g_idx] = evals_e[i]
                g_idx += 1
                
        # --- Artifact noise subspace ---
        if n_bad > 0:
            cov_noise = np.zeros((n_ch, n_ch), dtype=np.float64)
            for i in range(n_bad):
                col = VR_bad[:, i] * d_bad[i]
                for j in range(n_ch):
                    for k in range(n_ch):
                        cov_noise[j, k] += col[j] * VR_bad[k, i]
            
            cov_noise = (cov_noise + cov_noise.T) * 0.5
            
            max_val = 0.0
            for i in range(n_ch):
                for j in range(n_ch):
                    val = cov_noise[i, j]
                    if val < 0: val = -val
                    if val > max_val: max_val = val
                        
            if max_val < 1e-12:
                Y1_n = eye_n_pc.copy()
            else:
                Y1_n = np.ascontiguousarray(cov_noise) @ np.ascontiguousarray(template)
            
            # Use ascontiguousarray to prevent numba warnings and qr failure on non-contiguous memory
            Y1_n_contig = np.ascontiguousarray(Y1_n)
            Q1_n, _ = np.linalg.qr(Y1_n_contig)
            
            Q1_n_contig = np.ascontiguousarray(Q1_n)
            cov_n_contig = np.ascontiguousarray(cov_noise)
            basis_n, _ = np.linalg.qr(cov_n_contig @ Q1_n_contig)
            
            basis_n_contig = np.ascontiguousarray(basis_n)
            noi_sims[e] = subspace_similarity(basis_n_contig, np.ascontiguousarray(template), n_pc)
            
        # --- Clean signal subspace ---
        if n_good > 0:
            cov_signal = np.zeros((n_ch, n_ch), dtype=np.float64)
            for i in range(n_good):
                col = VR_good[:, i] * d_good[i]
                for j in range(n_ch):
                    for k in range(n_ch):
                        cov_signal[j, k] += col[j] * VR_good[k, i]
            
            cov_signal = (cov_signal + cov_signal.T) * 0.5
            
            cov_s_contig = np.ascontiguousarray(cov_signal)
            Y1_s = cov_s_contig @ np.ascontiguousarray(template)
            
            Y1_s_contig = np.ascontiguousarray(Y1_s)
            Q1_s, _ = np.linalg.qr(Y1_s_contig)
            
            Q1_s_contig = np.ascontiguousarray(Q1_s)
            basis_s, _ = np.linalg.qr(cov_s_contig @ Q1_s_contig)
            
            basis_s_contig = np.ascontiguousarray(basis_s)
            sig_sims[e] = subspace_similarity(basis_s_contig, np.ascontiguousarray(template), n_pc)
            
    return sig_sims, noi_sims


def _sensai_score_from_gevd(
    all_eigenvalues: np.ndarray,
    all_eigenvectors: np.ndarray,
    reference_cov: np.ndarray,
    reference_eigenvectors: np.ndarray,
    threshold: float,
    n_pc: int,
    noise_multiplier: float,
) -> tuple[float, float, float]:
    """Fast SENSAI score using cached GEVD and 2-step QR subspace iteration."""
    template = np.ascontiguousarray(reference_eigenvectors[:, :n_pc])

    # Precompute reference_cov @ eigenvectors for all epochs at once
    all_VR = np.einsum("ij,ejk->eik", reference_cov, all_eigenvectors)
    abs_evals = np.abs(all_eigenvalues)

    sig_sims, noi_sims = _numba_sensai_score_loop(
        np.ascontiguousarray(abs_evals),
        np.ascontiguousarray(all_VR),
        template,
        threshold,
        n_pc
    )

    sig_sim = np.mean(sig_sims) * 100.0
    noi_sim = np.mean(noi_sims) * 100.0
    score = sig_sim - (noise_multiplier * noi_sim)

    return float(score), float(sig_sim), float(noi_sim)


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


def _sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier):
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
        sig_sims, noi_sims = _numba_sensai_score_loop(
            abs_evals,
            all_VR,
            template,
            threshold,
            n_pc,
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
        eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
        sig_sims, noi_sims = _numba_sensai_score_loop(
            abs_evals,
            all_VR,
            template,
            eigen_threshold,
            n_pc,
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
    eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs
