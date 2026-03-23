import numpy as np
from mne.parallel import parallel_func
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from ..gedai.decompose import _clean_epochs


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
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A, _ = np.linalg.qr(A)
    B, _ = np.linalg.qr(B)

    S = np.linalg.svd(A.T @ B, compute_uv=False)
    S_clipped = np.clip(S, -1.0, 1.0)
    angles_rad = np.arccos(S_clipped)
    return np.sort(angles_rad)


def _sensai_to_eigen(sensai_value, eigenvalues, percentile=95):
    """Convert a SENSAI score (0-105 scale) to an eigenvalue threshold.

    Parameters
    ----------
    sensai_value : float
        SENSAI threshold on the 0–105 scale.
    eigenvalues : np.ndarray
        All per-epoch GEVD eigenvalues (used to calibrate the scale).
    percentile : float
        Percentile of the log-eigenvalue distribution used as the reference
        point.  MATLAB uses **98** for EEG and **99** for MEG (the original
        Python code was 95).  Default is 95 (backward compatible).

    Returns
    -------
    eigenvalue : float
        The corresponding raw eigenvalue threshold.
    """
    all_diagonals = np.abs(eigenvalues.T.flatten())
    valid_diags = all_diagonals[all_diagonals > 0]
    if len(valid_diags) == 0:
        return 0.0

    log_eig = np.log10(valid_diags)
    min_log = np.min(log_eig)

    offset = np.abs(min_log) + 1.0
    shifted_log = log_eig + offset

    T1 = (105 - sensai_value) / 100
    threshold1 = T1 * np.percentile(shifted_log, percentile)

    eigenvalue = 10 ** (threshold1 - offset)
    return float(eigenvalue)


def _eigen_to_sensai(eigenvalue, eigenvalues, percentile=95):
    """Inverse of :func:`_sensai_to_eigen` — convert an eigenvalue threshold
    back to the SENSAI 0–105 scale (used for plotting).

    Parameters
    ----------
    eigenvalue : float
        Raw eigenvalue threshold.
    eigenvalues : np.ndarray
        All per-epoch GEVD eigenvalues.
    percentile : float
        Must match the value used in the forward conversion.  Default 95.
    """
    all_diagonals = np.abs(eigenvalues.T.flatten())
    valid_diags = all_diagonals[all_diagonals > 0]
    if len(valid_diags) == 0:
        return 0.0

    log_eig = np.log10(valid_diags)
    min_log = np.min(log_eig)

    offset = np.abs(min_log) + 1.0
    shifted_log = log_eig + offset

    if eigenvalue <= 0:
        return 105.0

    threshold1 = np.log10(eigenvalue) + offset
    percentile_val = np.percentile(shifted_log, percentile)

    if percentile_val == 0:
        return 105.0

    T1 = threshold1 / percentile_val
    sensai_value = 105 - T1 * 100
    return float(sensai_value)


# ---------------------------------------------------------------------------
# Fast analytical SENSAI score — port of MATLAB clean_SENSAI.m
# ---------------------------------------------------------------------------

def _sensai_score_fast(
    threshold,
    epochs_eigenvalues,
    epochs_eigenvectors,
    reference_cov_reg,
    evecs_reference,
    n_pc,
    noise_multiplier,
):
    """Compute the SENSAI score analytically from pre-computed GEVD results.

    This is a direct port of the MATLAB ``clean_SENSAI.m`` / ``SENSAI.m``
    pipeline.  Instead of reconstructing time series and re-computing
    covariances from data (as the legacy ``_sensai_score`` does), this
    function exploits the GEVD B-orthonormality property::

        V⁻¹ = Vᵀ @ B_reg

    to reconstruct signal and noise covariances analytically:

        V_bad_rows  = V_bad.T  @ B_reg          # (K_bad,  N)
        cov_noise   = V_bad_rows.T @ (V_bad_rows  * d_bad [:, None])

        V_good_rows = V_good.T @ B_reg          # (K_good, N)
        cov_signal  = V_good_rows.T @ (V_good_rows * d_good[:, None])

    No time series are reconstructed.  The GEVD is computed **once** per
    band (outside this function) and the results are reused across every
    threshold evaluation, yielding large speed gains especially for low-
    frequency bands with many samples per epoch.

    Parameters
    ----------
    threshold : float
        Eigenvalue threshold.  Components with ``|eigenvalue| >= threshold``
        are classified as artifacts.
    epochs_eigenvalues : np.ndarray, shape (n_epochs, n_channels)
        Per-epoch GEVD eigenvalues (ascending order, from ``scipy.linalg.eigh``).
    epochs_eigenvectors : np.ndarray, shape (n_epochs, n_channels, n_channels)
        Per-epoch GEVD eigenvectors (columns = generalised eigenvectors).
    reference_cov_reg : np.ndarray, shape (n_channels, n_channels)
        Regularised reference covariance matrix (the **B** matrix of the GEVD,
        already regularised before calling ``_fit_single_band``).
    evecs_reference : np.ndarray, shape (n_channels, n_pc)
        Top ``n_pc`` eigenvectors of the reference covariance (template
        subspace), pre-computed once per band outside the threshold loop.
    n_pc : int
        Number of principal components used for subspace similarity.
    noise_multiplier : float
        Noise multiplier for the SENSAI score.

    Returns
    -------
    score : float
    signal_subspace_similarity : float
    noise_subspace_similarity : float
    """
    n_epochs = len(epochs_eigenvalues)
    signal_similarities = np.empty(n_epochs)
    noise_similarities = np.empty(n_epochs)

    for e in range(n_epochs):
        eigenvalues = epochs_eigenvalues[e]    # (n_ch,)
        eigenvectors = epochs_eigenvectors[e]  # (n_ch, n_ch)

        bad_mask = np.abs(eigenvalues) >= threshold
        good_mask = ~bad_mask

        # --- Noise covariance: artifact components ---
        if np.any(bad_mask):
            V_bad = eigenvectors[:, bad_mask]               # (n_ch, K_bad)
            V_bad_rows = V_bad.T @ reference_cov_reg        # (K_bad, n_ch)
            d_bad = np.abs(eigenvalues[bad_mask])           # (K_bad,)
            # cov_noise = Σ_k d_k * v_k^T v_k
            cov_noise = V_bad_rows.T @ (V_bad_rows * d_bad[:, np.newaxis])
            cov_noise = (cov_noise + cov_noise.T) * 0.5    # enforce symmetry
            _, evecs_noise = eigh(cov_noise)
            evecs_noise = evecs_noise[:, ::-1][:, :n_pc]
            angles = subspace_angles(evecs_noise, evecs_reference)
            noise_similarities[e] = np.prod(np.cos(angles))
        else:
            noise_similarities[e] = 0.0

        # --- Signal covariance: clean components ---
        if np.any(good_mask):
            V_good = eigenvectors[:, good_mask]             # (n_ch, K_good)
            V_good_rows = V_good.T @ reference_cov_reg     # (K_good, n_ch)
            d_good = np.abs(eigenvalues[good_mask])        # (K_good,)
            cov_signal = V_good_rows.T @ (V_good_rows * d_good[:, np.newaxis])
            cov_signal = (cov_signal + cov_signal.T) * 0.5
            _, evecs_signal = eigh(cov_signal)
            evecs_signal = evecs_signal[:, ::-1][:, :n_pc]
            angles = subspace_angles(evecs_signal, evecs_reference)
            signal_similarities[e] = np.prod(np.cos(angles))
        else:
            signal_similarities[e] = 0.0

    signal_subspace_similarity = 100.0 * np.mean(signal_similarities)
    noise_subspace_similarity  = 100.0 * np.mean(noise_similarities)
    score = signal_subspace_similarity - noise_multiplier * noise_subspace_similarity
    return score, signal_subspace_similarity, noise_subspace_similarity


def _sensai_gridsearch_fast(
    epochs_eigenvalues,
    epochs_eigenvectors,
    reference_cov_reg,
    evecs_reference,
    n_pc,
    noise_multiplier,
    eigen_thresholds,
    n_jobs=1,
):
    """Grid-search SENSAI threshold using the fast analytical scorer.

    Parameters
    ----------
    epochs_eigenvalues : np.ndarray, shape (n_epochs, n_channels)
    epochs_eigenvectors : np.ndarray, shape (n_epochs, n_channels, n_channels)
    reference_cov_reg : np.ndarray, shape (n_channels, n_channels)
    evecs_reference : np.ndarray, shape (n_channels, n_pc)
    n_pc : int
    noise_multiplier : float
    eigen_thresholds : list of float
    n_jobs : int

    Returns
    -------
    best_threshold : float
    sensai_data : list of [threshold, score, signal_ss, noise_ss]
    """
    if n_jobs == 1:
        runs = [
            _sensai_score_fast(
                threshold,
                epochs_eigenvalues,
                epochs_eigenvectors,
                reference_cov_reg,
                evecs_reference,
                n_pc,
                noise_multiplier,
            )
            for threshold in eigen_thresholds
        ]
    else:
        parallel, p_fun, _ = parallel_func(
            _sensai_score_fast, n_jobs, total=len(eigen_thresholds)
        )
        runs = parallel(
            p_fun(
                threshold,
                epochs_eigenvalues,
                epochs_eigenvectors,
                reference_cov_reg,
                evecs_reference,
                n_pc,
                noise_multiplier,
            )
            for threshold in eigen_thresholds
        )

    best_threshold = eigen_thresholds[np.argmax([run[0] for run in runs])]
    sensai_data = [
        [eigen_thresholds[r], runs[r][0], runs[r][1], runs[r][2]]
        for r in range(len(runs))
    ]
    return best_threshold, sensai_data


def _sensai_optimize_fast(
    epochs_eigenvalues,
    epochs_eigenvectors,
    reference_cov_reg,
    evecs_reference,
    n_pc,
    noise_multiplier,
    bounds,
    percentile=95,
):
    """Optimize SENSAI threshold using the fast analytical scorer.

    Parameters
    ----------
    epochs_eigenvalues : np.ndarray, shape (n_epochs, n_channels)
    epochs_eigenvectors : np.ndarray, shape (n_epochs, n_channels, n_channels)
    reference_cov_reg : np.ndarray, shape (n_channels, n_channels)
    evecs_reference : np.ndarray, shape (n_channels, n_pc)
    n_pc : int
    noise_multiplier : float
    bounds : tuple of (float, float)
        SENSAI-scale bounds for the scalar minimisation.
    percentile : float
        Percentile of the log-eigenvalue distribution used for threshold
        conversion.  Use **98** for EEG, **99** for MEG (MATLAB defaults).
        Default is 95 (backward compatible).

    Returns
    -------
    eigen_threshold : float
    runs : list of [eigen_threshold, score, signal_ss, noise_ss]
    """
    runs = []

    def objective_function(sensai_threshold):
        eigen_threshold = _sensai_to_eigen(
            sensai_threshold, epochs_eigenvalues, percentile=percentile
        )
        score, sig_ss, noise_ss = _sensai_score_fast(
            eigen_threshold,
            epochs_eigenvalues,
            epochs_eigenvectors,
            reference_cov_reg,
            evecs_reference,
            n_pc,
            noise_multiplier,
        )
        runs.append([eigen_threshold, score, sig_ss, noise_ss])
        return -score

    result = minimize_scalar(objective_function, bounds=bounds, method="bounded")

    eigen_threshold = _sensai_to_eigen(result.x, epochs_eigenvalues, percentile=percentile)
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs


# ---------------------------------------------------------------------------
# Legacy API — kept for backward compatibility (used by fit_epochs path
# via direct epoch objects; not called in the spectral hot path).
# ---------------------------------------------------------------------------

def _sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier):
    """Compute the SENSAI score for a given threshold (legacy, epoch-based).

    .. deprecated::
        Prefer ``_sensai_score_fast`` which avoids time-series reconstruction.
    """
    epochs_data = epochs.get_data(verbose=False)
    epochs_clean, epochs_artefacts = _clean_epochs(
        epochs_data, reference_cov, threshold
    )

    reference_eigenvalues, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvalues = reference_eigenvalues[::-1]
    reference_eigenvectors = reference_eigenvectors[:, ::-1]
    reference_eigenvalues = reference_eigenvalues[:n_pc]
    reference_eigenvectors = reference_eigenvectors[:, :n_pc]

    signal_subspace_similarity = np.zeros((len(epochs_data), n_pc))
    noise_subspace_similarity = np.zeros((len(epochs_data), n_pc))

    for e, (epoch_clean_data, epoch_artefact_data) in enumerate(
        zip(epochs_clean, epochs_artefacts, strict=False)
    ):
        epoch_clean_covariance = np.cov(epoch_clean_data)
        _, epoch_clean_eigenvectors = eigh(epoch_clean_covariance)
        epoch_clean_eigenvectors = epoch_clean_eigenvectors[:, ::-1][:, :n_pc]
        angles = subspace_angles(epoch_clean_eigenvectors, reference_eigenvectors)
        signal_subspace_similarity[e] = np.prod(np.cos(angles))

        epoch_artefact_covariance = np.cov(epoch_artefact_data)
        _, epoch_artefact_eigenvectors = eigh(epoch_artefact_covariance)
        epoch_artefact_eigenvectors = epoch_artefact_eigenvectors[:, ::-1][:, :n_pc]
        angles = subspace_angles(epoch_artefact_eigenvectors, reference_eigenvectors)
        noise_subspace_similarity[e] = np.prod(np.cos(angles))

    signal_subspace_similarity = 100 * np.mean(signal_subspace_similarity)
    noise_subspace_similarity = 100 * np.mean(noise_subspace_similarity)

    score = signal_subspace_similarity - noise_multiplier * noise_subspace_similarity
    return score, signal_subspace_similarity, noise_subspace_similarity


def _sensai_gridsearch(
    epochs, reference_cov, n_pc, noise_multiplier, eigen_thresholds, n_jobs=1
):
    """Legacy grid-search (epoch-based). Prefer ``_sensai_gridsearch_fast``."""
    if n_jobs == 1:
        runs = [
            _sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier)
            for threshold in eigen_thresholds
        ]
    else:
        parallel, p_fun, _ = parallel_func(
            _sensai_score, n_jobs, total=len(eigen_thresholds)
        )
        runs = parallel(
            p_fun(epochs, threshold, reference_cov, n_pc, noise_multiplier)
            for threshold in eigen_thresholds
        )

    best_threshold = eigen_thresholds[np.argmax([run[0] for run in runs])]

    sensai_data = [
        [eigen_thresholds[r], runs[r][0], runs[r][1], runs[r][2]]
        for r in range(len(runs))
    ]
    return best_threshold, sensai_data


def _sensai_optimize(
    epochs, reference_cov, n_pc, noise_multiplier, epochs_eigenvalues, bounds
):
    """Legacy optimize (epoch-based). Prefer ``_sensai_optimize_fast``."""
    runs = []

    def objective_function(sensai_threshold):
        eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
        score, signal_subspace_similarity, noise_subspace_similarity = _sensai_score(
            epochs,
            eigen_threshold,
            reference_cov,
            n_pc=n_pc,
            noise_multiplier=noise_multiplier,
        )
        runs.append(
            [
                eigen_threshold,
                score,
                signal_subspace_similarity,
                noise_subspace_similarity,
            ]
        )
        return -score

    result = minimize_scalar(objective_function, bounds=bounds, method="bounded")

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    sensai_threshold = result.x
    eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs
