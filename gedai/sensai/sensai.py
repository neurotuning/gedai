import numpy as np
from mne.parallel import parallel_func
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


def _sensai_to_eigen(sensai_value, eigenvalues):
    all_diagonals = np.abs(eigenvalues.T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    T1 = (105 - sensai_value) / 100  # TODO: Is altered by data units (Vol vs uV)
    threshold1 = T1 * np.percentile(log_eig_val_all, 98)
    eigenvalue = np.exp(threshold1 - 100)
    return eigenvalue


def _eigen_to_sensai(eigenvalue, eigenvalues):
    all_diagonals = np.abs(eigenvalues.T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    threshold1 = np.log(eigenvalue) + 100
    T1 = threshold1 / np.percentile(log_eig_val_all, 98)
    sensai_value = 105 - T1 * 100
    return sensai_value


def _sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier):
    """Compute the SENSAI score for given threshold.

    Exploits the GEVD B-orthonormality property::

        V⁻¹ = Vᵀ @ B

    to derive signal and noise covariances analytically without reconstructing
    time series::

        V_bad_rows = V_bad.T @ B  # (K_bad,  N)
        cov_noise = V_bad_rows.T @ (V_bad_rows * d_bad[:, None])

        V_good_rows = V_good.T @ B  # (K_good, N)
        cov_signal = V_good_rows.T @ (V_good_rows * d_good[:, None])

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    threshold : float
        Eigenvalue threshold. Components with ``|eigenvalue| >= threshold``
        are classified as artifacts.
    reference_cov : np.ndarray, shape (n_channels, n_channels)
        Reference covariance matrix (B matrix of the GEVD).
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
    epochs_data = epochs.get_data(verbose=False)

    # Top n_pc eigenvectors of the reference covariance (template subspace)
    _, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvectors = reference_eigenvectors[:, ::-1][:, :n_pc]

    # Initialize arrays for storing similarities
    signal_subspace_similarity = np.zeros(len(epochs_data))
    noise_subspace_similarity = np.zeros(len(epochs_data))

    for e, epoch_data in enumerate(epochs_data):
        covariance = np.cov(epoch_data)
        eigenvalues, eigenvectors = eigh(covariance, reference_cov, check_finite=False)

        bad_mask = np.abs(eigenvalues) >= threshold
        good_mask = ~bad_mask

        # --- Artefact noise subspace ---
        if np.any(bad_mask):
            V_bad = eigenvectors[:, bad_mask]  # (n_ch, K_bad)
            V_bad_rows = V_bad.T @ reference_cov  # (K_bad, n_ch)
            d_bad = np.abs(eigenvalues[bad_mask])  # (K_bad,)
            epoch_artefact_covariance = V_bad_rows.T @ (
                V_bad_rows * d_bad[:, np.newaxis]
            )
            epoch_artefact_covariance = (
                epoch_artefact_covariance + epoch_artefact_covariance.T
            ) * 0.5
            _, epoch_artefact_eigenvectors = eigh(epoch_artefact_covariance)
            epoch_artefact_eigenvectors = epoch_artefact_eigenvectors[:, ::-1][:, :n_pc]
            angles = subspace_angles(
                epoch_artefact_eigenvectors, reference_eigenvectors
            )
            noise_subspace_similarity[e] = np.prod(np.cos(angles))

        # --- Clean signal subspace ---
        if np.any(good_mask):
            V_good = eigenvectors[:, good_mask]  # (n_ch, K_good)
            V_good_rows = V_good.T @ reference_cov  # (K_good, n_ch)
            d_good = np.abs(eigenvalues[good_mask])  # (K_good,)
            epoch_clean_covariance = V_good_rows.T @ (
                V_good_rows * d_good[:, np.newaxis]
            )
            epoch_clean_covariance = (
                epoch_clean_covariance + epoch_clean_covariance.T
            ) * 0.5
            _, epoch_clean_eigenvectors = eigh(epoch_clean_covariance)
            epoch_clean_eigenvectors = epoch_clean_eigenvectors[:, ::-1][:, :n_pc]
            angles = subspace_angles(epoch_clean_eigenvectors, reference_eigenvectors)
            signal_subspace_similarity[e] = np.prod(np.cos(angles))

    # Compute the mean similarity for signal and noise subspaces
    signal_subspace_similarity = 100 * np.mean(signal_subspace_similarity)
    noise_subspace_similarity = 100 * np.mean(noise_subspace_similarity)

    # Compute the final score
    score = signal_subspace_similarity - noise_multiplier * noise_subspace_similarity
    return score, signal_subspace_similarity, noise_subspace_similarity


def _sensai_gridsearch(
    epochs,
    reference_cov,
    n_pc,
    noise_multiplier,
    eigen_thresholds,
    n_jobs=1,
    verbose=None,
):
    if n_jobs == 1:
        runs = [
            _sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier)
            for threshold in eigen_thresholds
        ]
    else:
        parallel, p_fun, _ = parallel_func(
            _sensai_score, n_jobs, total=len(eigen_thresholds), verbose=verbose
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
    # sort runs
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs
