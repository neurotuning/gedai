import numpy as np
from mne.parallel import parallel_func
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from ..gedai.decompose import clean_epochs


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
    T1 = (105 - sensai_value) / 100
    threshold1 = T1 * np.percentile(log_eig_val_all, 95)
    eigenvalue = np.exp(threshold1 - 100)
    return eigenvalue


def _eigen_to_sensai(eigenvalue, eigenvalues):
    all_diagonals = np.abs(eigenvalues.T.flatten())
    log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
    threshold1 = np.log(eigenvalue) + 100
    T1 = threshold1 / np.percentile(log_eig_val_all, 95)
    sensai_value = 105 - T1 * 100
    return sensai_value
    

def sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier):
    epochs_data = epochs.get_data(verbose=False)
    epochs_clean, epochs_artefacts = clean_epochs(epochs_data, reference_cov, threshold)

    # Top n_pc components of reference_cov
    reference_eigenvalues, reference_eigenvectors = eigh(reference_cov)
    reference_eigenvalues = reference_eigenvalues[::-1]
    reference_eigenvectors = reference_eigenvectors[:, ::-1]
    reference_eigenvalues = reference_eigenvalues[:n_pc]
    reference_eigenvectors = reference_eigenvectors[:, :n_pc]

    # Initialize arrays for storing similarities
    signal_subspace_similarity = np.zeros((len(epochs_data), n_pc))
    noise_subspace_similarity = np.zeros((len(epochs_data), n_pc))

    for e, (epoch_clean_data, epoch_artefact_data) in enumerate(zip(epochs_clean, epochs_artefacts, strict=False)):
        # Clean signal subspace
        epoch_clean_covariance = np.cov(epoch_clean_data)
        _, epoch_clean_eigenvectors = eigh(epoch_clean_covariance)
        epoch_clean_eigenvectors = epoch_clean_eigenvectors[:, ::-1][:, :n_pc]
        angles = subspace_angles(epoch_clean_eigenvectors, reference_eigenvectors)
        signal_subspace_similarity[e] = np.prod(np.cos(angles))

        # Artefact noise subspace
        epoch_artefact_covariance = np.cov(epoch_artefact_data)
        _, epoch_artefact_eigenvectors = eigh(epoch_artefact_covariance)
        epoch_artefact_eigenvectors = epoch_artefact_eigenvectors[:, ::-1][:, :n_pc]
        angles = subspace_angles(epoch_artefact_eigenvectors, reference_eigenvectors)
        noise_subspace_similarity[e] = np.prod(np.cos(angles))

    # Compute the mean similarity for signal and noise subspaces
    signal_subspace_similarity = 100 * np.mean(signal_subspace_similarity)
    noise_subspace_similarity = 100 * np.mean(noise_subspace_similarity)

    # Compute the final score
    score = signal_subspace_similarity - noise_multiplier * noise_subspace_similarity
    return score, signal_subspace_similarity, noise_subspace_similarity


def sensai_gridsearch(epochs, reference_cov, n_pc, noise_multiplier, eigen_thresholds, n_jobs=1):
    if n_jobs == 1:
        runs = [
            sensai_score(epochs, threshold, reference_cov, n_pc, noise_multiplier)
            for threshold in eigen_thresholds
        ]
    else:
        parallel, p_fun, _ = parallel_func(
                        sensai_score, n_jobs, total=len(eigen_thresholds)
                    )
        runs = parallel(
            p_fun(epochs, threshold, reference_cov, n_pc, noise_multiplier)
            for threshold in eigen_thresholds
        )

    best_threshold = eigen_thresholds[np.argmax([run[0] for run in runs])]

    sensai_data = [[eigen_thresholds[r], runs[r][0], runs[r][1], runs[r][2]] for r in range(len(runs))]
    return best_threshold, sensai_data

def sensai_optimize(epochs, epochs_eigenvalues, reference_cov, n_pc, noise_multiplier, bounds):
    runs = []

    def objective_function(sensai_threshold):
        eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
        score, signal_subspace_similarity, noise_subspace_similarity = sensai_score(epochs, eigen_threshold, reference_cov, n_pc=n_pc, noise_multiplier=noise_multiplier)
        runs.append([eigen_threshold, score, signal_subspace_similarity, noise_subspace_similarity])
        return -score
    
    result = minimize_scalar(objective_function, bounds=bounds, method='bounded')

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    sensai_threshold = result.x
    eigen_threshold = _sensai_to_eigen(sensai_threshold, epochs_eigenvalues)
    # sort runs
    runs.sort(key=lambda x: x[0])
    return eigen_threshold, runs