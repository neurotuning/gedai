import numpy as np
from scipy.linalg import eigh

from ..utils._torch_backend import clean_epochs_batched_torch, resolve_engine


def _clean_epochs(epochs_data, reference_cov, threshold, engine="numpy"):
    resolved = resolve_engine(engine)
    if resolved == "torch":
        return clean_epochs_batched_torch(epochs_data, reference_cov, threshold)

    # Reconstruct data
    cleaned_epochs = np.zeros_like(epochs_data)
    artefact_epochs = np.zeros_like(epochs_data)

    for e, epoch_data in enumerate(epochs_data):
        covariance = np.cov(epoch_data)
        eigenvalues, eigenvectors = eigh(covariance, reference_cov, check_finite=True)

        eigvecs_filtered = eigenvectors.copy()
        signal_mask = np.abs(eigenvalues) < threshold
        eigvecs_filtered[:, signal_mask] = 0

        # Direct Regularized Reference Covariance Projection:
        # Since V^T * C_ref * V = I, the spatial maps are C_ref * V.
        # Artifact projection: C_ref * V_art * (V_art^T * X)
        artifact_tc = eigvecs_filtered.T @ epoch_data
        artefact_data = reference_cov @ (eigvecs_filtered @ artifact_tc)

        artefact_epochs[e] = artefact_data
        cleaned_epochs[e] = epoch_data - artefact_data

    return (cleaned_epochs, artefact_epochs)
