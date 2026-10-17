import numpy as np
from scipy.linalg import eigh

from ..utils._checks import ensure_engine
from ..utils._torch_backend import (
    batched_gevd_cholesky,
    clean_epochs_batched_torch,
    robust_cholesky_gevd,
)

__all__ = [
    "_clean_epochs",
    "robust_cholesky_gevd",
    "batched_gevd_cholesky",
]


def _clean_epochs(
    epochs_data,
    reference_cov,
    threshold=None,
    engine="auto",
):
    resolved = ensure_engine(engine)
    if resolved == "torch":
        return clean_epochs_batched_torch(
            epochs_data,
            reference_cov,
            threshold=threshold,
        )

    if threshold is None:
        raise ValueError("threshold must be provided for deartifacting.")

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
