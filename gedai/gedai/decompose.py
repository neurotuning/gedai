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
    T1=None,
    percentile=None,
):
    resolved = ensure_engine(engine)
    if resolved == "torch":
        return clean_epochs_batched_torch(
            epochs_data,
            reference_cov,
            threshold=threshold,
            T1=T1,
            percentile=percentile,
        )

    # Numpy fallback: compute dynamic chunk threshold if T1 & percentile are provided
    if T1 is not None and percentile is not None:
        all_evals = []
        for epoch_data in epochs_data:
            cov_ep = np.cov(epoch_data)
            evs = eigh(cov_ep, reference_cov, eigvals_only=True, check_finite=True)
            all_evals.append(evs)
        all_diags = np.abs(np.concatenate(all_evals))
        pos = all_diags[all_diags > 0]
        if len(pos) > 0:
            log_evals = np.log(pos) + 100.0
            chunk_prctile = float(np.percentile(log_evals, percentile))
            eff_thresh = float(np.exp(T1 * chunk_prctile - 100.0))
        else:
            eff_thresh = threshold if threshold is not None else 1.0
    else:
        eff_thresh = threshold

    # Reconstruct data
    cleaned_epochs = np.zeros_like(epochs_data)
    artefact_epochs = np.zeros_like(epochs_data)

    for e, epoch_data in enumerate(epochs_data):
        covariance = np.cov(epoch_data)
        eigenvalues, eigenvectors = eigh(covariance, reference_cov, check_finite=True)

        eigvecs_filtered = eigenvectors.copy()
        signal_mask = np.abs(eigenvalues) < eff_thresh
        eigvecs_filtered[:, signal_mask] = 0

        # Direct Regularized Reference Covariance Projection:
        # Since V^T * C_ref * V = I, the spatial maps are C_ref * V.
        # Artifact projection: C_ref * V_art * (V_art^T * X)
        artifact_tc = eigvecs_filtered.T @ epoch_data
        artefact_data = reference_cov @ (eigvecs_filtered @ artifact_tc)

        artefact_epochs[e] = artefact_data
        cleaned_epochs[e] = epoch_data - artefact_data

    return (cleaned_epochs, artefact_epochs)
