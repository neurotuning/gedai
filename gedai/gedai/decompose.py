import numpy as np
from scipy.linalg import eigh


def _clean_epochs(epochs_data, reference_cov, threshold):
    # Reconstruct data
    cleaned_epochs = np.zeros_like(epochs_data)
    artefact_epochs = np.zeros_like(epochs_data)

    for e, epoch_data in enumerate(epochs_data):
        covariance = np.cov(epoch_data)
        eigenvalues, eigenvectors = eigh(covariance, reference_cov, check_finite=True)

        # Compute spatial maps
        maps = np.linalg.pinv(eigenvectors).T
        eigenvectors_filtered = eigenvectors.copy()

        # Zero out components with small eigenvalues
        for v, val in enumerate(eigenvalues):
            if abs(val) < threshold:
                maps[:, v] = 0
                eigenvectors_filtered[:, v] = 0

        # Reconstruct artifact signal
        spatial_filter = np.dot(maps, eigenvectors_filtered.T)
        artefact_data = spatial_filter @ epoch_data

        artefact_epochs[e] = artefact_data
        cleaned_epochs[e] = epoch_data - artefact_data

    return (cleaned_epochs, artefact_epochs)
