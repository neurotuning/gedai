"""PyTorch CPU backend acceleration for pyGEDAI linear algebra operations."""

from __future__ import annotations

import numpy as np

from ._imports import import_optional_dependency


def has_torch() -> bool:
    """Check if PyTorch is installed and importable."""
    return import_optional_dependency("torch", raise_error=False) is not None


def resolve_engine(engine: str = "numpy") -> str:
    """Validate and resolve backend engine ('numpy', 'torch', 'auto').

    Parameters
    ----------
    engine : str
        The requested computation engine ('numpy', 'torch', or 'auto').

    Returns
    -------
    resolved : str
        The resolved engine ('numpy' or 'torch').
    """
    if not isinstance(engine, str):
        raise TypeError(f"engine must be a string, got {type(engine).__name__}")

    engine_lower = engine.lower()
    if engine_lower not in ("numpy", "torch", "auto"):
        raise ValueError(
            f"Invalid engine '{engine}'. Expected one of 'numpy', 'torch', or 'auto'."
        )

    if engine_lower == "auto":
        return "torch" if has_torch() else "numpy"

    if engine_lower == "torch":
        if not has_torch():
            raise ImportError(
                "Missing optional dependency 'torch'. Use 'pip install torch' "
                "or 'pip install gedai[torch]' to run with the PyTorch engine."
            )
        return "torch"

    return "numpy"


def robust_cholesky_gevd(
    cov_a,
    cov_b,
    eps_jitter: float = 1e-6,
):
    """Batched Generalized Eigendecomposition with robust Cholesky factorization.

    Solves the generalized eigenvalue problem A V = B V Lambda for symmetric
    matrices cov_a and symmetric positive-definite (or regularized) reference cov_b.
    Satisfies V.mT @ B @ V = I and A @ V = B @ V @ diag(evals).

    Handles near-singular or non-strictly-positive-definite reference covariances
    via adaptive diagonal jitter fallback (cholesky_ex).

    Parameters
    ----------
    cov_a : torch.Tensor
        Symmetric matrix or batch of matrices with shape (..., n_channels, n_channels).
    cov_b : torch.Tensor
        Symmetric reference covariance matrix with shape (..., n_channels, n_channels)
        or (n_channels, n_channels).
    eps_jitter : float, default 1e-6
        Initial diagonal jitter scaling factor for fallback regularization.

    Returns
    -------
    evals : torch.Tensor
        Generalized eigenvalues in ascending order with shape (..., n_channels).
    evecs : torch.Tensor
        B-orthonormal eigenvectors with shape (..., n_channels, n_channels),
        where evecs[..., :, i] is the i-th eigenvector.
    """
    import torch

    # Enforce float64 for EEG/MEG linear algebra stability
    a = cov_a.to(torch.float64)
    b = cov_b.to(torch.float64)

    # Symmetrize inputs
    b = 0.5 * (b + b.mT)
    a = 0.5 * (a + a.mT)

    n_ch = b.shape[-1]

    # Robust Cholesky factorization with adaptive jitter fallback
    l_factor, info = torch.linalg.cholesky_ex(b)
    if bool((info != 0).any()):
        eye = torch.eye(n_ch, dtype=b.dtype, device=b.device)
        diag_mean = float(
            torch.diagonal(b, dim1=-2, dim2=-1).abs().mean().clamp_min(1e-12)
        )
        current_jitter = eps_jitter * diag_mean
        for _ in range(4):
            b_jittered = b + current_jitter * eye
            l_factor, info = torch.linalg.cholesky_ex(b_jittered)
            if not bool((info != 0).any()):
                break
            current_jitter *= 10.0
        if bool((info != 0).any()):
            raise RuntimeError(
                "Reference covariance matrix is not positive-definite even after "
                "adaptive jitter."
            )

    # Transform A to standard symmetric eigenproblem:
    # A_tilde = L^(-1) @ A @ L^(-T)
    c_mat = torch.linalg.solve_triangular(l_factor, a, upper=False)
    a_tilde = torch.linalg.solve_triangular(l_factor, c_mat.mT, upper=False).mT

    # Ensure exact numerical symmetry to prevent eigh drift
    a_tilde = 0.5 * (a_tilde + a_tilde.mT)

    # Standard symmetric eigendecomposition
    evals, y_mat = torch.linalg.eigh(a_tilde)

    # Back-transform eigenvectors: V = L^(-T) @ Y
    evecs = torch.linalg.solve_triangular(l_factor.mT, y_mat, upper=True)

    return evals, evecs


# Aliases for convenience and backward compatibility
batched_gevd_cholesky = robust_cholesky_gevd
gevd_torch = robust_cholesky_gevd


def precompute_gevd_torch(
    epochs_data: np.ndarray, reference_cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute GEVD across all epochs using PyTorch CPU.

    Parameters
    ----------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
    reference_cov : np.ndarray, shape (n_channels, n_channels)

    Returns
    -------
    all_eval : np.ndarray, shape (n_epochs, n_channels)
    all_evec : np.ndarray, shape (n_epochs, n_channels, n_channels)
    """
    import torch

    n_ep, n_ch, n_times = epochs_data.shape
    if n_times < 2:
        raise ValueError(
            "epochs_data must contain at least 2 time points per epoch to "
            "compute covariance."
        )

    x = torch.from_numpy(epochs_data).to(torch.float64)
    b = torch.from_numpy(reference_cov).to(torch.float64)

    # Vectorized batch covariance
    x_centered = x - x.mean(dim=-1, keepdim=True)
    covs = torch.bmm(x_centered, x_centered.mT) / (n_times - 1)

    evals, evecs = robust_cholesky_gevd(covs, b)

    return evals.cpu().numpy(), evecs.cpu().numpy()


def clean_epochs_batched_torch(
    epochs_data: np.ndarray, reference_cov: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Process all epochs in a single batched tensor pass on PyTorch CPU.

    Parameters
    ----------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
        Multi-channel epoch data.
    reference_cov : np.ndarray, shape (n_channels, n_channels)
        Regularized reference covariance matrix.
    threshold : float
        Eigenvalue threshold for artifact rejection.

    Returns
    -------
    cleaned_epochs : np.ndarray, shape (n_epochs, n_channels, n_times)
    artefact_epochs : np.ndarray, shape (n_epochs, n_channels, n_times)
    """
    import torch

    x = torch.from_numpy(epochs_data).to(torch.float64)
    b = torch.from_numpy(reference_cov).to(torch.float64)
    n_times = x.shape[-1]

    # Vectorized batch covariance
    x_centered = x - x.mean(dim=-1, keepdim=True)
    covs = torch.bmm(x_centered, x_centered.mT) / max(1, n_times - 1)

    # Batched GEVD
    evals, evecs = robust_cholesky_gevd(covs, b)

    # Filter artifact eigenvectors: signal_mask zeroes out columns where |eval| < thresh
    signal_mask = torch.abs(evals) < threshold  # shape (n_epochs, n_channels)
    evecs_filtered = torch.where(
        signal_mask.unsqueeze(1), torch.zeros_like(evecs), evecs
    )

    # Direct Regularized Reference Covariance Projection:
    # artifact_tc = V_art.T @ X
    # artefact_data = C_ref @ V_art @ artifact_tc
    artifact_tc = torch.bmm(evecs_filtered.mT, x)
    rec = torch.bmm(evecs_filtered, artifact_tc)
    artefact_data = torch.matmul(b, rec)
    cleaned_epochs = x - artefact_data

    return cleaned_epochs.cpu().numpy(), artefact_data.cpu().numpy()


def clean_continuous_stream_torch(
    stream: np.ndarray,
    reference_cov: np.ndarray,
    threshold: float,
    cosine_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean segmented continuous stream in PyTorch with cosine windowing.

    Parameters
    ----------
    stream : np.ndarray, shape (n_epochs, n_channels, epoch_samples)
    reference_cov : np.ndarray, shape (n_channels, n_channels)
    threshold : float
    cosine_weights : np.ndarray, shape (epoch_samples,)

    Returns
    -------
    clean_out : np.ndarray, shape (n_channels, n_epochs * epoch_samples)
    noise_out : np.ndarray, shape (n_channels, n_epochs * epoch_samples)
    """
    import torch

    n_ep, n_ch, epoch_samples = stream.shape
    half = epoch_samples // 2

    clean, noise = clean_epochs_batched_torch(stream, reference_cov, threshold)

    clean_t = torch.from_numpy(clean).to(torch.float64)
    noise_t = torch.from_numpy(noise).to(torch.float64)
    cw = torch.from_numpy(cosine_weights).to(torch.float64)

    if n_ep == 1:
        pass
    else:
        # First epoch: fade in second half
        clean_t[0, :, half:] *= cw[half:]
        noise_t[0, :, half:] *= cw[half:]

        # Last epoch: fade out first half
        clean_t[-1, :, :half] *= cw[:half]
        noise_t[-1, :, :half] *= cw[:half]

        # Interior epochs: full cosine window
        if n_ep > 2:
            clean_t[1:-1] *= cw.view(1, 1, -1)
            noise_t[1:-1] *= cw.view(1, 1, -1)

    # Permute from (n_ep, n_ch, epoch_samples) to (n_ch, n_ep * epoch_samples)
    clean_out = clean_t.permute(1, 0, 2).reshape(n_ch, n_ep * epoch_samples)
    noise_out = noise_t.permute(1, 0, 2).reshape(n_ch, n_ep * epoch_samples)

    return clean_out.cpu().numpy(), noise_out.cpu().numpy()
