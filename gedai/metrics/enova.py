"""ENOVA (Explained Noise Variance) and composite cleaning quality metrics."""

import numpy as np
from scipy.linalg import eigh, subspace_angles

from ..utils._checks import ensure_int


def compute_enova_per_epoch(
    clean: np.ndarray,
    noise: np.ndarray,
    epoch_samples: int,
) -> np.ndarray:
    """Compute Explained Noise Variance (ENOVA) per epoch.

    ENOVA = var(noise_epoch) / var(original_epoch)

    Parameters
    ----------
    clean : np.ndarray, shape (n_channels, n_times)
        Cleaned EEG signal.
    noise : np.ndarray, shape (n_channels, n_times)
        Removed noise/artifact signal (original = clean + noise).
    epoch_samples : int
        Number of samples per epoch.

    Returns
    -------
    enova : np.ndarray, shape (n_epochs,)
        ENOVA values per epoch in [0, inf).
    """
    n_times = clean.shape[-1]
    n_epochs = n_times // epoch_samples
    if n_epochs == 0:
        original = clean + noise
        var_o = float(np.mean(np.var(original, axis=-1)))
        var_n = float(np.mean(np.var(noise, axis=-1)))
        return np.array([var_n / var_o if var_o > 0 else 0.0], dtype=np.float32)

    enova = np.zeros(n_epochs, dtype=np.float32)
    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        orig_ep = clean[..., s:e] + noise[..., s:e]
        var_o = float(np.mean(np.var(orig_ep, axis=-1)))
        var_n = float(np.mean(np.var(noise[..., s:e], axis=-1)))
        enova[i] = var_n / var_o if var_o > 0 else 0.0
    return enova


def compute_enova_per_channel(
    clean: np.ndarray,
    noise: np.ndarray,
    epoch_samples: int,
) -> np.ndarray:
    """Compute per-channel ENOVA, averaged across epochs.

    Parameters
    ----------
    clean : np.ndarray, shape (n_channels, n_times)
        Cleaned EEG signal.
    noise : np.ndarray, shape (n_channels, n_times)
        Removed noise signal.
    epoch_samples : int
        Number of samples per epoch.

    Returns
    -------
    enova_ch : np.ndarray, shape (n_channels,)
        Average ENOVA for each channel.
    """
    n_ch, n_times = clean.shape
    n_epochs = n_times // epoch_samples

    if n_epochs == 0:
        original = clean + noise
        var_o = np.var(original, axis=1)
        var_n = np.var(noise, axis=1)
        return np.divide(
            var_n, var_o, out=np.zeros_like(var_n, dtype=np.float32), where=(var_o > 0)
        )

    enova_acc = np.zeros(n_ch, dtype=np.float64)
    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        orig_ep = clean[:, s:e] + noise[:, s:e]
        var_o = np.var(orig_ep, axis=1)
        var_n = np.var(noise[:, s:e], axis=1)
        enova_acc += np.divide(
            var_n, var_o, out=np.zeros_like(var_n, dtype=np.float64), where=(var_o > 0)
        )
    return (enova_acc / n_epochs).astype(np.float32)


def enova_summary(enova_per_epoch: np.ndarray) -> dict:
    """Compute summary statistics for an ENOVA array."""
    if len(enova_per_epoch) == 0:
        return {}
    return {
        "mean": float(np.mean(enova_per_epoch)),
        "median": float(np.median(enova_per_epoch)),
        "std": float(np.std(enova_per_epoch)),
        "min": float(np.min(enova_per_epoch)),
        "max": float(np.max(enova_per_epoch)),
    }


def compute_composite_sensai(
    clean: np.ndarray,
    noise: np.ndarray,
    sfreq: float,
    reference_cov: np.ndarray,
    epoch_size: float = 1.0,
    n_pc: int = 3,
) -> float:
    """Compute composite physical SENSAI score from clean and noise time series.

    Parameters
    ----------
    clean : np.ndarray, shape (n_channels, n_times)
        Cleaned signal.
    noise : np.ndarray, shape (n_channels, n_times)
        Noise signal.
    sfreq : float
        Sampling frequency in Hz.
    reference_cov : np.ndarray, shape (n_channels, n_channels)
        Reference covariance matrix.
    epoch_size : float
        Epoch length in seconds (default 1.0 s).
    n_pc : int
        Number of principal components for template subspace (default 3).

    Returns
    -------
    sensai_score : float
        Composite SENSAI score in percent.
    """
    n_pc = ensure_int(n_pc, "n_pc")
    if not 1 <= n_pc <= reference_cov.shape[0]:
        max_n_pc = reference_cov.shape[0]
        raise ValueError(
            f"n_pc must be an integer in the range [1, {max_n_pc}], got {n_pc!r}."
        )

    ref_cov = np.real(np.asarray(reference_cov, dtype=np.float64))
    ref_cov = (ref_cov + ref_cov.T) * 0.5
    lam = 0.05
    trace = np.trace(ref_cov) / ref_cov.shape[0] if ref_cov.shape[0] > 0 else 1.0
    ref_cov_reg = (1 - lam) * ref_cov + lam * trace * np.eye(ref_cov.shape[0])
    ref_cov_reg = (ref_cov_reg + ref_cov_reg.T) * 0.5

    _, ref_evecs = eigh(ref_cov_reg)
    ref_evecs = ref_evecs[:, ::-1][:, :n_pc]

    epoch_samples = max(1, round(sfreq * epoch_size))
    n_times = clean.shape[1]
    n_epochs = n_times // epoch_samples
    if n_epochs == 0:
        return float("nan")

    sig_sims = np.zeros(n_epochs, dtype=np.float64)
    noi_sims = np.zeros(n_epochs, dtype=np.float64)

    def _epoch_subspace_sim(ep_data: np.ndarray, var_tol: float) -> float:
        if ep_data.shape[1] < 2 or float(np.var(ep_data)) <= var_tol:
            return 0.0
        cov_ep = np.cov(ep_data)
        cov_ep = (cov_ep + cov_ep.T) * 0.5
        eigvals, eigvecs = eigh(cov_ep)
        max_eig = float(eigvals[-1])
        if max_eig <= 0.0 or int(np.sum(eigvals > 1e-12 * max_eig)) < n_pc:
            return 0.0
        basis = eigvecs[:, ::-1][:, :n_pc]
        angles = subspace_angles(basis, ref_evecs)
        return float(np.prod(np.cos(angles)))

    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        c_ep = clean[:, s:e]
        n_ep = noise[:, s:e]
        orig_var = float(np.var(c_ep + n_ep))
        if orig_var <= 0.0:
            continue
        var_tol = 1e-12 * orig_var
        sig_sims[i] = _epoch_subspace_sim(c_ep, var_tol)
        noi_sims[i] = _epoch_subspace_sim(n_ep, var_tol)

    signal_sim = 100.0 * float(np.mean(sig_sims))
    noise_sim = 100.0 * float(np.mean(noi_sims))
    return signal_sim - 1.0 * noise_sim
