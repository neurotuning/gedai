"""ENOVA (Explained Noise Variance) and composite cleaning quality metrics."""

import numpy as np
from scipy.linalg import eigh, subspace_angles


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
    n_times = clean.shape[1]
    n_epochs = n_times // epoch_samples
    if n_epochs == 0:
        original = clean + noise
        var_o = float(np.var(original))
        var_n = float(np.var(noise))
        return np.array([var_n / var_o if var_o > 0 else 0.0], dtype=np.float32)

    enova = np.zeros(n_epochs, dtype=np.float32)
    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        orig_ep = clean[:, s:e] + noise[:, s:e]
        var_o = float(np.var(orig_ep))
        var_n = float(np.var(noise[:, s:e]))
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

    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        c_ep = clean[:, s:e]
        n_ep = noise[:, s:e]

        if np.var(c_ep) > 0:
            cov_sig = np.cov(c_ep)
            cov_sig = (cov_sig + cov_sig.T) * 0.5
            _, eigvecs_sig = eigh(cov_sig)
            basis_sig = eigvecs_sig[:, ::-1][:, :n_pc]
            angles_sig = subspace_angles(basis_sig, ref_evecs)
            sig_sims[i] = float(np.prod(np.cos(angles_sig)))

        if np.var(n_ep) > 0:
            cov_noi = np.cov(n_ep)
            cov_noi = (cov_noi + cov_noi.T) * 0.5
            _, eigvecs_noi = eigh(cov_noi)
            basis_noi = eigvecs_noi[:, ::-1][:, :n_pc]
            angles_noi = subspace_angles(basis_noi, ref_evecs)
            noi_sims[i] = float(np.prod(np.cos(angles_noi)))

    signal_sim = 100.0 * float(np.mean(sig_sims))
    noise_sim = 100.0 * float(np.mean(noi_sims))
    return signal_sim - 1.0 * noise_sim
