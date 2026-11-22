"""ENOVA (Explained Noise Variance) and composite cleaning quality metrics."""

import mne
import numpy as np
from scipy.linalg import eigh, subspace_angles

from ..utils._checks import _check_type, ensure_int


def _epoch_var_floor(var_arr: np.ndarray, mean_sq: np.ndarray | float) -> np.ndarray:
    """Return scale-invariant variance floor to guard flat-signal roundoff."""
    tiny = float(np.finfo(np.float64).tiny)
    var_d = np.asarray(var_arr, dtype=np.float64)
    pos = var_d[var_d > tiny]
    ref_var = float(np.median(pos)) if pos.size > 0 else 0.0
    return np.maximum(
        np.maximum(1e-12 * ref_var, 1e-20 * np.asarray(mean_sq, dtype=np.float64)),
        tiny,
    )


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
    tiny = float(np.finfo(np.float64).tiny)
    if n_epochs == 0:
        original = clean + noise
        var_o = float(np.mean(np.var(original, axis=-1)))
        var_n = float(np.mean(np.var(noise, axis=-1)))
        mean_sq = float(np.mean(original * original))
        floor = max(1e-20 * mean_sq, tiny)
        return np.array(
            [var_n / var_o if (var_o > floor and var_n > floor) else 0.0],
            dtype=np.float32,
        )

    var_o_arr = np.zeros(n_epochs, dtype=np.float64)
    var_n_arr = np.zeros(n_epochs, dtype=np.float64)
    mean_sq_arr = np.zeros(n_epochs, dtype=np.float64)
    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        orig_ep = clean[..., s:e] + noise[..., s:e]
        var_o_arr[i] = float(np.mean(np.var(orig_ep, axis=-1)))
        var_n_arr[i] = float(np.mean(np.var(noise[..., s:e], axis=-1)))
        mean_sq_arr[i] = float(np.mean(orig_ep * orig_ep))

    floor = _epoch_var_floor(var_o_arr, mean_sq_arr)
    valid = (var_o_arr > floor) & (var_n_arr > floor)
    enova = np.divide(
        var_n_arr, var_o_arr, out=np.zeros(n_epochs, dtype=np.float64), where=valid
    )
    return enova.astype(np.float32)


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
        mean_sq = np.mean(original * original, axis=1)
        floor = _epoch_var_floor(var_o, mean_sq)
        valid = (var_o > floor) & (var_n > floor)
        return np.divide(
            var_n, var_o, out=np.zeros_like(var_n, dtype=np.float32), where=valid
        ).astype(np.float32)

    var_o_mat = np.zeros((n_ch, n_epochs), dtype=np.float64)
    var_n_mat = np.zeros((n_ch, n_epochs), dtype=np.float64)
    mean_sq_mat = np.zeros((n_ch, n_epochs), dtype=np.float64)
    for i in range(n_epochs):
        s = i * epoch_samples
        e = s + epoch_samples
        orig_ep = clean[:, s:e] + noise[:, s:e]
        var_o_mat[:, i] = np.var(orig_ep, axis=1)
        var_n_mat[:, i] = np.var(noise[:, s:e], axis=1)
        mean_sq_mat[:, i] = np.mean(orig_ep * orig_ep, axis=1)

    floor = _epoch_var_floor(var_o_mat, mean_sq_mat)
    valid = (var_o_mat > floor) & (var_n_mat > floor)
    ratio_mat = np.divide(
        var_n_mat, var_o_mat, out=np.zeros_like(var_o_mat), where=valid
    )
    return np.mean(ratio_mat, axis=1).astype(np.float32)


def annotate_bad_epochs(
    raw: mne.io.BaseRaw,
    enova_per_epoch: np.ndarray,
    epoch_duration: float = 1.0,
    threshold: float = 0.85,
    description: str = "BAD_gedai",
) -> mne.Annotations:
    """Build MNE Annotations marking epochs whose ENOVA exceeds ``threshold``.

    Contiguous bad epochs are merged into single annotation intervals aligned to
    ``raw.first_time`` and clipped to the recording duration.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw instance corresponding to ``enova_per_epoch``.
    enova_per_epoch : np.ndarray, shape (n_epochs,)
        Explained noise variance ratio per epoch.
    epoch_duration : float, default=1.0
        Duration (seconds) of each epoch in ``enova_per_epoch``.
    threshold : float, default=0.85
        ENOVA threshold above which an epoch is flagged bad.
    description : str, default="BAD_gedai"
        MNE annotation label (prefix ``"BAD_"`` enables native MNE rejection).

    Returns
    -------
    annotations : mne.Annotations
        Annotations object covering flagged bad segments.
    """
    _check_type(raw, (mne.io.BaseRaw,), "raw")
    _check_type(epoch_duration, (float, int), "epoch_duration")
    _check_type(threshold, ("numeric",), "threshold")
    _check_type(description, (str,), "description")
    if not np.isfinite(epoch_duration) or epoch_duration <= 0:
        raise ValueError(f"epoch_duration must be > 0, got {epoch_duration!r}.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold!r}.")

    enova_arr = np.asarray(enova_per_epoch, dtype=np.float64).ravel()
    bad_mask = enova_arr > float(threshold)
    orig_time = raw.annotations.orig_time if raw.annotations is not None else None
    if not np.any(bad_mask):
        return mne.Annotations([], [], [], orig_time=orig_time)

    max_dur = raw.n_times / float(raw.info["sfreq"])
    first_time = float(raw.first_time)
    onsets: list[float] = []
    durations: list[float] = []

    idx = 0
    n_ep = len(bad_mask)
    while idx < n_ep:
        if not bad_mask[idx]:
            idx += 1
            continue
        start_ep = idx
        while idx < n_ep and bad_mask[idx]:
            idx += 1
        end_ep = idx
        rel_onset = start_ep * float(epoch_duration)
        rel_end = min(end_ep * float(epoch_duration), max_dur)
        if rel_end > rel_onset:
            onsets.append(first_time + rel_onset)
            durations.append(rel_end - rel_onset)

    descriptions = [description] * len(onsets)
    return mne.Annotations(
        onset=onsets, duration=durations, description=descriptions, orig_time=orig_time
    )


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
        orig_var = float(np.mean(np.var(c_ep + n_ep, axis=-1)))
        if orig_var <= 0.0:
            continue
        var_tol = 1e-12 * orig_var
        sig_sims[i] = _epoch_subspace_sim(c_ep, var_tol)
        noi_sims[i] = _epoch_subspace_sim(n_ep, var_tol)

    signal_sim = 100.0 * float(np.mean(sig_sims))
    noise_sim = 100.0 * float(np.mean(noi_sims))
    return signal_sim - 1.0 * noise_sim
