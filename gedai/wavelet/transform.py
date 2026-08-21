import numpy as np
from mne.parallel import parallel_func

from ..utils._checks import _check_n_jobs
from ..utils._docs import fill_doc
from ._modwt import modwt, modwtmra


def _process_epoch_wavelet(epoch_data, wavelet, level):
    """Process a single epoch with wavelet transform.

    Parameters
    ----------
    epoch_data : np.ndarray
        Single epoch data with shape (n_channels, n_times).
    wavelet : str
        The type of wavelet to use.
    level : int
        The level of decomposition.

    Returns
    -------
    transformed_epoch : np.ndarray
        Transformed epoch with shape (n_channels, level+1, n_times).
    """
    n_channels, n_times = epoch_data.shape
    transformed_epoch = np.zeros((n_channels, level + 1, n_times))

    for c, ch_data in enumerate(epoch_data):
        coeffs = modwt(ch_data, wavelet, level)
        modwtmra_data = modwtmra(coeffs, wavelet)
        modwtmra_data = np.squeeze(modwtmra_data, axis=-1)
        transformed_epoch[c, :, :] = modwtmra_data

    return transformed_epoch


@fill_doc
def epochs_to_wavelet(data, sfreq, wavelet, level, n_jobs=None, verbose=None):
    """Apply MODWT to each epoch in the epochs object.

    Parameters
    ----------
    data : np.ndarray
        The epochs data with shape (n_epochs, n_channels, n_times).
    sfreq : float
        The sampling frequency of the data.
    wavelet : str
        The type of wavelet to use (e.g., 'haar', 'db4', etc.).
    level : int
        The level of decomposition. If 0, no decomposition is performed.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    transformed_data : np.ndarray
        The transformed data with shape (n_epochs, n_channels, level+1, n_times).
    freq_bands : list of tuple
        Frequency bands for each component, ordered to match transformed_data.
    levels : int
        The actual decomposition level used.
    """
    n_jobs = _check_n_jobs(n_jobs)
    n_epochs, n_channels, n_times = data.shape

    if level == 0:
        # No wavelet decomposition - return original data as single band
        transformed_data = data[:, :, np.newaxis, :]
        freq_bands = [(0, sfreq / 2)]
        levels = 0
    else:
        freq_bands = []

        # Approximation (index 0): lowest frequencies
        freq_bands.append((0, sfreq / (2 ** (level + 1))))

        # Details (indices 1 to level): from coarse to fine
        for i in range(level, 0, -1):
            fmin = sfreq / (2 ** (i + 1))
            fmax = sfreq / (2**i)
            freq_bands.append((fmin, fmax))

        # Parallelize the wavelet transform across epochs
        if n_jobs == 1:
            # Sequential processing
            transformed_data = np.zeros((n_epochs, n_channels, level + 1, n_times))
            for e, epoch in enumerate(data):
                transformed_data[e] = _process_epoch_wavelet(epoch, wavelet, level)
        else:
            # Parallel processing using MNE's parallel_func
            parallel, p_fun, n_jobs = parallel_func(
                _process_epoch_wavelet, n_jobs, total=n_epochs, verbose=verbose
            )
            transformed_epochs = parallel(
                p_fun(epoch, wavelet, level) for epoch in data
            )
            transformed_data = np.array(transformed_epochs)

        levels = level

    return transformed_data, freq_bands, levels


def compute_wavelet_level(
    sfreq: float,
    lowcut_hz: float = 0.5,
    n_times: int | None = None,
    wavelet_low_cutoff: float | None = None,
    cycles_per_wavelet: int | None = None,
) -> int:
    """Compute number of wavelet decomposition levels matching MATLAB GEDAI.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    lowcut_hz : float
        Low cutoff frequency in Hz (default 0.5 Hz).
    n_times : int | None
        Number of time samples in data (optional, used to bound max levels).
    wavelet_low_cutoff : float | None
        Alias for lowcut_hz.
    cycles_per_wavelet : int | None
        Number of cycles per wavelet band (for adaptive windowing).

    Returns
    -------
    level : int
        Recommended wavelet decomposition level.
    """
    if wavelet_low_cutoff is not None:
        lowcut_hz = wavelet_low_cutoff
    lowcut_hz = max(float(lowcut_hz), 0.01)
    ideal = int(np.ceil(np.log2(sfreq / lowcut_hz)))
    if n_times is not None:
        if cycles_per_wavelet is not None and cycles_per_wavelet > 0:
            max_possible = max(1, int(np.floor(np.log2(max(1.0, n_times / cycles_per_wavelet)))) - 1)
        else:
            max_possible = int(np.floor(np.log2(n_times)))
        return max(4, min(ideal, max_possible))
    return max(6, ideal)


def get_modwt_band_limits(sfreq: float, n_bands: int) -> list[tuple[float, float]]:
    """Frequency limits for each MODWT band (index 0 = finest/highest detail band).

    Band f: [sfreq / 2^(f+2), sfreq / 2^(f+1)]
    Approximation (last band): [0.0, sfreq / 2^n_bands]
    """
    limits = []
    for f in range(n_bands - 1):
        lo = sfreq / (2 ** (f + 2))
        hi = sfreq / (2 ** (f + 1))
        limits.append((lo, hi))
    limits.append((0.0, sfreq / (2 ** n_bands)))
    return limits


def _modwt_haar_single_band(data_T: np.ndarray, level: int, band_idx: int) -> np.ndarray:
    """Haar MODWT single-band reconstruction — exact port of MATLAB modwt_single_band.m.

    Uses circular shifts (np.roll) matching MATLAB circshift, with the same
    forward/inverse Haar filter bank. Returns the time-domain reconstructed
    signal for one wavelet band only (MRA reconstruction).

    Parameters
    ----------
    data_T : (n_times, n_ch) float64
        Samples x channels.
    level : int
        Decomposition level.
    band_idx : int
        0-indexed band (0 = finest detail D1, ..., level = approximation A_J).

    Returns
    -------
    (n_ch, n_times) float64
        Reconstructed band signal in time domain.
    """
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    target_band = band_idx + 1
    n_bands = level + 1
    data_T = data_T.astype(np.float64)

    # Forward decomposition
    current_approx = data_T
    max_level_needed = min(target_band, level)
    target_coefs = None

    for j in range(1, max_level_needed + 1):
        step = 2 ** (j - 1)
        shifted_approx = np.roll(current_approx, step, axis=0)
        if j == target_band:
            target_coefs = (shifted_approx - current_approx) * inv_sqrt2
        else:
            current_approx = (current_approx + shifted_approx) * inv_sqrt2

    if target_band == n_bands:
        target_coefs = current_approx

    # Inverse reconstruction
    current_recon = target_coefs.copy()

    if target_band == n_bands:
        for j in range(level, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = np.roll(current_recon, -step, axis=0)
            current_recon = 0.5 * inv_sqrt2 * (current_recon + A_shifted)
    else:
        j = target_band
        step = 2 ** (j - 1)
        D_shifted = np.roll(current_recon, -step, axis=0)
        current_recon = 0.5 * inv_sqrt2 * (D_shifted - current_recon)
        for j in range(target_band - 1, 0, -1):
            step = 2 ** (j - 1)
            A_shifted = np.roll(current_recon, -step, axis=0)
            current_recon = 0.5 * inv_sqrt2 * (current_recon + A_shifted)

    return current_recon.T


def _apply_wavelet_highpass_prefilter(
    data: np.ndarray, sfreq: float, lowcut_hz: float = 0.5
) -> np.ndarray:
    """Remove sub-lowcut_hz slow drift using continuous Haar MODWT decomposition.

    Decomposes the continuous multichannel signal into Haar wavelet bands and
    subtracts all sub-lowcut_hz bands before broadband GED covariance estimation,
    preventing slow drift and DC offsets from biasing broadband covariance.
    Matches MATLAB GEDAI.m lines 520-603.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Continuous multichannel data.
    sfreq : float
        Sampling frequency in Hz.
    lowcut_hz : float
        Highpass cutoff frequency (default 0.5 Hz).

    Returns
    -------
    filtered_data : np.ndarray, shape (n_channels, n_times)
        Data with sub-lowcut_hz drift subtracted.
    """
    if lowcut_hz is None or lowcut_hz <= 0:
        return data

    n_times = data.shape[1]
    hp_wavelet_levels = int(np.ceil(np.log2(sfreq / max(0.01, min(0.1, lowcut_hz))) - 1))
    hp_wavelet_levels = max(hp_wavelet_levels, 3)
    hp_wavelet_levels = min(hp_wavelet_levels, int(np.floor(np.log2(n_times))))
    n_bands_hp = hp_wavelet_levels + 1

    bands_to_hp_zero = [
        j for j in range(n_bands_hp) if (sfreq / (2 ** (j + 1))) <= lowcut_hz
    ]
    if not bands_to_hp_zero:
        return data

    low_freq_noise = np.zeros_like(data, dtype=np.float64)
    data_T = data.T.astype(np.float64)
    for b in bands_to_hp_zero:
        low_freq_noise += _modwt_haar_single_band(data_T, hp_wavelet_levels, b)

    return (data.astype(np.float64) - low_freq_noise).astype(data.dtype)




