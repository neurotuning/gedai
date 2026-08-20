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



