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
        # n_jobs=1: this function is already dispatched in parallel over epochs;
        # using n_jobs>1 here would cause oversubscription.
        coeffs = modwt(ch_data, wavelet, level, n_jobs=1)
        modwtmra_data = modwtmra(coeffs, wavelet, n_jobs=1)
        modwtmra_data = np.squeeze(modwtmra_data, axis=-1)
        transformed_epoch[c, :, :] = modwtmra_data

    return transformed_epoch


@fill_doc
def epochs_to_wavelet(epochs, wavelet, level, n_jobs=None, verbose=None):
    """Apply MODWT to each epoch in the epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to transform.
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

    epochs_data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = epochs_data.shape
    sfreq = epochs.info["sfreq"]

    if level == 0:
        # No wavelet decomposition - return original data as single band
        transformed_data = epochs_data[:, :, np.newaxis, :]
        freq_bands = [(0, sfreq / 2)]
        levels = 0
    else:
        # freq_bands matches MRA band order:
        #   [D_1(highest freq), D_2, ..., D_L(lowest freq detail), S_L(approx)]
        freq_bands = []
        for k in range(1, level + 1):
            freq_bands.append((sfreq / (2 ** (k + 1)), sfreq / (2 ** k)))
        freq_bands.append((0, sfreq / (2 ** (level + 1))))

        # Parallelize the wavelet transform across epochs
        if n_jobs == 1:
            # Sequential processing
            transformed_data = np.zeros((n_epochs, n_channels, level + 1, n_times))
            for e, epoch in enumerate(epochs_data):
                transformed_data[e] = _process_epoch_wavelet(epoch, wavelet, level)
        else:
            # Parallel processing using MNE's parallel_func
            parallel, p_fun, n_jobs = parallel_func(_process_epoch_wavelet, n_jobs)
            transformed_epochs = parallel(
                p_fun(epoch, wavelet, level) for epoch in epochs_data
            )
            transformed_data = np.array(transformed_epochs)

        levels = level

    return transformed_data, freq_bands, levels
