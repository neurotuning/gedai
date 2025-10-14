import numpy as np
import pywt
from ._modwt import modwt, modwtmra



def epochs_to_wavelet(epochs, wavelet_type, wavelet_level):
    """Apply MODWT to each epoch in the epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to transform.
    wavelet_type : str
        The type of wavelet to use (e.g., 'haar', 'db1', etc.).
    wavelet_level : int
        The level of decomposition.

    Returns
    -------
    transformed_data : np.ndarray
        The transformed data with shape  (n_epochs, n_channels, num_wavelet_levels, n_times).
    """
    epochs_data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = epochs_data.shape
    sfreq = epochs.info['sfreq']
    transformed_data = []

    max_level_allowed = pywt.swt_max_level(n_times)
    desired_level = int(np.floor(np.log2(sfreq))) - 1
    if desired_level < 1: desired_level = 1
    num_wavelet_levels = min(desired_level, max_level_allowed)

    freq_bands = [(sfreq / (2 ** (i + 1)), sfreq / (2 ** i)) for i in range(num_wavelet_levels)]

    transformed_data = np.zeros((n_epochs, n_channels, num_wavelet_levels + 1, n_times))
    for e, epoch in epochs_data:
        for c, ch_data in epoch:
            coeffs = modwt(ch_data, wavelet_type, wavelet_level)
            modwtmra_data = modwtmra(coeffs, wavelet_type)
            modwtmra_data = np.squeeze(modwtmra_data, axis=-1)
            transformed_data[e, c, :, :] = modwtmra_data
    return transformed_data, freq_bands
    
        