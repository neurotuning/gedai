import numpy as np
import pywt
from ._modwt import modwt, modwtmra



def epochs_to_wavelet(epochs, wavelet, level):
    """Apply MODWT to each epoch in the epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to transform.
    wavelet : str
        The type of wavelet to use (e.g., 'haar', 'db4', etc.).
    level : int
        The level of decomposition. If 0, no decomposition is performed.

    Returns
    -------
    transformed_data : np.ndarray
        The transformed data with shape (n_epochs, n_channels, level+1, n_times).
        When level=0: returns original data with shape (n_epochs, n_channels, 1, n_times).
        When level>0: Index 0 corresponds to approximation (lowest frequencies),
                      Indices 1 to level correspond to details from coarse to fine.
    freq_bands : list of tuple
        Frequency bands for each component, ordered to match transformed_data.
    levels : int
        The actual decomposition level used.
    """
    epochs_data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = epochs_data.shape
    sfreq = epochs.info['sfreq']

    if level == 0:
        # No wavelet decomposition - return original data as single band
        transformed_data = epochs_data[:, :, np.newaxis, :]
        freq_bands = [(0, sfreq / 2)]
        levels = 0
    else:
        # Calculate frequency bands matching MODWT MRA output order
        # MODWT MRA returns: [approximation, detail_level, detail_level-1, ..., detail_1]
        freq_bands = []
        
        # Approximation (index 0): lowest frequencies
        freq_bands.append((0, sfreq / (2 ** (level + 1))))
        
        # Details (indices 1 to level): from coarse to fine
        for i in range(level, 0, -1):
            fmin = sfreq / (2 ** (i + 1))
            fmax = sfreq / (2 ** i)
            freq_bands.append((fmin, fmax))

        transformed_data = np.zeros((n_epochs, n_channels, level + 1, n_times))
        
        for e, epoch in enumerate(epochs_data):
            for c, ch_data in enumerate(epoch):
                coeffs = modwt(ch_data, wavelet, level)
                modwtmra_data = modwtmra(coeffs, wavelet)
                modwtmra_data = np.squeeze(modwtmra_data, axis=-1)
                transformed_data[e, c, :, :] = modwtmra_data
        
        levels = level
    
    return transformed_data, freq_bands, levels

