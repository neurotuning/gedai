
import mne
import numpy as np
from gedai import Gedai

def pygedai_denoise_EEGLAB_data(eeg_data, sfreq):
    """
    Denoises EEG data using the pyGEDAI library.

    This function is designed to be called from MATLAB. It takes a NumPy array
    of EEG data and a sampling frequency, performs a two-stage GEDAI denoising,
    and returns the cleaned data as a NumPy array.

    Parameters
    ----------
    eeg_data : np.ndarray
        The EEG data matrix (channels x samples).
    sfreq : float
        The sampling frequency of the EEG data.

    Returns
    -------
    np.ndarray
        The denoised EEG data matrix.
    """
    # 1. Create an MNE RawArray object from the input data
    n_channels = eeg_data.shape[0]
    ch_names = [f"EEG {i+1:03}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # 2. Apply the two-stage GEDAI denoising process
    # Set common parameters
    n_jobs = 1
    
    # Stage 1: Broadband denoising
    raw.set_eeg_reference("average", projection=False, verbose=False)
    gedai = Gedai()
    gedai.fit_raw(raw, noise_multiplier=6., n_jobs=n_jobs, verbose=False)
    raw_corrected = gedai.transform_raw(raw, verbose=False)

    # Stage 2: Spectral denoising
    duration = 10  # seconds
    gedai_spectral = Gedai(wavelet_type='haar', wavelet_level=5)
    gedai_spectral.fit_raw(raw_corrected, noise_multiplier=3., duration=duration, n_jobs=n_jobs, verbose=False)
    raw_corrected_final = gedai_spectral.transform_raw(raw_corrected, duration=duration, verbose=False)

    # 3. Extract and return the denoised data as a NumPy array
    denoised_data = raw_corrected_final.get_data()
    
    return denoised_data
