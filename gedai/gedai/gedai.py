import os
import numpy as np
import mne
from mne import BaseEpochs
from mne.io import BaseRaw

import matplotlib.pyplot as plt

from scipy.linalg import eigh
import pywt
from pyriemann.estimation import Covariances

from typing import Optional

from ..utils._checks import check_type, _check_n_jobs
from ..utils._docs import fill_doc

from .decompose import clean_epochs
from .covariances import compute_distance_cov, compute_refcov
from ..sensai import sensai_gridsearch, sensai_optimize, scale_threshold


def create_cosine_weights(n_samples):
    """Create cosine weights for a single epoch, mimicking the MATLAB implementation."""
    u = np.arange(1, n_samples + 1)
    cos_win = 0.5 - 0.5 * np.cos(2 * u * np.pi / n_samples)
    return cos_win


def validate_method(method):
    check_type(method, (str,), 'method')
    if method not in ['gridsearch', 'optimize']:
        raise ValueError("Method must be either 'gridsearch' or 'optimize', got '{method}' instead.")


def epochs_to_wavelet(epochs, wavelet, level):
    # compute frequency bands
    freq_bands = []
    for i in range(1, level + 1):
        fmin = epochs.info['sfreq'] / (2 ** (i + 1))
        fmax = epochs.info['sfreq'] / (2 ** i)
        freq_bands.append((fmin, fmax))
    freq_bands.append((0, epochs.info['sfreq'] / (2 ** (level+1))))
    freq_bands = freq_bands[::-1]
    
    epochs_data = epochs.get_data()
    wavelet_signals = np.zeros((epochs_data.shape[0], epochs_data.shape[1], level+1, epochs_data.shape[2]))
    for e,epoch_data in enumerate(epochs_data):
        for c,channel_data in enumerate(epoch_data):
            coeffs = pywt.wavedec(channel_data, wavelet, level=level)
            for component in range(level+1):
                coeffs_single = [np.zeros_like(c) for c in coeffs]
                coeffs_single[component] = coeffs[component]  # Only keep the desired component
                # Reconstruct the signal using the modified coefficients
                comp = pywt.waverec(coeffs_single, wavelet)
                wavelet_signals[e, c, component] = comp[:len(channel_data)]
    return(wavelet_signals, freq_bands)


@fill_doc
class Gedai():
    r"""Generalized Eigenvalue De-Artifacting Instrument (GEDAI).

    Parameters
    ----------
    wavelet : int
        Wavelet to use for the decomposition. The default is 'db4' (Daubechies 4).
        See :py:func:`pywt.wavedec` for complete list of available wavelet values.
    wavelet_level : int
        Decomposition level (must be >= 0). The default is 0 (no decomposition).
        See :py:func:`pywt.wavedec` more details.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, wavelet_type='db4', wavelet_level=0):
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level

    @fill_doc
    def fit_epochs(self,
                    epochs: BaseEpochs,
                    noise_multiplier: float = 0.0,
                    covariance_method: str = 'leadfield',
                    method: str = 'gridsearch',
                    n_jobs: int = None,
                    verbose: Optional[str] = None):
        """Fit the GEDAI model to the epochs data.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs data to fit the model to.
        noise_multiplier : float
            The noise multiplier to use artefact threshold rejection optimization.
        method : str
            The method to use for threshold optimization. Can be either 'gridsearch' or 'optimize'.
        %(n_jobs)s
        %(verbose)s
        """
        check_type(epochs, (BaseEpochs,), 'epochs')
        check_type(noise_multiplier, (float,), 'noise_multiplier')
        validate_method(method)
        n_jobs = _check_n_jobs(n_jobs)
        # Compute reference covariance matrix
        if covariance_method == 'distance':
            reference_cov = compute_distance_cov(epochs)
            ch_names = epochs.info['ch_names']
        elif covariance_method == 'leadfield':
            mat = os.path.join(os.path.dirname(__file__), '../../gedai/data/fsavLEADFIELD_4_GEDAI.mat')
            reference_cov, ch_names = compute_refcov(epochs, mat)
        else:
            raise ValueError("Covariance method must be either 'distance' or 'leadfield'")

        # Tikhonov Regularization based on average diagonal power 
        avg_diag_power = np.trace(reference_cov) / reference_cov.shape[0]
        regularization_lambda = 0.05
        epsilon = regularization_lambda * avg_diag_power
        reference_cov = reference_cov + epsilon * np.eye(reference_cov.shape[0])

        # Broadband data
        epochs_wavelet, freq_bands = epochs_to_wavelet(epochs, wavelet=self.wavelet_type, level=self.wavelet_level)
        
        wavelets_fits = []
        for w, (fmin, fmax) in enumerate(freq_bands):
            wavelet_epochs_data = epochs_wavelet[:, :, w, :]

            # Compute eigenvalues
            epochs_eigenvalues = np.zeros((len(wavelet_epochs_data), wavelet_epochs_data.shape[1]))
            for e, wavelet_epoch_data in enumerate(wavelet_epochs_data):
                covariance = np.cov(wavelet_epoch_data)
                eigenvalues, _ = eigh(covariance, reference_cov, check_finite=True)
                epochs_eigenvalues[e] = eigenvalues
  
            wavelet_epochs = mne.EpochsArray(wavelet_epochs_data, epochs.info, tmin=epochs.tmin, verbose=False)
            if method == 'gridsearch':
                min_sensai_threshold, max_threshold, step = 0, 15, 0.1
                sensai_thresholds = np.arange(min_sensai_threshold, max_threshold, step)
                eigen_thresholds = [self._sensai_to_eigen(sensai_value, epochs_eigenvalues) for sensai_value in sensai_thresholds]
                threshold, runs = sensai_gridsearch(wavelet_epochs, reference_cov, n_pc=3, noise_multiplier=noise_multiplier, eigen_thresholds=eigen_thresholds, n_jobs=n_jobs)

            else:
                raise ValueError("Method must be 'gridsearch'")
            wavelet_fit = {'fmin': fmin, 'fmax': fmax, 'threshold': threshold, 'reference_cov': reference_cov, 'epochs_eigenvalues': epochs_eigenvalues, 'sensai_runs': runs}
            wavelets_fits.append(wavelet_fit)
        self.wavelets_fits = wavelets_fits

    @fill_doc
    def fit_raw(self,
                raw: BaseRaw,
                duration: Optional[float] = 1,
                overlap: float = 0.5,
                reject_by_annotation: Optional[bool] = False,
                covariance_method: str = 'leadfield',
                noise_multiplier: float = 1.0,
                method: str = 'gridsearch',
                n_jobs: int = None,
                verbose: Optional[str] = None):
        """Fit the GEDAI model to the raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        duration : float
            Duration of each epoch in seconds. Defaults to 1.
        overlap : float
            The overlap between epochs, in seconds. Must be ``0 <= overlap < duration``. Default is 0, i.e., no overlap. 
        %(reject_by_annotation_raw)s
        noise_multiplier : float
            The noise multiplier to use artefact threshold rejection optimization.
        method : str
            The method to use for threshold optimization. Can be either 'gridsearch' or 'optimize'.
        %(n_jobs)s
        %(verbose)s
        """
        check_type(raw, (BaseRaw,), 'raw')
        check_type(duration, (float, int,), 'duration')
        check_type(overlap, (float, int,), 'overlap')
        check_type(reject_by_annotation, (bool,), 'reject_by_annotation')
        check_type(noise_multiplier, (float,), 'noise_multiplier')
        validate_method(method)
        n_jobs = _check_n_jobs(n_jobs)
        epochs = mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap, reject_by_annotation=reject_by_annotation, preload=True, verbose=verbose)
        self.fit_epochs(epochs, noise_multiplier=noise_multiplier, covariance_method=covariance_method, method=method, n_jobs=n_jobs, verbose=verbose)

    @fill_doc
    def transform_epochs(self,
                         epochs: BaseEpochs,
                         verbose: Optional[str] = None):
        """Transform epochs data using the fitted model.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs to transform.
        %(verbose)s

        Returns
        -------
        epochs : mne.Epochs
            The transformed epochs.
        """
        check_type(epochs, (BaseEpochs,), 'epochs')

        epochs_wavelet, freq_bands = epochs_to_wavelet(epochs, wavelet=self.wavelet_type, level=self.wavelet_level)
        
        cleaned_epochs_wavelet = np.zeros_like(epochs_wavelet)
        for w, wavelet_fits in enumerate(self.wavelets_fits):
            wavelet_epochs_data = epochs_wavelet[:, :, w, :]
            cleaned_epochs, artefact_epochs = clean_epochs(wavelet_epochs_data, wavelet_fits['reference_cov'], wavelet_fits['threshold'])
            cleaned_epochs_wavelet[:, :, w, :] = cleaned_epochs
        
        # Recreate broadband signal
        cleaned_epochs_data = np.sum(cleaned_epochs_wavelet, axis=2)
        cleaned_epochs = mne.EpochsArray(cleaned_epochs_data, epochs.info, tmin=epochs.tmin, verbose=verbose)
        return cleaned_epochs

    @fill_doc
    def transform_raw(self, raw: BaseRaw,
                      duration: Optional[float] = 1,
                      overlap: float = 0.5,
                      verbose: Optional[str] = None):
        """Transform raw data using the fitted model.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        duration : float
            Duration of each epoch in seconds. Defaults to 1.
        overlap : float
            The overlap between epochs. Must be ``0 <= overlap < 1``. Default is 0.5 .
        %(verbose)s

        Returns
        -------
        raw_corrected : mne.io.BaseRaw
            The corrected raw data.
        """
        check_type(raw, (BaseRaw,), 'raw')
        check_type(duration, (float, int), 'duration')
        check_type(overlap, (float, int), 'overlap')

        raw_data = raw.get_data()
        n_channels, n_times = raw_data.shape

        window_size = raw.info['sfreq'] * duration
        window = create_cosine_weights(window_size)

        raw_corrected = np.zeros_like(raw_data)
        weight_sum = np.zeros_like(raw_data)

        step = int(window_size * (1 - overlap))
        starts = np.arange(0, n_times, step)
        for s, start in enumerate(starts[:-2]):
            end = int(min(start + window_size, n_times))
            actual_window_size = end - start
            segment = raw_data[:, start:end]
            segment_epoch = mne.EpochsArray(segment[np.newaxis], raw.info, verbose=False)
            # GEDAI
            corrected_epochs = self.transform_epochs(segment_epoch, verbose=False)
            corrected_segment = corrected_epochs.get_data()[0]
            corrected_segment *= window[:actual_window_size]
            raw_corrected[:, start:end] += corrected_segment
            weight_sum[:, start:end] += window[:actual_window_size]
        
        # Normalize the corrected signal by the weight sum
        weight_sum[weight_sum == 0] = 1 # Avoid division by zero
        raw_corrected /= weight_sum

        raw_corrected = mne.io.RawArray(raw_corrected, raw.info, verbose=False)
        return raw_corrected
    
    def _sensai_to_eigen(self, sensai_value, eigenvalues):
        all_diagonals = np.abs(eigenvalues.T.flatten())
        log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
        T1 = (105 - sensai_value) / 100
        threshold1 = T1 * np.percentile(log_eig_val_all, 95)
        eigenvalue = np.exp(threshold1 - 100)
        return eigenvalue

    def _eigen_to_sensai(self, eigenvalue, eigenvalues):
        all_diagonals = np.abs(eigenvalues.T.flatten())
        log_eig_val_all = np.log(all_diagonals[all_diagonals > 0]) + 100
        threshold1 = np.log(eigenvalue) + 100
        T1 = threshold1 / np.percentile(log_eig_val_all, 95)
        sensai_value = 105 - T1 * 100
        return sensai_value

    def plot_fit(self):
        wavelet_fits = self.wavelets_fits
        figs = []
        for w, wavelet_fit in enumerate(wavelet_fits):
            threshold = wavelet_fit['threshold']
            eigenvalues = wavelet_fit['epochs_eigenvalues']

            sensai_runs = wavelet_fit['sensai_runs']
            eigen_thresholds = [run[0] for run in sensai_runs]
            sensai_thresholds = [self._eigen_to_sensai(thresh, eigenvalues) for thresh in eigen_thresholds]

            sensai_score = [run[1] for run in sensai_runs]
            signal_score = [run[2] for run in sensai_runs]
            noise_score = [run[3] for run in sensai_runs]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].hist(eigenvalues.flatten(), bins=50, color='gray')
            axes[0].axvline(threshold, color='red', linestyle='--', label='Threshold')
            axes[0].set_xlabel('Eigenvalue')

            axes[1].plot(sensai_thresholds, sensai_score, label='SENSAI score', color='black')
            axes[1].plot(sensai_thresholds, signal_score, label='Signal similarity', color='blue')
            axes[1].plot(sensai_thresholds, noise_score, label='Noise similarity', color='red')
            axes[1].axvline(self._eigen_to_sensai(threshold, eigenvalues), color='green', linestyle='--', label='Threshold')
            axes[1].set_xlabel('SENSAI threshold')
            axes[1].legend()

            # Add second x-axis for eigenvalue thresholds
            ax2 = axes[1].twiny()
            ax2.set_xlim(axes[1].get_xlim())
            ax2.set_xticks(sensai_thresholds[::len(sensai_thresholds)//5])
            ax2.set_xticklabels([f"{eigen_thresholds[i]:.2e}" for i in range(0, len(eigen_thresholds), len(eigen_thresholds)//5)])
            ax2.set_xlabel('Eigenvalue Threshold')

            fig.suptitle(f'Band {w+1}: {wavelet_fit["fmin"]:.2f}-{wavelet_fit["fmax"]:.2f} Hz')
            figs.append(fig)

        return figs

