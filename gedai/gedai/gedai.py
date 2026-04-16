import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import BaseEpochs
from mne._fiff.pick import _picks_to_idx
from mne.io import BaseRaw
from mne.parallel import parallel_func
from scipy.linalg import eigh

from ..sensai.sensai import (
    _eigen_to_sensai,
    _sensai_gridsearch,
    _sensai_optimize,
    _sensai_to_eigen,
)
from ..utils._checks import _check_n_jobs, _check_picks_uniqueness, check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger, verbose
from ..wavelet.transform import epochs_to_wavelet
from .covariances import _ensure_cov, _pick_cov


def create_cosine_weights(n_samples):
    """Create cosine weights for a single epoch, mimicking the MATLAB implementation."""
    u = np.arange(1, n_samples + 1)
    cos_win = 0.5 - 0.5 * np.cos(2 * u * np.pi / n_samples)
    return cos_win


def compute_required_duration(wavelet_level, sfreq):
    """Compute the minimum epoch duration required for a given wavelet level.

    Parameters
    ----------
    wavelet_level : int
        The desired wavelet decomposition level.
    sfreq : float
        The sampling frequency in Hz.

    Returns
    -------
    duration : float
        Minimum duration in seconds required for the wavelet level.
    """
    if wavelet_level == 0:
        return 1.0  # Default for no decomposition
    # For SWT, minimum length is 2^(level+1)
    min_samples = 2 ** (wavelet_level + 1)
    duration = min_samples / sfreq
    return duration


def compute_closest_valid_duration(target_duration, wavelet_level, sfreq):
    """Compute the closest valid duration for a given wavelet level.

    For SWT to work at a given level, the signal length must be divisible by 2^level.
    This function finds the closest valid duration to the target duration.

    Parameters
    ----------
    target_duration : float
        The desired duration in seconds.
    wavelet_level : int
        The desired wavelet decomposition level.
    sfreq : float
        The sampling frequency in Hz.

    Returns
    -------
    valid_duration : float
        The closest valid duration in seconds.
    valid_samples : int
        The number of samples for the valid duration.
    """
    if wavelet_level == 0:
        # No constraint for level 0
        return target_duration, int(target_duration * sfreq)

    # Convert target duration to samples
    target_samples = int(target_duration * sfreq)

    # For SWT at level L, length must be divisible by 2^L
    divisor = 2**wavelet_level

    # Find the smallest valid number of samples >= target_samples.
    # A valid number of samples must be a multiple of the divisor.
    if target_samples % divisor == 0:
        valid_samples = target_samples
    else:
        # If not a multiple, round up to the next multiple of the divisor.
        valid_samples = ((target_samples // divisor) + 1) * divisor

    # Ensure we meet minimum length requirement (2^(level+1))
    min_samples = 2 ** (wavelet_level + 1)
    if valid_samples < min_samples:
        valid_samples = min_samples

    valid_duration = valid_samples / sfreq

    return valid_duration, valid_samples


def _check_sensai_method(method):
    check_type(method, (str,), "method")
    if method not in ["gridsearch", "optimize"]:
        raise ValueError(
            f"Method must be either 'gridsearch' or 'optimize', got '{method}' instead."
        )


@fill_doc
class Gedai:
    """Generalized Eigenvalue De-Artifacting Instrument (GEDAI).

    See :footcite:`Ros2025`.

    .. warning::
        For EEG channels, Gedai will set average reference internally
        to match the leadfield covariance reference.
        Gedai will not modify the input data in-place, but will create
        copies when necessary to ensure the original data remains unchanged.

    Parameters
    ----------
    wavelet_type : str
        Wavelet to use for the decomposition. The default is 'haar'.
        See :py:func:`pywt.wavedec` for complete list of available wavelet values.
    wavelet_level : int
        Decomposition level (must be >= 0). The default is 0 (no decomposition).
        If 0 (default), no wavelet decomposition is performed.
        See :py:func:`pywt.wavedec` more details.
    wavelet_low_cutoff : float | None
        If ``float``, zero out all wavelet levels (i.e frequency bands) whose upper
        frequency bound is below this cutoff frequency (in Hz).
        If ``None``, no frequency band is zeroed out. The default is ``None``.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, wavelet_type="haar", wavelet_level=0, wavelet_low_cutoff=None):
        self.fitted = False
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.wavelet_low_cutoff = wavelet_low_cutoff

        self._wavelets_fits = None
        self._reference_cov = None
        self._levels = None

    def _check_fit(self):
        """Check if the Gedai is fitted."""
        if not self.fitted:
            raise RuntimeError(
                f"Gedai must be fitted before using {self.__class__.__name__}"
            )
        # sanity-check
        assert self._wavelets_fits is not None
        assert self._reference_cov is not None
        assert self._levels is not None

    def _check_unfitted(self):
        """Check if the Gedai is unfitted."""
        if self.fitted:
            raise RuntimeError(
                f"Gedai must be unfitted before using {self.__class__.__name__}."
            )
        assert self._wavelets_fits is None
        assert self._reference_cov is None
        assert self._levels is None

    @fill_doc
    @verbose
    def fit_epochs(
        self,
        epochs: BaseEpochs,
        picks: list | str = "eeg",
        reference_cov: str = "leadfield",
        sensai_method: str = "optimize",
        noise_multiplier: float = 3.0,
        n_jobs: int = None,
        verbose: str | None = None,
    ):
        """Fit the GEDAI model to the epochs data.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs data to fit the model to.
        picks : str | list | slice
            Channels to include. Note that all channels selected must have the same
            type. Slices and lists of integers will be interpreted as channel indices.
            In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the given
            channels. Can also be the string values ``“all”`` to pick all channels, or
            ``“data”`` to pick data channels. The default is ``“eeg”`` to pick all
            EEG channels.
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        %(n_jobs)s
        %(verbose)s
        """
        self._check_unfitted()
        check_type(epochs, (BaseEpochs,), "epochs")
        _ensure_cov(reference_cov)
        _check_sensai_method(sensai_method)
        check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        # Data
        picks = _picks_to_idx(epochs.info, picks, none="all", exclude=[])
        _check_picks_uniqueness(epochs.info, picks)
        epochs = epochs.copy()
        epochs.load_data()
        epochs = epochs.pick(picks)
        logger.info("Setting average reference.")
        epochs.set_eeg_reference("average", projection=False)
        data = epochs.get_data()

        # reference covariance
        cov = _ensure_cov(reference_cov)
        cov = _pick_cov(cov, epochs.info["ch_names"])
        reference_cov = cov.data
        # Tikhonov Regularization based on average diagonal power
        # TODO. move ?
        avg_diag_power = np.trace(reference_cov) / reference_cov.shape[0]
        regularization_lambda = 0.05
        epsilon = regularization_lambda * avg_diag_power
        reference_cov = reference_cov + epsilon * np.eye(reference_cov.shape[0])

        # Wavelet Decomposition
        epochs_wavelet, freq_bands, levels = epochs_to_wavelet(
            data,
            sfreq=epochs.info["sfreq"],
            wavelet=self.wavelet_type,
            level=self.wavelet_level,
            n_jobs=n_jobs,
        )

        # Store the actual levels used for consistency in transform
        wavelets_fits = []
        for w, (fmin, fmax) in enumerate(freq_bands):
            wavelet_epochs_data = epochs_wavelet[:, :, w, :]

            epochs_eigenvalues = np.zeros(
                (len(wavelet_epochs_data), wavelet_epochs_data.shape[1])
            )
            for e, wavelet_epoch_data in enumerate(wavelet_epochs_data):
                covariance = np.cov(wavelet_epoch_data)
                eigenvalues, _ = eigh(covariance, reference_cov, check_finite=True)
                epochs_eigenvalues[e] = eigenvalues

            wavelet_epochs = mne.EpochsArray(
                wavelet_epochs_data, epochs.info, tmin=epochs.tmin, verbose=False
            )
            min_sensai_threshold, max_sensai_threshold, step = 0, 12, 0.1

            # Extend minThreshold for alpha range (7-13 Hz)
            center_freq = (fmin + fmax) / 2
            if 7 <= center_freq <= 13:
                min_sensai_threshold = -6
                logger.info(
                    f"Alpha range detected ({fmin:.1f}-{fmax:.1f} Hz):"
                    f" extending minThreshold to {min_sensai_threshold}"
                )

            n_pc = 3
            if sensai_method == "gridsearch":
                sensai_thresholds = np.arange(
                    min_sensai_threshold, max_sensai_threshold, step
                )
                eigen_thresholds = [
                    _sensai_to_eigen(sensai_value, epochs_eigenvalues)
                    for sensai_value in sensai_thresholds
                ]
                threshold, runs = _sensai_gridsearch(
                    wavelet_epochs,
                    reference_cov,
                    n_pc=n_pc,
                    noise_multiplier=noise_multiplier,
                    eigen_thresholds=eigen_thresholds,
                    n_jobs=n_jobs,
                )
            elif sensai_method == "optimize":
                sensai_threshold_bounds = (min_sensai_threshold, max_sensai_threshold)
                threshold, runs = _sensai_optimize(
                    wavelet_epochs,
                    reference_cov,
                    n_pc=n_pc,
                    noise_multiplier=noise_multiplier,
                    epochs_eigenvalues=epochs_eigenvalues,
                    bounds=sensai_threshold_bounds,
                )
            else:
                raise ValueError(
                    "Method must be either 'gridsearch' or 'optimize', "
                    f"got '{sensai_method}' instead."
                )
            # Store band_index to map back
            wavelet_fit = {
                "band_index": w,
                "fmin": fmin,
                "fmax": fmax,
                "threshold": threshold,
                "epochs_eigenvalues": epochs_eigenvalues,
                "sensai_runs": runs,
            }
            # ignore
            ignore = False
            if self.wavelet_low_cutoff is not None:
                if fmax < self.wavelet_low_cutoff:
                    ignore = True
                    logger.info(
                        f"Wavelet index {w} ({fmin:.2f}-{fmax:.2f} Hz) "
                        f"will be zeroed out during transformation "
                        f"because its upper frequency {fmax:.2f} Hz "
                        f"is below the low cutoff {self.wavelet_low_cutoff:.2f} Hz."
                    )
            wavelet_fit["ignore"] = ignore
            wavelets_fits.append(wavelet_fit)

        self._levels = levels
        self._wavelets_fits = wavelets_fits
        self._reference_cov = cov
        self.fitted = True

    @fill_doc
    @verbose
    def fit_raw(
        self,
        raw: BaseRaw,
        picks: list | str = "eeg",
        duration: float = 1.0,
        overlap: float = 0.5,
        reject_by_annotation: bool | None = False,
        reference_cov: str = "leadfield",
        sensai_method: str = "optimize",
        noise_multiplier: float = 3.0,
        n_jobs: int = None,
        verbose: str | None = None,
    ):
        """Fit the GEDAI model to the raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        picks : str | list | slice
            Channels to include. Note that all channels selected must have the same
            type. Slices and lists of integers will be interpreted as channel indices.
            In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the given
            channels. Can also be the string values ``“all”`` to pick all channels, or
            ``“data”`` to pick data channels. The default is ``“eeg”`` to pick all
            EEG channels.
        duration : float
            Duration of each epoch in seconds (default 1.0). Will be automatically
            adjusted to the closest valid duration for the wavelet level.
        overlap : float
            The overlap ratio between epochs (0 to 1). Default is 0.5 (50%% overlap).
            For example, 0.5 means 50%% overlap, 0.75 means 75%% overlap.
        reject_by_annotation : bool
            Whether to reject epochs based on annotations. Default is False.
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        %(n_jobs)s
        %(verbose)s
        """
        check_type(raw, (BaseRaw,), "raw")
        check_type(
            duration,
            (
                float,
                int,
            ),
            "duration",
        )
        check_type(
            overlap,
            (
                float,
                int,
            ),
            "overlap",
        )
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        reference_cov = _ensure_cov(reference_cov)
        _check_sensai_method(sensai_method)
        check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        # Adjust user's duration to closest valid duration
        valid_duration, valid_samples = compute_closest_valid_duration(
            duration, self.wavelet_level, raw.info["sfreq"]
        )
        if valid_duration != duration:
            logger.warn(
                f"Requested duration {duration:.3f}s adjusted to {valid_duration:.3f}s "
                f"({valid_samples} samples) to satisfy wavelet level"
                f" {self.wavelet_level} requirements."
            )
        duration = valid_duration

        # Convert overlap ratio to seconds for mne.make_fixed_length_epochs
        overlap_seconds = duration * overlap

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=duration,
            overlap=overlap_seconds,
            reject_by_annotation=reject_by_annotation,
            preload=True,
            verbose=verbose,
        )
        self.fit_epochs(
            epochs,
            picks=picks,
            noise_multiplier=noise_multiplier,
            reference_cov=reference_cov,
            sensai_method=sensai_method,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    @fill_doc
    @verbose
    def transform_epochs(
        self, epochs: BaseEpochs, n_jobs: int = None, verbose: str | None = None
    ):
        """Transform epochs data using the fitted model.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs to transform.
        %(n_jobs)s
        %(verbose)s

        Returns
        -------
        epochs : mne.Epochs
            The transformed epochs.
        """
        self._check_fit()
        check_type(epochs, (BaseEpochs,), "epochs")
        n_jobs = _check_n_jobs(n_jobs)

        # Data
        missing_ch = set(self.ch_names) - set(epochs.info["ch_names"])
        if len(missing_ch) > 0:
            raise ValueError(
                "The following channels are missing in the input inst but were "
                "present during fitting: "
                f"{missing_ch}. \n"
                "Please make sure to include the same channels during transform "
                "as were used during fit. \n"
                "See "
                f"{self.__class__.__name__}.ch_names "
                "for the list of channels used during fit."
            )
        extra_ch = set(epochs.info["ch_names"]) - set(self.ch_names)
        if len(extra_ch) > 0:
            raise ValueError(
                "The following channels are present in the input inst but were "
                "not present during fitting: "
                f"{extra_ch}. \n"
                "These channels will be ignored during transformation. \n"
                "Please make sure to include the same channels during transform "
                "as were used during fit. \n"
                "See "
                f"{self.__class__.__name__}.ch_names "
                "for the list of channels used during fit."
            )

        picks = _picks_to_idx(epochs.info, self.ch_names, none="all", exclude=[])
        epochs = epochs.copy()
        epochs.load_data()
        epochs = epochs.pick(picks)
        logger.info("Setting average reference.")
        epochs.set_eeg_reference("average", projection=False)
        data = epochs.get_data()

        # reference covariance
        reference_cov = self._reference_cov.data

        epochs_wavelet, freq_bands, levels = epochs_to_wavelet(
            data,
            sfreq=epochs.info["sfreq"],
            wavelet=self.wavelet_type,
            level=self.wavelet_level,
            n_jobs=n_jobs,
        )

        # Validate that the decomposition matches the fitted model
        if levels != self._levels:
            raise ValueError(
                f"Wavelet decomposition levels mismatch. \n"
                f"Model was fitted with levels {self._levels}, "
                f"but transform got levels {levels}. \n"
                f"This may happen if epoch lengths differ between fit and transform."
            )

        cleaned_epochs_wavelet = epochs_wavelet.copy()

        for wavelet_fit in self._wavelets_fits:
            band_idx = wavelet_fit["band_index"]
            ignore = wavelet_fit["ignore"]

            if ignore:
                # Zero out this band
                cleaned_epochs_wavelet[:, :, band_idx, :] = 0
            else:
                wavelet_epochs_data = epochs_wavelet[:, :, band_idx, :]
                threshold = wavelet_fit["threshold"]

                if n_jobs == 1:
                    # Sequential processing
                    for e, epoch_data in enumerate(wavelet_epochs_data):
                        cleaned_epochs_wavelet[e, :, band_idx, :] = (
                            _process_single_epoch(epoch_data, reference_cov, threshold)
                        )
                else:
                    # Parallel processing across epochs
                    parallel, p_fun, _ = parallel_func(_process_single_epoch, n_jobs)
                    cleaned_epochs_list = parallel(
                        p_fun(epoch_data, reference_cov, threshold)
                        for epoch_data in wavelet_epochs_data
                    )
                    cleaned_epochs_wavelet[:, :, band_idx, :] = np.array(
                        cleaned_epochs_list
                    )

        # Recreate broadband signal
        cleaned_epochs_data = np.sum(cleaned_epochs_wavelet, axis=2)
        cleaned_epochs = mne.EpochsArray(
            cleaned_epochs_data, epochs.info, tmin=epochs.tmin, verbose=verbose
        )
        return cleaned_epochs

    @fill_doc
    @verbose
    def transform_raw(
        self,
        raw: BaseRaw,
        duration: float = 1.0,
        overlap: float = 0.5,
        n_jobs: int = None,
        verbose: str | None = None,
    ):
        """Transform raw data using the fitted model.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        duration : float
            Duration of each epoch in seconds (default 1.0). Will be automatically
            adjusted to the closest valid duration for the wavelet level.
        overlap : float
            The overlap ratio between epochs (0 to 1). Default is 0.5 (50%% overlap).
            For example, 0.5 means 50%% overlap, 0.75 means 75%% overlap.
        %(n_jobs)s
        %(verbose)s

        Returns
        -------
        raw_corrected : mne.io.BaseRaw
            The corrected raw data.
        """
        check_type(raw, (BaseRaw,), "raw")
        check_type(duration, (float, int), "duration")
        check_type(overlap, (float, int), "overlap")
        n_jobs = _check_n_jobs(n_jobs)

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        # Adjust user's duration to closest valid duration
        valid_duration, valid_samples = compute_closest_valid_duration(
            duration, self.wavelet_level, raw.info["sfreq"]
        )
        if (
            abs(valid_duration - duration) > 1e-6
        ):  # Only warn if there's a significant difference
            logger.warn(
                f"Requested duration {duration:.3f}s adjusted to {valid_duration:.3f}s"
                f" ({valid_samples} samples) to satisfy wavelet level"
                f" {self.wavelet_level} requirements."
            )
        duration = valid_duration

        raw_data = raw.get_data(verbose=False)
        n_channels, n_times = raw_data.shape

        window_size = int(raw.info["sfreq"] * duration)
        window = create_cosine_weights(window_size)

        raw_corrected = np.zeros_like(raw_data)
        weight_sum = np.zeros_like(raw_data)

        step = int(window_size * (1 - overlap))
        starts = np.arange(0, n_times - window_size, step)
        starts = np.append(starts, n_times - window_size)

        # Batch all segments together for parallel processing
        all_segments = []
        for start in starts:
            segment = raw_data[:, start : start + window_size]
            all_segments.append(segment)

        # Convert to epochs array (n_epochs, n_channels, n_times)
        all_segments_array = np.array(all_segments)
        segments_epochs = mne.EpochsArray(all_segments_array, raw.info, verbose=verbose)

        # Process all segments at once with parallelization
        corrected_segments_epochs = self.transform_epochs(
            segments_epochs, n_jobs=n_jobs, verbose=verbose
        )
        corrected_segments = corrected_segments_epochs.get_data(verbose=verbose)

        # Apply windowing and reconstruct
        for s, start in enumerate(starts):
            corrected_segment = corrected_segments[s] * window
            raw_corrected[:, start : start + window_size] += corrected_segment
            weight_sum[:, start : start + window_size] += window

        # Normalize the corrected signal by the weight sum
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero
        raw_corrected /= weight_sum

        raw_corrected = mne.io.RawArray(raw_corrected, raw.info, verbose=verbose)
        return raw_corrected

    def plot_fit(self):
        """Plot the fitting results.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            The list of figures showing the fitting results.
        """
        wavelet_fits = self._wavelets_fits
        figs = []
        for w, wavelet_fit in enumerate(wavelet_fits):
            if wavelet_fit["ignore"]:
                continue
            threshold = wavelet_fit["threshold"]
            eigenvalues = wavelet_fit["epochs_eigenvalues"]

            sensai_runs = wavelet_fit["sensai_runs"]
            eigen_thresholds = [run[0] for run in sensai_runs]
            sensai_thresholds = [
                _eigen_to_sensai(thresh, eigenvalues) for thresh in eigen_thresholds
            ]

            sensai_score = [run[1] for run in sensai_runs]
            signal_score = [run[2] for run in sensai_runs]
            noise_score = [run[3] for run in sensai_runs]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].hist(eigenvalues.flatten(), bins=50, color="gray")
            axes[0].axvline(threshold, color="red", linestyle="--", label="Threshold")
            axes[0].set_xlabel("Eigenvalue")

            axes[1].plot(
                sensai_thresholds, sensai_score, label="SENSAI score", color="black"
            )
            axes[1].plot(
                sensai_thresholds, signal_score, label="Signal similarity", color="blue"
            )
            axes[1].plot(
                sensai_thresholds, noise_score, label="Noise similarity", color="red"
            )
            axes[1].axvline(
                _eigen_to_sensai(threshold, eigenvalues),
                color="green",
                linestyle="--",
                label="Threshold",
            )
            axes[1].set_xlabel("SENSAI threshold")
            axes[1].legend()

            fig.suptitle(
                f"Band {w + 1}: {wavelet_fit['fmin']:.2f}-{wavelet_fit['fmax']:.2f} Hz"
            )
            figs.append(fig)
            axes[1].axvline(
                _eigen_to_sensai(threshold, eigenvalues),
                color="green",
                linestyle="--",
                label="Threshold",
            )
            axes[1].set_xlabel("SENSAI threshold")
            axes[1].legend()

            fig.suptitle(
                f"Band {w + 1}: {wavelet_fit['fmin']:.2f}-{wavelet_fit['fmax']:.2f} Hz"
            )
            figs.append(fig)
        return figs

    @property
    def ch_names(self):
        """Get the channel names used during fitting."""
        self._check_fit()
        return self._reference_cov.ch_names


def _process_single_epoch(epoch_data, reference_cov, threshold):
    """Process a single epoch for cleaning.

    Parameters
    ----------
    epoch_data : np.ndarray
        Single epoch data with shape (n_channels, n_times).
    reference_cov : np.ndarray
        Reference covariance matrix.
    threshold : float
        Threshold for component selection.

    Returns
    -------
    cleaned_epoch : np.ndarray
        The cleaned epoch data.
    """
    covariance = np.cov(epoch_data)
    eigenvalues, eigenvectors = eigh(covariance, reference_cov, check_finite=True)

    # Compute spatial maps
    maps = np.linalg.pinv(eigenvectors).T
    eigenvectors_filtered = eigenvectors.copy()

    # Zero out components with small eigenvalues
    for v, val in enumerate(eigenvalues):
        if abs(val) < threshold:
            maps[:, v] = 0
            eigenvectors_filtered[:, v] = 0

    # Reconstruct artifact signal
    spatial_filter = np.dot(maps, eigenvectors_filtered.T)
    artefact_data = spatial_filter @ epoch_data
    cleaned_epoch = epoch_data - artefact_data

    return cleaned_epoch
