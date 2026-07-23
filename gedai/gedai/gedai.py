import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import BaseEpochs
from mne._fiff.pick import _picks_to_idx
from mne.io import BaseRaw
from mne.parallel import parallel_func
from scipy.linalg import eigh

from ..covariance.covariance import _ensure_cov, _pick_cov
from ..sensai.sensai import (
    _eigen_to_sensai,
    _sensai_gridsearch,
    _sensai_optimize,
    _sensai_to_eigen,
)
from ..utils._checks import _check_n_jobs, _check_picks_uniqueness, _check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger, verbose


def create_cosine_weights(n_samples):
    """Create cosine weights for a single epoch, mimicking the MATLAB implementation."""
    u = np.arange(1, n_samples + 1)
    cos_win = 0.5 - 0.5 * np.cos(2 * u * np.pi / n_samples)
    return cos_win


def _check_sensai_method(method):
    _check_type(method, (str,), "method")
    if method not in ["gridsearch"]:
        raise ValueError(f"Method must be 'gridsearch', got '{method}' instead.")


@fill_doc
class Gedai:
    """Generalized Eigenvalue De-Artifacting Instrument (GEDAI).

    This class implements the single band GEDAI workflow.
    For wavelet-based decomposition, band-wise denoising use :class:`MultibandGedai`.

    See :footcite:`Ros2025`.

    .. warning::
        For EEG channels, Gedai will set average reference internally
        to match the leadfield covariance reference.
        Gedai will not modify the input data in-place, but will create
        copies when necessary to ensure the original data remains unchanged.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self):
        self.fitted = False
        self._fit = None
        self._reference_cov = None
        self._info = None
        self._n_samples = None

    def _check_fit(self):
        """Check if the Gedai is fitted."""
        if not self.fitted:
            raise RuntimeError(
                f"Gedai must be fitted before using {self.__class__.__name__}"
            )
        assert self._fit is not None
        assert self._reference_cov is not None
        assert self._info is not None
        assert self._n_samples is not None

    def _check_unfitted(self):
        """Check if the Gedai is unfitted."""
        if self.fitted:
            raise RuntimeError(
                f"Gedai must be unfitted before using {self.__class__.__name__}."
            )
        assert self._fit is None
        assert self._reference_cov is None
        assert self._info is None
        assert self._n_samples is None

    @fill_doc
    @verbose
    def fit_epochs(
        self,
        epochs: BaseEpochs,
        picks: list | str = "eeg",
        reference_cov: str = "leadfield",
        sensai_method: str = "gridsearch",
        noise_multiplier: float = 3.0,
        n_jobs: int = None,
        verbose: str | None = None,
    ):
        """Fit the GEDAI model to the epochs data.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs data to fit the model to.
        %(picks)s
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        %(n_jobs)s
        %(verbose)s
        """
        self._check_unfitted()
        _check_type(epochs, (BaseEpochs,), "epochs")
        _ensure_cov(reference_cov)
        _check_sensai_method(sensai_method)
        _check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        picks = _picks_to_idx(epochs.info, picks, none="all", exclude=[])
        _check_picks_uniqueness(epochs.info, picks)
        epochs = epochs.copy()
        epochs.load_data()
        epochs = epochs.pick(picks, verbose=False)
        logger.info("Setting average reference.")
        epochs.set_eeg_reference("average", projection=False, verbose=False)
        data = epochs.get_data(verbose=False)

        cov = _ensure_cov(reference_cov)
        cov = _pick_cov(cov, epochs.info["ch_names"])
        reference_cov = cov.data

        avg_diag_power = np.trace(reference_cov) / reference_cov.shape[0]
        regularization_lambda = 0.05
        epsilon = regularization_lambda * avg_diag_power
        reference_cov = reference_cov + epsilon * np.eye(reference_cov.shape[0])
        cov.update(data=reference_cov)

        epochs_eigenvalues = np.zeros((len(data), data.shape[1]))
        for e, epoch_data in enumerate(data):
            covariance = np.cov(epoch_data)
            eigenvalues, _ = eigh(covariance, reference_cov, check_finite=True)
            epochs_eigenvalues[e] = eigenvalues

        fit_epochs = mne.EpochsArray(data, epochs.info, tmin=epochs.tmin, verbose=False)
        min_sensai_threshold, max_sensai_threshold, step = (
            -6,
            12,
            0.1,
        )  # MATLAB min_sensai_threshold -6 for f < 60Hz.
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
                fit_epochs,
                reference_cov,
                n_pc=n_pc,
                noise_multiplier=noise_multiplier,
                eigen_thresholds=eigen_thresholds,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        elif sensai_method == "optimize":
            sensai_threshold_bounds = (min_sensai_threshold, max_sensai_threshold)
            threshold, runs = _sensai_optimize(
                fit_epochs,
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

        self._fit = {
            "threshold": threshold,
            "epochs_eigenvalues": epochs_eigenvalues,
            "sensai_runs": runs,
        }
        self._reference_cov = cov
        self.fitted = True
        self._info = epochs.info.copy()
        self._n_samples = data.shape[-1]

    @fill_doc
    @verbose
    def fit_raw(
        self,
        raw: BaseRaw,
        picks: list | str = "eeg",
        duration: float = 1.0,
        overlap: float = 0.75,
        reject_by_annotation: bool | None = False,
        reference_cov: str = "leadfield",
        sensai_method: str = "gridsearch",
        noise_multiplier: float = 3.0,
        n_jobs: int = None,
        verbose: str | None = None,
    ):
        """Fit the GEDAI model to the raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        %(picks)s
        %(duration)s
        %(overlap)s
        %(reject_by_annotation)s
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        %(n_jobs)s
        %(verbose)s
        """
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(duration, (float, int), "duration")
        _check_type(overlap, (float, int), "overlap")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        _check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        reference_cov = _ensure_cov(reference_cov)
        _check_sensai_method(sensai_method)
        _check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        overlap_seconds = duration * overlap
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=duration,
            overlap=overlap_seconds,
            reject_by_annotation=reject_by_annotation,
            preload=True,
            verbose=False,
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
        _check_type(epochs, (BaseEpochs,), "epochs")
        n_jobs = _check_n_jobs(n_jobs)

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

        if epochs.info["sfreq"] != self._info["sfreq"]:
            raise ValueError(
                f"Sampling frequency mismatch between fitted model and input instance."
                f"nFitted model sfreq: {self._info['sfreq']} Hz, input instance sfreq:"
                f" {epochs.info['sfreq']} Hz."
            )

        if epochs.get_data(verbose=False).shape[-1] != self._n_samples:
            input_duration = (
                epochs.get_data(verbose=False).shape[-1] - 1
            ) / epochs.info["sfreq"]
            raise ValueError(
                f"Duration mismatch between fitted model and input instance. "
                f"Fitted model epoch duration: {self._duration} s, input instance "
                f"epoch duration: {input_duration} s."
                " Please make sure the epoch duration of the input instance "
                "matches the one of the data used during fit."
            )

        picks = _picks_to_idx(epochs.info, self.ch_names, none="all", exclude=[])
        epochs_copy = epochs.copy()
        epochs_copy.load_data()
        epochs_copy = epochs_copy.pick(picks)
        logger.info("Setting average reference.")
        epochs_copy.set_eeg_reference("average", projection=False)
        data = epochs_copy.get_data(verbose=False)

        reference_cov = self._reference_cov.data
        threshold = self._fit["threshold"]
        cleaned_epochs_data = np.zeros_like(data)

        if n_jobs == 1:
            for e, epoch_data in enumerate(data):
                cleaned_epochs_data[e] = _process_single_epoch(
                    epoch_data, reference_cov, threshold
                )
        else:
            parallel, p_fun, _ = parallel_func(
                _process_single_epoch, n_jobs, total=len(data), verbose=verbose
            )
            cleaned_epochs_list = parallel(
                p_fun(epoch_data, reference_cov, threshold) for epoch_data in data
            )
            cleaned_epochs_data = np.array(cleaned_epochs_list)

        cleaned_epochs = epochs.copy()
        cleaned_epochs._data = cleaned_epochs_data
        return cleaned_epochs

    @fill_doc
    @verbose
    def transform_raw(
        self,
        raw: BaseRaw,
        overlap: float = 0.75,
        n_jobs: int = None,
        verbose: str | None = None,
    ):
        """Transform raw data using the fitted model.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        %(overlap)s
        %(n_jobs)s
        %(verbose)s

        Returns
        -------
        raw_corrected : mne.io.BaseRaw
            The corrected raw data.
        """
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        n_jobs = _check_n_jobs(n_jobs)

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        raw_data = raw.get_data(verbose=False)
        n_channels, n_times = raw_data.shape

        window_size = self._n_samples
        window = create_cosine_weights(window_size)

        raw_corrected = np.zeros_like(raw_data)
        weight_sum = np.zeros_like(raw_data)

        step = int(window_size * (1 - overlap))
        starts = np.arange(0, n_times - window_size, step)
        starts = np.append(starts, n_times - window_size)

        all_segments = []
        for start in starts:
            segment = raw_data[:, start : start + window_size]
            all_segments.append(segment)

        all_segments_array = np.array(all_segments)
        segments_epochs = mne.EpochsArray(all_segments_array, raw.info, verbose=False)

        corrected_segments_epochs = self.transform_epochs(
            segments_epochs, n_jobs=n_jobs, verbose=False
        )
        corrected_segments = corrected_segments_epochs.get_data(verbose=False)

        for s, start in enumerate(starts):
            corrected_segment = corrected_segments[s] * window
            raw_corrected[:, start : start + window_size] += corrected_segment
            weight_sum[:, start : start + window_size] += window

        weight_sum[weight_sum == 0] = 1
        raw_corrected /= weight_sum

        raw_corrected = mne.io.RawArray(raw_corrected, raw.info, verbose=False)
        return raw_corrected

    def plot_fit(self):
        """Plot the fitting results.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            The list of figures showing the fitting results.
        """
        self._check_fit()
        threshold = self._fit["threshold"]
        eigenvalues = self._fit["epochs_eigenvalues"]
        sensai_runs = self._fit["sensai_runs"]
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
        fig.suptitle("Broadband GEDAI")

        return [fig]

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


# Backward-compatible import path: gedai.gedai.gedai.MultibandGedai
