import mne
import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw

from gedai.gedai._utils import (
    _check_fit_info,
    _prepare_epochs_fit,
    _prepare_epochs_transform,
    _prepare_raw_fit,
    _prepare_raw_transform,
)

from ..covariance.covariance import _ensure_cov, _pick_cov
from ..utils._checks import (
    _check_n_jobs,
    _check_type,
)
from ..utils._docs import fill_doc
from ..utils.logs import logger, verbose
from ..wavelet.transform import epochs_to_wavelet
from .gedai import Gedai, create_cosine_weights


def _ensure_wavelet_low_cutoff(wavelet_low_cutoff, filter_highpass, epoch_duration):
    duration_highpass = 2 / epoch_duration  # 2 cycles
    if wavelet_low_cutoff == "auto":
        if filter_highpass > duration_highpass:
            logger.info(
                f"Setting wavelet_low_cutoff to {filter_highpass} Hz based on "
                f"high-pass filter cutoff frequency in data info"
                f" (info['highpass'] = {filter_highpass} Hz)."
            )
            wavelet_low_cutoff = filter_highpass

        else:
            logger.info(
                f"Setting wavelet_low_cutoff to {duration_highpass} Hz based on "
                f"epoch duration and sampling frequency."
            )
            wavelet_low_cutoff = duration_highpass
    elif wavelet_low_cutoff is None:
        wavelet_low_cutoff = 0
    else:
        wavelet_low_cutoff = wavelet_low_cutoff

    if duration_highpass > wavelet_low_cutoff:
        logger.warning(
            f"wavelet_low_cutoff ({wavelet_low_cutoff:.2f} Hz) is below the "
            f"frequency cutoff ( {duration_highpass:.2f} Hz) that can be "
            f"resolved with an epoch duration of {1 / duration_highpass:.2f} Hz."
            "Lower frequency bands may not be well estimated. Consider "
            "increasing wavelet_low_cutoff or using longer window durations "
            "during fitting."
        )
    if filter_highpass > wavelet_low_cutoff:
        logger.warning(
            f"wavelet_low_cutoff ({wavelet_low_cutoff:.2f} Hz) is below the "
            f"high-pass filter cutoff frequency in data info (info['highpass'] "
            f"= {filter_highpass} Hz). Lower frequency bands will be keep "
            f" even if no signal of interest is expected in these bands."
        )
    return wavelet_low_cutoff


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


@fill_doc
class MultibandGedai:
    """Multiband Generalized Eigenvalue De-Artifacting Instrument.

    A multiband extension of the standard :class:`~gedai.gedai.Gedai` that applies
    GEDAI algorithm separately to different frequency bands (via wavelet decomposition).
    This approach allows for more targeted artifact removal while preserving
    neural signals.
    See :footcite:`Ros2025`.

    .. warning::
        For EEG channels, Gedai will set average reference internally
        to match the leadfield covariance reference.
        Gedai will not modify the input data in-place, but will create
        copies when necessary to ensure the original data remains unchanged.

    Parameters
    ----------
    %(wavelet_type)s
    %(wavelet_level)s

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, wavelet_type="haar", wavelet_level=8):
        _check_type(wavelet_level, (int,), "wavelet_level")
        _check_type(wavelet_type, (str,), "wavelet_type")
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level

        self.fitted = False
        self._wavelet_low_cutoff = None
        self._wavelets_fits = None
        self._reference_cov = None
        self._levels = None

    def _check_fit(self):
        """Check if the Gedai is fitted."""
        if not self.fitted:
            raise RuntimeError(
                f"Gedai must be fitted before using {self.__class__.__name__}"
            )
        assert self._wavelets_fits is not None
        for wavelet_fit in self._wavelets_fits:
            if not wavelet_fit["ignore"]:
                wavelet_fit["model"]._check_fit()
        assert self._reference_cov is not None
        assert self._levels is not None
        assert self._wavelet_low_cutoff is not None

    def _check_unfitted(self):
        """Check if the Gedai is unfitted."""
        if self.fitted:
            raise RuntimeError(
                f"Gedai must be unfitted before using {self.__class__.__name__}."
            )
        assert self._wavelets_fits is None
        assert self._reference_cov is None
        assert self._levels is None
        assert self._wavelet_low_cutoff is None

    @fill_doc
    @verbose
    def fit_epochs(
        self,
        epochs: BaseEpochs,
        picks: list | str = "eeg",
        reference_cov: str = "leadfield",
        sensai_method: str = "gridsearch",
        noise_multiplier: float = 3.0,
        wavelet_low_cutoff="auto",
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
        %(wavelet_low_cutoff)s
        %(n_jobs)s
        %(verbose)s
        """
        self._check_unfitted()
        _check_type(epochs, (BaseEpochs,), "epochs")
        _ensure_cov(reference_cov)
        _check_type(sensai_method, (str,), "sensai_method")
        _check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        epochs_fit = _prepare_epochs_fit(epochs, picks)
        data = epochs_fit.get_data()

        cov = _ensure_cov(reference_cov)
        cov = _pick_cov(cov, epochs_fit.info["ch_names"])

        epoch_duration = epochs_fit.tmax - epochs_fit.tmin
        wavelet_low_cutoff = _ensure_wavelet_low_cutoff(
            wavelet_low_cutoff, epochs_fit.info["highpass"], epoch_duration
        )

        epochs_wavelet, freq_bands, levels = epochs_to_wavelet(
            data,
            sfreq=epochs_fit.info["sfreq"],
            wavelet=self.wavelet_type,
            level=self.wavelet_level,
            n_jobs=n_jobs,
        )
        n_samples = epochs_wavelet.shape[-1]

        wavelets_fits = []
        for w, (fmin, fmax) in enumerate(freq_bands):
            if fmax < wavelet_low_cutoff:
                logger.info(
                    f"Wavelet index {w} ({fmin:.2f}-{fmax:.2f} Hz) "
                    f"will be zeroed out during transformation because its upper "
                    f"frequency {fmax:.2f} Hz is below the low cutoff "
                    f"{wavelet_low_cutoff:.2f} Hz."
                )
                wavelet_fit = {
                    "band_index": w,
                    "fmin": fmin,
                    "fmax": fmax,
                    "model": None,
                    "ignore": True,
                    "n_samples": n_samples,
                }
            else:
                wavelet_epochs_data = epochs_wavelet[:, :, w, :]
                wavelet_epochs = mne.EpochsArray(
                    wavelet_epochs_data, epochs.info, tmin=epochs.tmin, verbose=False
                )

                model = Gedai()
                model.fit_epochs(
                    wavelet_epochs,
                    picks="all",
                    reference_cov=cov.copy(),
                    sensai_method=sensai_method,
                    noise_multiplier=noise_multiplier,
                    n_jobs=n_jobs,
                    verbose=verbose,
                )

                wavelet_fit = {
                    "band_index": w,
                    "fmin": fmin,
                    "fmax": fmax,
                    "model": model,
                    "ignore": False,
                    "n_samples": n_samples,
                }
            wavelets_fits.append(wavelet_fit)

        self.fitted = True
        self._info = epochs_fit.info.copy()
        self._reference_cov = cov  # No regularization applied

        self._levels = levels
        self._wavelets_fits = wavelets_fits
        self._wavelet_low_cutoff = wavelet_low_cutoff


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
        wavelet_low_cutoff="auto",
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
        %(wavelet_low_cutoff)s
        %(n_jobs)s
        %(verbose)s
        """
        self._check_unfitted()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(duration, (float, int), "duration")
        _check_type(overlap, (float, int), "overlap")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        _check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        reference_cov = _ensure_cov(reference_cov)
        _check_type(sensai_method, (str,), "sensai_method")
        _check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        valid_duration, valid_samples = compute_closest_valid_duration(
            duration, self.wavelet_level, raw.info["sfreq"]
        )
        if valid_duration != duration:
            logger.warning(
                f"Requested duration {duration:.3f}s adjusted to {valid_duration:.3f}s "
                f"({valid_samples} samples) to satisfy wavelet level "
                f"{self.wavelet_level} requirements."
            )
        duration = valid_duration

        raw_fit = _prepare_raw_fit(raw, picks)

        overlap_seconds = duration * overlap
        epochs = mne.make_fixed_length_epochs(
            raw_fit,
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
            wavelet_low_cutoff=wavelet_low_cutoff,
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
        epochs_transformed : mne.Epochs
            The transformed epochs.
        """
        self._check_fit()
        _check_type(epochs, (BaseEpochs,), "epochs")
        n_jobs = _check_n_jobs(n_jobs)

        _check_fit_info(self, epochs)
        epochs_transform = _prepare_epochs_transform(epochs, self.ch_names)

        data = epochs_transform.get_data()

        epochs_wavelet, _, levels = epochs_to_wavelet(
            data,
            sfreq=epochs_transform.info["sfreq"],
            wavelet=self.wavelet_type,
            level=self.wavelet_level,
            n_jobs=n_jobs,
        )

        if levels != self._levels:
            raise ValueError(
                "Wavelet decomposition levels mismatch. \n"
                f"Model was fitted with levels {self._levels}, "
                f"but transform got levels {levels}. \n"
                "This may happen if epoch lengths differ between fit and transform."
            )

        cleaned_epochs_wavelet = epochs_wavelet.copy()
        for wavelet_fit in self._wavelets_fits:
            band_idx = wavelet_fit["band_index"]
            if wavelet_fit["ignore"]:
                cleaned_epochs_wavelet[:, :, band_idx, :] = 0
                continue

            wavelet_epochs_data = epochs_wavelet[:, :, band_idx, :]
            wavelet_epochs = mne.EpochsArray(
                wavelet_epochs_data,
                epochs_transform.info,
                tmin=epochs_transform.tmin,
                verbose=False,
            )
            cleaned_wavelet_epochs = wavelet_fit["model"].transform_epochs(
                wavelet_epochs,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            cleaned_epochs_wavelet[:, :, band_idx, :] = cleaned_wavelet_epochs.get_data(
                verbose=False
            )

        cleaned_epochs_data = np.sum(cleaned_epochs_wavelet, axis=2)
        epochs_transform._data = cleaned_epochs_data
        return epochs_transform

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
        raw_transformed : mne.io.BaseRaw
            The transformed raw data.
        """
        self._check_fit()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        n_jobs = _check_n_jobs(n_jobs)

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        _check_fit_info(self, raw)
        raw_transform = _prepare_raw_transform(raw, self.ch_names)

        raw_data = raw_transform.get_data(verbose=False)
        _, n_times = raw_data.shape

        # all models are fitted with the same duration
        n_samples = [
            fit["n_samples"] for fit in self._wavelets_fits if not fit["ignore"]
        ]
        window_size = max(n_samples)
        window = create_cosine_weights(window_size)

        raw_transformed_data = np.zeros_like(raw_data)
        weight_sum = np.zeros_like(raw_data)

        step = int(window_size * (1 - overlap))
        starts = np.arange(0, n_times - window_size, step)
        starts = np.append(starts, n_times - window_size)

        all_segments = []
        for start in starts:
            segment = raw_data[:, start : start + window_size]
            all_segments.append(segment)

        all_segments_array = np.array(all_segments)
        segments_epochs = mne.EpochsArray(all_segments_array, raw.info, verbose=verbose)

        corrected_segments_epochs = self.transform_epochs(
            segments_epochs, n_jobs=n_jobs, verbose=verbose
        )
        corrected_segments = corrected_segments_epochs.get_data(verbose=verbose)

        for s, start in enumerate(starts):
            corrected_segment = corrected_segments[s] * window
            raw_transformed_data[:, start : start + window_size] += corrected_segment
            weight_sum[:, start : start + window_size] += window

        weight_sum[weight_sum == 0] = 1
        raw_transformed_data /= weight_sum

        raw_transform._data = raw_transformed_data
        return raw_transform

    def plot_fit(self):
        """Plot the fitting results.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            A list of figures showing the fitting results for each wavelet band
            that was not ignored.
        """
        self._check_fit()
        wavelet_fits = self._wavelets_fits
        figs = []
        for w, wavelet_fit in enumerate(wavelet_fits):
            if wavelet_fit["ignore"]:
                continue
            fig = wavelet_fit["model"].plot_fit()[0]
            fig.suptitle(
                f"Band {w + 1}: {wavelet_fit['fmin']:.2f}-{wavelet_fit['fmax']:.2f} Hz"
            )
            figs.append(fig)
        return figs

    @property
    def ch_names(self):
        """Get the channel names used during fitting.

        Returns
        -------
        ch_names : list of str
            The channel names that were used during model fitting.
        """
        self._check_fit()
        return self._reference_cov.ch_names
