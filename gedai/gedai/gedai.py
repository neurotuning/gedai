import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.parallel import parallel_func
from scipy.linalg import eigh

from ..sensai.sensai import (
    _eigen_to_sensai,
    _sensai_gridsearch,
    _sensai_optimize,
    _sensai_to_eigen,
)
from ..utils._checks import _check_n_jobs, check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ..wavelet.transform import epochs_to_wavelet
from .covariances import _compute_refcov


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


def compute_epoch_sizes_per_band(freq_bands, epoch_size_in_cycles, sfreq, n_times):
    """Compute frequency-specific epoch sizes, mirroring the MATLAB GEDAI logic.

    For each wavelet band the epoch duration is chosen so that it spans exactly
    ``epoch_size_in_cycles`` wave cycles of the band's *lower* boundary frequency::

        epoch_size_s = epoch_size_in_cycles / lower_frequency_of_band

    The broadband band (index 0 when ``level == 0``, or the approximation
    sub-band) is treated separately and always receives a fixed 1-second epoch.

    The returned durations are adjusted so that the number of samples is even
    (to match the MATLAB implementation), and bands whose required epoch length
    exceeds the available data are skipped (flagged as such).

    Parameters
    ----------
    freq_bands : list of tuple
        List of ``(fmin, fmax)`` pairs in Hz, one per wavelet band.  The first
        entry is typically the approximation / broadband sub-band.
    epoch_size_in_cycles : float
        Number of wave cycles per epoch.  Default in MATLAB is ``12``.
    sfreq : float
        Sampling frequency in Hz.
    n_times : int
        Total number of time samples in the recording.

    Returns
    -------
    epoch_durations : list of float
        Epoch duration in seconds for each band.  Bands that are too short for
        the required epoch are assigned ``None``.
    """
    epoch_durations = []
    for fmin, fmax in freq_bands:
        # Broadband / approximation band: use a fixed 1-second epoch
        if fmin == 0 and fmax == freq_bands[0][1]:
            target_duration = 1.0
        else:
            # epoch_size_in_cycles cycles of the lower band boundary
            lower_freq = fmin if fmin > 0 else fmax / 2.0  # safety fallback
            target_duration = epoch_size_in_cycles / lower_freq

        # Round to the nearest *even* number of samples (MATLAB parity)
        target_samples = target_duration * sfreq
        rounded = round(target_samples)
        if rounded % 2 != 0:
            # Pick the closest even neighbour
            lo, hi = rounded - 1, rounded + 1
            rounded = lo if abs(target_samples - lo) < abs(target_samples - hi) else hi
        rounded = max(rounded, 2)  # at least 2 samples

        final_duration = rounded / sfreq

        # If the epoch exceeds available data, mark as too long
        if rounded > n_times:
            epoch_durations.append(None)
        else:
            epoch_durations.append(final_duration)

    return epoch_durations


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


def _check_reference_cov(reference_cov):
    if isinstance(reference_cov, str):
        if reference_cov == "leadfield":
            return
        if os.path.exists(reference_cov):
            return
        raise ValueError(
            "Reference covariance string must be 'leadfield' or a valid path, "
            f"got '{reference_cov}' instead."
        )
    check_type(reference_cov, (mne.Forward, np.ndarray), "reference_cov")


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
    epoch_size_in_cycles : float | None
        When not ``None``, enables **frequency-specific epoching** (ported from the
        MATLAB GEDAI implementation).  Each wavelet band is fitted using epochs whose
        duration equals ``epoch_size_in_cycles`` wave-cycles of the band's lower
        cutoff frequency::

            epoch_duration_s = epoch_size_in_cycles / lower_frequency_of_band

        This produces longer epochs for low-frequency bands (better statistical
        estimation) and shorter epochs for high-frequency bands.  The MATLAB
        default is ``12`` cycles.
        When ``None`` (default), a single fixed epoch duration is used for all
        bands, matching the original Python behaviour.

        .. note::
            This parameter only has an effect when calling :meth:`fit_raw`.
            When :meth:`fit_epochs` is called directly the epochs are already
            fixed-length; use :meth:`fit_raw` for frequency-specific epoching.
    alpha_range : tuple
        The frequency range in Hz for alpha-specific thresholding.
        Default is (7, 13) Hz.
    alpha_sensai_threshold : float
        The minimum SENSAI threshold to use within the alpha range.
        Default is -6.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        wavelet_type="haar",
        wavelet_level=0,
        wavelet_low_cutoff=None,
        epoch_size_in_cycles=None,
        alpha_range=(7, 13),
        alpha_sensai_threshold=-6,
    ):
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.wavelet_low_cutoff = wavelet_low_cutoff
        self.epoch_size_in_cycles = epoch_size_in_cycles
        self.alpha_range = alpha_range
        self.alpha_sensai_threshold = alpha_sensai_threshold

    def _fit_single_band(
        self,
        wavelet_epochs_data,
        band_index,
        fmin,
        fmax,
        epochs_info,
        epochs_tmin,
        reference_cov,
        sensai_method,
        noise_multiplier,
        n_jobs,
    ):
        """Fit a single wavelet band and return its fit dict.

        This is the inner loop extracted from :meth:`fit_epochs` so that
        :meth:`fit_raw` can call it per-band with differently-sized epochs
        when frequency-specific epoching is enabled.

        Parameters
        ----------
        wavelet_epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
            Raw wavelet-band data for this band (already extracted from the
            full wavelet decomposition).
        band_index : int
            Index of the band in the wavelet decomposition output.
        fmin, fmax : float
            Frequency bounds (Hz) of this band.
        epochs_info : mne.Info
            Info object to use when constructing :class:`mne.EpochsArray`.
        epochs_tmin : float
            ``tmin`` of the epochs.
        reference_cov : np.ndarray
            Regularised reference covariance matrix (already computed).
        sensai_method : str
            ``'gridsearch'`` or ``'optimize'``.
        noise_multiplier : float
            Noise multiplier for SENSAI scoring.
        n_jobs : int
            Number of parallel jobs.

        Returns
        -------
        wavelet_fit : dict
            Dictionary with keys ``band_index``, ``fmin``, ``fmax``,
            ``threshold``, ``reference_cov``, ``epochs_eigenvalues``,
            ``sensai_runs``, and ``ignore``.
        """
        epochs_eigenvalues = np.zeros(
            (len(wavelet_epochs_data), wavelet_epochs_data.shape[1])
        )
        for e, wavelet_epoch_data in enumerate(wavelet_epochs_data):
            covariance = np.cov(wavelet_epoch_data)
            eigenvalues, _ = eigh(covariance, reference_cov, check_finite=True)
            epochs_eigenvalues[e] = eigenvalues

        wavelet_epochs = mne.EpochsArray(
            wavelet_epochs_data, epochs_info, tmin=epochs_tmin, verbose=False
        )
        min_sensai_threshold, max_sensai_threshold, step = 0, 12, 0.1

        # Alpha support logic
        target_min, target_max = self.alpha_range
        if fmin < target_max and fmax > target_min:
            is_broadband = (fmax - fmin) > (target_max - target_min) * 5
            if not is_broadband or self.wavelet_level == 0:
                min_sensai_threshold = self.alpha_sensai_threshold
                logger.info(
                    f"Alpha range overlap detected ({fmin:.1f}-{fmax:.1f} Hz):"
                    f" extending minThreshold to {min_sensai_threshold}"
                )

        n_pc = 3
        if sensai_method == "gridsearch":
            sensai_thresholds = np.arange(min_sensai_threshold, max_sensai_threshold, step)
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

        # Build the fit dictionary
        wavelet_fit = {
            "band_index": band_index,
            "fmin": fmin,
            "fmax": fmax,
            "threshold": threshold,
            "reference_cov": reference_cov,
            "epochs_eigenvalues": epochs_eigenvalues,
            "sensai_runs": runs,
        }

        # Flag bands that fall below the low-cutoff
        ignore = False
        if self.wavelet_low_cutoff is not None:
            if fmax < self.wavelet_low_cutoff:
                ignore = True
                logger.info(
                    f"Wavelet index {band_index} ({fmin:.2f}-{fmax:.2f} Hz) "
                    f"will be zeroed out during transformation "
                    f"because its upper frequency {fmax:.2f} Hz "
                    f"is below the low cutoff {self.wavelet_low_cutoff:.2f} Hz."
                )
        wavelet_fit["ignore"] = ignore
        return wavelet_fit

    @fill_doc
    def fit_epochs(
        self,
        epochs: BaseEpochs,
        reference_cov: str | mne.Forward | np.ndarray = "leadfield",
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
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        %(n_jobs)s
        %(verbose)s

        Notes
        -----
        When ``epoch_size_in_cycles`` is set on the :class:`Gedai` object,
        frequency-specific epoching is only applied via :meth:`fit_raw`.
        Calling ``fit_epochs`` directly uses the fixed epoch length of
        the supplied ``epochs`` object for all bands.
        """
        check_type(epochs, (BaseEpochs,), "epochs")
        _check_reference_cov(reference_cov)
        _check_sensai_method(sensai_method)
        check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        # Set average reference if EEG is present
        if "eeg" in epochs.get_channel_types():
            logger.info("Setting average reference to match leadfield/forward reference.")
            epochs = epochs.copy()
            epochs.load_data()
            epochs.set_eeg_reference("average", projection=False)

        # Manage built-in leadfield
        if isinstance(reference_cov, str) and reference_cov == "leadfield":
            reference_cov = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "../data/fsavLEADFIELD_4_GEDAI.mat"
                )
            )

        reference_cov, ch_names = _compute_refcov(epochs, reference_cov)

        # Tikhonov Regularization based on average diagonal power
        avg_diag_power = np.trace(reference_cov) / reference_cov.shape[0]
        regularization_lambda = 0.05
        epsilon = regularization_lambda * avg_diag_power
        reference_cov = reference_cov + epsilon * np.eye(reference_cov.shape[0])

        # Broadband data
        epochs_wavelet, freq_bands, levels = epochs_to_wavelet(
            epochs, wavelet=self.wavelet_type, level=self.wavelet_level, n_jobs=n_jobs
        )

        # Store the actual levels used for consistency in transform
        self.levels_used = levels

        wavelets_fits = []
        for w, (fmin, fmax) in enumerate(freq_bands):
            wavelet_epochs_data = epochs_wavelet[:, :, w, :]
            wavelet_fit = self._fit_single_band(
                wavelet_epochs_data=wavelet_epochs_data,
                band_index=w,
                fmin=fmin,
                fmax=fmax,
                epochs_info=epochs.info,
                epochs_tmin=epochs.tmin,
                reference_cov=reference_cov,
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
            )
            wavelets_fits.append(wavelet_fit)
        self.wavelets_fits = wavelets_fits

    @fill_doc
    def fit_raw(
        self,
        raw: BaseRaw,
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
        duration : float
            Duration of each epoch in seconds (default 1.0). Will be automatically
            adjusted to the closest valid duration for the wavelet level.
            Ignored when ``epoch_size_in_cycles`` is set on the
            :class:`Gedai` object (frequency-specific epoching mode).
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

        Notes
        -----
        When ``epoch_size_in_cycles`` is not ``None`` on the :class:`Gedai`
        object, frequency-specific epoching is used: each wavelet band is
        fitted on epochs whose duration is
        ``epoch_size_in_cycles / lower_frequency_of_band`` seconds (mirroring
        the MATLAB GEDAI implementation).  The ``duration`` argument is then
        ignored for per-band fitting but is still used as the broadband epoch
        length.
        """
        check_type(raw, (BaseRaw,), "raw")
        check_type(duration, (float, int,), "duration")
        check_type(overlap, (float, int,), "overlap")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        _check_reference_cov(reference_cov)
        _check_sensai_method(sensai_method)
        check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        sfreq = raw.info["sfreq"]

        # ------------------------------------------------------------------
        # Resolve and regularise the reference covariance once, upfront
        # ------------------------------------------------------------------
        _ref_cov = reference_cov
        if isinstance(_ref_cov, str) and _ref_cov == "leadfield":
            _ref_cov = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "../data/fsavLEADFIELD_4_GEDAI.mat"
                )
            )

        # Adjust user's duration to closest valid duration
        valid_duration, valid_samples = compute_closest_valid_duration(
            duration, self.wavelet_level, sfreq
        )
        if valid_duration != duration:
            logger.warn(
                f"Requested duration {duration:.3f}s adjusted to {valid_duration:.3f}s "
                f"({valid_samples} samples) to satisfy wavelet level"
                f" {self.wavelet_level} requirements."
            )
        duration = valid_duration

        # ------------------------------------------------------------------
        # Frequency-specific epoching path
        # ------------------------------------------------------------------
        if self.epoch_size_in_cycles is not None:
            logger.info(
                f"Frequency-specific epoching enabled "
                f"({self.epoch_size_in_cycles} cycles per band)."
            )
            self._fit_raw_frequency_specific(
                raw=raw,
                broadband_duration=duration,
                overlap=overlap,
                reject_by_annotation=reject_by_annotation,
                reference_cov=_ref_cov,
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
            )
            return

        # ------------------------------------------------------------------
        # Fixed-duration path (original behaviour)
        # ------------------------------------------------------------------
        # Convert overlap ratio to seconds for mne.make_fixed_length_epochs
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
            noise_multiplier=noise_multiplier,
            reference_cov=reference_cov,
            sensai_method=sensai_method,
            n_jobs=n_jobs,
            verbose=False,
        )

    def _fit_raw_frequency_specific(
        self,
        raw: BaseRaw,
        broadband_duration: float,
        overlap: float,
        reject_by_annotation: bool,
        reference_cov,
        sensai_method: str,
        noise_multiplier: float,
        n_jobs: int,
    ):
        """Fit with frequency-specific epoch sizes (MATLAB port).

        Each wavelet band receives epochs sized to contain exactly
        ``self.epoch_size_in_cycles`` wave-cycles of the band's lower
        cutoff frequency.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data.
        broadband_duration : float
            Epoch duration used for the broadband (approximation) band.
        overlap : float
            Overlap ratio between consecutive epochs (0–1).
        reject_by_annotation : bool
            Whether to drop annotated segments.
        reference_cov : np.ndarray or str
            Already-resolved path or array (not the 'leadfield' sentinel).
        sensai_method : str
            ``'gridsearch'`` or ``'optimize'``.
        noise_multiplier : float
            Noise multiplier for SENSAI.
        n_jobs : int
            Number of parallel jobs.
        """
        sfreq = raw.info["sfreq"]
        n_times = raw.n_times

        # Set average reference on a working copy if EEG is present
        raw_work = raw
        if "eeg" in raw.get_channel_types():
            logger.info("Setting average reference to match leadfield/forward reference.")
            raw_work = raw.copy().load_data()
            raw_work.set_eeg_reference("average", projection=False)

        # Use a short broadband epoch to resolve the reference covariance
        _bb_epochs = mne.make_fixed_length_epochs(
            raw_work,
            duration=broadband_duration,
            overlap=broadband_duration * overlap,
            reject_by_annotation=reject_by_annotation,
            preload=True,
            verbose=False,
        )

        # Resolve the reference covariance from the broadband epochs
        from .covariances import _compute_refcov
        _ref_cov_arr, ch_names = _compute_refcov(_bb_epochs, reference_cov)
        avg_diag_power = np.trace(_ref_cov_arr) / _ref_cov_arr.shape[0]
        regularization_lambda = 0.05
        epsilon = regularization_lambda * avg_diag_power
        _ref_cov_arr = _ref_cov_arr + epsilon * np.eye(_ref_cov_arr.shape[0])

        # Determine the frequency bands for all wavelet levels
        # (Use a short dummy epoch to call epochs_to_wavelet and get freq_bands)
        _dummy_epochs_wavelet, freq_bands, levels = epochs_to_wavelet(
            _bb_epochs, wavelet=self.wavelet_type, level=self.wavelet_level, n_jobs=1
        )
        self.levels_used = levels

        # Compute per-band epoch durations
        epoch_durations = compute_epoch_sizes_per_band(
            freq_bands, self.epoch_size_in_cycles, sfreq, n_times
        )

        logger.info(
            "Frequency-specific epoch durations per band:\n"
            + "\n".join(
                f"  Band {i} ({fmin:.2f}-{fmax:.2f} Hz): "
                + (f"{d:.3f} s" if d is not None else "SKIPPED (data too short)")
                for i, ((fmin, fmax), d) in enumerate(zip(freq_bands, epoch_durations))
            )
        )

        wavelets_fits = []
        for w, ((fmin, fmax), band_duration) in enumerate(
            zip(freq_bands, epoch_durations)
        ):
            # Skip band if data is too short for the required epoch
            if band_duration is None:
                logger.warning(
                    f"Band {w} ({fmin:.2f}-{fmax:.2f} Hz): data too short for "
                    f"{self.epoch_size_in_cycles} cycles. Skipping (will pass through)."
                )
                # Create a pass-through fit (threshold=0, no cleaning)
                wavelet_fit = {
                    "band_index": w,
                    "fmin": fmin,
                    "fmax": fmax,
                    "threshold": 0.0,
                    "reference_cov": _ref_cov_arr,
                    "epochs_eigenvalues": np.array([]),
                    "sensai_runs": [],
                    "ignore": True,
                }
                wavelets_fits.append(wavelet_fit)
                continue

            # Also flag bands below the low-cutoff
            if self.wavelet_low_cutoff is not None and fmax < self.wavelet_low_cutoff:
                logger.info(
                    f"Band {w} ({fmin:.2f}-{fmax:.2f} Hz): below low cutoff "
                    f"{self.wavelet_low_cutoff:.2f} Hz — zeroing."
                )
                wavelet_fit = {
                    "band_index": w,
                    "fmin": fmin,
                    "fmax": fmax,
                    "threshold": 0.0,
                    "reference_cov": _ref_cov_arr,
                    "epochs_eigenvalues": np.array([]),
                    "sensai_runs": [],
                    "ignore": True,
                }
                wavelets_fits.append(wavelet_fit)
                continue

            logger.info(
                f"Fitting band {w} ({fmin:.2f}-{fmax:.2f} Hz) "
                f"with {band_duration:.3f} s epochs "
                f"({int(round(band_duration * sfreq))} samples)."
            )

            # Epoch the raw data with the band-specific duration
            band_overlap_s = band_duration * overlap
            band_epochs = mne.make_fixed_length_epochs(
                raw_work,
                duration=band_duration,
                overlap=band_overlap_s,
                reject_by_annotation=reject_by_annotation,
                preload=True,
                verbose=False,
            )

            # Wavelet-decompose the band-specific epochs
            band_epochs_wavelet, _, _ = epochs_to_wavelet(
                band_epochs,
                wavelet=self.wavelet_type,
                level=self.wavelet_level,
                n_jobs=n_jobs,
            )

            # Extract the single wavelet sub-band that corresponds to this band
            wavelet_epochs_data = band_epochs_wavelet[:, :, w, :]

            wavelet_fit = self._fit_single_band(
                wavelet_epochs_data=wavelet_epochs_data,
                band_index=w,
                fmin=fmin,
                fmax=fmax,
                epochs_info=band_epochs.info,
                epochs_tmin=band_epochs.tmin,
                reference_cov=_ref_cov_arr,
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
            )
            wavelets_fits.append(wavelet_fit)

        self.wavelets_fits = wavelets_fits

    @fill_doc
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
        check_type(epochs, (BaseEpochs,), "epochs")
        n_jobs = _check_n_jobs(n_jobs)

        # Set average reference if EEG is present
        if "eeg" in epochs.get_channel_types():
            logger.info("Setting average reference to match leadfield/forward reference.")
            epochs = epochs.copy()
            epochs.load_data()
            epochs.set_eeg_reference("average", projection=False)

        # Check if model was fitted
        if not hasattr(self, "wavelets_fits"):
            raise RuntimeError(
                "Model has not been fitted yet. Call fit_epochs() or fit_raw() first."
            )

        epochs_wavelet, freq_bands, levels = epochs_to_wavelet(
            epochs, wavelet=self.wavelet_type, level=self.wavelet_level, n_jobs=n_jobs
        )

        # Validate that the decomposition matches the fitted model
        if levels != self.levels_used:
            raise ValueError(
                f"Wavelet decomposition levels mismatch. \n"
                f"Model was fitted with levels {self.levels_used}, "
                f"but transform got levels {levels}. \n"
                f"This may happen if epoch lengths differ between fit and transform."
            )

        cleaned_epochs_wavelet = epochs_wavelet.copy()

        for wavelet_fit in self.wavelets_fits:
            band_idx = wavelet_fit["band_index"]
            ignore = wavelet_fit["ignore"]

            if ignore:
                # Zero out this band
                cleaned_epochs_wavelet[:, :, band_idx, :] = 0
            else:
                wavelet_epochs_data = epochs_wavelet[:, :, band_idx, :]
                reference_cov = wavelet_fit["reference_cov"]
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
        segments_epochs = mne.EpochsArray(all_segments_array, raw.info, verbose=False)

        # Process all segments at once with parallelization
        corrected_segments_epochs = self.transform_epochs(
            segments_epochs, n_jobs=n_jobs, verbose=False
        )
        corrected_segments = corrected_segments_epochs.get_data(verbose=False)

        # Apply windowing and reconstruct
        for s, start in enumerate(starts):
            corrected_segment = corrected_segments[s] * window
            raw_corrected[:, start : start + window_size] += corrected_segment
            weight_sum[:, start : start + window_size] += window

        # Normalize the corrected signal by the weight sum
        mask = weight_sum > 0
        raw_corrected[mask] /= weight_sum[mask]

        # Safely inject the new data into a copy of the original object
        # This guarantees first_samp, times, and bads are perfectly preserved
        raw_corrected_obj = raw.copy()
        raw_corrected_obj._data = raw_corrected

        return raw_corrected_obj

    def plot_fit(self):
        """Plot the fitting results.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            The list of figures showing the fitting results.
        """
        wavelet_fits = self.wavelets_fits
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
