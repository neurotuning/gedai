import mne
import numpy as np
from joblib import Parallel, delayed
from mne import BaseEpochs
from mne.io import BaseRaw

from gedai.gedai._utils import (
    _apply_eeg_reference,
    _check_bad_channel_threshold,
    _check_fit_info,
    _check_on_bad_channel,
    _detect_signal_type,
    _ensure_wavelet_low_cutoff,
    _format_summary_table,
    _interpolate_bad_channels,
    _prepare_epochs_fit,
    _prepare_epochs_transform,
    _prepare_raw_fit,
    _prepare_raw_transform,
    _run_two_pass_bad_channel_detection,
)

from ..covariance.covariance import _resolve_reference_and_cov
from ..metrics.enova import (
    annotate_bad_epochs,
    compute_composite_sensai,
    compute_enova_per_channel,
    compute_enova_per_epoch,
)
from ..utils._checks import (
    _check_n_jobs,
    _check_type,
    _ensure_strength,
    ensure_engine,
)
from ..utils._docs import fill_doc
from ..utils.logs import logger, verbose
from ..wavelet.transform import (
    _apply_wavelet_highpass_prefilter,
    _modwt_haar_single_band,
    compute_wavelet_level,
    epochs_to_wavelet,
    get_modwt_band_limits,
)
from .gedai import (
    Gedai,
    _check_bad_epoch_threshold,
    _check_smoothing_window,
    _clean_continuous_dual_stream,
    _get_channel_multiplier,
    _validate_reference_cov_arg,
)


def compute_required_duration(wavelet_level, sfreq):
    """Compute minimum duration required for multiband processing.

    Parameters
    ----------
    wavelet_level : int
        Wavelet decomposition level.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    duration : float
        Minimum duration in seconds (at least 2 samples).
    """
    return 2 / sfreq


def compute_closest_valid_duration(
    target_duration, wavelet_level, sfreq, n_channels=None, n_times=None
):
    """Compute a valid duration for multiband processing.

    The Haar MODWT decomposition operates on arbitrary sample lengths without
    dyadic length constraints. Duration is rounded to an even number of samples
    for dual-stream processing and satisfies channel-count covariance requirements.

    Parameters
    ----------
    target_duration : float
        The desired duration in seconds.
    wavelet_level : int
        Wavelet decomposition level.
    sfreq : float
        Sampling frequency in Hz.
    n_channels : int | None
        Number of channels used with the configured channel sample multiplier.
    n_times : int | None
        Total number of time samples available.

    Returns
    -------
    valid_duration : float
        The valid duration in seconds.
    valid_samples : int
        The number of samples for the valid duration.
    """
    samples = max(2, int(round(target_duration * sfreq)))
    if n_channels is not None:
        k_mult = _get_channel_multiplier()
        min_samples = int(np.ceil(k_mult * n_channels))
        if n_times is not None:
            min_samples = min(min_samples, n_times)
        samples = max(samples, min_samples)
    samples += samples % 2
    return samples / sfreq, samples


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
    wavelet_level : int or 'auto'
        The wavelet decomposition level. If 'auto', automatically computed from sfreq.
    broadband_pass : bool
        Whether to run an initial broadband GED pass before multiband
        wavelet decomposition.
    %(engine)s

    Attributes
    ----------
    fit_metrics_ : dict | None
        Optimization metrics populated after fitting (for example,
        ``"sensai_score"``), or ``None`` before fitting.
    metrics_ : dict | None
        Cleaning metrics populated after transforming (including
        ``"sensai_score"``, ``"mean_enova"``, ``"enova_per_epoch"``,
        and ``"enova_per_channel"``), or ``None`` before transforming.

    Notes
    -----
    Following MNE-Python conventions, inputs are expected in standard SI units
    (Volts for EEG; Tesla or Tesla/m for MEG). Internally, the reference
    covariance is trace-matched to the data covariance so fitted thresholds and
    cleaned outputs are invariant to uniform input amplitude rescaling
    (e.g., Volts versus microvolts).

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        wavelet_type="haar",
        wavelet_level="auto",
        broadband_pass=True,
        engine="auto",
    ):
        if wavelet_level != "auto":
            _check_type(wavelet_level, (int,), "wavelet_level")
        _check_type(wavelet_type, (str,), "wavelet_type")
        _check_type(broadband_pass, (bool,), "broadband_pass")
        self.wavelet_type = wavelet_type
        self._wavelet_level = wavelet_level
        self.broadband_pass = broadband_pass
        self.engine = ensure_engine(engine)

        self.fitted = False
        self._info = None
        self._wavelet_low_cutoff = None
        self._wavelets_fits = None
        self._reference_cov = None
        self._levels = None
        self._actual_wavelet_level = None
        self._broadband_model = None
        self.fit_metrics_ = None
        self.metrics_ = None
        self._on_bad_channel = "exclude"
        self._bad_channel_threshold = None
        self.bad_channels_ = []
        self.detected_bad_channels_ = []
        self.flat_channels_ = []
        self.flat_reference_channels_ = []
        self.noisy_channels_ = []
        self.enova_per_channel_ = None
        self.bad_epochs_ = None
        self._reference = "average"
        self._ref_spec = None
        self._reference_cov_arg = "leadfield"
        self._smoothing_window = None

    def _clone_unfitted(self):
        """Return an unfitted clone with the same initialization settings."""
        return MultibandGedai(
            wavelet_type=self.wavelet_type,
            wavelet_level=self._wavelet_level,
            broadband_pass=self.broadband_pass,
            engine=self.engine,
        )

    @property
    def wavelet_level(self):
        """Wavelet level (integer level if fitted, or configured setting)."""
        if self.fitted and self._actual_wavelet_level is not None:
            return self._actual_wavelet_level
        return self._wavelet_level

    @wavelet_level.setter
    def wavelet_level(self, value):
        if value != "auto":
            _check_type(value, (int,), "wavelet_level")
        self._wavelet_level = value

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
        reference_cov: str | mne.Covariance | mne.Forward = "leadfield",
        sensai_method: str = "optimize",
        strength: float | str = "auto",
        sensai_tol: float = 0.1,
        wavelet_low_cutoff: float | str | None = 0.1,
        n_pc: int | str = "auto",
        n_jobs: int = None,
        verbose: str | None = None,
        engine: str | None = None,
        on_bad_channel: str = "exclude",
        bad_channel_threshold: float | str | None = None,
        reference: str | None = "average",
    ):
        """Fit the GEDAI model to the epochs data.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs data to fit the model to.
        %(picks)s
        %(reference_cov)s
        %(sensai_method)s
        %(strength)s
        %(sensai_tol)s
        %(wavelet_low_cutoff)s
        %(n_pc)s
        %(n_jobs)s
        %(verbose)s
        engine : str | None
            Computation engine ('numpy', 'torch', or 'auto'). If None, uses
            the engine specified at initialization.
        %(on_bad_channel)s
        %(bad_channel_threshold)s
        %(reference)s
        """
        self._check_unfitted()
        _check_type(epochs, (BaseEpochs,), "epochs")
        current_engine = ensure_engine(engine) if engine is not None else self.engine
        _validate_reference_cov_arg(reference_cov)
        _check_type(sensai_method, (str,), "sensai_method")
        resolved_strength = _ensure_strength(strength)
        _check_type(sensai_tol, (float, int), "sensai_tol")
        if sensai_tol <= 0:
            raise ValueError(f"sensai_tol must be > 0, got {sensai_tol}")
        n_jobs = _check_n_jobs(n_jobs)
        on_bad_channel = _check_on_bad_channel(on_bad_channel)
        resolved_bad_thresh = _check_bad_channel_threshold(bad_channel_threshold)

        if resolved_bad_thresh is not None:
            _run_two_pass_bad_channel_detection(
                self,
                epochs,
                fit_method_name="fit_epochs",
                transform_method_name="transform_epochs",
                picks=picks,
                on_bad_channel=on_bad_channel,
                threshold=resolved_bad_thresh,
                fit_kwargs=dict(
                    reference_cov=reference_cov,
                    sensai_method=sensai_method,
                    strength=resolved_strength,
                    sensai_tol=sensai_tol,
                    wavelet_low_cutoff=wavelet_low_cutoff,
                    n_pc=n_pc,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    engine=current_engine,
                    reference=reference,
                ),
                transform_kwargs=dict(
                    n_jobs=n_jobs,
                    engine=current_engine,
                ),
            )
            return

        epochs_fit, excluded_bads = _prepare_epochs_fit(
            epochs,
            picks,
            on_bad_channel=on_bad_channel,
            return_bads=True,
            apply_reference=False,
        )
        cov, ref_spec = _resolve_reference_and_cov(
            reference_cov, epochs_fit.info, reference=reference
        )
        _apply_eeg_reference(epochs_fit, ref_spec)
        if ref_spec.get("is_external_ref", False):
            logger.info(
                "Aligned leadfield to external recording reference '%s'.",
                ref_spec.get("external_ref_label", ""),
            )
        data = epochs_fit.get_data()

        epoch_duration = epochs_fit.tmax - epochs_fit.tmin
        wavelet_low_cutoff = _ensure_wavelet_low_cutoff(
            wavelet_low_cutoff, epochs_fit.info["highpass"], epoch_duration
        )

        # Automatic wavelet level resolution if requested
        if self.wavelet_level == "auto":
            actual_wavelet_level = compute_wavelet_level(
                epochs_fit.info["sfreq"],
                wavelet_low_cutoff=wavelet_low_cutoff,
                n_times=data.shape[-1],
            )
        else:
            actual_wavelet_level = self.wavelet_level
        self._actual_wavelet_level = actual_wavelet_level

        # Broadband pre-cleaning pass if requested
        if self.broadband_pass:
            signal_type = _detect_signal_type(epochs_fit.info)
            bb_bounds = (-4.0, 8.0) if signal_type == "meg" else (-4.0, 12.0)
            logger.info("Running broadband GEDAI pre-cleaning pass on epochs...")
            broadband_model = Gedai(engine=current_engine)
            broadband_model.fit_epochs(
                epochs_fit,
                picks="all",
                reference_cov=cov.copy(),
                sensai_method=sensai_method,
                strength=resolved_strength,
                sensai_bounds=bb_bounds,
                sensai_tol=sensai_tol,
                n_pc=n_pc,
                n_jobs=n_jobs,
                verbose=verbose,
                reference="none",
            )
            epochs_precleaned = broadband_model.transform_epochs(
                epochs_fit, n_jobs=n_jobs, verbose=False
            )
            data = epochs_precleaned.get_data()
            self._broadband_model = broadband_model

        epochs_wavelet, freq_bands, levels = epochs_to_wavelet(
            data,
            sfreq=epochs_fit.info["sfreq"],
            wavelet=self.wavelet_type,
            level=actual_wavelet_level,
            n_jobs=n_jobs,
        )
        n_samples = epochs_wavelet.shape[-1]

        wavelets_fits = []
        for w, (fmin, fmax) in enumerate(freq_bands):
            if wavelet_low_cutoff >= 0.5 and fmax <= wavelet_low_cutoff:
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
                    "sensai_bounds": (0.0, 10.0),
                    "enova": 0.0,
                }
            else:
                signal_type = _detect_signal_type(epochs_fit.info)
                center_freq = (fmin + fmax) / 2.0
                lowcut = wavelet_low_cutoff
                min_thresh = (
                    -6.0 if (min(lowcut, fmax / 2.0) <= center_freq <= 60.0) else 0.0
                )
                max_thresh = 8.0 if signal_type == "meg" else 12.0
                band_bounds = (min_thresh, max_thresh)

                wavelet_epochs_data = epochs_wavelet[:, :, w, :]
                wavelet_epochs = mne.EpochsArray(
                    wavelet_epochs_data,
                    epochs_fit.info,
                    tmin=epochs_fit.tmin,
                    verbose=False,
                )

                model = Gedai(engine=current_engine)
                model._is_wavelet_band = True
                model.fit_epochs(
                    wavelet_epochs,
                    picks="all",
                    reference_cov=cov.copy(),
                    sensai_method=sensai_method,
                    strength=resolved_strength,
                    sensai_bounds=band_bounds,
                    sensai_tol=sensai_tol,
                    n_pc=n_pc,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    reference="none",
                )

                wavelet_fit = {
                    "band_index": w,
                    "fmin": fmin,
                    "fmax": fmax,
                    "model": model,
                    "ignore": False,
                    "n_samples": n_samples,
                    "sensai_bounds": band_bounds,
                    "sensai": model.fit_metrics_["sensai_score"]
                    if model.fit_metrics_
                    else 0.0,
                    "enova": 0.0,
                }
            wavelets_fits.append(wavelet_fit)

        self.fitted = True
        self._info = epochs_fit.info.copy()
        self._reference_cov = cov

        self._levels = levels
        self._wavelets_fits = wavelets_fits
        self._wavelet_low_cutoff = wavelet_low_cutoff

        sensai_scores = [
            wf["model"].fit_metrics_["sensai_score"]
            for wf in wavelets_fits
            if not wf.get("ignore", False) and wf.get("model") is not None
        ]
        self.fit_metrics_ = {
            "sensai_score": float(np.mean(sensai_scores)) if sensai_scores else 0.0,
            "wavelets_fits": wavelets_fits,
        }
        self._on_bad_channel = on_bad_channel
        self._bad_channel_threshold = None
        self._reference = ref_spec["mode"]
        self._ref_spec = ref_spec
        self._reference_cov_arg = (
            reference_cov
            if isinstance(reference_cov, (str, mne.Forward))
            else "leadfield"
        )
        self._smoothing_window = None
        self.bad_channels_ = list(excluded_bads)
        self.detected_bad_channels_ = []
        self.flat_channels_ = []
        self.flat_reference_channels_ = []
        self.noisy_channels_ = []
        self.enova_per_channel_ = None
        self.bad_epochs_ = None

    @fill_doc
    @verbose
    def fit_raw(
        self,
        raw: BaseRaw,
        picks: list | str = "eeg",
        duration: float = 1.0,
        overlap: float = 0.5,
        reject_by_annotation: bool | None = False,
        reference_cov: str | mne.Covariance | mne.Forward = "leadfield",
        sensai_method: str = "optimize",
        strength: float | str = "auto",
        sensai_tol: float = 0.1,
        wavelet_low_cutoff: float | str | None = 0.1,
        n_pc: int | str = "auto",
        n_jobs: int = None,
        verbose: str | None = None,
        engine: str | None = None,
        on_bad_channel: str = "exclude",
        bad_channel_threshold: float | str | None = None,
        reference: str | None = "average",
        smoothing_window: float | None = None,
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
        %(strength)s
        %(sensai_tol)s
        %(wavelet_low_cutoff)s
        %(n_pc)s
        %(n_jobs)s
        %(verbose)s
        engine : str | None
            Computation engine ('numpy', 'torch', or 'auto'). If None, uses
            the engine specified at initialization.
        %(on_bad_channel)s
        %(bad_channel_threshold)s
        %(reference)s
        %(smoothing_window)s

        Notes
        -----
        The continuous raw-data path uses circular Haar MODWT.
        Calibration durations use even-sample rounding and the configured
        channel-based covariance minimum.
        """
        self._check_unfitted()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(duration, (float, int), "duration")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        current_engine = ensure_engine(engine) if engine is not None else self.engine
        _check_type(overlap, (float, int), "overlap")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        _check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        _validate_reference_cov_arg(reference_cov)
        _check_type(sensai_method, (str,), "sensai_method")
        resolved_strength = _ensure_strength(strength)
        _check_type(sensai_tol, (float, int), "sensai_tol")
        if sensai_tol <= 0:
            raise ValueError(f"sensai_tol must be > 0, got {sensai_tol}")
        resolved_smooth = _check_smoothing_window(smoothing_window)
        n_jobs = _check_n_jobs(n_jobs)
        on_bad_channel = _check_on_bad_channel(on_bad_channel)
        resolved_bad_thresh = _check_bad_channel_threshold(bad_channel_threshold)

        if resolved_bad_thresh is not None:
            _run_two_pass_bad_channel_detection(
                self,
                raw,
                fit_method_name="fit_raw",
                transform_method_name="transform_raw",
                picks=picks,
                on_bad_channel=on_bad_channel,
                threshold=resolved_bad_thresh,
                fit_kwargs=dict(
                    duration=duration,
                    overlap=overlap,
                    reject_by_annotation=reject_by_annotation,
                    reference_cov=reference_cov,
                    sensai_method=sensai_method,
                    strength=resolved_strength,
                    sensai_tol=sensai_tol,
                    wavelet_low_cutoff=wavelet_low_cutoff,
                    n_pc=n_pc,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    engine=current_engine,
                    reference=reference,
                    smoothing_window=resolved_smooth,
                ),
                transform_kwargs=dict(
                    overlap=overlap,
                    n_jobs=n_jobs,
                    engine=current_engine,
                ),
            )
            return

        raw_fit, excluded_bads = _prepare_raw_fit(
            raw,
            picks,
            on_bad_channel=on_bad_channel,
            return_bads=True,
            apply_reference=False,
        )
        sfreq = raw_fit.info["sfreq"]

        cov, ref_spec = _resolve_reference_and_cov(
            reference_cov, raw_fit.info, reference=reference
        )
        _apply_eeg_reference(raw_fit, ref_spec)
        if ref_spec.get("is_external_ref", False):
            logger.info(
                "Aligned leadfield to external recording reference '%s'.",
                ref_spec.get("external_ref_label", ""),
            )
        filter_cutoff = raw_fit.info["highpass"]
        wavelet_low_cutoff = _ensure_wavelet_low_cutoff(
            wavelet_low_cutoff, filter_cutoff, duration
        )

        # Single-stage pre-broadband wavelet high-pass pre-filter (or global
        # channel mean centering when wavelet_low_cutoff <= 0)
        if wavelet_low_cutoff > 0:
            logger.info(
                f"Applying wavelet HP pre-filter (sub-{wavelet_low_cutoff:.2f} Hz)..."
            )
            raw_fit._data = _apply_wavelet_highpass_prefilter(
                raw_fit._data,
                sfreq,
                lowcut_hz=wavelet_low_cutoff,
                engine=current_engine,
            )
        else:
            raw_fit._data = raw_fit._data - np.mean(
                raw_fit._data, axis=1, keepdims=True
            )

        if self.wavelet_level == "auto":
            actual_wavelet_level = compute_wavelet_level(
                sfreq,
                lowcut_hz=wavelet_low_cutoff if wavelet_low_cutoff > 0 else 0.5,
                n_times=raw_fit.n_times,
            )
        else:
            actual_wavelet_level = self.wavelet_level
        self._actual_wavelet_level = actual_wavelet_level

        valid_duration, valid_samples = compute_closest_valid_duration(
            duration,
            actual_wavelet_level,
            sfreq,
            n_channels=len(raw_fit.ch_names),
            n_times=raw_fit.n_times,
        )
        if valid_duration != duration:
            logger.warning(
                f"Requested duration {duration:.3f}s adjusted to {valid_duration:.3f}s "
                f"({valid_samples} samples) to satisfy sample-count and "
                "covariance requirements."
            )
        duration = valid_duration

        # Broadband pre-cleaning pass if requested
        signal_type = _detect_signal_type(raw_fit.info)
        bb_bounds = (-4.0, 8.0) if signal_type == "meg" else (-4.0, 12.0)
        if self.broadband_pass:
            logger.info("Running broadband GEDAI pass...")
            broadband_model = Gedai(engine=current_engine)
            broadband_model.fit_raw(
                raw_fit,
                picks="all",
                duration=duration,
                overlap=overlap,
                reject_by_annotation=reject_by_annotation,
                reference_cov=cov.copy(),
                sensai_method=sensai_method,
                strength=resolved_strength,
                sensai_bounds=bb_bounds,
                sensai_tol=sensai_tol,
                highpass_prefilter=None,
                n_pc=n_pc,
                n_jobs=n_jobs,
                verbose=verbose,
                reference="none",
                smoothing_window=resolved_smooth,
            )
            raw_for_multiband = broadband_model.transform_raw(
                raw_fit, overlap=overlap, n_jobs=n_jobs, verbose=False
            )
            self._broadband_model = broadband_model
        else:
            raw_for_multiband = raw_fit

        # Continuous single-band MODWT decomposition & band fitting
        band_limits = get_modwt_band_limits(sfreq, actual_wavelet_level + 1)
        raw_data_fit = raw_for_multiband.get_data(verbose=False)
        epoch_samples = valid_samples
        n_ep = raw_data_fit.shape[1] // epoch_samples

        items = list(enumerate(band_limits))
        if n_jobs == 1 or len(items) <= 1:
            wavelets_fits = [
                self._fit_wavelet_band(
                    item,
                    raw_data_fit,
                    raw_fit.info,
                    duration,
                    epoch_samples,
                    n_ep,
                    actual_wavelet_level,
                    wavelet_low_cutoff,
                    cov,
                    sensai_method,
                    resolved_strength,
                    sensai_tol=sensai_tol,
                    n_pc=n_pc,
                    engine=current_engine,
                    smoothing_window=resolved_smooth,
                )
                for item in items
            ]
        else:
            wavelets_fits = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._fit_wavelet_band)(
                    item,
                    raw_data_fit,
                    raw_fit.info,
                    duration,
                    epoch_samples,
                    n_ep,
                    actual_wavelet_level,
                    wavelet_low_cutoff,
                    cov,
                    sensai_method,
                    resolved_strength,
                    sensai_tol=sensai_tol,
                    n_pc=n_pc,
                    engine=current_engine,
                    smoothing_window=resolved_smooth,
                )
                for item in items
            )

        self.fitted = True
        self._info = raw_fit.info.copy()
        self._reference_cov = cov
        self._levels = actual_wavelet_level
        self._wavelets_fits = wavelets_fits
        self._wavelet_low_cutoff = wavelet_low_cutoff

        sensai_scores = [
            wf["model"].fit_metrics_["sensai_score"]
            for wf in wavelets_fits
            if not wf.get("ignore", False) and wf.get("model") is not None
        ]
        self.fit_metrics_ = {
            "sensai_score": float(np.mean(sensai_scores)) if sensai_scores else 0.0,
            "wavelets_fits": wavelets_fits,
        }
        self._on_bad_channel = on_bad_channel
        self._bad_channel_threshold = None
        self._reference = ref_spec["mode"]
        self._ref_spec = ref_spec
        self._reference_cov_arg = (
            reference_cov
            if isinstance(reference_cov, (str, mne.Forward))
            else "leadfield"
        )
        self._smoothing_window = resolved_smooth
        self.bad_channels_ = list(excluded_bads)
        self.detected_bad_channels_ = []
        self.flat_channels_ = []
        self.flat_reference_channels_ = []
        self.noisy_channels_ = []
        self.enova_per_channel_ = None
        self.bad_epochs_ = None

    def _fit_wavelet_band(
        self,
        item,
        raw_data_fit,
        raw_fit_info,
        duration,
        epoch_samples,
        n_ep,
        actual_wavelet_level,
        wavelet_low_cutoff,
        cov,
        sensai_method,
        strength,
        sensai_tol=0.1,
        n_pc="auto",
        engine=None,
        smoothing_window=None,
    ):
        """Fit a single wavelet band model."""
        current_engine = ensure_engine(engine) if engine is not None else self.engine
        w, (fmin, fmax) = item
        if wavelet_low_cutoff >= 0.5 and fmax <= wavelet_low_cutoff:
            return {
                "band_index": w,
                "fmin": fmin,
                "fmax": fmax,
                "model": None,
                "ignore": True,
                "duration": duration,
                "n_samples": epoch_samples,
                "sensai_bounds": (0.0, 12.0),
                "enova": 0.0,
            }

        band_data = _modwt_haar_single_band(
            raw_data_fit.T, actual_wavelet_level, w, engine=current_engine
        )
        signal_type = _detect_signal_type(raw_fit_info)
        center_freq = (fmin + fmax) / 2.0
        lowcut = wavelet_low_cutoff
        min_thresh = -6.0 if (min(lowcut, fmax / 2.0) <= center_freq <= 60.0) else 0.0
        max_thresh = 8.0 if signal_type == "meg" else 12.0
        band_bounds = (min_thresh, max_thresh)

        model = Gedai(engine=current_engine)
        model._is_wavelet_band = True
        if smoothing_window is not None:
            model._fit_continuous_band_array(
                band_data,
                raw_fit_info,
                cov=cov,
                duration=duration,
                sensai_method=sensai_method,
                strength=strength,
                sensai_bounds=band_bounds,
                sensai_tol=sensai_tol,
                n_pc=n_pc,
                engine=current_engine,
                n_jobs=1,
                smoothing_window=smoothing_window,
            )
        else:
            if n_ep > 0:
                band_epochs_data = (
                    band_data[:, : n_ep * epoch_samples]
                    .reshape(raw_fit_info["nchan"], n_ep, epoch_samples)
                    .transpose(1, 0, 2)
                )
            else:
                band_epochs_data = band_data[np.newaxis, :, :]

            wavelet_epochs = mne.EpochsArray(
                band_epochs_data,
                raw_fit_info,
                tmin=0.0,
                verbose=False,
            )
            model.fit_epochs(
                wavelet_epochs,
                picks="all",
                reference_cov=cov.copy(),
                sensai_method=sensai_method,
                strength=strength,
                sensai_bounds=band_bounds,
                sensai_tol=sensai_tol,
                n_pc=n_pc,
                n_jobs=1,
                verbose=False,
                reference="none",
            )

        sensai_score = model.fit_metrics_["sensai_score"] if model.fit_metrics_ else 0.0

        return {
            "band_index": w,
            "fmin": fmin,
            "fmax": fmax,
            "model": model,
            "duration": duration,
            "n_samples": epoch_samples,
            "ignore": False,
            "sensai_bounds": band_bounds,
            "sensai": sensai_score,
            "enova": 0.0,
        }

    @fill_doc
    @verbose
    def transform_epochs(
        self,
        epochs: BaseEpochs,
        n_jobs: int = None,
        verbose: str | None = None,
        engine: str | None = None,
        interpolate_bads: bool = False,
        compute_metrics: bool = True,
        bad_epoch_threshold: float | None = None,
    ):
        """Transform epochs data using the fitted model.

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs to transform.
        %(n_jobs)s
        %(verbose)s
        engine : str | None
            Computation engine ('numpy', 'torch', or 'auto'). If None, uses
            the engine specified at initialization.
        %(interpolate_bads)s
        %(compute_metrics)s
        %(bad_epoch_threshold)s

        Returns
        -------
        epochs_transformed : mne.Epochs
            The transformed epochs.
        """
        self._check_fit()
        _check_type(epochs, (BaseEpochs,), "epochs")
        _check_type(interpolate_bads, (bool,), "interpolate_bads")
        _check_type(compute_metrics, (bool,), "compute_metrics")
        bad_ep_thresh = _check_bad_epoch_threshold(bad_epoch_threshold)
        n_jobs = _check_n_jobs(n_jobs)

        _check_fit_info(self, epochs)
        epochs_transform = _prepare_epochs_transform(
            epochs,
            self.ch_names,
            on_bad_channel=self._on_bad_channel,
            excluded_bads=self.bad_channels_,
            apply_reference=False,
        )
        _apply_eeg_reference(epochs_transform, self._ref_spec)
        need_orig = compute_metrics or bad_ep_thresh is not None
        orig_data = epochs_transform.get_data().copy() if need_orig else None

        current_engine = ensure_engine(engine) if engine is not None else self.engine
        if self.broadband_pass and self._broadband_model is not None:
            epochs_input = self._broadband_model.transform_epochs(
                epochs_transform,
                n_jobs=n_jobs,
                verbose=False,
                engine=current_engine,
                compute_metrics=False,
            )
        else:
            epochs_input = epochs_transform

        data = epochs_input.get_data()

        actual_level = self._actual_wavelet_level or self.wavelet_level
        epochs_wavelet, _, levels = epochs_to_wavelet(
            data,
            sfreq=epochs_transform.info["sfreq"],
            wavelet=self.wavelet_type,
            level=actual_level,
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
                engine=current_engine,
                compute_metrics=False,
            )
            cleaned_epochs_wavelet[:, :, band_idx, :] = cleaned_wavelet_epochs.get_data(
                verbose=False
            )

        cleaned_epochs_data = np.sum(cleaned_epochs_wavelet, axis=2)
        epochs_transform._data = cleaned_epochs_data

        enova_ep = None
        if need_orig:
            orig_2d = orig_data.transpose(1, 0, 2).reshape(orig_data.shape[1], -1)
            clean_2d = cleaned_epochs_data.transpose(1, 0, 2).reshape(
                cleaned_epochs_data.shape[1], -1
            )
            noise_2d = orig_2d - clean_2d
            ep_samples = orig_data.shape[-1]
            enova_ep = compute_enova_per_epoch(clean_2d, noise_2d, ep_samples)
            if compute_metrics:
                enova_ch = compute_enova_per_channel(clean_2d, noise_2d, ep_samples)
                sensai_val = compute_composite_sensai(
                    clean_2d,
                    noise_2d,
                    epochs_transform.info["sfreq"],
                    self._reference_cov.data,
                    n_pc=min(3, orig_data.shape[1]),
                )
                self.metrics_ = {
                    "enova_per_epoch": enova_ep,
                    "enova_per_channel": enova_ch,
                    "mean_enova": (
                        float(np.mean(enova_ep)) if len(enova_ep) > 0 else 0.0
                    ),
                    "sensai_score": sensai_val,
                }
        if bad_ep_thresh is not None and enova_ep is not None:
            self.bad_epochs_ = np.asarray(enova_ep > bad_ep_thresh, dtype=bool)

        if interpolate_bads and self.bad_channels_:
            return _interpolate_bad_channels(
                epochs,
                epochs_transform,
                self.bad_channels_,
                reference=self._reference,
                reference_cov=self._reference_cov_arg,
            )
        return epochs_transform

    @fill_doc
    @verbose
    def transform_raw(
        self,
        raw: BaseRaw,
        overlap: float = 0.5,
        n_jobs: int = None,
        verbose: str | None = None,
        engine: str | None = None,
        interpolate_bads: bool = False,
        compute_metrics: bool = True,
        bad_epoch_threshold: float | None = None,
    ):
        """Transform raw data using the fitted model.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        %(overlap)s
        %(n_jobs)s
        %(verbose)s
        engine : str | None
            Computation engine ('numpy', 'torch', or 'auto'). If None, uses
            the engine specified at initialization.
        %(interpolate_bads)s
        %(compute_metrics)s
        %(bad_epoch_threshold)s

        Returns
        -------
        raw_transformed : mne.io.BaseRaw
            The transformed raw data.
        """
        self._check_fit()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        _check_type(interpolate_bads, (bool,), "interpolate_bads")
        _check_type(compute_metrics, (bool,), "compute_metrics")
        bad_ep_thresh = _check_bad_epoch_threshold(bad_epoch_threshold)
        n_jobs = _check_n_jobs(n_jobs)

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        current_engine = ensure_engine(engine) if engine is not None else self.engine
        _check_fit_info(self, raw)
        raw_transform = _prepare_raw_transform(
            raw,
            self.ch_names,
            on_bad_channel=self._on_bad_channel,
            excluded_bads=self.bad_channels_,
            apply_reference=False,
        )
        _apply_eeg_reference(raw_transform, self._ref_spec)
        sfreq = raw_transform.info["sfreq"]

        # Mirror the single-stage high-pass pre-filtering (or mean centering)
        # applied during fit_raw
        if self._wavelet_low_cutoff is not None and self._wavelet_low_cutoff > 0:
            raw_transform._data = _apply_wavelet_highpass_prefilter(
                raw_transform._data,
                sfreq,
                lowcut_hz=self._wavelet_low_cutoff,
                engine=current_engine,
            )
        else:
            raw_transform._data = raw_transform._data - np.mean(
                raw_transform._data, axis=1, keepdims=True
            )

        need_orig = compute_metrics or bad_ep_thresh is not None
        orig_data = raw_transform.get_data(verbose=False).copy() if need_orig else None

        # Broadband pre-cleaning if model was fitted with broadband_pass
        if self.broadband_pass and self._broadband_model is not None:
            raw_input = self._broadband_model.transform_raw(
                raw_transform,
                overlap=overlap,
                n_jobs=n_jobs,
                verbose=False,
                engine=current_engine,
                compute_metrics=False,
            )
        else:
            raw_input = raw_transform

        sfreq = raw_transform.info["sfreq"]
        raw_data = raw_input.get_data(verbose=False)
        _, n_times = raw_data.shape

        actual_level = self._actual_wavelet_level or self.wavelet_level

        if n_jobs == 1 or len(self._wavelets_fits) <= 1:
            band_results = [
                self._transform_wavelet_band(
                    wf,
                    raw_data,
                    sfreq,
                    actual_level,
                    engine=current_engine,
                    compute_metrics=compute_metrics,
                )
                for wf in self._wavelets_fits
            ]
        else:
            band_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._transform_wavelet_band)(
                    wf,
                    raw_data,
                    sfreq,
                    actual_level,
                    engine=current_engine,
                    compute_metrics=compute_metrics,
                )
                for wf in self._wavelets_fits
            )

        raw_transformed_data = np.zeros_like(raw_data)
        for _i, (clean_band, _enova_band, _sensai_band) in enumerate(band_results):
            raw_transformed_data += clean_band

        enova_ep = None
        if need_orig:
            noise_data = orig_data - raw_transformed_data
            ep_samples = max(1, round(sfreq * 1.0))
            enova_ep = compute_enova_per_epoch(
                raw_transformed_data, noise_data, ep_samples
            )
            if compute_metrics:
                enova_ch = compute_enova_per_channel(
                    raw_transformed_data, noise_data, ep_samples
                )
                sensai_val = compute_composite_sensai(
                    raw_transformed_data,
                    noise_data,
                    sfreq,
                    self._reference_cov.data,
                    n_pc=min(3, orig_data.shape[0]),
                )
                self.metrics_ = {
                    "enova_per_epoch": enova_ep,
                    "enova_per_channel": enova_ch,
                    "mean_enova": (
                        float(np.mean(enova_ep)) if len(enova_ep) > 0 else 0.0
                    ),
                    "sensai_score": sensai_val,
                }

        raw_transform._data = raw_transformed_data

        if bad_ep_thresh is not None and enova_ep is not None:
            self.bad_epochs_ = np.asarray(enova_ep > bad_ep_thresh, dtype=bool)
            ann = annotate_bad_epochs(
                raw_transform,
                enova_ep,
                epoch_duration=ep_samples / sfreq,
                threshold=bad_ep_thresh,
            )
            if len(ann) > 0:
                raw_transform.set_annotations(raw_transform.annotations + ann)

        if interpolate_bads and self.bad_channels_:
            return _interpolate_bad_channels(
                raw,
                raw_transform,
                self.bad_channels_,
                reference=self._reference,
                reference_cov=self._reference_cov_arg,
            )
        return raw_transform

    def _transform_wavelet_band(
        self,
        wavelet_fit,
        raw_data,
        sfreq,
        actual_level,
        engine=None,
        compute_metrics=True,
    ):
        """Transform one wavelet band using continuous MODWT cleaning."""
        current_engine = ensure_engine(engine) if engine is not None else self.engine
        band_idx = wavelet_fit["band_index"]
        ignore = wavelet_fit["ignore"]

        if ignore:
            return np.zeros_like(raw_data), 0.0, 0.0

        band_data = _modwt_haar_single_band(
            raw_data.T, actual_level, band_idx, engine=current_engine
        )
        epoch_duration = wavelet_fit.get("duration", 1.0)
        if epoch_duration is None:
            epoch_duration = 1.0
        thresh1, thresh2 = wavelet_fit["model"]._get_transform_thresholds(
            raw_data.shape[1], sfreq, epoch_duration
        )

        band_ref_cov = (
            wavelet_fit["model"]._reference_cov.data
            if wavelet_fit["model"] is not None
            else self._reference_cov.data
        )
        clean_band, noise_band = _clean_continuous_dual_stream(
            band_data,
            sfreq=sfreq,
            reference_cov=band_ref_cov,
            epoch_duration=epoch_duration,
            threshold=thresh1,
            threshold_stream2=thresh2,
            engine=current_engine,
        )
        if not compute_metrics:
            return clean_band, 0.0, 0.0
        ep_samples_band = max(1, round(sfreq * 1.0))
        enova_band = float(
            np.mean(compute_enova_per_epoch(clean_band, noise_band, ep_samples_band))
        )
        runs = wavelet_fit["model"]._fit.get("sensai_runs", [])
        sensai_band = max(r[1] for r in runs) if len(runs) > 0 else 0.0
        return clean_band, enova_band, sensai_band

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

    @property
    def summary(self) -> str:
        """Formatted ASCII summary table of the model fitting metrics.

        Returns
        -------
        summary_str : str
            Formatted ASCII summary table.
        """
        self._check_fit()
        table_str = _format_summary_table(self)
        return table_str

    def fit_summary(self) -> str:
        """Compatibility wrapper for the fitted-model summary.

        Returns
        -------
        summary_str : str
            Formatted ASCII summary table.
        """
        return self.summary

    def plot_sensai(
        self,
        raw_before: BaseRaw,
        raw_after: BaseRaw | None = None,
        epoch_duration_sec: float = 1.0,
        n_pc: int = 3,
        show: bool = True,
    ):
        """Plot SENSAI subspace similarity and manifold classification.

        Replicates MATLAB's SENSAI_visualization.m with side-by-side
        Before/After subspace projections, LDA decision boundary shading,
        marginal KDE distributions, ENOVA-weighted Welch PSD, and per-channel
        ENOVA topoplot.

        Parameters
        ----------
        raw_before : mne.io.BaseRaw
            Original EEG recording before denoising.
        raw_after : mne.io.BaseRaw | None
            Cleaned EEG recording after denoising. If None, automatically computed.
        epoch_duration_sec : float
            Epoch duration in seconds (default 1.0s).
        n_pc : int
            Number of principal components for SSI calculation (default 3 for EEG).
        show : bool
            Whether to call plt.show() or return the figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The SENSAI visualization figure.
        metrics : dict
            A dictionary containing the computed SENSAI and ENOVA metrics.
        """
        from ..viz.sensai_viz import plot_sensai_visualization

        self._check_fit()
        auto_transformed = raw_after is None
        if auto_transformed:
            raw_after = self.transform_raw(raw_before, verbose=False)

        ch_names = list(self.ch_names)
        raw_before_eval = raw_before.copy().pick(ch_names).reorder_channels(ch_names)
        if getattr(self, "_ref_spec", None) is not None:
            _apply_eeg_reference(raw_before_eval, self._ref_spec)
        raw_after_eval = raw_after.copy().pick(ch_names).reorder_channels(ch_names)

        score = (
            self.metrics_.get("sensai_score")
            if auto_transformed and self.metrics_
            else (self.fit_metrics_.get("sensai_score") if self.fit_metrics_ else None)
        )
        mean_enova = (
            self.metrics_.get("mean_enova")
            if auto_transformed and self.metrics_
            else None
        )
        enova_per_ch = (
            self.metrics_.get("enova_per_channel")
            if auto_transformed and self.metrics_
            else None
        )
        if enova_per_ch is not None and len(enova_per_ch) != len(ch_names):
            enova_per_ch = None

        return plot_sensai_visualization(
            raw_before=raw_before_eval,
            raw_after=raw_after_eval,
            reference_cov=self._reference_cov,
            epoch_duration_sec=epoch_duration_sec,
            n_pc=n_pc,
            sensai_score=score,
            mean_enova=mean_enova,
            enova_per_channel=enova_per_ch,
            title_suffix=f"{self.__class__.__name__}",
            show=show,
        )

    def __repr__(self) -> str:
        """Return a compact representation of the model status."""
        status = "fitted" if self.fitted else "unfitted"
        metrics_info = ""
        if getattr(self, "metrics_", None) is not None:
            sensai = self.metrics_.get("sensai_score", 0.0)
            enova = self.metrics_.get("mean_enova", 0.0) * 100
            metrics_info = f", SENSAI={sensai:.2f}%, Mean ENOVA={enova:.2f}%"
        return f"<{self.__class__.__name__} ({status}{metrics_info})>"
