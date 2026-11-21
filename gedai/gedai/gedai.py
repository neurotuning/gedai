import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.parallel import parallel_func
from scipy.interpolate import Akima1DInterpolator
from scipy.linalg import eigh

from gedai.gedai._utils import (
    _apply_eeg_reference,
    _check_bad_channel_threshold,
    _check_fit_info,
    _check_on_bad_channel,
    _detect_signal_type,
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
from ..sensai.sensai import (
    _artifact_mask,
    _compute_default_n_pc,
    _eigen_to_sensai,
    _precompute_gevd,
    _scale_invariant_eigh,
    _sensai_gridsearch,
    _sensai_optimize,
    _sensai_score_loop,
    _sensai_to_eigen,
    compute_z3_noise_multiplier,
)
from ..utils._checks import (
    _check_n_jobs,
    _check_type,
    _ensure_noise_multiplier,
    _resolve_noise_multiplier_spec,
    ensure_engine,
    ensure_int,
)
from ..utils._docs import fill_doc
from ..utils._torch_backend import (
    clean_continuous_stream_torch,
    clean_epochs_batched_torch,
)
from ..utils.logs import logger, verbose
from ..wavelet.transform import _apply_wavelet_highpass_prefilter


def create_cosine_weights(n_samples):
    """Create cosine weights for a single epoch, mimicking the MATLAB implementation."""
    u = np.arange(1, n_samples + 1)
    cos_win = 0.5 - 0.5 * np.cos(2 * u * np.pi / n_samples)
    return cos_win


def _check_sensai_method(sensai_method):
    _check_type(sensai_method, (str,), "sensai_method")
    if sensai_method not in ["gridsearch", "optimize"]:
        raise ValueError(
            "sensai_method must be either 'gridsearch' or 'optimize', "
            f"got {sensai_method}"
        )


def _check_smoothing_window(smoothing_window: float | int | None) -> float | None:
    """Validate sliding-window adaptation duration in seconds."""
    if smoothing_window is None:
        return None
    if isinstance(smoothing_window, bool):
        raise TypeError(
            "smoothing_window must be a positive float or None, got bool."
        )
    _check_type(smoothing_window, ("numeric",), "smoothing_window")
    val = float(smoothing_window)
    if np.isnan(val) or val <= 0:
        raise ValueError(
            "smoothing_window must be > 0 (or None / np.inf), "
            f"got {smoothing_window!r}."
        )
    if np.isinf(val):
        return None
    return val


def _check_bad_epoch_threshold(
    bad_epoch_threshold: float | int | None,
) -> float | None:
    """Validate optional ENOVA threshold for bad-epoch annotation/flagging."""
    if bad_epoch_threshold is None:
        return None
    if isinstance(bad_epoch_threshold, bool):
        raise TypeError(
            "bad_epoch_threshold must be a positive float or None, got bool."
        )
    _check_type(bad_epoch_threshold, ("numeric",), "bad_epoch_threshold")
    val = float(bad_epoch_threshold)
    if not np.isfinite(val) or val <= 0:
        raise ValueError(
            f"bad_epoch_threshold must be finite and > 0, got {bad_epoch_threshold!r}."
        )
    return val


def _validate_reference_cov_arg(reference_cov) -> None:
    """Validate ``reference_cov`` type and supported string keywords."""
    _check_type(reference_cov, (str, mne.Covariance, mne.Forward), "reference_cov")
    if isinstance(reference_cov, str):
        norm = reference_cov.strip().lower()
        if norm not in ("leadfield", "projection", "projected", "dipole"):
            raise ValueError(
                "Reference covariance must be 'leadfield', 'projection', "
                "an mne.Covariance, or an mne.Forward instance; "
                f"got '{reference_cov}' instead."
            )


def _pad_and_segment_continuous(
    data: np.ndarray, sfreq: float, epoch_duration: float
) -> tuple[np.ndarray, int, int, int]:
    """Reflect-pad continuous 2D data and return non-overlapping epochs."""
    n_ch, orig_len = data.shape
    epoch_samples = max(2, int(round(epoch_duration * sfreq)))
    if epoch_samples % 2 != 0:
        epoch_samples += 1
    rem = orig_len % epoch_samples
    pad_len = (epoch_samples - rem) % epoch_samples
    if pad_len > 0:
        data_padded = np.pad(data, ((0, 0), (0, pad_len)), mode="reflect")
    else:
        data_padded = data
    total_len = data_padded.shape[1]
    n_ep = total_len // epoch_samples
    epochs_arr = (
        data_padded[:, : n_ep * epoch_samples]
        .reshape(n_ch, n_ep, epoch_samples)
        .transpose(1, 0, 2)
    )
    return (
        np.ascontiguousarray(epochs_arr, dtype=np.float64),
        epoch_samples,
        n_ep,
        orig_len,
    )


def _interp_window_curve(
    window_centers: np.ndarray, values: np.ndarray, n_ep: int
) -> np.ndarray:
    """Smooth and makima-interpolate window-level curves onto epoch indices."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    if len(vals) <= 1:
        return np.full(n_ep, float(vals[0]) if len(vals) else 0.0, dtype=np.float64)
    if len(vals) >= 3:
        padded_v = np.pad(vals, 1, mode="edge")
        vals = (padded_v[:-2] + padded_v[1:-1] + padded_v[2:]) / 3.0
    padded_c = np.concatenate(
        [[0.0], np.asarray(window_centers, dtype=np.float64), [float(n_ep - 1)]]
    )
    padded_y = np.concatenate([[vals[0]], vals, [vals[-1]]])
    u_c, u_idx = np.unique(padded_c, return_index=True)
    u_y = padded_y[u_idx]
    target_x = np.arange(n_ep, dtype=np.float64)
    if len(u_c) == 1:
        return np.full(n_ep, float(u_y[0]), dtype=np.float64)
    if len(u_c) == 2:
        return np.interp(target_x, u_c, u_y)
    return Akima1DInterpolator(u_c, u_y, method="makima")(target_x)


def _get_channel_multiplier() -> float:
    """Return the channel sample constraint multiplier (default 1.0)."""
    value = os.environ.get("GEDAI_CHANNEL_MULTIPLIER")
    if value is None:
        return 1.0
    try:
        return float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"GEDAI_CHANNEL_MULTIPLIER must be a parseable float, got {value!r}."
        ) from exc


@fill_doc
class Gedai:
    """Generalized Eigenvalue De-Artifacting Instrument.

    See :footcite:`Ros2025`.

    Parameters
    ----------
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
        engine: str = "auto",
    ):
        self.engine = ensure_engine(engine)
        self.fitted = False
        self._fit = None
        self._info = None
        self._reference_cov = None
        self._n_samples = None
        self._duration = None
        self._highpass_prefilter = None
        self._signal_type = None
        self._n_pc = None
        self._percentile = None
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
        self._dipolar_guardrail = False
        self._smoothing_window = None

    def _clone_unfitted(self):
        """Return an unfitted clone with the same initialization settings."""
        return Gedai(engine=self.engine)

    def _check_fit(self):
        """Check if the Gedai is fitted."""
        if not self.fitted:
            raise RuntimeError(
                f"Gedai must be fitted before using {self.__class__.__name__}"
            )
        assert self._fit is not None
        assert self._info is not None
        assert self._reference_cov is not None
        assert self._n_samples is not None
        assert self._duration is not None

    def _check_unfitted(self):
        """Check if the Gedai is unfitted."""
        if self.fitted:
            raise RuntimeError(
                f"Gedai must be unfitted before using {self.__class__.__name__}."
            )
        assert self._fit is None
        assert self._info is None
        assert self._reference_cov is None
        assert self._n_samples is None
        assert self._duration is None

    @fill_doc
    @verbose
    def fit_epochs(
        self,
        epochs: BaseEpochs,
        picks: list | str = "eeg",
        reference_cov: str | mne.Covariance | mne.Forward = "leadfield",
        sensai_method: str = "optimize",
        noise_multiplier: float | str = "auto",
        sensai_bounds: tuple[float, float] = (-6.0, 12.0),
        sensai_tol: float = 0.1,
        n_pc: int | str = "auto",
        n_jobs: int = None,
        verbose: str | None = None,
        engine: str | None = None,
        on_bad_channel: str = "exclude",
        bad_channel_threshold: float | str | None = None,
        reference: str | None = "average",
        dipolar_guardrail: bool = False,
    ):
        """Fit the GEDAI model to the epochs.

        Parameters
        ----------
        epochs : mne.BaseEpochs
            The epochs to fit the model to.
        %(picks)s
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        sensai_bounds : tuple of float
            The (min, max) bounds for the SENSAI search threshold. Default (-6.0, 12.0).
        %(sensai_tol)s
        %(n_pc)s
        %(n_jobs)s
        %(verbose)s
        engine : str | None
            Computation engine ('numpy', 'torch', or 'auto'). If None, uses
            engine specified at initialization.
        %(on_bad_channel)s
        %(bad_channel_threshold)s
        %(reference)s
        %(dipolar_guardrail)s
        """
        self._check_unfitted()
        current_engine = ensure_engine(engine) if engine is not None else self.engine
        _check_type(epochs, (BaseEpochs,), "epochs")
        _check_sensai_method(sensai_method)
        noise_multiplier_spec = _ensure_noise_multiplier(noise_multiplier)
        _check_type(sensai_bounds, (tuple, list), "sensai_bounds")
        _check_type(sensai_tol, (float, int), "sensai_tol")
        if sensai_tol <= 0:
            raise ValueError(f"sensai_tol must be > 0, got {sensai_tol}")
        _check_type(dipolar_guardrail, (bool,), "dipolar_guardrail")
        n_jobs = _check_n_jobs(n_jobs)
        on_bad_channel = _check_on_bad_channel(on_bad_channel)
        resolved_bad_thresh = _check_bad_channel_threshold(bad_channel_threshold)
        _validate_reference_cov_arg(reference_cov)

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
                    noise_multiplier=noise_multiplier_spec,
                    sensai_bounds=sensai_bounds,
                    sensai_tol=sensai_tol,
                    n_pc=n_pc,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    engine=current_engine,
                    reference=reference,
                    dipolar_guardrail=dipolar_guardrail,
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
        reference_cov_arr = cov.data.copy()

        signal_type = _detect_signal_type(epochs_fit.info)

        # Scale reference_cov to match data covariance trace so GEVD
        # eigenvalues and SENSAI thresholds are invariant to input units
        # (e.g., Volts vs microvolts) and reference covariance scaling.
        centered = data - data.mean(axis=-1, keepdims=True)
        denom = max(1, data.shape[-1] - 1)
        data_cov_trace = float(
            np.mean(np.sum(centered * centered, axis=(1, 2)) / denom)
        )
        ref_cov_trace = float(np.trace(reference_cov_arr))
        if ref_cov_trace > 0 and data_cov_trace > 0:
            reference_cov_arr *= data_cov_trace / ref_cov_trace

        avg_diag_power = np.trace(reference_cov_arr) / reference_cov_arr.shape[0]
        regularization_lambda = 0.05
        reference_cov_arr = (1.0 - regularization_lambda) * reference_cov_arr + (
            regularization_lambda * avg_diag_power
        ) * np.eye(reference_cov_arr.shape[0])
        reference_cov_arr = (reference_cov_arr + reference_cov_arr.T) * 0.5
        cov.update(data=reference_cov_arr)

        all_eval, all_evec = _precompute_gevd(
            data, reference_cov_arr, engine=current_engine
        )
        epochs_eigenvalues = all_eval
        percentile = 99 if signal_type == "meg" else 98
        if n_pc == "auto":
            resolved_n_pc = _compute_default_n_pc(
                reference_cov_arr, signal_type=signal_type, data=data
            )
        else:
            resolved_n_pc = ensure_int(n_pc, "n_pc")
            if not 1 <= resolved_n_pc <= reference_cov_arr.shape[0]:
                max_n_pc = reference_cov_arr.shape[0]
                raise ValueError(
                    f"n_pc must be an integer in the range [1, {max_n_pc}], "
                    f"got {n_pc!r}."
                )

        min_sensai_threshold, max_sensai_threshold = (
            float(sensai_bounds[0]),
            float(sensai_bounds[1]),
        )
        fixed_nm, z3_scale = _resolve_noise_multiplier_spec(noise_multiplier_spec)
        if fixed_nm is None:
            _, ref_evecs = eigh(reference_cov_arr)
            template = ref_evecs[:, ::-1][:, :resolved_n_pc]
            m_eff, *_ = compute_z3_noise_multiplier(
                all_eval,
                all_evec,
                reference_cov_arr,
                template,
                resolved_n_pc,
                min_threshold=min_sensai_threshold,
                epoch_samples=data.shape[-1],
            )
            resolved_noise_multiplier = float(z3_scale * m_eff)
        else:
            resolved_noise_multiplier = fixed_nm

        fit_epochs = mne.EpochsArray(
            data, epochs_fit.info, tmin=epochs.tmin, verbose=False
        )
        step = 0.1

        if sensai_method == "gridsearch":
            sensai_thresholds = np.arange(
                min_sensai_threshold, max_sensai_threshold, step
            )
            eigen_thresholds = [
                _sensai_to_eigen(
                    sensai_value, epochs_eigenvalues, percentile=percentile
                )
                for sensai_value in sensai_thresholds
            ]
            threshold, sensai_value, runs = _sensai_gridsearch(
                fit_epochs,
                reference_cov_arr,
                n_pc=resolved_n_pc,
                noise_multiplier=resolved_noise_multiplier,
                eigen_thresholds=eigen_thresholds,
                n_jobs=n_jobs,
                verbose=verbose,
                all_eval=all_eval,
                all_evec=all_evec,
                sensai_thresholds=sensai_thresholds,
                signal_type=signal_type,
                engine=current_engine,
                dipolar_guardrail=dipolar_guardrail,
            )
        elif sensai_method == "optimize":
            sensai_threshold_bounds = (min_sensai_threshold, max_sensai_threshold)
            threshold, sensai_value, runs = _sensai_optimize(
                fit_epochs,
                reference_cov_arr,
                n_pc=resolved_n_pc,
                noise_multiplier=resolved_noise_multiplier,
                epochs_eigenvalues=epochs_eigenvalues,
                bounds=sensai_threshold_bounds,
                all_eval=all_eval,
                all_evec=all_evec,
                percentile=percentile,
                signal_type=signal_type,
                engine=current_engine,
                sensai_tol=sensai_tol,
                is_wavelet_band=getattr(self, "_is_wavelet_band", False),
                dipolar_guardrail=dipolar_guardrail,
            )
        else:
            raise ValueError(
                "Method must be either 'gridsearch' or 'optimize', "
                f"got '{sensai_method}' instead."
            )

        # Compute dimensionless scale factor T1 matching MATLAB clean_EEG.m
        T1 = float((105.0 - sensai_value) / 100.0)

        best_run = min(runs, key=lambda x: abs(x[0] - threshold)) if runs else None
        self.fit_metrics_ = {
            "sensai_score": best_run[1] if best_run else 0.0,
            "signal_similarity": best_run[2] if best_run else 0.0,
            "noise_similarity": best_run[3] if best_run else 0.0,
            "threshold": threshold,
            "sensai_value": sensai_value,
            "T1": T1,
            "noise_multiplier": float(resolved_noise_multiplier),
        }

        self._fit = {
            "threshold": threshold,
            "epochs_eigenvalues": epochs_eigenvalues,
            "sensai_runs": runs,
            "sensai_bounds": (min_sensai_threshold, max_sensai_threshold),
            "sensai_value": sensai_value,
            "T1": T1,
            "percentile": percentile,
        }
        self.T1_ = T1
        self.percentile_ = percentile

        self.fitted = True
        self._info = epochs_fit.info.copy()
        self._reference_cov = cov
        self._signal_type = signal_type
        self._n_pc = resolved_n_pc
        self._percentile = percentile

        self._n_samples = data.shape[-1]
        self._duration = (self._n_samples - 1) / self._info["sfreq"]
        self._on_bad_channel = on_bad_channel
        self._bad_channel_threshold = None
        self._reference = ref_spec["mode"]
        self._ref_spec = ref_spec
        self._reference_cov_arg = (
            reference_cov
            if isinstance(reference_cov, (str, mne.Forward))
            else "leadfield"
        )
        self._dipolar_guardrail = dipolar_guardrail
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
        noise_multiplier: float | str = "auto",
        sensai_bounds: tuple[float, float] = (-6.0, 12.0),
        sensai_tol: float = 0.1,
        highpass_prefilter: float | None = 0.1,
        n_pc: int | str = "auto",
        n_jobs: int = None,
        verbose: str | None = None,
        engine: str | None = None,
        on_bad_channel: str = "exclude",
        bad_channel_threshold: float | str | None = None,
        reference: str | None = "average",
        smoothing_window: float | None = None,
        dipolar_guardrail: bool = False,
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
        sensai_bounds : tuple of float
            The (min, max) bounds for the SENSAI search threshold. Default (-6.0, 12.0).
        %(sensai_tol)s
        highpass_prefilter : float | None
            Wavelet high-pass pre-filtering cutoff frequency in Hz (default 0.1 Hz).
        %(n_pc)s
        %(n_jobs)s
        %(verbose)s
        engine : str | None
            Computation engine ('numpy', 'torch', or 'auto').
        %(on_bad_channel)s
        %(bad_channel_threshold)s
        %(reference)s
        %(smoothing_window)s
        %(dipolar_guardrail)s
        """
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(duration, (float, int), "duration")
        current_engine = ensure_engine(engine) if engine is not None else self.engine
        _check_type(overlap, (float, int), "overlap")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        _check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        _validate_reference_cov_arg(reference_cov)
        _check_sensai_method(sensai_method)
        noise_multiplier_spec = _ensure_noise_multiplier(noise_multiplier)
        _check_type(sensai_bounds, (tuple, list), "sensai_bounds")
        _check_type(sensai_tol, (float, int), "sensai_tol")
        if sensai_tol <= 0:
            raise ValueError(f"sensai_tol must be > 0, got {sensai_tol}")
        _check_type(dipolar_guardrail, (bool,), "dipolar_guardrail")
        resolved_smooth = _check_smoothing_window(smoothing_window)
        n_jobs = _check_n_jobs(n_jobs)
        on_bad_channel = _check_on_bad_channel(on_bad_channel)
        resolved_bad_thresh = _check_bad_channel_threshold(bad_channel_threshold)
        self._check_unfitted()

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
                    noise_multiplier=noise_multiplier_spec,
                    sensai_bounds=sensai_bounds,
                    sensai_tol=sensai_tol,
                    highpass_prefilter=highpass_prefilter,
                    n_pc=n_pc,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    engine=current_engine,
                    reference=reference,
                    smoothing_window=resolved_smooth,
                    dipolar_guardrail=dipolar_guardrail,
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
        cov, ref_spec = _resolve_reference_and_cov(
            reference_cov, raw_fit.info, reference=reference
        )
        _apply_eeg_reference(raw_fit, ref_spec)
        if ref_spec.get("is_external_ref", False):
            logger.info(
                "Aligned leadfield to external recording reference '%s'.",
                ref_spec.get("external_ref_label", ""),
            )

        # Enforce at least k*C samples to avoid rank deficiency and
        # ill-conditioning in high-density arrays
        k_mult = _get_channel_multiplier()
        min_samples = min(int(np.ceil(k_mult * len(raw_fit.ch_names))), raw_fit.n_times)
        if duration * raw_fit.info["sfreq"] < min_samples:
            duration = min_samples / raw_fit.info["sfreq"]

        if highpass_prefilter is not None and highpass_prefilter > 0:
            if (
                raw_fit.info["highpass"] is None
                or raw_fit.info["highpass"] < highpass_prefilter
            ):
                raw_fit._data = _apply_wavelet_highpass_prefilter(
                    raw_fit._data,
                    raw_fit.info["sfreq"],
                    lowcut_hz=highpass_prefilter,
                    engine=current_engine,
                )
        elif highpass_prefilter is not None and highpass_prefilter <= 0:
            raw_fit._data = raw_fit._data - np.mean(
                raw_fit._data, axis=1, keepdims=True
            )
        self._highpass_prefilter = highpass_prefilter

        if resolved_smooth is not None:
            self._fit_sliding_window_raw(
                raw_fit.get_data(verbose=False),
                raw_fit.info,
                cov=cov,
                ref_spec=ref_spec,
                reference_cov_arg=reference_cov,
                duration=duration,
                sensai_method=sensai_method,
                noise_multiplier_spec=noise_multiplier_spec,
                sensai_bounds=sensai_bounds,
                sensai_tol=sensai_tol,
                n_pc=n_pc,
                n_jobs=n_jobs,
                engine=current_engine,
                smoothing_window=resolved_smooth,
                dipolar_guardrail=dipolar_guardrail,
                on_bad_channel=on_bad_channel,
                excluded_bads=excluded_bads,
            )
            return

        overlap_seconds = duration * overlap
        epochs = mne.make_fixed_length_epochs(
            raw_fit,
            duration=duration,
            overlap=overlap_seconds,
            reject_by_annotation=reject_by_annotation,
            preload=True,
            verbose=False,
        )
        self.fit_epochs(
            epochs,
            picks=raw_fit.ch_names,
            reference_cov=cov,
            sensai_method=sensai_method,
            noise_multiplier=noise_multiplier_spec,
            sensai_bounds=sensai_bounds,
            sensai_tol=sensai_tol,
            n_pc=n_pc,
            n_jobs=n_jobs,
            verbose=verbose,
            engine=current_engine,
            on_bad_channel="ignore",
            bad_channel_threshold=None,
            reference="none",
            dipolar_guardrail=dipolar_guardrail,
        )
        self._reference = ref_spec["mode"]
        self._ref_spec = ref_spec
        self._reference_cov_arg = (
            reference_cov
            if isinstance(reference_cov, (str, mne.Forward))
            else "leadfield"
        )
        self._dipolar_guardrail = dipolar_guardrail
        self._smoothing_window = None
        self._on_bad_channel = on_bad_channel
        self._bad_channel_threshold = None
        self.bad_channels_ = list(excluded_bads)
        self.detected_bad_channels_ = []
        self.flat_channels_ = []
        self.flat_reference_channels_ = []
        self.noisy_channels_ = []
        self.enova_per_channel_ = None
        self.bad_epochs_ = None

    def _fit_continuous_band_array(
        self,
        band_data: np.ndarray,
        info: mne.Info,
        cov: mne.Covariance,
        duration: float,
        sensai_method: str = "optimize",
        noise_multiplier: float | str = "auto",
        sensai_bounds: tuple[float, float] = (-6.0, 12.0),
        sensai_tol: float = 0.1,
        n_pc: int | str = "auto",
        engine: str = "auto",
        n_jobs: int = 1,
        smoothing_window: float | None = None,
        dipolar_guardrail: bool = False,
    ) -> None:
        """Fit an already-referenced continuous sub-band array (used by multiband)."""
        self._check_unfitted()
        noise_multiplier_spec = _ensure_noise_multiplier(noise_multiplier)
        ref_spec = {
            "mode": "none",
            "label": "none",
            "is_rest": False,
            "is_external_ref": False,
            "external_ref_label": "",
            "R": np.eye(band_data.shape[0], dtype=np.float64),
            "rest_operator": None,
            "required_indices": [],
            "required_channels": [],
        }
        self._highpass_prefilter = None
        resolved_smooth = _check_smoothing_window(smoothing_window)
        if resolved_smooth is None:
            sfreq = float(info["sfreq"])
            epoch_samples = max(2, int(round(duration * sfreq)))
            if epoch_samples % 2 != 0:
                epoch_samples += 1
            n_ep = band_data.shape[1] // epoch_samples
            if n_ep > 0:
                ep_arr = (
                    band_data[:, : n_ep * epoch_samples]
                    .reshape(band_data.shape[0], n_ep, epoch_samples)
                    .transpose(1, 0, 2)
                )
            else:
                ep_arr = band_data[np.newaxis, :, :]
            epochs_band = mne.EpochsArray(ep_arr, info, tmin=0.0, verbose=False)
            self.fit_epochs(
                epochs_band,
                picks="all",
                reference_cov=cov.copy(),
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier_spec,
                sensai_bounds=sensai_bounds,
                sensai_tol=sensai_tol,
                n_pc=n_pc,
                n_jobs=n_jobs,
                verbose=False,
                engine=engine,
                reference="none",
                dipolar_guardrail=dipolar_guardrail,
            )
            return

        self._fit_sliding_window_raw(
            band_data,
            info,
            cov=cov.copy(),
            ref_spec=ref_spec,
            reference_cov_arg="leadfield",
            duration=duration,
            sensai_method=sensai_method,
            noise_multiplier_spec=noise_multiplier_spec,
            sensai_bounds=sensai_bounds,
            sensai_tol=sensai_tol,
            n_pc=n_pc,
            n_jobs=n_jobs,
            engine=engine,
            smoothing_window=resolved_smooth,
            dipolar_guardrail=dipolar_guardrail,
            on_bad_channel="ignore",
            excluded_bads=[],
        )

    def _fit_sliding_window_raw(
        self,
        raw_data: np.ndarray,
        info: mne.Info,
        cov: mne.Covariance,
        ref_spec: dict,
        reference_cov_arg,
        duration: float,
        sensai_method: str,
        noise_multiplier_spec: float | str,
        sensai_bounds: tuple[float, float],
        sensai_tol: float,
        n_pc: int | str,
        n_jobs: int,
        engine: str,
        smoothing_window: float,
        dipolar_guardrail: bool,
        on_bad_channel: str,
        excluded_bads: list[str],
    ) -> None:
        """Fit smooth non-stationary per-epoch threshold curves on continuous data."""
        sfreq = float(info["sfreq"])
        signal_type = _detect_signal_type(info)
        percentile = 99 if signal_type == "meg" else 98
        epochs_arr, epoch_samples, n_ep, _ = _pad_and_segment_continuous(
            raw_data, sfreq, duration
        )

        # Match MATLAB GEDAI_per_band trace matching on the continuous recording
        cov_fit = cov.copy()
        ref_arr = cov_fit.data.astype(np.float64, copy=True)
        centered = raw_data - raw_data.mean(axis=1, keepdims=True)
        denom = max(1, raw_data.shape[1] - 1)
        data_cov_trace = float(np.sum(centered * centered) / denom)
        ref_trace = float(np.trace(ref_arr))
        if ref_trace > 0.0 and data_cov_trace > 0.0:
            ref_arr *= data_cov_trace / ref_trace
        avg_diag = np.trace(ref_arr) / ref_arr.shape[0]
        lam = 0.05
        ref_arr = (1.0 - lam) * ref_arr + (lam * avg_diag) * np.eye(ref_arr.shape[0])
        ref_arr = (ref_arr + ref_arr.T) * 0.5
        cov_fit.update(data=ref_arr)

        if n_pc == "auto":
            resolved_n_pc = _compute_default_n_pc(
                ref_arr, signal_type=signal_type, data=raw_data
            )
        else:
            resolved_n_pc = ensure_int(n_pc, "n_pc")
            if not 1 <= resolved_n_pc <= ref_arr.shape[0]:
                raise ValueError(
                    f"n_pc must be an integer in the range [1, {ref_arr.shape[0]}], "
                    f"got {n_pc!r}."
                )

        _, ref_evecs = eigh(ref_arr)
        template = np.ascontiguousarray(ref_evecs[:, ::-1][:, :resolved_n_pc])
        fixed_nm, z3_scale = _resolve_noise_multiplier_spec(noise_multiplier_spec)

        min_b, max_b = float(sensai_bounds[0]), float(sensai_bounds[1])
        epoch_dur = epoch_samples / sfreq
        window_epochs = min(n_ep, max(1, int(round(smoothing_window / epoch_dur))))
        step_epochs = max(1, int(round(window_epochs / 2.0)))
        if n_ep <= window_epochs:
            num_windows = 1
        else:
            num_windows = max(1, int(np.ceil((n_ep - window_epochs) / step_epochs)) + 1)

        window_centers = np.zeros(num_windows, dtype=np.float64)
        opt_sv = np.zeros(num_windows, dtype=np.float64)
        win_log_prctile = np.zeros(num_windows, dtype=np.float64)
        tiny = float(np.finfo(np.float64).tiny)
        last_runs = []
        nm_values: list[float] = []

        all_eval_full = np.empty((n_ep, ref_arr.shape[0]), dtype=np.float64)
        all_evec_full = np.empty(
            (n_ep, ref_arr.shape[0], ref_arr.shape[0]), dtype=np.float64
        )
        computed_mask = np.zeros(n_ep, dtype=bool)

        for w in range(num_windows):
            idx_start = w * step_epochs
            idx_end = min(n_ep, idx_start + window_epochs)
            if (
                w == num_windows - 1
                and (idx_end - idx_start) < (window_epochs / 2.0)
                and num_windows > 1
            ):
                idx_start = max(0, n_ep - window_epochs)
                idx_end = n_ep
            window_centers[w] = 0.5 * (idx_start + (idx_end - 1))

            ep_sub = epochs_arr[idx_start:idx_end]
            eval_sub, evec_sub = _precompute_gevd(ep_sub, ref_arr, engine=engine)
            all_eval_full[idx_start:idx_end] = eval_sub
            all_evec_full[idx_start:idx_end] = evec_sub
            computed_mask[idx_start:idx_end] = True

            if fixed_nm is None:
                m_eff, *_ = compute_z3_noise_multiplier(
                    eval_sub,
                    evec_sub,
                    ref_arr,
                    template,
                    resolved_n_pc,
                    min_threshold=min_b,
                    epoch_samples=epoch_samples,
                )
                nm_w = float(z3_scale * m_eff)
            else:
                nm_w = fixed_nm
            nm_values.append(nm_w)

            mags_w = np.abs(eval_sub.ravel())
            max_mag_w = float(np.max(mags_w)) if mags_w.size else 0.0
            if max_mag_w > 0.0:
                valid_w = np.maximum(mags_w, max(1e-12 * max_mag_w, tiny))
                win_log_prctile[w] = float(
                    np.percentile(np.log(valid_w) + 100.0, percentile)
                )
            else:
                win_log_prctile[w] = 100.0

            if sensai_method == "gridsearch":
                sv_grid = np.arange(min_b, max_b, 0.1)
                ev_grid = [
                    _sensai_to_eigen(sv, eval_sub, percentile=percentile)
                    for sv in sv_grid
                ]
                _, sv_w, last_runs = _sensai_gridsearch(
                    ep_sub,
                    ref_arr,
                    n_pc=resolved_n_pc,
                    noise_multiplier=nm_w,
                    eigen_thresholds=ev_grid,
                    n_jobs=n_jobs,
                    verbose=False,
                    all_eval=eval_sub,
                    all_evec=evec_sub,
                    sensai_thresholds=sv_grid,
                    signal_type=signal_type,
                    engine=engine,
                    dipolar_guardrail=dipolar_guardrail,
                )
            else:
                _, sv_w, last_runs = _sensai_optimize(
                    ep_sub,
                    ref_arr,
                    n_pc=resolved_n_pc,
                    noise_multiplier=nm_w,
                    epochs_eigenvalues=eval_sub,
                    bounds=(min_b, max_b),
                    all_eval=eval_sub,
                    all_evec=evec_sub,
                    percentile=percentile,
                    signal_type=signal_type,
                    engine=engine,
                    sensai_tol=sensai_tol,
                    is_wavelet_band=getattr(self, "_is_wavelet_band", False),
                    dipolar_guardrail=dipolar_guardrail,
                )
            opt_sv[w] = sv_w

        # Fill any gaps if step > window (normally step <= window/2 so all True)
        if not np.all(computed_mask):
            miss = ~computed_mask
            ev_m, evec_m = _precompute_gevd(epochs_arr[miss], ref_arr, engine=engine)
            all_eval_full[miss] = ev_m
            all_evec_full[miss] = evec_m

        sv_curve = np.clip(
            _interp_window_curve(window_centers, opt_sv, n_ep), min_b, max_b
        )
        logp_curve = _interp_window_curve(window_centers, win_log_prctile, n_ep)
        thresh_stream1 = np.exp(((105.0 - sv_curve) / 100.0) * logp_curve - 100.0)

        if n_ep > 1:
            sv_2 = 0.5 * (sv_curve[:-1] + sv_curve[1:])
            logp_2 = 0.5 * (logp_curve[:-1] + logp_curve[1:])
            thresh_stream2 = np.exp(((105.0 - sv_2) / 100.0) * logp_2 - 100.0)
        else:
            thresh_stream2 = thresh_stream1.copy()

        # Global whole-recording SENSAI score under the per-epoch threshold curve
        all_VR = np.ascontiguousarray(np.einsum("ij,ejk->eik", ref_arr, all_evec_full))
        abs_all = np.ascontiguousarray(np.abs(all_eval_full))
        sig_ep = np.zeros(n_ep, dtype=np.float64)
        noi_ep = np.zeros(n_ep, dtype=np.float64)
        for e in range(n_ep):
            s_e, n_e = _sensai_score_loop(
                abs_all[e : e + 1],
                all_VR[e : e + 1],
                template,
                float(thresh_stream1[e]),
                n_pc=resolved_n_pc,
                signal_type=signal_type,
                dipolar_guardrail=dipolar_guardrail,
            )
            sig_ep[e] = s_e[0]
            noi_ep[e] = n_e[0]
        mean_sig = float(np.mean(sig_ep) * 100.0)
        mean_noi = float(np.mean(noi_ep) * 100.0)
        mean_nm = float(np.mean(nm_values))
        overall_sensai = float(mean_sig - mean_nm * mean_noi)

        scalar_thresh = float(np.median(thresh_stream1))
        scalar_sv = float(np.median(sv_curve))
        t1_val = float((105.0 - scalar_sv) / 100.0)

        self.fit_metrics_ = {
            "sensai_score": overall_sensai,
            "signal_similarity": mean_sig,
            "noise_similarity": mean_noi,
            "threshold": scalar_thresh,
            "sensai_value": scalar_sv,
            "T1": t1_val,
            "noise_multiplier": mean_nm,
        }
        self._fit = {
            "threshold": scalar_thresh,
            "threshold_curve": thresh_stream1,
            "threshold_curve_stream2": thresh_stream2,
            "sensai_curve": sv_curve,
            "window_centers_sec": (window_centers + 0.5) * epoch_dur,
            "smoothing_window": float(smoothing_window),
            "n_ep_fitted": int(n_ep),
            "epochs_eigenvalues": all_eval_full,
            "sensai_runs": last_runs,
            "sensai_bounds": (min_b, max_b),
            "sensai_value": scalar_sv,
            "T1": t1_val,
            "percentile": percentile,
        }
        self.T1_ = t1_val
        self.percentile_ = percentile

        self.fitted = True
        self._info = info.copy()
        self._reference_cov = cov_fit
        self._signal_type = signal_type
        self._n_pc = resolved_n_pc
        self._percentile = percentile
        self._n_samples = epoch_samples
        self._duration = (epoch_samples - 1) / sfreq
        self._reference = ref_spec["mode"]
        self._ref_spec = ref_spec
        self._reference_cov_arg = (
            reference_cov_arg
            if isinstance(reference_cov_arg, (str, mne.Forward))
            else "leadfield"
        )
        self._dipolar_guardrail = dipolar_guardrail
        self._smoothing_window = float(smoothing_window)
        self._on_bad_channel = on_bad_channel
        self._bad_channel_threshold = None
        self.bad_channels_ = list(excluded_bads)
        self.detected_bad_channels_ = []
        self.flat_channels_ = []
        self.flat_reference_channels_ = []
        self.noisy_channels_ = []
        self.enova_per_channel_ = None
        self.bad_epochs_ = None

    def _get_transform_thresholds(
        self, n_times: int, sfreq: float, epoch_duration: float
    ) -> tuple[float | np.ndarray, np.ndarray | None]:
        """Return scalar threshold or per-epoch curve pair when length matches."""
        curve1 = self._fit.get("threshold_curve")
        curve2 = self._fit.get("threshold_curve_stream2")
        if curve1 is not None:
            epoch_samples = max(2, int(round(epoch_duration * sfreq)))
            if epoch_samples % 2 != 0:
                epoch_samples += 1
            n_ep = int(np.ceil(n_times / epoch_samples))
            if n_ep == self._fit.get("n_ep_fitted") and len(curve1) == n_ep:
                return curve1, curve2
            logger.info(
                "Input duration (%d epochs) differs from fitted sliding-window "
                "duration (%s epochs); applying median scalar threshold.",
                n_ep,
                self._fit.get("n_ep_fitted"),
            )
        return self._fit["threshold"], None

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

        _check_fit_info(self, epochs)
        epochs_transform = _prepare_epochs_transform(
            epochs,
            self.ch_names,
            on_bad_channel=self._on_bad_channel,
            excluded_bads=self.bad_channels_,
            apply_reference=False,
        )
        _apply_eeg_reference(epochs_transform, self._ref_spec)

        data = epochs_transform.get_data()

        reference_cov = self._reference_cov.data
        threshold = self._fit["threshold"]
        cleaned_epochs_data = np.zeros_like(data)

        resolved_engine = ensure_engine(engine) if engine is not None else self.engine
        if resolved_engine == "torch":
            cleaned_epochs_data, _ = clean_epochs_batched_torch(
                data,
                reference_cov,
                threshold=threshold,
            )
        elif n_jobs == 1:
            for e, epoch_data in enumerate(data):
                cleaned_epochs_data[e] = _process_single_epoch(
                    epoch_data,
                    reference_cov,
                    threshold=threshold,
                )
        else:
            parallel, p_fun, _ = parallel_func(
                _process_single_epoch, n_jobs, total=len(data), verbose=verbose
            )
            cleaned_epochs_list = parallel(
                p_fun(
                    epoch_data,
                    reference_cov,
                    threshold=threshold,
                )
                for epoch_data in data
            )
            cleaned_epochs_data = np.array(cleaned_epochs_list)

        epochs_transform._data = cleaned_epochs_data

        enova_ep = None
        if compute_metrics or bad_ep_thresh is not None:
            orig_2d = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
            clean_2d = cleaned_epochs_data.transpose(1, 0, 2).reshape(
                cleaned_epochs_data.shape[1], -1
            )
            noise_2d = orig_2d - clean_2d
            ep_samples = data.shape[-1]
            enova_ep = compute_enova_per_epoch(clean_2d, noise_2d, ep_samples)
            if compute_metrics:
                enova_ch = compute_enova_per_channel(clean_2d, noise_2d, ep_samples)
                n_pc = min(3 if self._n_pc is None else self._n_pc, clean_2d.shape[0])
                sensai_val = compute_composite_sensai(
                    clean_2d,
                    noise_2d,
                    epochs_transform.info["sfreq"],
                    reference_cov,
                    n_pc=n_pc,
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
        current_engine = ensure_engine(engine) if engine is not None else self.engine

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        _check_fit_info(self, raw)
        raw_transform = _prepare_raw_transform(
            raw,
            self.ch_names,
            on_bad_channel=self._on_bad_channel,
            excluded_bads=self.bad_channels_,
            apply_reference=False,
        )
        _apply_eeg_reference(raw_transform, self._ref_spec)

        if (
            getattr(self, "_highpass_prefilter", None) is not None
            and self._highpass_prefilter > 0
        ):
            if (
                raw_transform.info["highpass"] is None
                or raw_transform.info["highpass"] < self._highpass_prefilter
            ):
                raw_transform._data = _apply_wavelet_highpass_prefilter(
                    raw_transform._data,
                    raw_transform.info["sfreq"],
                    lowcut_hz=self._highpass_prefilter,
                    engine=current_engine,
                )
        elif (
            getattr(self, "_highpass_prefilter", None) is not None
            and self._highpass_prefilter <= 0
        ):
            raw_transform._data = raw_transform._data - np.mean(
                raw_transform._data, axis=1, keepdims=True
            )

        raw_data = raw_transform.get_data(verbose=False)
        sfreq = raw_transform.info["sfreq"]
        epoch_duration = (
            self._duration
            if hasattr(self, "_duration") and self._duration > 0
            else 1.0
        )
        thresh1, thresh2 = self._get_transform_thresholds(
            raw_data.shape[1], sfreq, epoch_duration
        )

        # Fast dual-stream continuous broadband cleaning
        clean_data, noise_data = _clean_continuous_dual_stream(
            raw_data,
            sfreq=sfreq,
            reference_cov=self._reference_cov.data,
            epoch_duration=epoch_duration,
            threshold=thresh1,
            threshold_stream2=thresh2,
            engine=current_engine,
        )

        enova_ep = None
        if compute_metrics or bad_ep_thresh is not None:
            ep_samples = max(1, round(sfreq * 1.0))
            enova_ep = compute_enova_per_epoch(clean_data, noise_data, ep_samples)
            if compute_metrics:
                enova_ch = compute_enova_per_channel(clean_data, noise_data, ep_samples)
                n_pc = min(3 if self._n_pc is None else self._n_pc, clean_data.shape[0])
                sensai_val = compute_composite_sensai(
                    clean_data,
                    noise_data,
                    sfreq,
                    self._reference_cov.data,
                    n_pc=n_pc,
                )
                self.metrics_ = {
                    "enova_per_epoch": enova_ep,
                    "enova_per_channel": enova_ch,
                    "mean_enova": (
                        float(np.mean(enova_ep)) if len(enova_ep) > 0 else 0.0
                    ),
                    "sensai_score": sensai_val,
                }

        raw_transform._data = clean_data

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
    def threshold(self):
        """Get the eigenvalue threshold used for cleaning."""
        self._check_fit()
        return self._fit["threshold"]

    @property
    def ch_names(self):
        """Get the channel names used during fitting."""
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
        and marginal KDE distributions.

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
        if raw_after is None:
            raw_after = self.transform_raw(raw_before, verbose=False)

        score = self.metrics_.get("sensai_score") if self.metrics_ else None
        mean_enova = self.metrics_.get("mean_enova") if self.metrics_ else None

        return plot_sensai_visualization(
            raw_before=raw_before,
            raw_after=raw_after,
            reference_cov=self._reference_cov.data,
            epoch_duration_sec=epoch_duration_sec,
            n_pc=n_pc,
            sensai_score=score,
            mean_enova=mean_enova,
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


def _process_single_epoch(
    epoch_data,
    reference_cov,
    threshold=None,
):
    """Process a single epoch for cleaning using direct reference covariance projection.

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
    if threshold is None:
        raise ValueError("threshold must be provided for deartifacting.")

    covariance = np.cov(epoch_data)
    eigenvalues, eigenvectors = _scale_invariant_eigh(
        covariance, reference_cov, check_finite=True
    )

    eigvecs_filtered = eigenvectors.copy()
    signal_mask = ~_artifact_mask(eigenvalues, threshold)
    eigvecs_filtered[:, signal_mask] = 0

    # Direct Regularized Reference Covariance Projection:
    # Since V^T * C_ref * V = I, the inverse transpose (spatial maps) is C_ref * V.
    # Therefore, artifact projection is: C_ref * V_art * (V_art^T * X)
    artifact_tc = eigvecs_filtered.T @ epoch_data
    artefact_data = reference_cov @ (eigvecs_filtered @ artifact_tc)
    cleaned_epoch = epoch_data - artefact_data

    return cleaned_epoch


def _clean_continuous_dual_stream(
    data: np.ndarray,
    sfreq: float,
    reference_cov: np.ndarray,
    epoch_duration: float,
    threshold: float | np.ndarray | None = None,
    engine: str = "numpy",
    threshold_stream2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean continuous multi-channel data using dual-stream epoching with 50% shift.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Continuous multi-channel data.
    sfreq : float
        Sampling rate in Hz.
    reference_cov : np.ndarray, shape (n_channels, n_channels)
        Reference covariance matrix.
    epoch_duration : float
        Epoch length in seconds.
    threshold : float | np.ndarray
        Eigenvalue threshold (scalar or per-epoch array for stream 1).
    engine : str, default 'numpy'
        Computation engine ('numpy', 'torch', or 'auto').
    threshold_stream2 : np.ndarray | None
        Optional per-epoch thresholds for shifted stream 2.

    Returns
    -------
    clean : np.ndarray, shape (n_channels, n_times)
    noise : np.ndarray, shape (n_channels, n_times)
    """
    if threshold is None:
        raise ValueError("threshold must be provided for continuous cleaning.")

    resolved_engine = ensure_engine(engine)
    n_ch, orig_len = data.shape
    epoch_samples = max(2, int(round(epoch_duration * sfreq)))
    if epoch_samples % 2 != 0:
        epoch_samples += 1
    half = epoch_samples // 2

    # Pad data to multiple of epoch_samples
    rem = orig_len % epoch_samples
    pad_len = (epoch_samples - rem) % epoch_samples
    if pad_len > 0:
        data_padded = np.pad(data, ((0, 0), (0, pad_len)), mode="reflect")
    else:
        data_padded = data
    total_len = data_padded.shape[1]

    # Stream 1: non-overlapping epochs
    n_ep1 = total_len // epoch_samples
    stream1 = (
        data_padded[:, : n_ep1 * epoch_samples]
        .reshape(n_ch, n_ep1, epoch_samples)
        .transpose(1, 0, 2)
    )

    # Stream 2: shifted by half-epoch
    shifted_data = data_padded[:, half : total_len - half]
    n_ep2 = shifted_data.shape[1] // epoch_samples
    if n_ep2 > 0:
        stream2 = (
            shifted_data[:, : n_ep2 * epoch_samples]
            .reshape(n_ch, n_ep2, epoch_samples)
            .transpose(1, 0, 2)
        )
    else:
        stream2 = None

    cw = create_cosine_weights(epoch_samples)

    def _process_stream(stream, stream_thresh):
        if resolved_engine == "torch":
            return clean_continuous_stream_torch(
                stream,
                reference_cov,
                threshold=stream_thresh,
                cosine_weights=cw,
            )
        n_ep = len(stream)
        thresh_arr = np.asarray(stream_thresh, dtype=np.float64)
        is_per_epoch = thresh_arr.ndim > 0
        clean_out = np.zeros((n_ch, n_ep * epoch_samples), dtype=np.float64)
        noise_out = np.zeros((n_ch, n_ep * epoch_samples), dtype=np.float64)

        for i in range(n_ep):
            ep = stream[i].astype(np.float64)
            t_i = float(thresh_arr[i]) if is_per_epoch else float(thresh_arr)
            c = _process_single_epoch(ep, reference_cov, t_i)
            n = ep - c
            if i == 0:
                c[:, half:] *= cw[half:]
                n[:, half:] *= cw[half:]
            elif i == n_ep - 1:
                c[:, :half] *= cw[:half]
                n[:, :half] *= cw[:half]
            else:
                c *= cw
                n *= cw
            s = i * epoch_samples
            clean_out[:, s : s + epoch_samples] = c
            noise_out[:, s : s + epoch_samples] = n
        return clean_out, noise_out

    clean1, noise1 = _process_stream(stream1, threshold)
    clean_total = clean1[:, :total_len].copy()
    noise_total = noise1[:, :total_len].copy()

    if stream2 is not None and len(stream2) > 0:
        thresh_2 = threshold_stream2 if threshold_stream2 is not None else threshold
        if np.asarray(thresh_2).ndim > 0 and len(thresh_2) != len(stream2):
            thresh_2 = np.asarray(thresh_2, dtype=np.float64)[: len(stream2)]
        clean2, noise2 = _process_stream(stream2, thresh_2)
        len2 = clean2.shape[1]
        end2 = len2 - half
        clean2[:, :half] *= cw[:half]
        clean2[:, end2:] *= cw[half:]
        noise2[:, :half] *= cw[:half]
        noise2[:, end2:] *= cw[half:]

        clean_total[:, half : half + len2] += clean2
        noise_total[:, half : half + len2] += noise2

    clean_final = clean_total[:, :orig_len]
    noise_final = noise_total[:, :orig_len]
    return clean_final, noise_final


# Backward-compatible import path: gedai.gedai.gedai.MultibandGedai
