import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.parallel import parallel_func
from scipy.linalg import eigh

from gedai.gedai._utils import (
    _detect_signal_type,
    _check_fit_info,
    _format_summary_table,
    _prepare_epochs_fit,
    _prepare_epochs_transform,
    _prepare_raw_fit,
    _prepare_raw_transform,
)

from ..covariance.covariance import _ensure_cov, _pick_cov
from ..metrics.enova import (
    compute_composite_sensai,
    compute_enova_per_channel,
    compute_enova_per_epoch,
)
from ..sensai.sensai import (
    _compute_default_n_pc,
    _eigen_to_sensai,
    _precompute_gevd,
    _sensai_gridsearch,
    _sensai_optimize,
    _sensai_to_eigen,
)
from ..utils._checks import (
    _check_n_jobs,
    _check_type,
    _ensure_noise_multiplier,
)
from ..utils._docs import fill_doc
from ..utils.logs import verbose
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


@fill_doc
class Gedai:
    """Generalized Eigenvalue De-Artifacting Instrument.

    See :footcite:`deCheveigne2018`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
    ):
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
        picks: list | str | None = None,
        reference_cov: str | mne.Covariance | mne.Forward = "leadfield",
        sensai_method: str = "optimize",
        noise_multiplier: float | str = "auto",
        sensai_bounds: tuple[float, float] = (-6.0, 12.0),
        n_pc: int | str = "auto",
        n_jobs: int = None,
        verbose: str | None = None,
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
        %(n_pc)s
        %(n_jobs)s
        %(verbose)s
        """
        self._check_unfitted()
        _check_type(epochs, (BaseEpochs,), "epochs")
        _check_sensai_method(sensai_method)
        noise_multiplier = _ensure_noise_multiplier(noise_multiplier)
        _check_type(sensai_bounds, (tuple, list), "sensai_bounds")
        n_jobs = _check_n_jobs(n_jobs)

        epochs_fit = _prepare_epochs_fit(epochs, picks)
        data = epochs_fit.get_data()

        cov = _ensure_cov(reference_cov).copy()
        cov = _pick_cov(cov, epochs_fit.info["ch_names"])
        reference_cov = cov.data.copy()

        # Scale reference_cov to match data scale
        centered = data - data.mean(axis=-1, keepdims=True)
        denom = max(1, data.shape[-1] - 1)
        data_cov_trace = float(np.mean(np.sum(centered * centered, axis=(1, 2)) / denom))
        ref_cov_trace = float(np.trace(reference_cov))
        if ref_cov_trace > 0 and data_cov_trace > 0:
            reference_cov *= data_cov_trace / ref_cov_trace

        avg_diag_power = np.trace(reference_cov) / reference_cov.shape[0]
        regularization_lambda = 0.05
        reference_cov = (1.0 - regularization_lambda) * reference_cov + (
            regularization_lambda * avg_diag_power
        ) * np.eye(reference_cov.shape[0])
        reference_cov = (reference_cov + reference_cov.T) * 0.5
        cov.update(data=reference_cov)

        all_eval, all_evec = _precompute_gevd(data, reference_cov)
        epochs_eigenvalues = all_eval

        signal_type = _detect_signal_type(epochs_fit.info)
        percentile = 99 if signal_type == "meg" else 98
        if n_pc == "auto":
            resolved_n_pc = _compute_default_n_pc(
                reference_cov, signal_type=signal_type, data=data
            )
        else:
            resolved_n_pc = int(n_pc)

        fit_epochs = mne.EpochsArray(
            data, epochs_fit.info, tmin=epochs.tmin, verbose=False
        )
        min_sensai_threshold, max_sensai_threshold = (
            float(sensai_bounds[0]),
            float(sensai_bounds[1]),
        )
        step = 0.1

        if sensai_method == "gridsearch":
            sensai_thresholds = np.arange(
                min_sensai_threshold, max_sensai_threshold, step
            )
            eigen_thresholds = [
                _sensai_to_eigen(sensai_value, epochs_eigenvalues, percentile=percentile)
                for sensai_value in sensai_thresholds
            ]
            threshold, runs = _sensai_gridsearch(
                fit_epochs,
                reference_cov,
                n_pc=resolved_n_pc,
                noise_multiplier=noise_multiplier,
                eigen_thresholds=eigen_thresholds,
                n_jobs=n_jobs,
                verbose=verbose,
                all_eval=all_eval,
                all_evec=all_evec,
                signal_type=signal_type,
            )
        elif sensai_method == "optimize":
            sensai_threshold_bounds = (min_sensai_threshold, max_sensai_threshold)
            threshold, runs = _sensai_optimize(
                fit_epochs,
                reference_cov,
                n_pc=resolved_n_pc,
                noise_multiplier=noise_multiplier,
                epochs_eigenvalues=epochs_eigenvalues,
                bounds=sensai_threshold_bounds,
                all_eval=all_eval,
                all_evec=all_evec,
                percentile=percentile,
                signal_type=signal_type,
            )
        else:
            raise ValueError(
                "Method must be either 'gridsearch' or 'optimize', "
                f"got '{sensai_method}' instead."
            )

        best_run = max(runs, key=lambda x: x[1]) if runs else None
        self.fit_metrics_ = {
            "sensai_score": best_run[1] if best_run else 0.0,
            "signal_similarity": best_run[2] if best_run else 0.0,
            "noise_similarity": best_run[3] if best_run else 0.0,
            "threshold": threshold,
        }

        self._fit = {
            "threshold": threshold,
            "epochs_eigenvalues": epochs_eigenvalues,
            "sensai_runs": runs,
            "sensai_bounds": (min_sensai_threshold, max_sensai_threshold),
        }

        self.fitted = True
        self._info = epochs_fit.info.copy()
        self._reference_cov = cov
        self._signal_type = signal_type
        self._n_pc = resolved_n_pc
        self._percentile = percentile

        self._n_samples = data.shape[-1]
        self._duration = (self._n_samples - 1) / self._info["sfreq"]

    @fill_doc
    @fill_doc
    @verbose
    def fit_raw(
        self,
        raw: BaseRaw,
        picks: list | str | None = None,
        duration: float = 1.0,
        overlap: float = 0.5,
        reject_by_annotation: bool | None = False,
        reference_cov: str | mne.Covariance | mne.Forward = "leadfield",
        sensai_method: str = "optimize",
        noise_multiplier: float | str = "auto",
        sensai_bounds: tuple[float, float] = (-6.0, 12.0),
        highpass_prefilter: float | None = 0.1,
        n_pc: int | str = "auto",
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
        sensai_bounds : tuple of float
            The (min, max) bounds for the SENSAI search threshold. Default (-6.0, 12.0).
        highpass_prefilter : float | None
            Wavelet high-pass pre-filtering cutoff frequency in Hz (default 0.1 Hz).
        %(n_pc)s
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
        noise_multiplier = _ensure_noise_multiplier(noise_multiplier)
        _check_type(sensai_bounds, (tuple, list), "sensai_bounds")
        n_jobs = _check_n_jobs(n_jobs)

        raw_fit = _prepare_raw_fit(raw, picks)

        if highpass_prefilter is not None and highpass_prefilter > 0:
            if (
                raw_fit.info["highpass"] is None
                or raw_fit.info["highpass"] < highpass_prefilter
            ):
                raw_fit._data = _apply_wavelet_highpass_prefilter(
                    raw_fit._data, raw_fit.info["sfreq"], lowcut_hz=highpass_prefilter
                )
        self._highpass_prefilter = highpass_prefilter

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
            picks=picks,
            noise_multiplier=noise_multiplier,
            reference_cov=reference_cov,
            sensai_method=sensai_method,
            sensai_bounds=sensai_bounds,
            n_pc=n_pc,
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
        epochs_transform = _prepare_epochs_transform(epochs, self.ch_names)

        data = epochs_transform.get_data()

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

        epochs_transform._data = cleaned_epochs_data

        orig_2d = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
        clean_2d = cleaned_epochs_data.transpose(1, 0, 2).reshape(
            cleaned_epochs_data.shape[1], -1
        )
        noise_2d = orig_2d - clean_2d
        ep_samples = data.shape[-1]
        enova_ep = compute_enova_per_epoch(clean_2d, noise_2d, ep_samples)
        enova_ch = compute_enova_per_channel(clean_2d, noise_2d, ep_samples)
        sensai_val = compute_composite_sensai(
            clean_2d, noise_2d, epochs_transform.info["sfreq"], reference_cov, n_pc=(self._n_pc or 3)
        )
        self.metrics_ = {
            "enova_per_epoch": enova_ep,
            "enova_per_channel": enova_ch,
            "mean_enova": float(np.mean(enova_ep)) if len(enova_ep) > 0 else 0.0,
            "sensai_score": sensai_val,
        }
        return epochs_transform

    @fill_doc
    @verbose
    def transform_raw(
        self,
        raw: BaseRaw,
        overlap: float = 0.5,
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
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        n_jobs = _check_n_jobs(n_jobs)

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        _check_fit_info(self, raw)
        raw_transform = _prepare_raw_transform(raw, self.ch_names)

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
                )

        raw_data = raw_transform.get_data(verbose=False)
        sfreq = raw_transform.info["sfreq"]
        threshold = self._fit["threshold"]

        # Fast dual-stream continuous broadband cleaning
        clean_data, _ = _clean_continuous_dual_stream(
            raw_data,
            sfreq=sfreq,
            reference_cov=self._reference_cov.data,
            epoch_duration=self._duration
            if hasattr(self, "_duration") and self._duration > 0
            else 1.0,
            threshold=threshold,
        )

        raw_transform.get_data(verbose=False).copy()
        raw_transform._data = clean_data

        if verbose in (True, 1, "INFO", "info", "DEBUG", "debug") or (
            isinstance(verbose, int) and not isinstance(verbose, bool) and verbose >= 1
        ):
            pass

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

    def fit_summary(self) -> str:
        """Print and return a formatted summary table of the model fitting metrics.

        Returns
        -------
        summary_str : str
            Formatted ASCII summary table.
        """
        self._check_fit()
        table_str = _format_summary_table(self)
        return table_str

    summary = fit_summary

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


def _process_single_epoch(epoch_data, reference_cov, threshold):
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
    covariance = np.cov(epoch_data)
    eigenvalues, eigenvectors = eigh(covariance, reference_cov, check_finite=True)

    eigvecs_filtered = eigenvectors.copy()
    signal_mask = np.abs(eigenvalues) < threshold
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
    threshold: float,
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
    threshold : float
        Eigenvalue threshold.

    Returns
    -------
    clean : np.ndarray, shape (n_channels, n_times)
    noise : np.ndarray, shape (n_channels, n_times)
    """
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

    def _process_stream(stream):
        n_ep = len(stream)
        clean_out = np.zeros((n_ch, n_ep * epoch_samples), dtype=np.float64)
        noise_out = np.zeros((n_ch, n_ep * epoch_samples), dtype=np.float64)
        for i in range(n_ep):
            ep = stream[i].astype(np.float64)
            c = _process_single_epoch(ep, reference_cov, threshold)
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

    clean1, noise1 = _process_stream(stream1)
    clean_total = clean1[:, :total_len].copy()
    noise_total = noise1[:, :total_len].copy()

    if stream2 is not None and len(stream2) > 0:
        clean2, noise2 = _process_stream(stream2)
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
