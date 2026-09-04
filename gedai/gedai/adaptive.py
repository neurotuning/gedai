import mne
import numpy as np
from joblib import Parallel, delayed
from mne.io import BaseRaw

from gedai.gedai._utils import (
    _check_fit_info,
    _detect_signal_type,
    _ensure_wavelet_low_cutoff,
    _format_summary_table,
    _prepare_raw_fit,
    _prepare_raw_transform,
)

from ..covariance.covariance import _ensure_cov, _pick_cov
from ..metrics.enova import compute_enova_per_epoch
from ..utils._checks import (
    _check_n_jobs,
    _check_type,
    _ensure_noise_multiplier,
)
from ..utils._docs import fill_doc
from ..utils.logs import logger, verbose
from ..wavelet.transform import (
    _apply_wavelet_highpass_prefilter,
    _modwt_haar_single_band,
    compute_wavelet_level,
)
from .gedai import Gedai, _clean_continuous_dual_stream


def _compute_wavelet_parameters(sfreq, level, cycles_per_wavelet):
    """Compute wavelet band metadata matching MODWT ordering."""
    _check_type(cycles_per_wavelet, (float, int), "cycles_per_wavelet")
    if cycles_per_wavelet <= 0:
        raise ValueError(
            f"cycles_per_wavelet must be strictly positive, got {cycles_per_wavelet}"
        )
    if level < 0:
        raise ValueError(f"wavelet_level must be >= 0, got {level}")

    # In MODWT: Band 0 is the finest detail D1 [sfreq/4, sfreq/2],
    # ...; Band level is approx [0, sfreq/2^(level+1)].
    wavelet_parameters = []
    for band_index in range(level + 1):
        if band_index == level:
            fmin = 0.0
            fmax = sfreq / (2 ** (level + 1))
            lower_freq = sfreq / (2 ** (level + 2))
        else:
            fmin = sfreq / (2 ** (band_index + 2))
            fmax = sfreq / (2 ** (band_index + 1))
            lower_freq = fmin

        target_duration = 1.0 / max(lower_freq, 0.01) * cycles_per_wavelet
        n_samples = max(2, int(round(target_duration * sfreq)))
        wavelet_parameters.append(
            {
                "band_index": band_index,
                "fmin": fmin,
                "fmax": fmax,
                "duration": target_duration,
                "n_samples": n_samples,
                "ignore": False,
            }
        )
    return wavelet_parameters


@fill_doc
class AdaptiveMultibandGedai:
    """Adaptive Multiband Generalized Eigenvalue De-Artifacting Instrument.

    A extension of :class:`~gedai.gedai.MultibandGedai` that uses
    adaptive window lengths for each wavelet band to ensure
    wavelet decomposition has enough cycles for accurate decomposition.

    See :footcite:`Ros2025`.

    .. note::
        Since different epoch lengths are used for each wavelet band,
        it is not possible to fit the model using :class:`~mne.Epochs`.
        If you want to use epochs, consider using  :class:`~gedai.gedai.MultibandGedai`
        instead, which uses fixed epoch lengths across all wavelet bands.

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
    %(cycles_per_wavelet)s
    broadband_pass : bool
        Whether to run an initial broadband GED pass before multiband
        wavelet decomposition.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        wavelet_type="haar",
        wavelet_level="auto",
        cycles_per_wavelet=10,
        broadband_pass=True,
    ):
        if wavelet_level != "auto":
            _check_type(wavelet_level, (int,), "wavelet_level")
        _check_type(wavelet_type, (str,), "wavelet_type")
        _check_type(cycles_per_wavelet, (int,), "cycles_per_wavelet")
        _check_type(broadband_pass, (bool,), "broadband_pass")

        self.wavelet_type = wavelet_type
        self._wavelet_level = wavelet_level
        self.cycles_per_wavelet = cycles_per_wavelet
        self.broadband_pass = broadband_pass

        self.fitted = False

        self._wavelets_fits = None
        self._reference_cov = None
        self._wavelet_low_cutoff = None
        self._actual_wavelet_level = None
        self._broadband_model = None
        self.metrics_ = None

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
        assert self._wavelet_low_cutoff is not None

    def _check_unfitted(self):
        """Check if the Gedai is unfitted."""
        if self.fitted:
            raise RuntimeError(
                f"Gedai must be unfitted before using {self.__class__.__name__}."
            )
        assert self._wavelets_fits is None
        assert self._reference_cov is None
        assert self._wavelet_low_cutoff is None

    @fill_doc
    @verbose
    def fit_raw(
        self,
        raw: BaseRaw,
        picks: list | str = "eeg",
        overlap: float = 0.5,
        reject_by_annotation: bool | None = False,
        reference_cov: str | mne.Covariance | mne.Forward = "leadfield",
        sensai_method: str = "optimize",
        noise_multiplier: float | str = "auto",
        wavelet_low_cutoff: str | float | None = 0.5,
        n_pc: int | str = "auto",
        n_jobs: int = None,
        verbose: str | None = None,
    ):
        """Fit the model to raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit the model to.
        %(picks)s
        %(overlap)s
        %(reject_by_annotation)s
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        %(wavelet_low_cutoff)s
        %(n_pc)s
        %(n_jobs)s
        %(verbose)s
        """
        self._check_unfitted()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        _check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        reference_cov = _ensure_cov(reference_cov)
        _check_type(sensai_method, (str,), "sensai_method")
        noise_multiplier = _ensure_noise_multiplier(noise_multiplier)
        n_jobs = _check_n_jobs(n_jobs)

        raw_fit = _prepare_raw_fit(raw, picks)
        sfreq = raw_fit.info["sfreq"]

        # Obligatory 0.1 Hz wavelet high-pass pre-filter on input data
        raw_fit._data = _apply_wavelet_highpass_prefilter(
            raw_fit._data, sfreq, lowcut_hz=0.1
        )

        cov = _pick_cov(reference_cov, raw_fit.info)
        wavelet_low_cutoff = _ensure_wavelet_low_cutoff(
            wavelet_low_cutoff, raw_fit.info["highpass"]
        )

        # Automatic wavelet level calculation adaptively matching low cutoff
        if self.wavelet_level == "auto":
            actual_wavelet_level = compute_wavelet_level(
                sfreq,
                lowcut_hz=wavelet_low_cutoff if wavelet_low_cutoff > 0 else 0.5,
                n_times=raw_fit.n_times,
                cycles_per_wavelet=self.cycles_per_wavelet,
            )
        else:
            actual_wavelet_level = self.wavelet_level
        self._actual_wavelet_level = actual_wavelet_level

        wavelet_parameters = _compute_wavelet_parameters(
            sfreq,
            actual_wavelet_level,
            cycles_per_wavelet=self.cycles_per_wavelet,
        )

        # Broadband pre-cleaning pass with wavelet HP pre-filter if requested
        signal_type = _detect_signal_type(raw_fit.info)
        bb_bounds = (-4.0, 8.0) if signal_type == "meg" else (-4.0, 12.0)
        if self.broadband_pass:
            logger.info(
                "Applying wavelet HP pre-filter "
                f"(sub-{wavelet_low_cutoff:.2f} Hz) and running "
                "broadband GEDAI pass..."
            )
            raw_fit._data = _apply_wavelet_highpass_prefilter(
                raw_fit._data, sfreq, lowcut_hz=wavelet_low_cutoff
            )
            broadband_model = Gedai()
            broadband_model.fit_raw(
                raw_fit,
                picks="all",
                duration=1.0,
                overlap=overlap,
                reject_by_annotation=reject_by_annotation,
                reference_cov=cov.copy(),
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                sensai_bounds=bb_bounds,
                n_pc=n_pc,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            raw_for_multiband = broadband_model.transform_raw(
                raw_fit, overlap=overlap, n_jobs=n_jobs, verbose=False
            )
            self._broadband_model = broadband_model
        else:
            raw_for_multiband = raw_fit

        raw_data_fit = raw_for_multiband.get_data(verbose=False)

        if n_jobs == 1 or len(wavelet_parameters) <= 1:
            wavelets_fits = [
                self._fit_wavelet_band(
                    p,
                    raw_data_fit,
                    raw_fit.info,
                    raw_for_multiband.n_times,
                    sfreq,
                    actual_wavelet_level,
                    wavelet_low_cutoff,
                    cov,
                    sensai_method,
                    noise_multiplier,
                    n_pc=n_pc,
                )
                for p in wavelet_parameters
            ]
        else:
            wavelets_fits = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._fit_wavelet_band)(
                    p,
                    raw_data_fit,
                    raw_fit.info,
                    raw_for_multiband.n_times,
                    sfreq,
                    actual_wavelet_level,
                    wavelet_low_cutoff,
                    cov,
                    sensai_method,
                    noise_multiplier,
                    n_pc=n_pc,
                )
                for p in wavelet_parameters
            )

        self.fitted = True
        self._info = raw_fit.info.copy()
        self._reference_cov = cov
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

    def _fit_wavelet_band(
        self,
        wavelet_parameter,
        raw_data_fit,
        raw_fit_info,
        n_times,
        sfreq,
        actual_wavelet_level,
        wavelet_low_cutoff,
        cov,
        sensai_method,
        noise_multiplier,
        n_pc="auto",
    ):
        """Fit a single adaptive wavelet band model."""
        w = wavelet_parameter["band_index"]
        fmin = wavelet_parameter["fmin"]
        fmax = wavelet_parameter["fmax"]
        target_duration = wavelet_parameter["duration"]

        if fmax <= wavelet_low_cutoff:
            return {
                "band_index": w,
                "fmin": fmin,
                "fmax": fmax,
                "model": None,
                "duration": target_duration,
                "n_samples": 0,
                "ignore": True,
                "sensai_bounds": (0.0, 12.0),
                "enova": 0.0,
            }

        max_duration = n_times / sfreq / 3.0
        duration = min(target_duration, max(max_duration, 0.5))
        epoch_samples = max(2, int(round(duration * sfreq)))
        if epoch_samples % 2 != 0:
            epoch_samples += 1

        band_data = _modwt_haar_single_band(raw_data_fit.T, actual_wavelet_level, w)
        n_ep = band_data.shape[1] // epoch_samples
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

        signal_type = _detect_signal_type(raw_fit_info)
        center_freq = (fmin + fmax) / 2.0
        lowcut = (
            wavelet_low_cutoff
            if (wavelet_low_cutoff is not None and wavelet_low_cutoff > 0)
            else 0.5
        )
        min_thresh = -6.0 if (lowcut <= center_freq <= 60.0) else 0.0
        max_thresh = 8.0 if signal_type == "meg" else 12.0
        band_bounds = (min_thresh, max_thresh)

        model = Gedai()
        model.fit_epochs(
            wavelet_epochs,
            picks="all",
            reference_cov=cov.copy(),
            sensai_method=sensai_method,
            noise_multiplier=noise_multiplier,
            sensai_bounds=band_bounds,
            n_pc=n_pc,
            n_jobs=1,
            verbose=False,
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
    def transform_raw(
        self,
        raw: BaseRaw,
        overlap: float = 0.5,
        n_jobs: int = None,
        verbose: str | None = None,
    ) -> BaseRaw:
        """Apply the Adaptive Multiband GEDAI transform to raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to transform.
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
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        n_jobs = _check_n_jobs(n_jobs)

        _check_fit_info(self, raw)
        raw_transform = _prepare_raw_transform(raw, self.ch_names)
        sfreq = raw_transform.info["sfreq"]

        # Mirror the 0.1 Hz high-pass pre-filtering applied during fit_raw
        raw_transform._data = _apply_wavelet_highpass_prefilter(
            raw_transform._data, sfreq, lowcut_hz=0.1
        )

        if self.broadband_pass and self._broadband_model is not None:
            raw_transform._data = _apply_wavelet_highpass_prefilter(
                raw_transform._data,
                sfreq,
                lowcut_hz=self._wavelet_low_cutoff,
            )
            raw_input = self._broadband_model.transform_raw(
                raw_transform, overlap=overlap, n_jobs=n_jobs, verbose=False
            )
        else:
            raw_input = raw_transform

        sfreq = raw_transform.info["sfreq"]
        raw_data = raw_input.get_data(verbose=False)
        _, n_times = raw_data.shape

        actual_level = self._actual_wavelet_level or self.wavelet_level

        if n_jobs == 1 or len(self._wavelets_fits) <= 1:
            band_results = [
                self._transform_wavelet_band(wf, raw_data, sfreq, actual_level)
                for wf in self._wavelets_fits
            ]
        else:
            band_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._transform_wavelet_band)(wf, raw_data, sfreq, actual_level)
                for wf in self._wavelets_fits
            )

        raw_transformed_data = np.zeros_like(raw_data)
        for _i, (clean_band, _enova_band, _sensai_band) in enumerate(band_results):
            raw_transformed_data += clean_band

        raw_transform._data = raw_transformed_data

        if verbose in (True, 1, "INFO", "info", "DEBUG", "debug") or (
            isinstance(verbose, int) and not isinstance(verbose, bool) and verbose >= 1
        ):
            pass

        return raw_transform

    def _transform_wavelet_band(self, wavelet_fit, raw_data, sfreq, actual_level):
        """Transform a single adaptive wavelet band."""
        band_idx = wavelet_fit["band_index"]
        ignore = wavelet_fit["ignore"]

        if ignore:
            return np.zeros_like(raw_data), 0.0, 0.0

        band_data = _modwt_haar_single_band(raw_data.T, actual_level, band_idx)
        threshold = wavelet_fit["model"].threshold
        epoch_duration = wavelet_fit["duration"]

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
            threshold=threshold,
        )
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
            A list of figures showing the fitting results for each wavelet band.
        """
        self._check_fit()
        wavelet_fits = self._wavelets_fits
        figs = []
        for w, wavelet_fit in enumerate(wavelet_fits):
            if not wavelet_fit["ignore"]:
                fig = wavelet_fit["model"].plot_fit()[0]
                fig.suptitle(
                    f"Band {w + 1}: {wavelet_fit['fmin']:.2f}-"
                    f"{wavelet_fit['fmax']:.2f} Hz"
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
