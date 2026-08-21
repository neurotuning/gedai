import mne
import numpy as np
from joblib import Parallel, delayed
from mne.io import BaseRaw

from gedai.gedai._utils import (
    _check_fit_info,
    _format_summary_table,
    _prepare_raw_fit,
    _prepare_raw_transform,
)

from ..covariance.covariance import _ensure_cov, _pick_cov
from ..sensai.sensai import (
    compute_composite_sensai,
    compute_enova_per_channel,
    compute_enova_per_epoch,
)
from ..utils._checks import (
    _check_n_jobs,
    _check_type,
    _parse_noise_multiplier,
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
from .gedai import Gedai, _clean_continuous_dual_stream, create_cosine_weights
from .multiband import compute_closest_valid_duration


def _compute_wavelet_parameters(sfreq, level, cycles_per_wavelet):
    """Compute wavelet band metadata matching MODWT band ordering (0 = highest detail)."""
    _check_type(cycles_per_wavelet, (float, int), "cycles_per_wavelet")
    if cycles_per_wavelet <= 0:
        raise ValueError(
            f"cycles_per_wavelet must be strictly positive, got {cycles_per_wavelet}"
        )
    if level < 0:
        raise ValueError(f"wavelet_level must be >= 0, got {level}")

    # In MODWT: Band 0 is finest detail D1 [sfreq/4, sfreq/2], ..., Band level is approx [0, sfreq/2^(level+1)]
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
        Whether to run an initial broadband GED pass before multiband wavelet decomposition.

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
        reference_cov: str = "leadfield",
        sensai_method: str = "gridsearch",
        noise_multiplier: float | str = "auto",
        wavelet_low_cutoff: str | float | None = "auto",
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
        noise_multiplier = _parse_noise_multiplier(noise_multiplier)
        n_jobs = _check_n_jobs(n_jobs)

        raw_fit = _prepare_raw_fit(raw, picks)

        cov = _pick_cov(reference_cov, raw_fit.info["ch_names"])
        sfreq = raw_fit.info["sfreq"]

        filter_cutoff = raw_fit.info["highpass"]
        if wavelet_low_cutoff == "auto":
            if filter_cutoff is not None and filter_cutoff > 0:
                wavelet_low_cutoff = float(filter_cutoff)
            else:
                wavelet_low_cutoff = 0.5
        elif wavelet_low_cutoff is None:
            wavelet_low_cutoff = 0.0
        else:
            wavelet_low_cutoff = float(wavelet_low_cutoff)

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
        if self.broadband_pass:
            logger.info(
                f"Applying wavelet HP pre-filter (sub-{wavelet_low_cutoff:.2f} Hz) and running broadband GEDAI pass..."
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
                sensai_bounds=(-4.0, 12.0),
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

        def _fit_single_wavelet_band(wavelet_parameter):
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

            max_duration = raw_for_multiband.n_times / sfreq / 3.0
            duration = min(target_duration, max(max_duration, 0.5))
            epoch_samples = max(2, int(round(duration * sfreq)))
            if epoch_samples % 2 != 0:
                epoch_samples += 1

            band_data = _modwt_haar_single_band(raw_data_fit.T, actual_wavelet_level, w)
            n_ep = band_data.shape[1] // epoch_samples
            if n_ep > 0:
                band_epochs_data = band_data[:, : n_ep * epoch_samples].reshape(
                    raw_fit.info["nchan"], n_ep, epoch_samples
                ).transpose(1, 0, 2)
            else:
                band_epochs_data = band_data[np.newaxis, :, :]

            wavelet_epochs = mne.EpochsArray(
                band_epochs_data,
                raw_fit.info,
                tmin=0.0,
                verbose=False,
            )

            center_freq = (fmin + fmax) / 2.0
            band_bounds = (-6.0, 12.0) if (0.8 <= center_freq <= 60.0) else (0.0, 12.0)

            model = Gedai()
            model.fit_epochs(
                wavelet_epochs,
                picks="all",
                reference_cov=cov.copy(),
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                sensai_bounds=band_bounds,
                n_jobs=1,
                verbose=False,
            )

            return {
                "band_index": w,
                "fmin": fmin,
                "fmax": fmax,
                "model": model,
                "duration": duration,
                "n_samples": epoch_samples,
                "ignore": False,
                "sensai_bounds": band_bounds,
                "enova": model.metrics_["mean_enova"] if model.metrics_ else 0.0,
            }

        if n_jobs == 1 or len(wavelet_parameters) <= 1:
            wavelets_fits = [_fit_single_wavelet_band(p) for p in wavelet_parameters]
        else:
            wavelets_fits = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fit_single_wavelet_band)(p) for p in wavelet_parameters
            )

        self.fitted = True
        self._info = raw_fit.info.copy()
        self._reference_cov = cov  # No regularization applied

        self._wavelets_fits = wavelets_fits
        self._wavelet_low_cutoff = wavelet_low_cutoff

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
        self._check_fit()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        n_jobs = _check_n_jobs(n_jobs)

        _check_fit_info(self, raw)
        raw_transform = _prepare_raw_transform(raw, self.ch_names)

        if self.broadband_pass and self._broadband_model is not None:
            raw_transform._data = _apply_wavelet_highpass_prefilter(
                raw_transform._data,
                raw_transform.info["sfreq"],
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
        raw_transformed_data = np.zeros_like(raw_data)

        # Process each wavelet band using fast continuous MODWT + dual-stream adaptive cleaning
        def _transform_single_wavelet_band(wavelet_fit):
            band_idx = wavelet_fit["band_index"]
            fmin = wavelet_fit["fmin"]
            fmax = wavelet_fit["fmax"]
            ignore = wavelet_fit["ignore"]

            if ignore:
                return np.zeros_like(raw_data), 0.0, 0.0

            band_data = _modwt_haar_single_band(raw_data.T, actual_level, band_idx)
            threshold = wavelet_fit["model"].threshold
            epoch_duration = wavelet_fit["duration"]

            clean_band, noise_band = _clean_continuous_dual_stream(
                band_data,
                sfreq=sfreq,
                reference_cov=self._reference_cov.data,
                epoch_duration=epoch_duration,
                threshold=threshold,
            )
            ep_samples_band = max(1, round(sfreq * 1.0))
            enova_band = float(np.mean(compute_enova_per_epoch(clean_band, noise_band, ep_samples_band)))
            runs = wavelet_fit["model"]._fit.get("sensai_runs", [])
            sensai_band = max(r[1] for r in runs) if len(runs) > 0 else 0.0
            return clean_band, enova_band, sensai_band

        if n_jobs == 1 or len(self._wavelets_fits) <= 1:
            band_results = [_transform_single_wavelet_band(wf) for wf in self._wavelets_fits]
        else:
            band_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_transform_single_wavelet_band)(wf) for wf in self._wavelets_fits
            )

        raw_transformed_data = np.zeros_like(raw_data)
        for i, (clean_band, enova_band, sensai_band) in enumerate(band_results):
            raw_transformed_data += clean_band
            self._wavelets_fits[i]["enova"] = enova_band
            if sensai_band > 0:
                self._wavelets_fits[i]["sensai"] = sensai_band

        original_data = raw_transform.get_data(verbose=False).copy()
        raw_transform._data = raw_transformed_data

        noise_data = original_data - raw_transformed_data
        ep_samples = max(1, round(raw_transform.info["sfreq"] * 1.0))
        enova_ep = compute_enova_per_epoch(raw_transformed_data, noise_data, ep_samples)
        enova_ch = compute_enova_per_channel(raw_transformed_data, noise_data, ep_samples)
        sensai_val = compute_composite_sensai(
            raw_transformed_data, noise_data, raw_transform.info["sfreq"], self._reference_cov.data
        )
        self.metrics_ = {
            "enova_per_epoch": enova_ep,
            "enova_per_channel": enova_ch,
            "mean_enova": float(np.mean(enova_ep)) if len(enova_ep) > 0 else 0.0,
            "sensai_score": sensai_val,
        }

        if verbose in (True, 1, "INFO", "info", "DEBUG", "debug") or (
            isinstance(verbose, int) and not isinstance(verbose, bool) and verbose >= 1
        ):
            print(_format_summary_table(self))

        return raw_transform

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

    def summary(self) -> str:
        """Print and return a formatted summary table of the model fitting and denoising metrics.

        Returns
        -------
        summary_str : str
            Formatted ASCII summary table.
        """
        self._check_fit()
        table_str = _format_summary_table(self)
        print(table_str)
        return table_str

    def plot_sensai(
        self,
        raw_before: BaseRaw,
        raw_after: BaseRaw | None = None,
        epoch_duration_sec: float = 1.0,
        n_pc: int = 3,
        show: bool = True,
    ):
        """Plot 2D SENSAI Subspace Similarity vs Epoch Power Scatter & Manifold Classification.

        Replicates MATLAB's SENSAI_visualization.m with side-by-side Before/After
        subspace projections, LDA decision boundary shading, and marginal KDE distributions.

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
        metrics : dict
        """
        from ..viz.sensai import plot_sensai_visualization

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
        status = "fitted" if self.fitted else "unfitted"
        metrics_info = ""
        if getattr(self, "metrics_", None) is not None:
            sensai = self.metrics_.get("sensai_score", 0.0)
            enova = self.metrics_.get("mean_enova", 0.0) * 100
            metrics_info = f", SENSAI={sensai:.2f}%, Mean ENOVA={enova:.2f}%"
        return f"<{self.__class__.__name__} ({status}{metrics_info})>"


