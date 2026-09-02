import mne
import numpy as np
from joblib import Parallel, delayed
from mne import BaseEpochs
from mne.io import BaseRaw

from gedai.gedai._utils import (
    _check_fit_info,
    _detect_signal_type,
    _ensure_wavelet_low_cutoff,
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
    epochs_to_wavelet,
    get_modwt_band_limits,
)
from .gedai import Gedai, _clean_continuous_dual_stream


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
    wavelet_level : int or 'auto'
        The wavelet decomposition level. If 'auto', automatically computed from sfreq.
    broadband_pass : bool
        Whether to run an initial broadband GED pass before multiband
        wavelet decomposition.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, wavelet_type="haar", wavelet_level="auto", broadband_pass=True):
        if wavelet_level != "auto":
            _check_type(wavelet_level, (int,), "wavelet_level")
        _check_type(wavelet_type, (str,), "wavelet_type")
        _check_type(broadband_pass, (bool,), "broadband_pass")
        self.wavelet_type = wavelet_type
        self._wavelet_level = wavelet_level
        self.broadband_pass = broadband_pass

        self.fitted = False
        self._wavelet_low_cutoff = None
        self._wavelets_fits = None
        self._reference_cov = None
        self._levels = None
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
        noise_multiplier: float | str = "auto",
        wavelet_low_cutoff: float | str | None = 0.5,
        n_pc: int | str = "auto",
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
        %(n_pc)s
        %(n_jobs)s
        %(verbose)s
        """
        self._check_unfitted()
        _check_type(epochs, (BaseEpochs,), "epochs")
        _ensure_cov(reference_cov)
        _check_type(sensai_method, (str,), "sensai_method")
        noise_multiplier = _ensure_noise_multiplier(noise_multiplier)
        n_jobs = _check_n_jobs(n_jobs)

        epochs_fit = _prepare_epochs_fit(epochs, picks)
        data = epochs_fit.get_data()

        cov = _ensure_cov(reference_cov)
        cov = _pick_cov(cov, epochs_fit.info)

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
            logger.info("Running broadband GEDAI pre-cleaning pass on epochs...")
            broadband_model = Gedai()
            broadband_model.fit_epochs(
                epochs_fit,
                picks="all",
                reference_cov=cov.copy(),
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
                verbose=verbose,
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
                    "sensai_bounds": (0.0, 10.0),
                    "enova": 0.0,
                }
            else:
                center_freq = (fmin + fmax) / 2.0
                band_bounds = (
                    (-6.0, 12.0) if (0.8 <= center_freq <= 60.0) else (0.0, 10.0)
                )

                wavelet_epochs_data = epochs_wavelet[:, :, w, :]
                wavelet_epochs = mne.EpochsArray(
                    wavelet_epochs_data,
                    epochs_fit.info,
                    tmin=epochs_fit.tmin,
                    verbose=False,
                )

                model = Gedai()
                model.fit_epochs(
                    wavelet_epochs,
                    picks="all",
                    reference_cov=cov.copy(),
                    sensai_method=sensai_method,
                    noise_multiplier=noise_multiplier,
                    sensai_bounds=band_bounds,
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
        noise_multiplier: float | str = "auto",
        wavelet_low_cutoff: float | str | None = 0.5,
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
        %(wavelet_low_cutoff)s
        %(n_pc)s
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
        cov = _ensure_cov(reference_cov)
        _check_type(sensai_method, (str,), "sensai_method")
        noise_multiplier = _ensure_noise_multiplier(noise_multiplier)
        n_jobs = _check_n_jobs(n_jobs)

        raw_fit = _prepare_raw_fit(raw, picks)
        sfreq = raw_fit.info["sfreq"]

        # Obligatory 0.1 Hz wavelet high-pass pre-filter on input data
        raw_fit._data = _apply_wavelet_highpass_prefilter(
            raw_fit._data, sfreq, lowcut_hz=0.1
        )

        cov = _pick_cov(cov, raw_fit.info)
        filter_cutoff = raw_fit.info["highpass"]
        wavelet_low_cutoff = _ensure_wavelet_low_cutoff(
            wavelet_low_cutoff, filter_cutoff, duration
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
            duration, actual_wavelet_level, sfreq
        )
        if valid_duration != duration:
            logger.warning(
                f"Requested duration {duration:.3f}s adjusted to {valid_duration:.3f}s "
                f"({valid_samples} samples) to satisfy wavelet level "
                f"{actual_wavelet_level} requirements."
            )
        duration = valid_duration

        # Broadband pre-cleaning pass with wavelet HP pre-filter if requested
        signal_type = _detect_signal_type(raw_fit.info)
        bb_bounds = (-4.0, 8.0) if signal_type == "meg" else (-4.0, 10.0)
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
                duration=duration,
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

        # Continuous single-band MODWT decomposition & band fitting
        band_limits = get_modwt_band_limits(sfreq, actual_wavelet_level + 1)
        raw_data_fit = raw_for_multiband.get_data(verbose=False)
        epoch_samples = max(2, int(round(duration * sfreq)))
        if epoch_samples % 2 != 0:
            epoch_samples += 1
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
                    noise_multiplier,
                    n_pc=n_pc,
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
                    noise_multiplier,
                    n_pc=n_pc,
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
        noise_multiplier,
        n_pc="auto",
    ):
        """Fit a single wavelet band model."""
        w, (fmin, fmax) = item
        if fmax <= wavelet_low_cutoff:
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

        band_data = _modwt_haar_single_band(raw_data_fit.T, actual_wavelet_level, w)
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
        if signal_type == "meg":
            # In MODWT, w=0 is the finest (highest-frequency) detail band
            # (e.g., EMG/sensor noise)
            # and w=1 is the second-highest detail band. Wider negative bounds
            # are used here
            # to capture high-frequency MEG noise.
            band_bounds = (-6.0, 8.0) if w in (0, 1) else (0.0, 10.0)
        else:
            band_bounds = (-6.0, 12.0) if (0.8 <= center_freq <= 60.0) else (0.0, 10.0)

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

        if self.broadband_pass and self._broadband_model is not None:
            epochs_input = self._broadband_model.transform_epochs(
                epochs_transform, n_jobs=n_jobs, verbose=False
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
            )
            cleaned_epochs_wavelet[:, :, band_idx, :] = cleaned_wavelet_epochs.get_data(
                verbose=False
            )

        cleaned_epochs_data = np.sum(cleaned_epochs_wavelet, axis=2)
        epochs_transform._data = cleaned_epochs_data

        orig_data = epochs_transform.get_data()
        orig_2d = orig_data.transpose(1, 0, 2).reshape(orig_data.shape[1], -1)
        clean_2d = cleaned_epochs_data.transpose(1, 0, 2).reshape(
            cleaned_epochs_data.shape[1], -1
        )
        noise_2d = orig_2d - clean_2d
        ep_samples = orig_data.shape[-1]
        enova_ep = compute_enova_per_epoch(clean_2d, noise_2d, ep_samples)
        enova_ch = compute_enova_per_channel(clean_2d, noise_2d, ep_samples)
        sensai_val = compute_composite_sensai(
            clean_2d, noise_2d, epochs_transform.info["sfreq"], self._reference_cov.data
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
        self._check_fit()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        n_jobs = _check_n_jobs(n_jobs)

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        _check_fit_info(self, raw)
        raw_transform = _prepare_raw_transform(raw, self.ch_names)
        sfreq = raw_transform.info["sfreq"]

        # Mirror the 0.1 Hz high-pass pre-filtering applied during fit_raw
        raw_transform._data = _apply_wavelet_highpass_prefilter(
            raw_transform._data, sfreq, lowcut_hz=0.1
        )

        # Broadband pre-cleaning if model was fitted with broadband_pass
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
        """Transform one wavelet band using continuous MODWT cleaning."""
        band_idx = wavelet_fit["band_index"]
        ignore = wavelet_fit["ignore"]

        if ignore:
            return np.zeros_like(raw_data), 0.0, 0.0

        band_data = _modwt_haar_single_band(raw_data.T, actual_level, band_idx)
        threshold = wavelet_fit["model"].threshold
        epoch_duration = wavelet_fit.get("duration", 1.0)
        if epoch_duration is None:
            epoch_duration = 1.0

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
