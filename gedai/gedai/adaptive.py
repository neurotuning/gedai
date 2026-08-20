import mne
import numpy as np
from mne.io import BaseRaw

from gedai.gedai._utils import (
    _check_fit_info,
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
)
from ..utils._docs import fill_doc
from ..utils.logs import logger, verbose
from ..wavelet.transform import compute_wavelet_level, epochs_to_wavelet
from .gedai import Gedai, create_cosine_weights
from .multiband import compute_closest_valid_duration


def _compute_wavelet_parameters(sfreq, level, cycles_per_wavelet):
    """Compute wavelet band metadata matching ``epochs_to_wavelet`` ordering."""
    _check_type(cycles_per_wavelet, (float, int), "cycles_per_wavelet")
    if cycles_per_wavelet <= 0:
        raise ValueError(
            f"cycles_per_wavelet must be strictly positive, got {cycles_per_wavelet}"
        )
    if level < 0:
        raise ValueError(f"wavelet_level must be >= 0, got {level}")

    # Index 0 is approximation, then details from coarse to fine.
    band_definitions = [(0.0, sfreq / (2 ** (level + 1)), level + 1)]
    for i in range(level, 0, -1):
        fmin = sfreq / (2 ** (i + 1))
        fmax = sfreq / (2**i)
        cycle_power = i
        band_definitions.append((fmin, fmax, cycle_power))

    wavelet_parameters = []
    for band_index, (fmin, fmax, _cycle_power) in enumerate(band_definitions):
        if fmin == 0.0:
            target_duration = None
            ignore = True
            n_samples = None
            duration = None
        else:
            target_duration = 1 / fmin * cycles_per_wavelet
            ignore = False
            duration, n_samples = compute_closest_valid_duration(
                target_duration,
                level,
                sfreq,
            )
        wavelet_parameters.append(
            {
                "band_index": band_index,
                "fmin": fmin,
                "fmax": fmax,
                "duration": duration,
                "n_samples": n_samples,
                "ignore": ignore,
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
        wavelet_level=8,
        cycles_per_wavelet=12,
        broadband_pass=False,
    ):
        if wavelet_level != "auto":
            _check_type(wavelet_level, (int,), "wavelet_level")
        _check_type(wavelet_type, (str,), "wavelet_type")
        _check_type(cycles_per_wavelet, (int,), "cycles_per_wavelet")
        _check_type(broadband_pass, (bool,), "broadband_pass")

        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.cycles_per_wavelet = cycles_per_wavelet
        self.broadband_pass = broadband_pass

        self.fitted = False

        self._wavelets_fits = None
        self._reference_cov = None
        self._wavelet_low_cutoff = None
        self._actual_wavelet_level = None
        self._broadband_model = None
        self.metrics_ = None

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
        overlap: float = 0.75,
        reject_by_annotation: bool | None = False,
        reference_cov: str = "leadfield",
        sensai_method: str = "gridsearch",
        noise_multiplier: float = 3.0,
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
        _check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        raw_fit = _prepare_raw_fit(raw, picks)

        cov = _pick_cov(reference_cov, raw_fit.info["ch_names"])
        sfreq = raw_fit.info["sfreq"]

        # Automatic wavelet level calculation
        if self.wavelet_level == "auto":
            actual_wavelet_level = compute_wavelet_level(
                sfreq,
                lowcut_hz=0.5,
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

        filter_cutoff = raw_fit.info["highpass"]
        duration_cutoff = wavelet_parameters[0]["fmax"]

        if wavelet_low_cutoff is None:
            wavelet_low_cutoff = 0

        if wavelet_low_cutoff == "auto":
            if filter_cutoff > duration_cutoff:
                wavelet_low_cutoff = filter_cutoff
                logger.info(
                    "Automatically setting ``wavelet_low_cutoff`` to "
                    f"{wavelet_low_cutoff:.2f} Hz based on raw.info['highpass'] "
                    f"({filter_cutoff:.2f} Hz)."
                )
            else:
                wavelet_low_cutoff = duration_cutoff
                logger.info(
                    "Automatically setting ``wavelet_low_cutoff`` to "
                    f"{wavelet_low_cutoff:.2f} Hz based on wavelet_level "
                    f"{actual_wavelet_level}."
                )

        if wavelet_low_cutoff < duration_cutoff:
            logger.warning(
                f"``wavelet_low_cutoff`` ({wavelet_low_cutoff:.2f} Hz) is below the "
                f"frequency cutoff ( {duration_cutoff:.2f} Hz) that can be "
                f"resolved with the chosen cycles_per_wavelet "
                f"({self.cycles_per_wavelet})."
            )

        # Broadband pre-cleaning pass if requested
        if self.broadband_pass:
            logger.info("Running broadband GEDAI pre-cleaning pass on raw data...")
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
                n_jobs=n_jobs,
                verbose=verbose,
            )
            raw_for_multiband = broadband_model.transform_raw(
                raw_fit, overlap=overlap, n_jobs=n_jobs, verbose=False
            )
            self._broadband_model = broadband_model
        else:
            raw_for_multiband = raw_fit

        wavelets_fits = []
        for wavelet_parameter in wavelet_parameters:
            w = wavelet_parameter["band_index"]
            fmin = wavelet_parameter["fmin"]
            fmax = wavelet_parameter["fmax"]
            duration = wavelet_parameter["duration"]
            n_samples = wavelet_parameter["n_samples"]

            if fmax <= wavelet_low_cutoff:
                ignore = True
            elif n_samples is not None and n_samples > raw_for_multiband.n_times:
                logger.warning(
                    f"Adaptive duration for wavelet index {w} "
                    f"({duration:.3f}s / {n_samples} samples) is longer than "
                    f"the available raw recording ({raw_for_multiband.n_times} samples). "
                    "Marking band as ignored."
                )
                ignore = True
            else:
                ignore = False

            if ignore:
                logger.info(
                    f"Skipping wavelet index {w} ({fmin:.2f}-{fmax:.2f} Hz) "
                    "because its upper frequency cutoff is below the low "
                    f"frequency cutoff ({wavelet_low_cutoff} Hz)."
                )
                wavelets_fits.append(
                    {
                        "band_index": w,
                        "fmin": fmin,
                        "fmax": fmax,
                        "model": None,
                        "duration": duration,
                        "n_samples": n_samples,
                        "ignore": ignore,
                        "sensai_bounds": (0.0, 12.0),
                        "enova": 0.0,
                    }
                )
            else:
                logger.info(
                    f"Adaptive wavelet index {w} ({fmin:.2f}-{fmax:.2f} Hz): "
                    f"duration={duration:.3f}s ({n_samples} samples)."
                )

                if n_samples > raw_for_multiband.n_times:
                    raise ValueError(
                        f"Adaptive duration for wavelet index {w} "
                        f"({duration:.3f}s / {n_samples} samples) is longer than "
                        f"the available raw recording ({raw_for_multiband.n_times} samples)."
                    )

                overlap_seconds = duration * overlap
                epochs = mne.make_fixed_length_epochs(
                    raw_for_multiband,
                    duration=duration,
                    overlap=overlap_seconds,
                    reject_by_annotation=reject_by_annotation,
                    preload=True,
                    verbose=False,
                )

                epochs_wavelet, freq_bands, _ = epochs_to_wavelet(
                    epochs.get_data(verbose=False),
                    sfreq=sfreq,
                    wavelet=self.wavelet_type,
                    level=actual_wavelet_level,
                    n_jobs=n_jobs,
                )

                freq_fmin, freq_fmax = freq_bands[w]
                if abs(freq_fmin - fmin) > 1e-12 or abs(freq_fmax - fmax) > 1e-12:
                    raise RuntimeError(
                        "Wavelet frequency band mismatch while building "
                        "adaptive epochs."
                    )

                wavelet_epochs_data = epochs_wavelet[:, :, w, :]
                del epochs_wavelet
                wavelet_epochs = mne.EpochsArray(
                    wavelet_epochs_data,
                    epochs.info,
                    tmin=epochs.tmin,
                    verbose=False,
                )
                del wavelet_epochs_data

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
                    n_jobs=n_jobs,
                    verbose=verbose,
                )

                wavelets_fits.append(
                    {
                        "band_index": w,
                        "fmin": fmin,
                        "fmax": fmax,
                        "model": model,
                        "duration": duration,
                        "n_samples": epochs.times.size,
                        "ignore": ignore,
                        "sensai_bounds": band_bounds,
                        "enova": model.metrics_["mean_enova"] if model.metrics_ else 0.0,
                    }
                )

        self.fitted = True
        self._info = raw_fit.info.copy()
        self._reference_cov = cov # No regularization applied

        self._wavelets_fits = wavelets_fits
        self._wavelet_low_cutoff = wavelet_low_cutoff

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

        if self.broadband_pass and self._broadband_model is not None:
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
        for wavelet_fit in self._wavelets_fits:
            band_idx = wavelet_fit["band_index"]
            window_size = wavelet_fit["n_samples"]
            fmin = wavelet_fit["fmin"]
            fmax = wavelet_fit["fmax"]
            ignore = wavelet_fit["ignore"]
            if ignore:
                logger.info(
                    f"Zeroing wavelet index {band_idx} ({fmin:.2f}-{fmax:.2f} Hz)"
                )
            else:
                window = create_cosine_weights(window_size)
                step = int(window_size * (1 - overlap))
                starts = np.arange(0, n_times - window_size, step)
                starts = np.append(starts, n_times - window_size)

                all_segments = []
                for start in starts:
                    segment = raw_data[:, start : start + window_size]
                    all_segments.append(segment)

                all_segments_array = np.array(all_segments)
                segments_wavelet, freq_bands, _ = epochs_to_wavelet(
                    all_segments_array,
                    sfreq=sfreq,
                    wavelet=self.wavelet_type,
                    level=actual_level,
                    n_jobs=n_jobs,
                )
                del all_segments_array

                freq_fmin, freq_fmax = freq_bands[band_idx]
                if abs(freq_fmin - fmin) > 1e-12 or abs(freq_fmax - fmax) > 1e-12:
                    raise RuntimeError(
                        "Wavelet frequency band mismatch while building "
                        "adaptive epochs."
                    )

                segments_band = segments_wavelet[:, :, band_idx, :]
                segments_epochs = mne.EpochsArray(
                    segments_band,
                    raw_transform.info,
                    tmin=0.0,
                    verbose=False,
                )
                del segments_wavelet

                corrected_segments_epochs = wavelet_fit["model"].transform_epochs(
                    segments_epochs,
                    n_jobs=n_jobs,
                    verbose=verbose,
                )
                corrected_segments = corrected_segments_epochs.get_data(verbose=False)
                del corrected_segments_epochs

                # Reconstruct the corrected wavelet band
                weight_sum = np.zeros_like(raw_data)
                band_corrected = np.zeros_like(raw_data)
                for s, start in enumerate(starts):
                    start = int(start)
                    corrected_segment = corrected_segments[s] * window
                    band_corrected[:, start : start + window_size] += corrected_segment
                    weight_sum[:, start : start + window_size] += window

                weight_sum[weight_sum == 0] = 1
                band_corrected /= weight_sum
                raw_transformed_data += band_corrected

        raw_transform._data = raw_transformed_data

        noise_data = raw_data - raw_transformed_data
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
