import mne
import numpy as np
from mne._fiff.pick import _picks_to_idx
from mne.io import BaseRaw

from ..utils._checks import _check_n_jobs, _check_picks_uniqueness, _check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger, verbose
from ..wavelet.transform import epochs_to_wavelet
from .covariances import _ensure_cov, _pick_cov
from .gedai import Gedai, create_cosine_weights
from .multiband import compute_closest_valid_duration


def _compute_wavelet_parameters(sfreq, level, cycles_per_wavelet=2):
    """Compute wavelet band metadata matching ``epochs_to_wavelet`` ordering."""
    _check_type(cycles_per_wavelet, (float, int), "cycles_per_wavelet")
    if cycles_per_wavelet <= 0:
        raise ValueError(
            "cycles_per_wavelet must be strictly positive, "
            f"got {cycles_per_wavelet}"
        )
    if level < 0:
        raise ValueError(f"wavelet_level must be >= 0, got {level}")

    # Index 0 is approximation, then details from coarse to fine.
    band_definitions = [(0.0, sfreq / (2 ** (level + 1)), level + 1)]
    for i in range(level, 0, -1):
        band_definitions.append((sfreq / (2 ** (i + 1)), sfreq / (2**i), i))

    wavelet_parameters = []
    for band_index, (fmin, fmax, cycle_power) in enumerate(band_definitions):
        target_duration = 1 / fmin * cycles_per_wavelet  if fmin > 0 else 10.0
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
                "samples": int(n_samples),
            }
        )
    return wavelet_parameters


def _compute_window_starts(n_times, window_size, overlap):
    """Compute overlap-add window starts ensuring full coverage."""
    if window_size > n_times:
        raise ValueError(
            "Window size is larger than the available recording length. "
            f"window_size={window_size}, n_times={n_times}."
        )

    step = int(window_size * (1 - overlap))
    if step <= 0:
        raise ValueError(
            "overlap produces a zero-sized step. "
            f"Got overlap={overlap} for window_size={window_size}."
        )

    starts = np.arange(0, n_times - window_size + 1, step)
    if starts[-1] != n_times - window_size:
        starts = np.append(starts, n_times - window_size)
    return starts


@fill_doc
class AdaptativeMultibandGedai:
    """Generalized Eigenvalue De-Artifacting Instrument (GEDAI).

    A extension of the :py:class:`~gedai.gedai.MultibandGedai` method that
    uses adaptive epoch durations for each wavelet band. Tihs methods
    allows to better capture the frequency content of each wavelet band.
    See :footcite:`Ros2025`.

    
    .. note::
        Since epoch duration is adapted for each wavelet band, it
        is not possible to use :func:`~mne.Epochs` objects with
        this model. If you want to use :func:`~mne.Epochs` objects, 
        check the :py:class:`~gedai.gedai.MultibandGedai` class which
        uses a fixed epoch duration for all wavelet bands.

    .. warning::
        For EEG channels, Gedai will set average reference internally
        to match the leadfield covariance reference.
        Gedai will not modify the input data in-place, but will create
        copies when necessary to ensure the original data remains unchanged.

    Parameters
    ----------
    %(wavelet_type)s
    %(wavelet_level)s
    %(cycles_per_wavelet)s

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        wavelet_type="haar",
        wavelet_level=4,
        cycles_per_wavelet=2.0,
    ):
        self.fitted = False
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.cycles_per_wavelet = cycles_per_wavelet

        self._wavelets_fits = None
        self._reference_cov = None
        self._sfreq = None

    def _check_fit(self):
        """Check if the Gedai is fitted."""
        if not self.fitted:
            raise RuntimeError(
                f"Gedai must be fitted before using {self.__class__.__name__}"
            )
        assert self._wavelets_fits is not None
        assert self._reference_cov is not None

    def _check_unfitted(self):
        """Check if the Gedai is unfitted."""
        if self.fitted:
            raise RuntimeError(
                f"Gedai must be unfitted before using {self.__class__.__name__}."
            )
        assert self._wavelets_fits is None
        assert self._reference_cov is None

    @fill_doc
    @verbose
    def fit_raw(
        self,
        raw: BaseRaw,
        picks: list | str = "eeg",
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
        %(picks)s
        %(overlap)s
        %(reject_by_annotation)s
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
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

        picks = _picks_to_idx(raw.info, picks, none="all", exclude=[])
        _check_picks_uniqueness(raw.info, picks)

        raw_fit = raw.copy().load_data().pick(picks)
        sfreq = raw_fit.info["sfreq"]
        #TODO: check raw.info['highpass']
        logger.info("Setting average reference.")
        raw_fit.set_eeg_reference("average", projection=False)

        cov = _pick_cov(reference_cov, raw_fit.info["ch_names"])

        wavelet_parameters = _compute_wavelet_parameters(
                sfreq,
                self.wavelet_level,
                cycles_per_wavelet=self.cycles_per_wavelet,
            )

        wavelets_fits = []
        for wavelet_parameter in wavelet_parameters:
            w = wavelet_parameter["band_index"]
            fmin = wavelet_parameter["fmin"]
            fmax = wavelet_parameter["fmax"]
            duration = wavelet_parameter["duration"]
            samples = wavelet_parameter["samples"]

            logger.info(
                f"Adaptive wavelet index {w} ({fmin:.2f}-{fmax:.2f} Hz): "
                f"duration={duration:.3f}s ({samples} samples)."
            )

            if samples > raw_fit.n_times:
                raise ValueError(
                    f"Adaptive duration for wavelet index {w} "
                    f"({duration:.3f}s / {samples} samples) is longer than "
                    f"the available raw recording ({raw_fit.n_times} samples)."
                )

            overlap_seconds = duration * overlap
            epochs = mne.make_fixed_length_epochs(
                raw_fit,
                duration=duration,
                overlap=overlap_seconds,
                reject_by_annotation=reject_by_annotation,
                preload=True,
                verbose=verbose,
            )

            epochs_wavelet, freq_bands, _ = epochs_to_wavelet(
                epochs.get_data(),
                sfreq=sfreq,
                wavelet=self.wavelet_type,
                level=self.wavelet_level,
                n_jobs=n_jobs,
            )

            freq_fmin, freq_fmax = freq_bands[w]
            if abs(freq_fmin - fmin) > 1e-12 or abs(freq_fmax - fmax) > 1e-12:
                raise RuntimeError(
                    "Wavelet frequency band mismatch while building adaptive epochs."
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

            wavelets_fits.append(
                {
                    "band_index": w,
                    "fmin": fmin,
                    "fmax": fmax,
                    "model": model,
                    "duration": duration,
                    "samples": samples,
                }
            )

        self._wavelets_fits = wavelets_fits
        self._reference_cov = cov
        self._sfreq = sfreq
        self.fitted = True

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
        raw_corrected : mne.io.BaseRaw
            The corrected raw data.
        """
        self._check_fit()
        _check_type(raw, (BaseRaw,), "raw")
        _check_type(overlap, (float, int), "overlap")
        n_jobs = _check_n_jobs(n_jobs)

        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")

        missing_ch = set(self.ch_names) - set(raw.info["ch_names"])
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

        sfreq = raw.info["sfreq"]
        if self._sfreq != sfreq:
            raise ValueError(
                f"Sampling frequency of input raw ({sfreq} Hz) does not match "
                f"the sampling frequency of the data used during fit "
                f"({self._sfreq} Hz). You can resample the raw to "
                f"{self._sfreq} Hz before calling transform_raw."
            )
        picks = _picks_to_idx(raw.info, self.ch_names, none="all", exclude=[])
        raw_copy = raw.copy().pick(picks)
        raw_copy.load_data()
        raw_copy.set_eeg_reference("average", projection=False)

        raw_data = raw_copy.get_data(verbose=False)
        _, n_times = raw_data.shape

        raw_data_corrected = np.zeros_like(raw_data)
        for wavelet_fit in self._wavelets_fits:
            band_idx = wavelet_fit["band_index"]
            window_size = wavelet_fit["samples"]
            fmin = wavelet_fit["fmin"]
            fmax = wavelet_fit["fmax"]

            starts = _compute_window_starts(n_times, window_size, overlap)
            window = create_cosine_weights(window_size)

            all_segments = []
            for start in starts:
                start = int(start)
                segment = raw_data[:, start : start + window_size]
                all_segments.append(segment)

            all_segments_array = np.array(all_segments)
            segments_wavelet, freq_bands, _ = epochs_to_wavelet(
                all_segments_array,
                sfreq=sfreq,
                wavelet=self.wavelet_type,
                level=self.wavelet_level,
                n_jobs=n_jobs,
            )
            del all_segments_array

            freq_fmin, freq_fmax = freq_bands[band_idx]
            if abs(freq_fmin - fmin) > 1e-12 or abs(freq_fmax - fmax) > 1e-12:
                raise RuntimeError(
                    "Wavelet frequency band mismatch while building adaptive epochs."
                )

            segments_band = segments_wavelet[:, :, band_idx, :]
            segments_epochs = mne.EpochsArray(
                segments_band,
                raw_copy.info,
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
            raw_data_corrected += band_corrected

        raw_copy._data = raw_data_corrected
        return raw_copy

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
