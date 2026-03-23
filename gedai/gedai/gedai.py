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
    _sensai_gridsearch_fast,
    _sensai_optimize_fast,
    _sensai_to_eigen,
)
from ..utils._checks import _check_n_jobs, check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ..wavelet.transform import epochs_to_wavelet
from .covariances import _compute_refcov


def _detect_signal_type(info):
    """Automatically detect whether data is EEG or MEG from MNE channel info.

    Parameters
    ----------
    info : mne.Info
        The MNE info object of the data being processed.

    Returns
    -------
    signal_type : str
        ``'eeg'`` if EEG channels are present (and no MEG), ``'meg'`` if
        magnetometers or gradiometers are present.  When both types coexist
        the first MEG-like type found takes priority.  Falls back to
        ``'meg'`` when the channel type cannot be determined.
    """
    ch_types = set(info.get_channel_types())
    if "mag" in ch_types or "grad" in ch_types:
        return "meg"
    if "eeg" in ch_types:
        return "eeg"
    # Unknown channel types (e.g. generic / misc) — treat conservatively as MEG
    return "meg"


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
    signal_type : str
        Type of neural signal.  Accepted values:

        - ``'eeg'`` — Apply average referencing; use fixed 3 principal
          components for the reference subspace (SENSAI); use the 98th
          percentile of the log-eigenvalue distribution for threshold
          conversion (matching the MATLAB implementation).
        - ``'meg'`` — Skip average referencing; adaptively select the
          number of reference PCs that explain >=85%% of the reference
          covariance variance; use 4 PCs for the SSI comparison; use
          the 99th percentile for threshold conversion.
        - ``'auto'`` (default) — Automatically determined from the
          MNE channel types of the data at fit time.  Magnetometers
          (``mag``) and gradiometers (``grad``) map to ``'meg'``;
          ``eeg`` channels map to ``'eeg'``.
    highpass_cutoff : float or None
        If not ``None``, apply a MODWT-based high-pass filter at
        approximately this frequency (Hz) **before** any GEDAI fitting or
        transforming.  Default is ``0.1`` Hz.  Set to ``None`` to disable.
    preliminary_broadband_noise_multiplier : float or None
        When spectral GEDAI is active (``epoch_size_in_cycles`` is set),
        run a preliminary **broadband** single-pass GEDAI on the full signal
        with this noise multiplier **before** the per-band wavelet pass.
        This mirrors the MATLAB ``GEDAI.m`` pipeline which always performs a
        broadband denoising step (noise_multiplier ~6) prior to the wavelet
        analysis.  Default is ``6.0``.  Set to ``None`` to skip.

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
        signal_type="auto",
        highpass_cutoff=0.1,
        preliminary_broadband_noise_multiplier=6.0,
    ):
        self.wavelet_type = wavelet_type
        if wavelet_level != "auto" and (
            not isinstance(wavelet_level, int) or wavelet_level < 0
        ):
            raise ValueError(
                f"wavelet_level must be a non-negative int or 'auto', got {wavelet_level!r}."
            )
        self.wavelet_level = wavelet_level
        self.wavelet_low_cutoff = wavelet_low_cutoff
        self.epoch_size_in_cycles = epoch_size_in_cycles
        self.alpha_range = alpha_range
        self.alpha_sensai_threshold = alpha_sensai_threshold
        if signal_type not in ("eeg", "meg", "auto"):
            raise ValueError(
                f"signal_type must be 'eeg', 'meg', or 'auto', got '{signal_type}'."
            )
        self.signal_type = signal_type
        if highpass_cutoff is not None and highpass_cutoff <= 0:
            raise ValueError(
                f"highpass_cutoff must be a positive frequency in Hz or None, "
                f"got {highpass_cutoff}."
            )
        self.highpass_cutoff = highpass_cutoff
        self.preliminary_broadband_noise_multiplier = preliminary_broadband_noise_multiplier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _modwt_highpass(self, raw: BaseRaw) -> BaseRaw:
        """Return a copy of *raw* with sub-cutoff content removed via MODWT.

        Computes a Maximal Overlap Discrete Wavelet Transform at the minimum
        level ``L`` such that the approximation band covers
        ``[0, sfreq / 2^(L+1)]`` Hz <= ``self.highpass_cutoff``.  The
        approximation coefficients are zeroed and the signal is reconstructed
        by summing the remaining detail bands via MODWTMRA, producing a
        high-passed copy of *raw*.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data (not modified in-place).

        Returns
        -------
        raw_hp : mne.io.BaseRaw
            A copy of *raw* with frequencies below ~``highpass_cutoff`` Hz
            removed.
        """
        import math
        from ..wavelet._modwt import modwt, modwtmra

        sfreq = raw.info["sfreq"]
        cutoff = self.highpass_cutoff

        # Minimum level L such that  sfreq / 2^(L+1)  <=  cutoff
        level = max(1, int(np.ceil(math.log2(sfreq / cutoff))) - 1)
        actual_cutoff = sfreq / 2 ** (level + 1)
        print(
            f"[GEDAI] MODWT high-pass: level={level}, "
            f"cutoff ~{actual_cutoff:.4f} Hz "
            f"(requested {cutoff} Hz)",
            flush=True,
        )

        raw_data = raw.get_data()          # (n_ch, n_times)
        n_ch, n_times = raw_data.shape

        # Pad to a length divisible by 2^level (required by SWT)
        divisor = 2 ** level
        pad_to = int(np.ceil(n_times / divisor) * divisor)
        if pad_to > n_times:
            padded = np.pad(raw_data, ((0, 0), (0, pad_to - n_times)), mode="edge")
        else:
            padded = raw_data

        # MODWT decomposition — top-level call (not inside a parallel loop);
        # use all available cores across channels.
        wpt = modwt(padded.T, self.wavelet_type, level, n_jobs=-1)  # (n_bands, pad_to, n_ch)
        mra = modwtmra(wpt, self.wavelet_type, n_jobs=-1)            # (n_bands, pad_to, n_ch)
        del wpt, padded

        # Zero the approximation band (last index = lowest frequencies).
        # MRA order: [D_1(finest), ..., D_L(coarsest), S_L(approx)]
        mra[-1] = 0.0

        # Reconstruct from detail bands only, trim padding
        hp_data = np.sum(mra, axis=0)[:n_times, :].T    # (n_ch, n_times)
        del mra

        raw_hp = raw.copy()
        raw_hp._data = hp_data
        return raw_hp

    def _resolve_wavelet_level(self, sfreq: float) -> int:
        """Return the concrete wavelet decomposition level to use.

        If ``self.wavelet_level`` is an integer, returns it unchanged.
        If it is ``'auto'``, computes the minimum level *L* such that the
        lowest detail band just covers ``wavelet_low_cutoff``::

            L = ceil(log2(sfreq / wavelet_low_cutoff)) - 1

        Falls back to ``wavelet_low_cutoff = 0.5`` when not set.
        """
        if self.wavelet_level != "auto":
            return self.wavelet_level

        import math
        cutoff = self.wavelet_low_cutoff if self.wavelet_low_cutoff is not None else 0.5
        level = max(1, math.ceil(math.log2(sfreq / cutoff)) - 1)
        print(
            f"[GEDAI] wavelet_level='auto': sfreq={sfreq} Hz, "
            f"cutoff={cutoff} Hz -> level={level} "
            f"(lowest band ~{sfreq / 2 ** (level + 1):.3f}-"
            f"{sfreq / 2 ** level:.3f} Hz)",
            flush=True,
        )
        return level

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
        signal_type="meg",
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
            Info object to use when constructing :class:`mne.EpochsArray`
            (retained for API compatibility; no longer used in the fast path).
        epochs_tmin : float
            ``tmin`` of the epochs (retained for API compat; unused in fast path).
        reference_cov : np.ndarray
            Regularised reference covariance matrix (already computed).
        sensai_method : str
            ``'gridsearch'`` or ``'optimize'``.
        noise_multiplier : float
            Noise multiplier for SENSAI scoring.
        n_jobs : int
            Number of parallel jobs.
        signal_type : str
            ``'eeg'`` or ``'meg'`` (already resolved from ``'auto'`` by the
            caller).  Controls the number of reference PCs and the percentile
            used for eigenvalue ↔ SENSAI conversion (mirroring the MATLAB
            ``GEDAI_per_band.m`` and ``clean_SENSAI.m`` logic).

        Returns
        -------
        wavelet_fit : dict
            Dictionary with keys ``band_index``, ``fmin``, ``fmax``,
            ``threshold``, ``reference_cov``, ``epochs_eigenvalues``,
            ``sensai_runs``, and ``ignore``.
        """
        n_epochs, n_channels, _ = wavelet_epochs_data.shape
        epochs_eigenvalues  = np.zeros((n_epochs, n_channels))
        epochs_eigenvectors = np.zeros((n_epochs, n_channels, n_channels))

        # Single GEVD pass — eigenvalues AND eigenvectors stored for reuse
        # across all threshold evaluations (avoids re-running eigh per threshold).
        for e, wavelet_epoch_data in enumerate(wavelet_epochs_data):
            covariance = np.cov(wavelet_epoch_data)
            eigenvalues, eigenvectors = eigh(covariance, reference_cov, check_finite=True)
            epochs_eigenvalues[e]  = eigenvalues
            epochs_eigenvectors[e] = eigenvectors

        # ------------------------------------------------------------------
        # Signal-type dependent parameters (mirrors MATLAB GEDAI_per_band.m
        # and clean_SENSAI.m).
        # ------------------------------------------------------------------
        if signal_type == "eeg":
            # EEG: fixed 3 PCs for both reference template and SSI comparison;
            # 98th percentile for eigenvalue <-> SENSAI conversion.
            refcov_n_pc = 3
            ssi_n_pc    = 3
            eigen_percentile = 98
        else:  # 'meg'
            # MEG: adaptive PCs for reference template (>=85 % variance);
            # 4 PCs for SSI comparison; 99th percentile.
            all_evals = eigh(reference_cov, eigvals_only=True)[::-1]  # descending
            cumvar = np.cumsum(all_evals) / np.sum(all_evals)
            refcov_n_pc = int(np.searchsorted(cumvar, 0.85) + 1)
            refcov_n_pc = max(1, min(refcov_n_pc, n_channels - 1))
            ssi_n_pc    = 4
            eigen_percentile = 99
            if refcov_n_pc < ssi_n_pc:
                # The adaptive threshold chose fewer PCs than needed for SSI
                # comparison (common for small sensor arrays).  Raise refcov_n_pc
                # to use as many PCs as available, up to ssi_n_pc.
                refcov_n_pc = min(ssi_n_pc, n_channels - 1)
                # If the array is so small that even n_channels-1 < ssi_n_pc,
                # reduce ssi_n_pc to match.
                ssi_n_pc = min(ssi_n_pc, refcov_n_pc)
                logger.info(
                    f"MEG: small sensor array ({n_channels} ch) — using "
                    f"{refcov_n_pc} reference PCs and {ssi_n_pc} SSI PCs."
                )

        # Pre-compute template (reference) subspace once per band.
        _, evecs_reference = eigh(reference_cov)
        evecs_reference = evecs_reference[:, ::-1][:, :refcov_n_pc]  # (n_ch, n_pc)

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

        if sensai_method == "gridsearch":
            sensai_thresholds = np.arange(min_sensai_threshold, max_sensai_threshold, step)
            eigen_thresholds = [
                _sensai_to_eigen(sensai_value, epochs_eigenvalues,
                                 percentile=eigen_percentile)
                for sensai_value in sensai_thresholds
            ]
            threshold, runs = _sensai_gridsearch_fast(
                epochs_eigenvalues,
                epochs_eigenvectors,
                reference_cov,
                evecs_reference,
                n_pc=ssi_n_pc,
                noise_multiplier=noise_multiplier,
                eigen_thresholds=eigen_thresholds,
                n_jobs=n_jobs,
            )
        elif sensai_method == "optimize":
            sensai_threshold_bounds = (min_sensai_threshold, max_sensai_threshold)
            threshold, runs = _sensai_optimize_fast(
                epochs_eigenvalues,
                epochs_eigenvectors,
                reference_cov,
                evecs_reference,
                n_pc=ssi_n_pc,
                noise_multiplier=noise_multiplier,
                bounds=sensai_threshold_bounds,
                percentile=eigen_percentile,
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

        # Resolve signal type (auto-detect from channel types if needed)
        signal_type = self.signal_type
        if signal_type == "auto":
            signal_type = _detect_signal_type(epochs.info)
            logger.info(f"Auto-detected signal type: '{signal_type}'.")

        # Set average reference for EEG only
        if signal_type == "eeg" and "eeg" in epochs.get_channel_types():
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
                signal_type=signal_type,
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

        # ------------------------------------------------------------------
        # Step 0: MODWT high-pass pre-processing
        # ------------------------------------------------------------------
        if self.highpass_cutoff is not None:
            raw = self._modwt_highpass(raw)

        # Resolve signal type (auto-detect from channel types if needed)
        signal_type = self.signal_type
        if signal_type == "auto":
            signal_type = _detect_signal_type(raw.info)
            logger.info(f"Auto-detected signal type: '{signal_type}'.")


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

        # ------------------------------------------------------------------
        # Frequency-specific epoching path
        # ------------------------------------------------------------------
        # The duration here is only used for broadband reference-covariance
        # epochs, which are NOT wavelet-decomposed, so no 2^level adjustment
        # is needed or appropriate.
        if self.epoch_size_in_cycles is not None:
            logger.info(
                f"Frequency-specific epoching enabled "
                f"({self.epoch_size_in_cycles} cycles per band)."
            )
            self._fit_raw_frequency_specific(
                raw=raw,
                broadband_duration=duration,   # raw user-supplied value, no adjustment
                overlap=overlap,
                reject_by_annotation=reject_by_annotation,
                reference_cov=_ref_cov,
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
                signal_type=signal_type,
            )
            return

        # ------------------------------------------------------------------
        # Fixed-duration path (original behaviour): adjustment IS needed here
        # because epochs are individually wavelet-decomposed.
        # ------------------------------------------------------------------
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
        signal_type: str = "meg",
    ):
        """Fit with frequency-specific epoch sizes (MATLAB port).

        Architecture (matching MATLAB GEDAI exactly):

        1. Run a **single** MODWT on the full raw signal to get the MRA for
           all bands at once.
        2. For each band, **slice** the band's MRA output into epochs sized
           to contain ``self.epoch_size_in_cycles`` wave-cycles of the band's
           lower cutoff frequency.
        3. Fit GEDAI on those band-specific epoch slices.

        This mirrors MATLAB's ``modwt_single_band`` + ``GEDAI_per_band`` loop
        and avoids the O(n_bands) MODWT passes of the naive per-band approach.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data.
        broadband_duration : float
            Epoch duration used to resolve the reference covariance (broadband
            approximation band).
        overlap : float
            Overlap ratio between consecutive epochs (0–1).
        reject_by_annotation : bool
            Whether to drop annotated segments when making broadband epochs.
        reference_cov : np.ndarray or str
            Already-resolved path or array (not the ``'leadfield'`` sentinel).
        sensai_method : str
            ``'gridsearch'`` or ``'optimize'``.
        noise_multiplier : float
            Noise multiplier for SENSAI scoring.
        n_jobs : int
            Number of parallel jobs.
        """
        from ..wavelet._modwt import modwt, modwtmra

        sfreq = raw.info["sfreq"]
        n_times = raw.n_times

        # ------------------------------------------------------------------
        # 1. Average reference for EEG only (MEG skips this step)
        # ------------------------------------------------------------------
        raw_work = raw
        if signal_type == "eeg" and "eeg" in raw.get_channel_types():
            logger.info("Setting average reference to match leadfield/forward reference.")
            raw_work = raw.copy().load_data()
            raw_work.set_eeg_reference("average", projection=False)

        # ------------------------------------------------------------------
        # 2. Resolve reference covariance using broadband fixed-length epochs
        # ------------------------------------------------------------------
        bb_epochs = mne.make_fixed_length_epochs(
            raw_work,
            duration=broadband_duration,
            overlap=broadband_duration * overlap,
            reject_by_annotation=reject_by_annotation,
            preload=True,
            verbose=False,
        )
        from .covariances import _compute_refcov
        _ref_cov_arr, _ = _compute_refcov(bb_epochs, reference_cov)
        avg_diag_power = np.trace(_ref_cov_arr) / _ref_cov_arr.shape[0]
        regularization_lambda = 0.05
        epsilon = regularization_lambda * avg_diag_power
        _ref_cov_arr = _ref_cov_arr + epsilon * np.eye(_ref_cov_arr.shape[0])

        # ------------------------------------------------------------------
        # 2b. Preliminary broadband GEDAI pass (mirrors MATLAB GEDAI.m)
        #     Run a single-band broadband GEDAI on the full signal before
        #     the per-band wavelet analysis, using a high noise_multiplier
        #     to strip large-amplitude artefacts first.
        # ------------------------------------------------------------------
        if self.preliminary_broadband_noise_multiplier is not None:
            print(
                f"[GEDAI] Preliminary broadband pass "
                f"(noise_multiplier={noise_multiplier:.1f})...",
                flush=True,
            )
            _bb_gedai = Gedai(
                wavelet_type=self.wavelet_type,
                wavelet_level=0,
                signal_type="meg",   # avg-ref already applied to raw_work
                highpass_cutoff=None,  # HP already applied
                preliminary_broadband_noise_multiplier=None,  # no recursion
            )
            raw_work = _bb_gedai.fit_transform_raw(
                raw_work,
                reference_cov=_ref_cov_arr,
                noise_multiplier=noise_multiplier,
                sensai_method=sensai_method,
            )

        # ------------------------------------------------------------------
        # 3. Compute frequency bands directly from sfreq and wavelet_level
        #    (mirrors the band structure produced by epochs_to_wavelet)
        # ------------------------------------------------------------------
        level = self._resolve_wavelet_level(sfreq)
        # freq_bands matches MRA band order:
        #   [D_1(highest freq), D_2, ..., D_L(lowest freq detail), S_L(approx)]
        freq_bands = []
        for k in range(1, level + 1):
            freq_bands.append((sfreq / 2 ** (k + 1), sfreq / 2 ** k))
        freq_bands.append((0.0, sfreq / 2 ** (level + 1)))
        self.levels_used = level

        # ------------------------------------------------------------------
        # 4. Single MODWT pass over the full signal
        #    modwt input : (n_samples, n_channels)
        #    modwt output: (n_bands, n_samples, n_channels)
        #    modwtmra returns the time-domain MRA reconstruction per band,
        #    same shape as the input coefficients.
        # ------------------------------------------------------------------
        print(
            f"Running single MODWT (level {level}) on full signal "
            f"({n_times} samples × {raw_work.info['nchan']} channels)...",
            flush=True,
        )
        raw_data = raw_work.get_data()  # (n_channels, n_times)

        # pywt.swt requires signal length divisible by 2^level.
        # Pad to the nearest valid length with edge values, then trim afterward.
        divisor = 2 ** level
        pad_to = int(np.ceil(n_times / divisor) * divisor)
        if pad_to > n_times:
            print(
                f"  Padding signal from {n_times} to {pad_to} samples "
                f"(nearest multiple of 2^{level}={divisor}) for SWT.",
                flush=True,
            )
            raw_data_padded = np.pad(
                raw_data, ((0, 0), (0, pad_to - n_times)), mode="edge"
            )
        else:
            raw_data_padded = raw_data

        wpt = modwt(raw_data_padded.T, self.wavelet_type, level, n_jobs=-1)   # (n_bands, pad_to, n_channels)
        mra = modwtmra(wpt, self.wavelet_type, n_jobs=-1)                      # (n_bands, pad_to, n_channels)
        mra = mra[:, :n_times, :]   # trim back to original length
        del wpt, raw_data, raw_data_padded  # free memory early


        # ------------------------------------------------------------------
        # 5. Compute per-band epoch durations
        # ------------------------------------------------------------------
        epoch_durations = compute_epoch_sizes_per_band(
            freq_bands, self.epoch_size_in_cycles, sfreq, n_times
        )

        import time

        # ------------------------------------------------------------------
        # 6. Fit each wavelet band on its own frequency-appropriate epochs
        # ------------------------------------------------------------------
        _pass_through = dict(
            threshold=0.0,
            reference_cov=_ref_cov_arr,
            epochs_eigenvalues=np.array([]),
            sensai_runs=[],
            ignore=True,
        )

        wavelets_fits = []
        for w, ((fmin, fmax), band_duration) in enumerate(
            zip(freq_bands, epoch_durations)
        ):
            base_fit = {"band_index": w, "fmin": fmin, "fmax": fmax}

            # --- Skip: data too short for the required epoch ---
            if band_duration is None:
                print(
                    f"  Band {w} ({fmin:.2f}–{fmax:.2f} Hz): data too short "
                    f"for {self.epoch_size_in_cycles} cycles — skipping.",
                    flush=True,
                )
                wavelets_fits.append({**base_fit, **_pass_through})
                continue

            # --- Skip: below the low-frequency cutoff ---
            if self.wavelet_low_cutoff is not None and fmax < self.wavelet_low_cutoff:
                print(
                    f"  Band {w} ({fmin:.2f}–{fmax:.2f} Hz): zeroed "
                    f"(below {self.wavelet_low_cutoff} Hz cutoff).",
                    flush=True,
                )
                wavelets_fits.append({**base_fit, **_pass_through})
                continue

            n_epoch_samples = int(round(band_duration * sfreq))
            step = max(1, int(n_epoch_samples * (1.0 - overlap)))
            n_epochs_approx = max(1, (n_times - n_epoch_samples) // step + 1)

            epoch_duration_s = n_epoch_samples / sfreq
            print(
                f"  Band {w} ({fmin:.2f}–{fmax:.2f} Hz): "
                f"{epoch_duration_s:.2f} s/epoch, ~{n_epochs_approx} epochs... ",
                end="", flush=True,
            )
            _t0 = time.time()

            # Extract this band's MRA: (n_times, n_channels) → (n_channels, n_times)
            band_signal = mra[w].T  # (n_channels, n_times)

            # Sliding-window epoch slicing directly on the MRA band signal
            starts = list(range(0, n_times - n_epoch_samples, step))
            # Ensure the very last segment is always included
            if not starts or starts[-1] + n_epoch_samples < n_times:
                starts.append(n_times - n_epoch_samples)

            wavelet_epochs_data = np.stack(
                [band_signal[:, s: s + n_epoch_samples] for s in starts],
                axis=0,
            )  # (n_epochs, n_channels, n_epoch_samples)

            wavelet_fit = self._fit_single_band(
                wavelet_epochs_data=wavelet_epochs_data,
                band_index=w,
                fmin=fmin,
                fmax=fmax,
                epochs_info=bb_epochs.info,
                epochs_tmin=0.0,
                reference_cov=_ref_cov_arr,
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
                signal_type=signal_type,
            )
            wavelets_fits.append(wavelet_fit)
            print(f"done ({time.time() - _t0:.1f} s)", flush=True)

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

        # ------------------------------------------------------------------
        # Fast path for spectral GEDAI: single MODWT on full signal
        # ------------------------------------------------------------------
        if self.epoch_size_in_cycles is not None:
            return self._transform_raw_frequency_specific(raw, n_jobs=n_jobs, verbose=verbose)

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

    def _transform_raw_frequency_specific(
        self,
        raw: BaseRaw,
        n_jobs: int = None,
        verbose=None,
    ):
        """Fast transform path for spectral GEDAI (mirrors _fit_raw_frequency_specific).

        Runs a **single** MODWT on the full signal, applies the fitted per-band
        GEDAI filter sample-by-sample in MRA space, then sums the bands to
        reconstruct the cleaned signal.  This avoids the O(n_windows × SWT)
        cost of the naive sliding-window approach.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to transform (should be the same object passed to fit_raw).
        n_jobs : int
            Number of parallel jobs (currently unused; reserved for future use).
        verbose : str | None
            Verbosity.

        Returns
        -------
        raw_corrected_obj : mne.io.BaseRaw
            A copy of *raw* with the cleaned data.
        """
        from ..wavelet._modwt import modwt, modwtmra

        if not hasattr(self, "wavelets_fits"):
            raise RuntimeError(
                "Model has not been fitted yet. Call fit_raw() first."
            )

        sfreq = raw.info["sfreq"]
        raw_data = raw.get_data(verbose=False)   # (n_channels, n_times)
        n_channels, n_times = raw_data.shape
        level = self.wavelet_level

        # ------------------------------------------------------------------
        # 1. Pad → MODWT → MRA  (same logic as in fit)
        # ------------------------------------------------------------------
        divisor = 2 ** level
        pad_to = int(np.ceil(n_times / divisor) * divisor)
        if pad_to > n_times:
            raw_data_padded = np.pad(
                raw_data, ((0, 0), (0, pad_to - n_times)), mode="edge"
            )
        else:
            raw_data_padded = raw_data

        print(
            f"[transform_raw] Running single MODWT (level {level}) on full signal "
            f"({n_times} samples × {n_channels} channels)...",
            flush=True,
        )
        wpt = modwt(raw_data_padded.T, self.wavelet_type, level)  # (n_bands, pad_to, n_ch)
        mra = modwtmra(wpt, self.wavelet_type)                     # (n_bands, pad_to, n_ch)
        mra = mra[:, :n_times, :]   # trim back to original length
        del wpt, raw_data_padded

        # ------------------------------------------------------------------
        # 2. Apply per-band GEDAI filter directly in MRA space
        # ------------------------------------------------------------------
        # For each band we apply the fitted threshold to every sample of that
        # band's MRA signal.  We use a single "epoch" = the entire band signal
        # so that the spatial filter is estimated globally (consistent with how
        # the broadband fixed-duration path works for transform, and simple).
        # A more faithful approach would be to use overlapping windows per band,
        # but the spatial filter only removes *spatial* artefact directions and
        # does not change with time, so a single global application is exact.

        cleaned_mra = mra.copy()  # (n_bands, n_times, n_channels)

        print("[transform_raw] Applying per-band GEDAI filter...", flush=True)
        for wavelet_fit in self.wavelets_fits:
            w = wavelet_fit["band_index"]
            ignore = wavelet_fit["ignore"]

            if ignore:
                cleaned_mra[w] = 0.0
                continue

            reference_cov = wavelet_fit["reference_cov"]
            threshold = wavelet_fit["threshold"]

            # band_signal: (n_channels, n_times)
            band_signal = mra[w].T
            cleaned_band = _process_single_epoch(band_signal, reference_cov, threshold)
            cleaned_mra[w] = cleaned_band.T   # back to (n_times, n_channels)

            print(
                f"  Band {w} ({wavelet_fit['fmin']:.2f}–{wavelet_fit['fmax']:.2f} Hz): done",
                flush=True,
            )

        # ------------------------------------------------------------------
        # 3. Sum bands → reconstructed clean signal
        # ------------------------------------------------------------------
        # cleaned_mra : (n_bands, n_times, n_channels)
        raw_corrected = np.sum(cleaned_mra, axis=0).T   # (n_channels, n_times)
        del mra, cleaned_mra

        raw_corrected_obj = raw.copy()
        raw_corrected_obj._data = raw_corrected
        print("[transform_raw] Done.", flush=True)
        return raw_corrected_obj

    @fill_doc
    def fit_transform_raw(
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
        """Fit and transform raw data in a single MODWT pass (spectral mode only).

        This method is functionally equivalent to calling :meth:`fit_raw` followed
        by :meth:`transform_raw`, but is significantly faster and more memory-efficient
        when ``epoch_size_in_cycles`` is set: the MODWT is computed **once** and each
        wavelet band is fitted and applied immediately before moving on to the next
        band.  The cleaned bands are accumulated directly into the output array, so no
        full ``cleaned_mra`` copy is ever allocated.

        For the fixed-duration (broadband) path this method simply delegates to
        :meth:`fit_raw` followed by :meth:`transform_raw` since the savings do not
        apply there.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The raw data to fit and transform.
        duration : float
            Epoch duration in seconds (default 1.0). Ignored when
            ``epoch_size_in_cycles`` is set.
        overlap : float
            Overlap ratio between consecutive epochs (0–1). Default is 0.5.
        reject_by_annotation : bool
            Whether to drop annotated segments when making broadband epochs.
            Default is False.
        %(reference_cov)s
        %(sensai_method)s
        %(noise_multiplier)s
        %(n_jobs)s
        %(verbose)s

        Returns
        -------
        raw_corrected : mne.io.BaseRaw
            A copy of *raw* with the cleaned data injected.

        Notes
        -----
        After this call :attr:`wavelets_fits` is populated exactly as after
        :meth:`fit_raw`, so :meth:`plot_fit` and a subsequent :meth:`transform_raw`
        on a different dataset will work as expected.
        """
        check_type(raw, (BaseRaw,), "raw")
        check_type(duration, (float, int), "duration")
        check_type(overlap, (float, int), "overlap")
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
        check_type(reject_by_annotation, (bool,), "reject_by_annotation")
        _check_reference_cov(reference_cov)
        _check_sensai_method(sensai_method)
        check_type(noise_multiplier, (float,), "noise_multiplier")
        n_jobs = _check_n_jobs(n_jobs)

        # Resolve signal type (auto-detect from channel types if needed)
        signal_type = self.signal_type
        if signal_type == "auto":
            signal_type = _detect_signal_type(raw.info)
            logger.info(f"Auto-detected signal type: '{signal_type}'.")

        # ------------------------------------------------------------------
        # Broadband path: no single-pass savings — just delegate
        # ------------------------------------------------------------------
        if self.epoch_size_in_cycles is None:
            self.fit_raw(
                raw,
                duration=duration,
                overlap=overlap,
                reject_by_annotation=reject_by_annotation,
                reference_cov=reference_cov,
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            return self.transform_raw(raw, duration=duration, overlap=overlap,
                                      n_jobs=n_jobs, verbose=verbose)

        # ------------------------------------------------------------------
        # Spectral path: single MODWT, fit+transform per band
        # ------------------------------------------------------------------
        return self._fit_transform_raw_frequency_specific(
            raw=raw,
            broadband_duration=duration,
            overlap=overlap,
            reject_by_annotation=reject_by_annotation,
            reference_cov=reference_cov,
            sensai_method=sensai_method,
            noise_multiplier=noise_multiplier,
            n_jobs=n_jobs,
            signal_type=signal_type,
        )

    def _fit_transform_raw_frequency_specific(
        self,
        raw: BaseRaw,
        broadband_duration: float,
        overlap: float,
        reject_by_annotation: bool,
        reference_cov,
        sensai_method: str,
        noise_multiplier: float,
        n_jobs: int,
        signal_type: str = "meg",
    ):
        """Single-pass fit + transform for spectral GEDAI.

        Runs MODWT **once**, then for each wavelet band:
        1. Fits the GEDAI spatial filter on frequency-appropriate epoch slices.
        2. Immediately applies the filter to the full band MRA signal.
        3. Accumulates the cleaned band directly into the output array.

        The band's MRA slice is zeroed in-place after accumulation to reduce
        the active memory footprint as the loop progresses.

        Parameters
        ----------
        raw : mne.io.BaseRaw
        broadband_duration : float
        overlap : float
        reject_by_annotation : bool
        reference_cov : np.ndarray or str (already resolved)
        sensai_method : str
        noise_multiplier : float
        n_jobs : int
        """
        import time
        from ..wavelet._modwt import modwt, modwtmra

        sfreq = raw.info["sfreq"]
        n_times = raw.n_times
        level = self._resolve_wavelet_level(sfreq)

        # ------------------------------------------------------------------
        # Step 0: MODWT high-pass pre-processing
        # ------------------------------------------------------------------
        if self.highpass_cutoff is not None:
            raw = self._modwt_highpass(raw)

        # ------------------------------------------------------------------
        # 1. Average reference for EEG only (MEG skips this step)
        # ------------------------------------------------------------------
        raw_work = raw
        if signal_type == "eeg" and "eeg" in raw.get_channel_types():
            logger.info("Setting average reference to match leadfield/forward reference.")
            raw_work = raw.copy().load_data()
            raw_work.set_eeg_reference("average", projection=False)

        # ------------------------------------------------------------------
        # 2. Resolve reference covariance (broadband fixed-length epochs)
        # ------------------------------------------------------------------
        _ref_cov = reference_cov
        if isinstance(_ref_cov, str) and _ref_cov == "leadfield":
            _ref_cov = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "../data/fsavLEADFIELD_4_GEDAI.mat"
                )
            )

        bb_epochs = mne.make_fixed_length_epochs(
            raw_work,
            duration=broadband_duration,
            overlap=broadband_duration * overlap,
            reject_by_annotation=reject_by_annotation,
            preload=True,
            verbose=False,
        )
        from .covariances import _compute_refcov
        _ref_cov_arr, _ = _compute_refcov(bb_epochs, _ref_cov)
        avg_diag_power = np.trace(_ref_cov_arr) / _ref_cov_arr.shape[0]
        regularization_lambda = 0.05
        epsilon = regularization_lambda * avg_diag_power
        _ref_cov_arr += epsilon * np.eye(_ref_cov_arr.shape[0])

        # ------------------------------------------------------------------
        # 2b. Preliminary broadband GEDAI pass (mirrors MATLAB GEDAI.m)
        # ------------------------------------------------------------------
        _bb_enova = None
        if self.preliminary_broadband_noise_multiplier is not None:
            print(
                f"[GEDAI] Preliminary broadband pass "
                f"(noise_multiplier={noise_multiplier:.1f})...",
                flush=True,
            )
            _bb_data_before = raw_work.get_data()
            _bb_gedai = Gedai(
                wavelet_type=self.wavelet_type,
                wavelet_level=0,
                signal_type="meg",   # avg-ref already applied to raw_work
                highpass_cutoff=None,  # HP already applied
                preliminary_broadband_noise_multiplier=None,  # no recursion
            )
            raw_work = _bb_gedai.fit_transform_raw(
                raw_work,
                reference_cov=_ref_cov_arr,
                noise_multiplier=noise_multiplier,
                sensai_method=sensai_method,
            )
            _bb_data_after = raw_work.get_data()
            _bb_var = float(np.var(_bb_data_before))
            _bb_enova = float(np.var(_bb_data_before - _bb_data_after) / _bb_var) if _bb_var > 0 else 0.0
            del _bb_data_before, _bb_data_after

        # ------------------------------------------------------------------
        # 3. Frequency bands
        # ------------------------------------------------------------------
        # freq_bands matches MRA band order:
        #   [D_1(highest freq), D_2, ..., D_L(lowest freq detail), S_L(approx)]
        freq_bands = []
        for k in range(1, level + 1):
            freq_bands.append((sfreq / 2 ** (k + 1), sfreq / 2 ** k))
        freq_bands.append((0.0, sfreq / 2 ** (level + 1)))
        self.levels_used = level

        # ------------------------------------------------------------------
        # 4. Single MODWT pass
        # ------------------------------------------------------------------
        raw_data = raw_work.get_data()          # (n_channels, n_times)
        n_channels = raw_data.shape[0]

        divisor = 2 ** level
        pad_to = int(np.ceil(n_times / divisor) * divisor)
        if pad_to > n_times:
            print(
                f"Running single MODWT (level {level}) on full signal "
                f"({n_times} samples × {n_channels} channels)...\n"
                f"  Padding signal from {n_times} to {pad_to} samples "
                f"(nearest multiple of 2^{level}={divisor}) for SWT.",
                flush=True,
            )
            raw_data_padded = np.pad(
                raw_data, ((0, 0), (0, pad_to - n_times)), mode="edge"
            )
        else:
            print(
                f"Running single MODWT (level {level}) on full signal "
                f"({n_times} samples × {n_channels} channels)...",
                flush=True,
            )
            raw_data_padded = raw_data

        wpt = modwt(raw_data_padded.T, self.wavelet_type, level)  # (n_bands, pad_to, n_ch)
        mra = modwtmra(wpt, self.wavelet_type)                     # (n_bands, pad_to, n_ch)
        mra = mra[:, :n_times, :]   # trim to original length: (n_bands, n_times, n_ch)
        del wpt, raw_data, raw_data_padded  # free raw arrays early

        # ------------------------------------------------------------------
        # 5. Per-band epoch sizes
        # ------------------------------------------------------------------
        epoch_durations = compute_epoch_sizes_per_band(
            freq_bands, self.epoch_size_in_cycles, sfreq, n_times
        )



        # ------------------------------------------------------------------
        # 6. Fit + transform each band, accumulate directly into output
        # ------------------------------------------------------------------
        _pass_through = dict(
            threshold=0.0,
            reference_cov=_ref_cov_arr,
            epochs_eigenvalues=np.array([]),
            sensai_runs=[],
            ignore=True,
        )

        wavelets_fits = []
        # Accumulate cleaned bands directly — no cleaned_mra copy needed
        raw_corrected = np.zeros((n_channels, n_times), dtype=mra.dtype)
        _band_table = []  # (band_idx, fmin, fmax, epoch_s, enova, status)

        for w, ((fmin, fmax), band_duration) in enumerate(
            zip(freq_bands, epoch_durations)
        ):
            base_fit = {"band_index": w, "fmin": fmin, "fmax": fmax}

            # --- Skip: data too short ---
            if band_duration is None:
                print(
                    f"  Band {w} ({fmin:.2f}-{fmax:.2f} Hz): data too short "
                    f"for {self.epoch_size_in_cycles} cycles - skipping.",
                    flush=True,
                )
                wavelets_fits.append({**base_fit, **_pass_through})
                _band_table.append((w, fmin, fmax, None, None, "too short"))
                mra[w] = 0.0  # free working memory in-place
                continue

            # --- Skip: below low-frequency cutoff (zeroed band) ---
            if self.wavelet_low_cutoff is not None and fmax < self.wavelet_low_cutoff:
                print(
                    f"  Band {w} ({fmin:.2f}-{fmax:.2f} Hz): zeroed "
                    f"(below {self.wavelet_low_cutoff} Hz cutoff).",
                    flush=True,
                )
                wavelets_fits.append({**base_fit, **_pass_through})
                _band_table.append((w, fmin, fmax, None, None, "zeroed"))
                mra[w] = 0.0  # zeroed band contributes nothing to output
                continue

            n_epoch_samples = int(round(band_duration * sfreq))
            step = max(1, int(n_epoch_samples * (1.0 - overlap)))
            n_epochs_approx = max(1, (n_times - n_epoch_samples) // step + 1)
            epoch_duration_s = n_epoch_samples / sfreq

            print(
                f"  Band {w} ({fmin:.2f}-{fmax:.2f} Hz): "
                f"{epoch_duration_s:.2f} s/epoch, ~{n_epochs_approx} epochs... ",
                end="", flush=True,
            )
            _t0 = time.time()

            band_signal = mra[w].T  # (n_channels, n_times) — view into mra

            # Build epoch slices for fitting
            starts = list(range(0, n_times - n_epoch_samples, step))
            if not starts or starts[-1] + n_epoch_samples < n_times:
                starts.append(n_times - n_epoch_samples)

            wavelet_epochs_data = np.stack(
                [band_signal[:, s: s + n_epoch_samples] for s in starts],
                axis=0,
            )  # (n_epochs, n_channels, n_epoch_samples)

            # --- FIT ---
            wavelet_fit = self._fit_single_band(
                wavelet_epochs_data=wavelet_epochs_data,
                band_index=w,
                fmin=fmin,
                fmax=fmax,
                epochs_info=bb_epochs.info,
                epochs_tmin=0.0,
                reference_cov=_ref_cov_arr,
                sensai_method=sensai_method,
                noise_multiplier=noise_multiplier,
                n_jobs=n_jobs,
                signal_type=signal_type,
            )
            wavelets_fits.append(wavelet_fit)
            del wavelet_epochs_data  # free epoch slices immediately

            # --- TRANSFORM: apply filter to full band signal ---
            cleaned_band = _process_single_epoch(
                band_signal, wavelet_fit["reference_cov"], wavelet_fit["threshold"]
            )

            # ENOVA: variance of removed noise / variance of original band
            band_var = float(np.var(band_signal))
            enova = float(np.var(band_signal - cleaned_band) / band_var) if band_var > 0 else 0.0
            wavelet_fit["enova"] = enova
            _band_table.append((w, fmin, fmax, epoch_duration_s, enova, "ok"))

            # Accumulate into output; then free this band's MRA slice
            raw_corrected += cleaned_band          # (n_channels, n_times)
            mra[w] = 0.0                           # zero in-place to reduce footprint
            del cleaned_band

            print(f"done ({time.time() - _t0:.1f} s)", flush=True)

        del mra  # full MRA now freed

        # ------------------------------------------------------------------
        # Summary table (mirrors MATLAB GEDAI.m per-band output)
        # ------------------------------------------------------------------
        _COL = 62
        _hdr = f"  {'Band':>4}  {'Center(Hz)':>10}  {'Range(Hz)':>17}  {'Epoch(s)':>9}  {'ENOVA(%)':>9}"
        print("\n" + "=" * _COL)
        print(_hdr)
        print("-" * _COL)
        # Broadband row
        if _bb_enova is not None:
            print(f"  {'BB':>4}  {'---':>10}  {'broadband':>17}  {'1.00':>9}  {_bb_enova * 100:>9.2f}")
        # Per-band rows
        for _w, _fmin, _fmax, _dur, _enova, _status in _band_table:
            _center = (_fmin + _fmax) / 2.0
            _range_str = f"{_fmin:.3f}-{_fmax:.3f}"
            if _status == "ok":
                print(f"  {_w:>4}  {_center:>10.3f}  {_range_str:>17}  {_dur:>9.2f}  {_enova * 100:>9.2f}")
            elif _status == "zeroed":
                print(f"  {_w:>4}  {_center:>10.3f}  {_range_str:>17}  {'---':>9}  {'zeroed':>9}")
            else:
                print(f"  {_w:>4}  {_center:>10.3f}  {_range_str:>17}  {'---':>9}  {'skipped':>9}")
        print("=" * _COL + "\n")

        self.wavelets_fits = wavelets_fits

        # ------------------------------------------------------------------
        # 7. Inject cleaned data into a copy of the original Raw object
        # ------------------------------------------------------------------
        raw_corrected_obj = raw.copy()
        raw_corrected_obj._data = raw_corrected
        print("Done.", flush=True)
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
