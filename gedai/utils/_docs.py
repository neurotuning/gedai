"""Fill docstrings to avoid redundant docstrings in multiple files.

Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from mne.utils.docs import docdict as docdict_mne

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# -- Documentation dictionary ----------------------------------------------------------
docdict: dict[str, str] = dict()

_KEYS_MNE: tuple[str, ...] = (
    "n_jobs",
    "picks_all",
    "random_state",
    "tmin_raw",
    "tmax_raw",
    "reject_by_annotation_raw",
)

for key in _KEYS_MNE:
    entry: str = docdict_mne[key]
    if ".. versionchanged::" in entry:
        entry = entry.replace(".. versionchanged::", ".. versionchanged:: MNE ")
    if ".. versionadded::" in entry:
        entry = entry.replace(".. versionadded::", ".. versionadded:: MNE ")
    docdict[key] = entry
del key

# Override n_jobs to avoid fragile intersphinx resolution of joblib in some builds.
# -- A ---------------------------------------------------------------------------------
# -- B ---------------------------------------------------------------------------------
docdict["bad_channel_threshold"] = """
bad_channel_threshold : float | str | None, default=None
    Threshold for automatic two-pass bad-channel detection and exclusion based
    on per-channel Explained Noise Variance (ENOVA), matching MATLAB GEDAI.
    If ``None`` (default) or ``np.inf``, automatic bad-channel detection is
    disabled. If ``"auto"``, uses ``0.95``. If a positive float (typically in
    ``(0, 1]``), performs a pre-pass to detect flat channels and a Pass 1
    fit/transform to flag channels whose ENOVA exceeds the threshold, then
    re-fits Pass 2 on the remaining good channels."""
docdict["bad_epoch_threshold"] = """
bad_epoch_threshold : float | None, default=None
    ENOVA threshold (e.g., ``0.85``) above which 1-second epochs are flagged
    as severely corrupted. On ``transform_raw``, appends ``"BAD_gedai"``
    annotations to the returned instance (without destructively removing samples)
    and populates ``self.bad_epochs_``. On ``transform_epochs``, populates
    boolean mask ``self.bad_epochs_``. When ``None`` (default), no bad-epoch
    annotations are added."""
# -- C ---------------------------------------------------------------------------------
docdict["compute_metrics"] = """
compute_metrics : bool, default=True
    If ``True`` (default), computes post-transform ENOVA and SENSAI quality
    metrics and stores them in ``self.metrics_``. Set to ``False`` in
    latency-critical real-time streaming loops to skip post-hoc metric
    computation and leave ``self.metrics_`` unchanged."""
docdict["cycles_per_wavelet"] = """
cycles_per_wavelet : int
    Minimum number of cycles targeted per wavelet band.
    Lower-frequency bands use longer epochs to satisfy
    this target. The default is ``12``."""
# -- D ---------------------------------------------------------------------------------
docdict["duration"] = """
duration : float
    Duration of each epoch in seconds. The default is ``1.0``."""
# -- E ---------------------------------------------------------------------------------
docdict["engine"] = """
engine : "auto" | "torch" | "numpy", default="auto"
    Computation engine for generalized eigendecomposition, batched SENSAI evaluation,
    and wavelet decomposition. If ``"auto"`` (default), uses PyTorch CPU
    acceleration when installed, and falls back cleanly to NumPy otherwise."""
# -- F ---------------------------------------------------------------------------------
# -- G ---------------------------------------------------------------------------------
# -- H ---------------------------------------------------------------------------------
# -- I ---------------------------------------------------------------------------------
docdict["interpolate_bads"] = """
interpolate_bads : bool, default=False
    If ``True`` and bad channels were excluded during fitting (recorded in
    ``self.bad_channels_``), reconstructs those channels in the output via
    MNE spherical spline interpolation (``interpolate_bads``) and, for EEG,
    re-applies an average reference across the restored montage. Requires
    valid 3D channel locations on the input instance."""
# -- J ---------------------------------------------------------------------------------
# -- K ---------------------------------------------------------------------------------
# -- L ---------------------------------------------------------------------------------
# -- M ---------------------------------------------------------------------------------
# -- N ---------------------------------------------------------------------------------
docdict["n_jobs"] = """
n_jobs : int | None
    The number of jobs to run in parallel. If ``1`` or ``None`` (default), computations
    are run serially. If ``-1``, all available CPU cores are used."""
docdict["n_pc"] = """
n_pc : int | "auto", default="auto"
    The number of principal components to retain for the SENSAI artifact template
    subspace similarity evaluation.
    If ``"auto"`` (default):

    - For **EEG**: Automatically uses ``3`` principal components for rank-normalized
      3D volumetric dipolar subspace similarity.
    - For **MEG**: Automatically prescans the GEVD artifact eigenvalue spectrum
      to select ``2`` or ``3`` principal components adaptively depending on
      ambient room noise harmonics versus localized bursts.

    If an :class:`int`, uses the specified number of principal components."""
# -- O ---------------------------------------------------------------------------------
docdict["on_bad_channel"] = """
on_bad_channel : "exclude" | "ignore", default="exclude"
    How to handle channels marked in ``inst.info['bads']`` when fitting:

    - ``"exclude"`` (default): Drops channels listed in ``info['bads']`` from
      the fit (even if explicitly named in ``picks``) and logs a warning.
      At transform time, raises a ``ValueError`` if any fitted channel in
      ``self.ch_names`` is marked bad on the input instance.
    - ``"ignore"``: Clears ``info['bads']`` on an internal copy of the data
      and logs a warning so all selected channels are average-referenced and
      processed together during both fitting and transformation."""
docdict["overlap"] = """
overlap : float
    The overlap ratio between consecutive epochs, between ``0`` and ``1``.
    The default is ``0.5`` (50%% overlap). For example, ``0.5`` means 50%%
    overlap and ``0.75`` means 75%% overlap."""
# -- P ---------------------------------------------------------------------------------
docdict["picks"] = """
picks : str | list | slice
    Channels to include. Note that all channels selected must have the same
    type. Slices and lists of integers will be interpreted as channel indices.
    In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the given
    channels. Can also be the string values ``"all"`` to pick all channels, or
    ``"data"`` to pick data channels. The default is ``"eeg"`` to pick all
    EEG channels."""
# -- Q ---------------------------------------------------------------------------------
# -- R ---------------------------------------------------------------------------------
docdict["reference"] = """
reference : str | None, default="average"
    EEG reference convention applied consistently to both data and leadfield /
    reference covariance (ignored for MEG):

    - ``"average"`` (default): Average reference.
    - ``"REST"``: Reference Electrode Standardization Technique (infinity reference;
      requires a leadfield/Forward).
    - ``"M1+M2"`` / ``"TP9+TP10"`` / ``"A1+A2"``: Linked mastoids / earlobes.
    - ``"none"``: No re-referencing.
    - Any channel label (e.g., ``"Cz"``): Re-reference to that channel (or adjust the
      leadfield directly if it is an external 10-05 reference not in the montage)."""
docdict["reference_cov"] = """
reference_cov : str | mne.Covariance | mne.Forward
    The reference covariance to use. If ``'leadfield'``, uses the bundled 343-channel
    OpenMEEG BEM covariance (with automatic 3D dipole-kernel projection fallback when
    electrode labels are non-10-20 and 3D coordinates are present). If
    ``'projection'``, always computes 3D dipole-kernel projection from channel
    positions. Also accepts an :class:`mne.Covariance` or :class:`mne.Forward`.
    See :func:`~gedai.covariance.compute_covariance_from_forward` and
    :func:`~gedai.covariance.compute_covariance_from_dipole_projection`."""
docdict["reject_by_annotation"] = """
reject_by_annotation : bool
    Whether annotated bad segments should be rejected.
    It is recommended to set this to ``False`` for fitting, since the
    algorithm needs to learn from bad segments.
    The default is ``False``."""
# -- S ---------------------------------------------------------------------------------
docdict["sensai_method"] = """
sensai_method : str
    The method to use for threshold optimization.
    Can be ``'optimize'`` (default, continuous scalar minimization) or
    ``'gridsearch'``.
"""
docdict["sensai_tol"] = """
sensai_tol : float
    Absolute tolerance (``xatol``) for threshold optimization when
    ``sensai_method='optimize'``. Defaults to ``0.1`` for fast convergence.
    Can be set to smaller values (e.g. ``0.01``) for finer search precision.
"""
docdict["smoothing_window"] = """
smoothing_window : float | None, default=None
    Duration in seconds of overlapping calibration windows for sliding-window
    threshold adaptation during ``fit_raw``, intended for long recordings with
    non-stationary background EEG signal (e.g., sleep, epilepsy, fatigue).
    Thresholds vary smoothly across the recording via Akima (makima)
    interpolation. When ``None`` or ``np.inf`` (default), fits a single
    stationary threshold."""
docdict["strength"] = """
strength : float | str, default="auto"
    Denoising strength balancing artifact suppression and neural preservation
    via closed-form adaptive Z3 spectral weighting (higher = stronger cleaning):

    - ``"auto"`` (default) : Standard balance (``7.0``).
    - ``"auto+"`` : More aggressive denoising (``8.0``).
    - ``"auto-"`` : More conservative denoising (``6.0``).
    - ``float`` in ``[0, 10]``: Continuous perceptual strength (higher = stronger
      cleaning, matching MATLAB GEDAI's log-linearized ``0``-``10`` scale)."""
# -- T ---------------------------------------------------------------------------------
# -- U ---------------------------------------------------------------------------------
# -- V ---------------------------------------------------------------------------------
docdict["verbose"] = """
verbose : int | str | bool | None
    Control verbosity of the logging output. If ``None``, use the default
    verbosity level (``"INFO"``).
    If a boolean, ``True`` corresponds to ``"INFO"`` and ``False`` corresponds
    to ``"WARNING"``.
    The default is ``None`` (interpreted as ``"INFO"``)."""

# -- W ---------------------------------------------------------------------------------
docdict["wavelet_level"] = """
wavelet_level : int
    Wavelet decomposition level. Must be greater than ``0``.
    The default is ``4``."""
docdict["wavelet_low_cutoff"] = """
wavelet_low_cutoff : float | str | None, default=0.1
    Low-frequency cutoff in Hz for wavelet high-pass pre-filtering and subband
    exclusion. Pre-filtering and global channel mean centering apply only to
    Raw inputs. Epochs inputs use this parameter for decomposition-level
    selection and subband exclusion, but receive neither preprocessing step;
    at ``0.1`` Hz, all epoch bands are retained and sub-0.1 Hz drift is not
    removed before broadband fitting. The modes below describe Raw behavior:

    - ``0.1`` (default): Applies a 0.1 Hz Haar wavelet high-pass pre-filter before
      broadband GED estimation while anchoring spectral MRA decomposition at 0.5 Hz
      and retaining/spatially cleaning the 0.1-0.5 Hz scaling band. Recommended for
      general-purpose EEG/MEG and standard ERP studies.
    - ``0.0`` or ``None``: Disables wavelet high-pass pre-filtering completely
      (applying only global channel mean centering) and retains all spectral bands
      down to DC. Recommended for pre-stimulus baseline / anticipatory ERP analyses
      (e.g., CNV, RP, LPP) or when data have already been causally high-pass filtered.
    - ``>= 0.5`` (e.g., ``0.5``): Applies a sub-cutoff wavelet high-pass pre-filter
      and zeroes out all spectral MRA bands with upper frequency bound
      ``<= wavelet_low_cutoff``. Recommended for resting-state recordings with
      severe slow electrode drift.
    - ``"auto"``: Automatically resolves to ``max(0.1, raw.info['highpass'])`` when
      ``raw.info['highpass'] > 0``, else ``0.1`` Hz."""
docdict["wavelet_type"] = """
wavelet_type : str
    Wavelet to use for the decomposition. The default is ``'haar'``.
    See :py:func:`pywt.wavedec` for the list of available wavelets."""
# -- X ---------------------------------------------------------------------------------
# -- Y ---------------------------------------------------------------------------------
# -- Z ---------------------------------------------------------------------------------
# -- Documentation functions -----------------------------------------------------------
docdict_indented: dict[int, dict[str, str]] = dict()


def fill_doc(f: Callable[..., Any]) -> Callable[..., Any]:
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of (modified in place).

    Returns
    -------
    f : callable
        The function, potentially with an updated __doc__.
    """
    docstring = f.__doc__
    if not docstring:
        return f

    lines = docstring.splitlines()
    indent_count = _indentcount_lines(lines)

    try:
        indented = docdict_indented[indent_count]
    except KeyError:
        indent = " " * indent_count
        docdict_indented[indent_count] = indented = dict()

        for name, docstr in docdict.items():
            lines = [
                indent + line if k != 0 else line
                for k, line in enumerate(docstr.strip().splitlines())
            ]
            indented[name] = "\n".join(lines)

    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")

    return f


def _indentcount_lines(lines: list[str]) -> int:
    """Minimum indent for all lines in line list.

    >>> lines = [" one", "  two", "   three"]
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [" one"]
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(["    "])
    0
    """
    indent = sys.maxsize
    for k, line in enumerate(lines):
        if k == 0:
            continue
        line_stripped = line.lstrip()
        if line_stripped:
            indent = min(indent, len(line) - len(line_stripped))
    return indent


def copy_doc(source: Callable[..., Any]) -> Callable[..., Any]:
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the function
    wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This decorator
    can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : callable
        The function to copy the docstring from.

    Returns
    -------
    wrapper : callable
        The decorated function.

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B(A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         '''this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise RuntimeError(
                f"The docstring from {source.__name__} could not be copied because it "
                "was empty."
            )
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func

    return wrapper
