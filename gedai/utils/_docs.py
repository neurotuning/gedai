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
# -- C ---------------------------------------------------------------------------------
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
# -- F ---------------------------------------------------------------------------------
# -- G ---------------------------------------------------------------------------------
# -- H ---------------------------------------------------------------------------------
# -- I ---------------------------------------------------------------------------------
# -- J ---------------------------------------------------------------------------------
# -- K ---------------------------------------------------------------------------------
# -- L ---------------------------------------------------------------------------------
# -- M ---------------------------------------------------------------------------------
# -- N ---------------------------------------------------------------------------------
docdict["n_jobs"] = """
n_jobs : int | None
    The number of jobs to run in parallel. If ``1`` or ``None`` (default), computations
    are run serially. If ``-1``, all available CPU cores are used."""
docdict["noise_multiplier"] = """
noise_multiplier : float | str
    The noise multiplier or string preset for artefact threshold rejection optimization.
    Supported string presets:
    - ``"auto"`` : Standard balance (noise_multiplier = 3.0, default).
    - ``"auto+"`` : More aggressive denoising (noise_multiplier = 1.5).
    - ``"auto-"`` : More conservative denoising (noise_multiplier = 6.0).
    Alternatively, a custom numerical float can be passed."""
# -- O ---------------------------------------------------------------------------------
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
docdict["reference_cov"] = """
reference_cov : str | mne.Covariance
    The reference covariance to use. If ``'leadfield'``, use a pre-computed covariance.
    The precomputed covariance if computed from a leadfield made using 1005 EEG channels
    layout and fsaverage head model.
    If :class:`mne.Covariance`, use a pre-computed covariance.
    See :func:`~gedai.covariance.compute_covariance_from_forward` for more
    details on how to compute a covariance from a forward solution."""
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
# -- T ---------------------------------------------------------------------------------
# -- U ---------------------------------------------------------------------------------
# -- V ---------------------------------------------------------------------------------
docdict["verbose"] = """
verbose : int | str | bool | None
    Sets the verbosity level. The verbosity increases gradually between ``"CRITICAL"``,
    ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``. If None is provided, the
    verbosity is set to ``"WARNING"``. If a bool is provided, the verbosity is set to
    ``"WARNING"`` for False and to ``"INFO"`` for True."""

# -- W ---------------------------------------------------------------------------------
docdict["wavelet_level"] = """
wavelet_level : int
    Wavelet decomposition level. Must be greater than ``0``.
    The default is ``4``."""
docdict["wavelet_low_cutoff"] = """
wavelet_low_cutoff : float | None
    If a float is provided, zero out all wavelet levels whose upper frequency
    bound is below this cutoff frequency in Hz. If ``None``, no frequency band
    is zeroed out. If ``"auto"``, the cutoff is automatically determined based
    on the info['highpass'] value of the fitted instance. While reading data
    from a file, info['highpass'] might be missing (i.e., equal to 0.0). If
    you know that your data has been high-pass filtered, make sure to set
    ``wavelet_low_cutoff`` to the high-pass cutoff frequency.
    The default is ``"auto"``."""
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
