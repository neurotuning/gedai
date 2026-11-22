"""Utility functions for checking types and values. Inspired from MNE."""

from __future__ import annotations

import logging
import multiprocessing as mp
import operator
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from mne import pick_info

from ._docs import fill_doc

if TYPE_CHECKING:
    from typing import Any


def ensure_int(item: Any, item_name: str | None = None) -> int:
    """Ensure a variable is an integer.

    Parameters
    ----------
    item : Any
        Item to check.
    item_name : str | None
        Name of the item to show inside the error message.

    Returns
    -------
    item : int
        Item validated and converted to a Python integer.

    Raises
    ------
    TypeError
        When the type of the item is not int.
    """
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(item, bool):
            raise TypeError
        item = int(operator.index(item))
    except TypeError:
        item_name = "Item" if item_name is None else f"'{item_name}'"
        raise TypeError(f"{item_name} must be an integer, got {type(item)} instead.")

    return item


class _IntLike:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool:
        try:
            ensure_int(other)
        except TypeError:
            return False
        else:
            return True


class _Callable:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool:
        return callable(other)


_types = {
    "numeric": (np.floating, float, _IntLike()),
    "path-like": (str, Path, os.PathLike),
    "int-like": (_IntLike(),),
    "callable": (_Callable(),),
    "array-like": (list, tuple, set, np.ndarray),
}


def _check_type(item: Any, types: tuple, item_name: str | None = None) -> None:
    """Check that item is an instance of types.

    Parameters
    ----------
    item : object
        Item to check.
    types : tuple of types | tuple of str
        Types to be checked against.
        If str, must be one of ('int-like', 'numeric', 'path-like', 'callable',
        'array-like').
    item_name : str | None
        Name of the item to show inside the error message.

    Raises
    ------
    TypeError
        When the type of the item is not one of the valid options.
    """
    check_types = sum(
        (
            (
                (type(None),)
                if type_ is None
                else (type_,)
                if not isinstance(type_, str)
                else _types[type_]
            )
            for type_ in types
        ),
        (),
    )

    if not isinstance(item, check_types):
        type_name = [
            (
                "None"
                if cls_ is None
                else cls_.__name__
                if not isinstance(cls_, str)
                else cls_
            )
            for cls_ in types
        ]
        if len(type_name) == 1:
            type_name = type_name[0]
        elif len(type_name) == 2:
            type_name = " or ".join(type_name)
        else:
            type_name[-1] = "or " + type_name[-1]
            type_name = ", ".join(type_name)
        item_name = "Item" if item_name is None else f"'{item_name}'"
        raise TypeError(
            f"{item_name} must be an instance of {type_name}, got {type(item)} instead."
        )


def _check_picks_uniqueness(info, picks):
    """Check that the provided picks yield a single channel type."""
    info = pick_info(info, picks, copy=True)
    if len(info.get_channel_types(unique=True)) != 1:
        ch_types = info.get_channel_types(unique=False)
        ch_types, counts = np.unique(ch_types, return_counts=True)
        channels_msg = ", ".join(
            "%s '%s' channel(s)" % t  # noqa: UP031
            for t in zip(counts, ch_types, strict=False)
        )
        raise ValueError(
            f"Only one datatype can be selected, but 'picks' results in {channels_msg}."
        )


def _check_value(
    item: Any,
    allowed_values: tuple | dict[Any, Any],
    item_name: str | None = None,
    extra: str | None = None,
) -> None:
    """Check the value of a parameter against a list of valid options.

    Parameters
    ----------
    item : object
        Item to check.
    allowed_values : tuple of objects | dict of objects
        Allowed values to be checked against. If a dictionary, checks against the keys.
    item_name : str | None
        Name of the item to show inside the error message.
    extra : str | None
        Extra string to append to the invalid value sentence, e.g. "when using DC mode".

    Raises
    ------
    ValueError
        When the value of the item is not one of the valid options.
    """
    if item not in allowed_values:
        item_name = "" if item_name is None else f" '{item_name}'"
        extra = "" if extra is None else " " + extra
        msg = (
            "Invalid value for the{item_name} parameter{extra}. "
            "{options}, but got {item!r} instead."
        )
        allowed_values = tuple(allowed_values)  # e.g., if a dict was given
        if len(allowed_values) == 1:
            options = f"The only allowed value is {repr(allowed_values[0])}"
        elif len(allowed_values) == 2:
            options = (
                f"Allowed values are {repr(allowed_values[0])} "
                f"and {repr(allowed_values[1])}"
            )
        else:
            options = "Allowed values are "
            options += ", ".join([f"{repr(v)}" for v in allowed_values[:-1]])
            options += f", and {repr(allowed_values[-1])}"
        raise ValueError(
            msg.format(item_name=item_name, extra=extra, options=options, item=item)
        )


@fill_doc
def _ensure_verbose(verbose: Any) -> int:
    """Ensure that the value of verbose is valid.

    Parameters
    ----------
    %(verbose)s

    Returns
    -------
    verbose : int
        The verbosity level as an integer.
    """
    logging_types = dict(
        DEBUG=logging.DEBUG,
        INFO=logging.INFO,
        WARNING=logging.WARNING,
        ERROR=logging.ERROR,
        CRITICAL=logging.CRITICAL,
    )

    _check_type(verbose, (bool, str, "int-like", None), item_name="verbose")

    if verbose is None:
        verbose = logging.INFO
    elif isinstance(verbose, str):
        verbose = verbose.upper()
        _check_value(verbose, logging_types, item_name="verbose")
        verbose = logging_types[verbose]
    elif isinstance(verbose, bool):
        if verbose:
            verbose = logging.INFO
        else:
            verbose = logging.WARNING
    elif isinstance(verbose, int):
        verbose = ensure_int(verbose)
        if verbose <= 0:
            raise ValueError(
                f"Argument 'verbose' can not be a negative integer, {verbose} is "
                "invalid."
            )

    return verbose


def _ensure_path(item: Any, must_exist: bool) -> Path:
    """Ensure a variable is a Path.

    Parameters
    ----------
    item : Any
        Item to check.
    must_exist : bool
        If True, the path must resolve to an existing file or directory.

    Returns
    -------
    path : Path
        Path validated and converted to a pathlib.Path object.
    """
    try:
        item = Path(item)
    except TypeError:
        try:
            str_ = f"'{str(item)}' "
        except Exception:
            str_ = ""
        raise TypeError(
            f"The provided path {str_}is invalid and can not be converted. Please "
            f"provide a str, an os.PathLike or a pathlib.Path object, not {type(item)}."
        )
    if must_exist and not item.exists():
        raise FileNotFoundError(f"The provided path '{str(item)}' does not exist.")
    return item


def _ensure_int(x, name="unknown", must_be="an int", *, extra=""):
    """Ensure a variable is an integer."""
    extra = f" {extra}" if extra else extra
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError(f"{name} must be {must_be}{extra}, got {type(x)}")
    return x


def _check_n_jobs(n_jobs):
    """Check n_jobs parameter.

    Check that n_jobs is a positive integer or a negative integer for all cores. CUDA is
    not supported.
    """
    if n_jobs is None:
        n_jobs = 1
    n_jobs = _ensure_int(n_jobs, "n_jobs", must_be="an int or None")
    if n_jobs <= 0:
        n_cores = mp.cpu_count()
        n_jobs_orig = n_jobs
        n_jobs = min(n_cores + n_jobs + 1, n_cores)
        if n_jobs <= 0:
            raise ValueError(
                f"If n_jobs has a non-positive value ({n_jobs_orig}), it must "
                f"not be less than the number of CPUs present ({n_cores})."
            )
    return n_jobs


_STRENGTH_KNOTS = np.array([0.1, 6.0, 7.0, 8.0, 10.0], dtype=np.float64)
_LOG10_SC_KNOTS = np.array(
    [1.0, 0.0, -np.log10(2.0), -np.log10(3.0), -1.0], dtype=np.float64
)


def strength_to_z3_scale(strength: float) -> float:
    """Convert perceptual denoising strength in ``[0, 10]`` to a Z3 scale factor.

    Uses the MATLAB GEDAI ``GEDAI_strength_to_sc`` piecewise log-linear mapping
    where ``6.0 -> 1.0`` (``"auto-"``), ``7.0 -> 0.5`` (``"auto"``),
    ``8.0 -> 1/3`` (``"auto+"``), and ``10.0 -> 0.1``.

    Parameters
    ----------
    strength : float
        Perceptual denoising strength in ``[0, 10]`` (higher = stronger).

    Returns
    -------
    sc : float
        Adaptive Z3 scale factor multiplying ``m_eff``.
    """
    if isinstance(strength, bool):
        raise TypeError("strength must be a float in [0, 10], got bool.")
    _check_type(strength, ("numeric",), "strength")
    s_val = float(strength)
    if not np.isfinite(s_val) or not (0.0 <= s_val <= 10.0):
        raise ValueError(
            f"strength must be finite and between 0 and 10, got {strength!r}."
        )
    if s_val < _STRENGTH_KNOTS[0]:
        slope = (_LOG10_SC_KNOTS[1] - _LOG10_SC_KNOTS[0]) / (
            _STRENGTH_KNOTS[1] - _STRENGTH_KNOTS[0]
        )
        log_sc = float(_LOG10_SC_KNOTS[0] + (s_val - _STRENGTH_KNOTS[0]) * slope)
    else:
        log_sc = float(np.interp(s_val, _STRENGTH_KNOTS, _LOG10_SC_KNOTS))
    return float(10.0**log_sc)


def z3_scale_to_strength(sc: float) -> float:
    """Convert adaptive Z3 scale ``sc > 0`` to perceptual strength in ``[0, 10]``.

    Inverse of :func:`strength_to_z3_scale` matching MATLAB ``GEDAI_sc_to_strength``.

    Parameters
    ----------
    sc : float
        Positive adaptive Z3 scale factor.

    Returns
    -------
    strength : float
        Corresponding perceptual denoising strength on the ``0``-``10`` scale.
    """
    if isinstance(sc, bool):
        raise TypeError("sc must be a positive float, got bool.")
    _check_type(sc, ("numeric",), "sc")
    sc_val = float(sc)
    if not np.isfinite(sc_val) or sc_val <= 0.0:
        raise ValueError(f"sc must be finite and > 0, got {sc!r}.")
    log_sc = float(np.log10(max(sc_val, 1e-12)))
    log_sc_asc = _LOG10_SC_KNOTS[::-1]
    s_asc = _STRENGTH_KNOTS[::-1]
    if log_sc <= log_sc_asc[0]:
        slope = (s_asc[1] - s_asc[0]) / (log_sc_asc[1] - log_sc_asc[0])
        return float(s_asc[0] + (log_sc - log_sc_asc[0]) * slope)
    if log_sc >= log_sc_asc[-1]:
        slope = (s_asc[-1] - s_asc[-2]) / (log_sc_asc[-1] - log_sc_asc[-2])
        return float(s_asc[-1] + (log_sc - log_sc_asc[-1]) * slope)
    return float(np.interp(log_sc, log_sc_asc, s_asc))


def _ensure_strength(value: float | int | str = "auto") -> float:
    """Validate and resolve denoising ``strength`` on the ``[0, 10]`` scale.

    Parameters
    ----------
    value : float | int | str
        - ``'auto'`` / ``'default'`` / ``'standard'`` : ``7.0`` (standard balance).
        - ``'auto+'`` / ``'aggressive'`` / ``'more'`` : ``8.0`` (stronger cleaning).
        - ``'auto-'`` / ``'conservative'`` / ``'less'`` : ``6.0`` (gentler cleaning).
        - float / int (or numeric string) in ``[0, 10]``: continuous perceptual
          strength (higher = stronger cleaning).

    Returns
    -------
    strength : float
        Validated perceptual denoising strength in ``[0.0, 10.0]``.
    """
    if isinstance(value, bool):
        raise TypeError("strength must be a float in [0, 10] or str, got bool.")
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("auto", "default", "standard"):
            return 7.0
        if val in ("auto+", "aggressive", "more"):
            return 8.0
        if val in ("auto-", "conservative", "less"):
            return 6.0
        try:
            s_val = float(val)
        except ValueError:
            raise ValueError(
                f"Unknown strength preset: '{value}'. "
                "Expected 'auto', 'auto+', 'auto-', or a number in [0, 10]."
            )
        if not np.isfinite(s_val) or not (0.0 <= s_val <= 10.0):
            raise ValueError(
                f"strength must be finite and between 0 and 10, got {value!r}."
            )
        return float(s_val)
    if isinstance(value, (int, float)):
        s_val = float(value)
        if not np.isfinite(s_val) or not (0.0 <= s_val <= 10.0):
            raise ValueError(
                f"strength must be finite and between 0 and 10, got {value!r}."
            )
        return s_val
    raise TypeError(
        f"strength must be a float in [0, 10] or str, got {type(value).__name__}"
    )


def ensure_engine(engine: str = "numpy") -> str:
    """Validate and resolve backend engine ('numpy', 'torch', 'auto').

    Parameters
    ----------
    engine : str
        The requested computation engine ('numpy', 'torch', or 'auto').

    Returns
    -------
    resolved : str
        The resolved engine ('numpy' or 'torch').
    """
    from ._imports import import_optional_dependency

    if not isinstance(engine, str):
        raise TypeError(f"engine must be a string, got {type(engine).__name__}")

    engine_lower = engine.lower()
    if engine_lower not in ("numpy", "torch", "auto"):
        raise ValueError(
            f"Invalid engine '{engine}'. Expected one of 'numpy', 'torch', or 'auto'."
        )

    has_torch = import_optional_dependency("torch", raise_error=False) is not None
    if engine_lower == "auto":
        return "torch" if has_torch else "numpy"

    if engine_lower == "torch":
        if not has_torch:
            raise ImportError(
                "Missing optional dependency 'torch'. Use 'pip install torch' "
                "or 'pip install gedai[torch]' to run with the PyTorch engine."
            )
        return "torch"

    return "numpy"
