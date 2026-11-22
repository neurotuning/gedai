from pathlib import Path
from typing import Any

from _typeshed import Incomplete

from ._docs import fill_doc as fill_doc

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

class _IntLike:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool: ...

class _Callable:
    @classmethod
    def __instancecheck__(cls, other: Any) -> bool: ...

_types: Incomplete

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

def _check_picks_uniqueness(info, picks) -> None:
    """Check that the provided picks yield a single channel type."""

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

def _ensure_verbose(verbose: Any) -> int:
    """Ensure that the value of verbose is valid.

    Parameters
    ----------
    verbose : int | str | bool | None
    Control verbosity of the logging output. If ``None``, use the default
    verbosity level (``"INFO"``).
    If a boolean, ``True`` corresponds to ``"INFO"`` and ``False`` corresponds
    to ``"WARNING"``.
    The default is ``None`` (interpreted as ``"INFO"``).

    Returns
    -------
    verbose : int
    The verbosity level as an integer.
    """

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

def _ensure_int(x, name: str = "unknown", must_be: str = "an int", *, extra: str = ""):
    """Ensure a variable is an integer."""

def _check_n_jobs(n_jobs):
    """Check n_jobs parameter.

    Check that n_jobs is a positive integer or a negative integer for all cores. CUDA is
    not supported.
    """

_STRENGTH_KNOTS: Incomplete
_LOG10_SC_KNOTS: Incomplete

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
