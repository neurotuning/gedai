import logging
from pathlib import Path

import numpy as np
import pytest

from .._checks import (
    _check_type,
    _check_value,
    _ensure_path,
    _ensure_verbose,
    ensure_int,
)


def test_ensure_int():
    """Test ensure_int checker."""
    # valids
    assert ensure_int(101) == 101

    # invalids
    with pytest.raises(TypeError, match="Item must be an int"):
        ensure_int(101.0)
    with pytest.raises(TypeError, match="Item must be an int"):
        ensure_int(True)
    with pytest.raises(TypeError, match="Item must be an int"):
        ensure_int([101])


def test_check_type():
    """Test _check_type checker."""
    # valids
    _check_type(101, ("int-like",))
    _check_type(101, ("int-like", str))
    _check_type("101.fif", ("path-like",))

    def foo():
        pass

    _check_type(foo, ("callable",))

    _check_type(101, ("numeric",))
    _check_type(101.0, ("numeric",))
    _check_type((1, 0, 1), ("array-like",))
    _check_type([1, 0, 1], ("array-like",))
    _check_type(np.array([1, 0, 1]), ("array-like",))

    # invalids
    with pytest.raises(TypeError, match="Item must be an instance of"):
        _check_type(101, (float,))
    with pytest.raises(TypeError, match="Item must be an instance of"):
        _check_type(101, ("array-like",))
    with pytest.raises(TypeError, match="'number' must be an instance of"):
        _check_type(101, (float,), "number")


def test__check_value():
    """Test _check_value checker."""
    # valids
    _check_value(5, (5,))
    _check_value(5, (5, 101))
    _check_value(5, [1, 2, 3, 4, 5])
    _check_value((1, 2), [(1, 2), (2, 3, 4, 5)])

    # invalids
    with pytest.raises(ValueError, match="Invalid value for the parameter."):
        _check_value(5, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="Invalid value for the 'number' parameter."):
        _check_value(5, [1, 2, 3, 4], "number")


def test__ensure_verbose():
    """Test _ensure_verbose checker."""
    # valids
    assert _ensure_verbose(12) == 12
    assert _ensure_verbose("INFO") == logging.INFO
    assert _ensure_verbose("DEBUG") == logging.DEBUG
    assert _ensure_verbose(True) == logging.INFO
    assert _ensure_verbose(False) == logging.WARNING
    assert _ensure_verbose(None) == logging.WARNING

    # invalids
    with pytest.raises(TypeError, match="must be an instance of"):
        _ensure_verbose(("INFO",))
    with pytest.raises(ValueError, match="Invalid value"):
        _ensure_verbose("101")
    with pytest.raises(ValueError, match="negative integer, -101 is invalid."):
        _ensure_verbose(-101)


def test_ensure_path():
    """Test ensure_path checker."""
    # valids
    cwd = Path.cwd()
    path = _ensure_path(cwd, must_exist=False)
    assert isinstance(path, Path)
    path = _ensure_path(cwd, must_exist=True)
    assert isinstance(path, Path)
    path = _ensure_path(str(cwd), must_exist=False)
    assert isinstance(path, Path)
    path = _ensure_path(str(cwd), must_exist=True)
    assert isinstance(path, Path)
    path = _ensure_path("101", must_exist=False)
    assert isinstance(path, Path)

    with pytest.raises(FileNotFoundError, match="does not exist."):
        _ensure_path("101", must_exist=True)

    # invalids
    with pytest.raises(TypeError, match="'101' is invalid"):
        _ensure_path(101, must_exist=False)

    class Foo:
        def __str__(self):
            pass

    with pytest.raises(TypeError, match="path is invalid"):
        _ensure_path(Foo(), must_exist=False)
