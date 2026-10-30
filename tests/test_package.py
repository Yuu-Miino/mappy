"""Package-level tests: public API and documentation examples."""

from __future__ import annotations

import doctest
import importlib

import pytest

import mappy_ds
from mappy_ds import MappyError, ModeNotTerminatedError, SymbolicEvaluationError


def test_version_is_string():
    assert isinstance(mappy_ds.__version__, str)
    assert mappy_ds.__version__


def test_all_is_importable():
    assert len(set(mappy_ds.__all__)) == len(mappy_ds.__all__)
    for name in mappy_ds.__all__:
        assert hasattr(mappy_ds, name), name
    namespace: dict[str, object] = {}
    exec("from mappy_ds import *", namespace)
    assert set(mappy_ds.__all__) <= set(namespace)


def test_exception_hierarchy():
    for exc in (
        ModeNotTerminatedError,
        SymbolicEvaluationError,
        mappy_ds.TransitionError,
        mappy_ds.UnknownModeError,
    ):
        assert issubclass(exc, MappyError)


@pytest.mark.parametrize(
    "module",
    [
        "mappy_ds",
        "mappy_ds.functions",
        "mappy_ds.maps",
        "mappy_ds.modes",
        "mappy_ds.trajectory",
        "mappy_ds.exceptions",
        "mappy_ds._typing",
    ],
)
def test_doctests(module):
    mod = importlib.import_module(module)
    result = doctest.testmod(
        mod, optionflags=doctest.NORMALIZE_WHITESPACE, verbose=False, report=True
    )
    assert result.failed == 0
