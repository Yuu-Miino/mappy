"""Exception hierarchy of :mod:`mappy_ds`.

Every exception raised deliberately by the package derives from
:class:`MappyError`, so callers can catch the whole family at once.
"""

from __future__ import annotations

__all__ = [
    "MappyError",
    "ModeNotTerminatedError",
    "SymbolicEvaluationError",
    "TransitionError",
    "UnknownModeError",
]


class MappyError(Exception):
    """Base class of all errors raised by :mod:`mappy_ds`."""


class SymbolicEvaluationError(MappyError):
    """A system function could not be evaluated with symbolic arguments.

    System functions are traced once with SymPy symbols in order to derive
    their Jacobian and Hessian. Operations that need concrete numbers
    (``numpy.cos`` on symbols, ``if y[0] > 0`` branches, ...) break the trace.
    Use the corresponding ``sympy`` functions (``sympy.cos``, ...) instead.
    """


class UnknownModeError(MappyError, KeyError):
    """A mode name does not appear in the mode set of a Poincaré map."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"mode {self.name!r} is not defined"


class TransitionError(MappyError):
    """The transition rules of a Poincaré map are inconsistent.

    Raised when a map is constructed with a rule that does not fit its mode
    (missing rule, wrong number of entries, unknown border name, mismatched
    dimensions, no reachable end mode) or, at run time, when a mode ends
    through a border that has no successor.
    """

    def __init__(
        self, mode: str, border: int | str | None = None, reason: str | None = None
    ) -> None:
        super().__init__(mode, border, reason)
        self.mode = mode
        self.border = border
        self.reason = reason

    def __str__(self) -> str:
        if self.reason is not None:
            return f"transition from mode {self.mode!r}: {self.reason}"
        if self.border is None:
            return f"no transition is defined from mode {self.mode!r}"
        return (
            f"no transition is defined from mode {self.mode!r} via border "
            f"{self.border!r}"
        )


class ModeNotTerminatedError(MappyError):
    """A continuous mode reached ``max_interval`` without hitting a border."""

    def __init__(self, mode: str, max_interval: float) -> None:
        super().__init__(mode, max_interval)
        self.mode = mode
        self.max_interval = max_interval

    def __str__(self) -> str:
        return (
            f"continuous mode {self.mode!r} did not reach any border within "
            f"max_interval={self.max_interval}"
        )
