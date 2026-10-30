"""Natural-parameter continuation driven by a user-supplied step function.

:func:`continuation` sweeps one parameter linearly between two values and
calls a *step function* at each point, seeding every step with the result of
the previous one. Specialised continuations built on it are available in
:mod:`mappy_ds.root` (:func:`~mappy_ds.root.trace_cycle` and
:func:`~mappy_ds.root.trace_local_bf`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np

from ._typing import Params

__all__ = ["ContinuationStep", "StepFunctionResult", "continuation"]

T = TypeVar("T")


@dataclass
class ContinuationStep(Generic[T]):
    """One successful step of a parameter continuation.

    Attributes
    ----------
    y : T
        Value of the continued variable found at this step.
    params : dict
        Parameters at this step (a copy owned by the step).
    info : dict
        Extra data returned by the step function; empty if none.
    """

    y: T
    params: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepFunctionResult(Generic[T]):
    """Return value of a step function passed to :func:`continuation`.

    Attributes
    ----------
    success : bool
        Whether the step succeeded. The continuation stops at the first
        unsuccessful step.
    y : T or None
        Value of the continued variable found by the step; it seeds the next
        step.
    params : dict or None
        Parameters after the step, e.g. with a solved parameter updated.
        ``None`` keeps the parameters the step function was called with.
    info : dict or None
        Extra data to store in :attr:`ContinuationStep.info`.
    """

    success: bool
    y: T | None = None
    params: dict[str, Any] | None = None
    info: dict[str, Any] | None = None


def continuation(
    fun: Callable[[T, dict[str, Any]], StepFunctionResult[T]],
    y0: T,
    params: Params,
    param_key: str,
    end_value: float,
    *,
    resolution: int = 100,
    show_progress: bool = False,
    callback: Callable[[ContinuationStep[T]], None] | None = None,
) -> list[ContinuationStep[T]]:
    """Continue the solution of a step function along a parameter.

    ``params[param_key]`` is swept linearly from its current value to
    ``end_value`` in ``resolution`` points, both ends included. At each point
    ``fun(y, params)`` is called, where ``y`` and ``params`` are those of the
    previous successful step (``y0`` and ``params`` at the first point) with
    ``params[param_key]`` set to the current value. The sweep stops at the
    first unsuccessful step.

    Parameters
    ----------
    fun : callable
        Step function ``fun(y, params) -> StepFunctionResult``. It receives a
        dictionary it may modify freely.
    y0 : T
        Initial value of the continued variable.
    params : mapping
        Initial parameters; it is copied and never modified.
    param_key : str
        Key of the parameter to continue.
    end_value : float
        Final value of ``params[param_key]``.
    resolution : int, optional
        Number of continuation points including both ends, by default ``100``.
        Must be at least ``2``.
    show_progress : bool, optional
        Print one status line per successful step to standard error, by
        default ``False``.
    callback : callable, optional
        Called with each successful :class:`ContinuationStep` as soon as it is
        found.

    Returns
    -------
    list of ContinuationStep
        The successful steps in order; shorter than ``resolution`` if a step
        failed.

    Raises
    ------
    ValueError
        If ``resolution`` is less than ``2``.

    See Also
    --------
    mappy_ds.root.trace_cycle, mappy_ds.root.trace_local_bf

    Examples
    --------
    Follow the positive square root of ``p`` from ``p = 1`` to ``p = 4``:

    >>> from mappy_ds.continuation import StepFunctionResult, continuation
    >>> def step(y, p):
    ...     return StepFunctionResult(True, p["p"] ** 0.5)
    >>> steps = continuation(step, 1.0, {"p": 1.0}, "p", 4.0, resolution=4)
    >>> [(s.params["p"], s.y) for s in steps]
    [(1.0, 1.0), (2.0, 1.4142135623730951), (3.0, 1.7320508075688772), (4.0, 2.0)]
    """
    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}")

    p: dict[str, Any] = dict(params)
    values = np.linspace(float(p[param_key]), float(end_value), resolution)
    width = len(str(resolution))
    y = y0
    found: list[ContinuationStep[T]] = []

    try:
        for i, value in enumerate(values):
            p[param_key] = float(value)
            ret = fun(y, dict(p))
            if not ret.success or ret.y is None:
                break
            y = ret.y
            if ret.params is not None:
                p = dict(ret.params)
            step = ContinuationStep(
                y, dict(p), {} if ret.info is None else dict(ret.info)
            )
            found.append(step)
            if callback is not None:
                callback(step)
            if show_progress:
                print(
                    f"SUCCESS {i + 1:{width}d}/{resolution} "
                    f"{param_key}={p[param_key]:+.10g} y={y}",
                    end="\r",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if show_progress:
            print(file=sys.stderr)
    return found
