"""Periodic points (cycles) of maps and their continuation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import root

from .._typing import Jacobian, Params, State, StateLike, as_state
from ..continuation import ContinuationStep, StepFunctionResult, continuation
from ..maps import Map

__all__ = ["CycleResult", "find_cycle", "trace_cycle"]


@dataclass
class CycleResult:
    """Result of :func:`find_cycle`.

    Attributes
    ----------
    success : bool
        Whether the root finder converged.
    y : numpy.ndarray or None
        The periodic point, or ``None`` if the search failed.
    eigvals : numpy.ndarray or None
        Characteristic multipliers: eigenvalues (complex) of the Jacobian
        matrix of the ``period``-th iterate of the map at ``y``.
    eigvecs : numpy.ndarray or None
        Corresponding eigenvectors (complex), as columns.
    nfev : int
        Number of evaluations of the objective function.
    residual : numpy.ndarray
        :math:`T^n(y) - y` at the last iterate of the root finder.
    message : str
        Termination message of :func:`scipy.optimize.root`.
    """

    success: bool
    y: State | None
    eigvals: npt.NDArray[np.complex128] | None
    eigvecs: npt.NDArray[np.complex128] | None
    nfev: int
    residual: State
    message: str


def find_cycle(
    diffeo: Map,
    y0: StateLike,
    params: Params | None = None,
    *,
    period: int = 1,
    **root_options: Any,
) -> CycleResult:
    """Find a periodic point of a map.

    Solves :math:`T^n(y) - y = 0` with :func:`scipy.optimize.root`, where
    :math:`T` is ``map`` and :math:`n` is ``period``, using the analytic
    Jacobian :math:`DT^n(y) - I` provided by the map.

    Parameters
    ----------
    diffeo : Map
        The map, e.g. a :class:`~mappy_ds.Diffeomorphism` or a
        :class:`~mappy_ds.PoincareMap`.
    y0 : array_like
        Initial guess of the periodic point.
    params : mapping, optional
        Parameters of the map.
    period : int, optional
        Period :math:`n` of the cycle, by default ``1`` (a fixed point).
    **root_options
        Forwarded to :func:`scipy.optimize.root` (``method``, ``tol``,
        ``options``, ...).

    Returns
    -------
    CycleResult

    See Also
    --------
    trace_cycle : Continue a periodic point along a parameter.

    Examples
    --------
    Fixed point of the Hénon map:

    >>> import numpy as np
    >>> from mappy_ds import Diffeomorphism, mapping
    >>> from mappy_ds.root import find_cycle
    >>> @mapping(2, param_keys=["a", "b"])
    ... def henon(y, p):
    ...     x, u = y
    ...     return np.array([1 - p["a"] * x**2 + u, p["b"] * x])
    >>> res = find_cycle(Diffeomorphism(henon), [0.6, 0.2], {"a": 1.4, "b": 0.3})
    >>> res.success
    True
    >>> res.y.round(6)
    array([0.631354, 0.189406])
    >>> np.sort(res.eigvals.real).round(6)
    array([-1.923739,  0.155946])
    """
    _y0 = as_state(y0)
    identity = np.eye(_y0.size)

    def objective(y: State) -> tuple[State, Jacobian]:
        res = diffeo.image_detail(y, params, period, jacobian=True)
        assert res.jac is not None
        return res.y - y, res.jac - identity

    sol = root(objective, _y0, jac=True, **root_options)
    success = bool(sol.success)
    y1: State | None = None
    eigvals = eigvecs = None
    if success:
        y1 = np.asarray(sol.x, dtype=np.float64)
        detail = diffeo.image_detail(y1, params, period, jacobian=True)
        eigvals, eigvecs = detail.eigvals, detail.eigvecs
    return CycleResult(
        success=success,
        y=y1,
        eigvals=eigvals,
        eigvecs=eigvecs,
        nfev=int(sol.get("nfev", 0)),
        residual=np.asarray(sol.fun, dtype=np.float64),
        message=str(sol.message),
    )


def trace_cycle(
    diffeo: Map,
    y0: StateLike,
    params: Params,
    param_key: str,
    end_value: float,
    *,
    period: int = 1,
    resolution: int = 100,
    show_progress: bool = False,
    callback: Callable[[ContinuationStep[State]], None] | None = None,
    **root_options: Any,
) -> list[ContinuationStep[State]]:
    """Continue a periodic point along a parameter.

    Calls :func:`find_cycle` at each point of a :func:`~mappy_ds.continuation`
    of ``params[param_key]`` from its current value to ``end_value``, starting
    each search from the previous periodic point.

    Parameters
    ----------
    diffeo : Map
        The map.
    y0 : array_like
        Initial guess of the periodic point at the initial parameters.
    params : mapping
        Initial parameters; not modified.
    param_key : str
        Key of the parameter to continue.
    end_value : float
        Final value of the parameter.
    period : int, optional
        Period of the cycle, by default ``1``.
    resolution : int, optional
        Number of continuation points including both ends, by default ``100``.
    show_progress : bool, optional
        Print progress to standard error, by default ``False``.
    callback : callable, optional
        Called with each successful step.
    **root_options
        Forwarded to :func:`scipy.optimize.root`.

    Returns
    -------
    list of ContinuationStep
        One step per periodic point found. ``info`` holds ``eigvals``,
        ``eigvecs``, ``nfev`` and ``residual`` as in :class:`CycleResult`.
        The list stops at the first parameter value where the search failed.

    See Also
    --------
    find_cycle, mappy_ds.continuation.continuation
    """

    def step(y: State, p: dict[str, Any]) -> StepFunctionResult[State]:
        res = find_cycle(diffeo, y, p, period=period, **root_options)
        return StepFunctionResult(
            res.success,
            res.y,
            None,
            {
                "eigvals": res.eigvals,
                "eigvecs": res.eigvecs,
                "nfev": res.nfev,
                "residual": res.residual,
            },
        )

    return continuation(
        step,
        as_state(y0),
        params,
        param_key,
        end_value,
        resolution=resolution,
        show_progress=show_progress,
        callback=callback,
    )
