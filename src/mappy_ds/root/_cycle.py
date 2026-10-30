"""Periodic points (cycles) of maps and their continuation."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import root

from .._typing import Jacobian, Params, State, StateLike, as_state
from ..continuation import (
    ArclengthPoint,
    ContinuationStep,
    StepFunctionResult,
    arclength_continuation,
    continuation,
)
from ..maps import Map

__all__ = ["CycleResult", "find_cycle", "trace_cycle", "trace_cycle_arclength"]


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
    :math:`T` is ``diffeo`` and :math:`n` is ``period``, using the analytic
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
    >>> @mapping(2)
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


def _classify_events(
    prev_tangent: np.ndarray,
    tangent: np.ndarray,
    prev_jac: Jacobian,
    jac: Jacobian,
    prev_eig: npt.NDArray[np.complex128],
    eig: npt.NDArray[np.complex128],
) -> tuple[str, ...]:
    """Bifurcations between two consecutive points, from sign changes."""
    dim = jac.shape[0]
    eye = np.eye(dim)
    events: list[str] = []
    fold_turn = prev_tangent[-1] * tangent[-1] < 0
    det_minus = np.linalg.det(prev_jac - eye) * np.linalg.det(jac - eye) < 0
    if fold_turn:
        events.append("fold")
    elif det_minus:
        events.append("branch_point")
    if np.linalg.det(prev_jac + eye) * np.linalg.det(jac + eye) < 0:
        events.append("period_doubling")
    # A complex pair crossing the unit circle changes the number of
    # multipliers outside it by two without det(J ∓ I) changing sign.
    out_prev = int(np.sum(np.abs(prev_eig) > 1.0))
    out_now = int(np.sum(np.abs(eig) > 1.0))
    complex_now = np.any(np.abs(eig.imag) > 1e-12 * (1 + np.abs(eig.real)))
    if abs(out_now - out_prev) == 2 and complex_now and not (fold_turn or det_minus):
        events.append("neimark_sacker")
    return tuple(events)


def trace_cycle_arclength(
    diffeo: Map,
    y0: StateLike,
    params: Params,
    param_key: str,
    *,
    period: int = 1,
    direction: int = 1,
    bounds: tuple[float, float] | None = None,
    step: float = 0.01,
    step_min: float = 1e-6,
    step_max: float = 0.1,
    max_steps: int = 1000,
    show_progress: bool = False,
    callback: Callable[[ContinuationStep[State]], None] | None = None,
) -> list[ContinuationStep[State]]:
    r"""Follow a branch of periodic points by pseudo-arclength continuation.

    Unlike :func:`trace_cycle`, which prescribes the parameter values, this
    continuation treats the periodic point and the parameter together as a
    point :math:`u = (\bm y, \lambda)` on the curve :math:`T^n(\bm y, \lambda)
    - \bm y = 0` and advances along the curve itself, so folds (saddle-node
    bifurcations), where the branch turns back in :math:`\lambda`, are
    passed and the continuation carries on along the other side. The
    tangent of the curve is the null vector of :math:`[DT^n - I \mid
    \partial T^n/\partial\lambda]`, available analytically from
    :attr:`MapResult.jac <mappy_ds.MapResult.jac>` and
    :attr:`MapResult.jac_p <mappy_ds.MapResult.jac_p>`.

    Bifurcations along the branch are detected from sign changes between
    consecutive points and recorded in ``info["events"]`` of the later
    point:

    * ``"fold"`` — the tangent's parameter component changes sign;
    * ``"branch_point"`` — ``det(DTⁿ - I)`` changes sign without a fold
      (transcritical or pitchfork; the continuation follows one branch);
    * ``"period_doubling"`` — ``det(DTⁿ + I)`` changes sign;
    * ``"neimark_sacker"`` — a complex pair of multipliers crosses the unit
      circle.

    Refine an event with :func:`~mappy_ds.root.find_local_bf`, starting from
    either of the two points and ``theta = 0`` (fold), ``pi`` (period
    doubling) or the argument of the critical pair (Neimark–Sacker).

    Parameters
    ----------
    diffeo : Map
        The map.
    y0 : array_like
        Initial guess of the periodic point at ``params``.
    params : mapping
        Initial parameters; not modified.
    param_key : str
        Key of the continuation parameter :math:`\lambda`.
    period : int, optional
        Period of the cycle, by default ``1``.
    direction : int, optional
        ``+1`` to start with :math:`\lambda` increasing, ``-1`` decreasing;
        by default ``1``.
    bounds : tuple of float, optional
        Stop when :math:`\lambda` leaves ``[lo, hi]``.
    step, step_min, step_max : float, optional
        Initial, minimum and maximum arclength step in the
        :math:`(\bm y, \lambda)` space; see
        :func:`~mappy_ds.continuation.arclength_continuation`.
    max_steps : int, optional
        Maximum number of points after the first, by default ``1000``.
    show_progress : bool, optional
        Print one line per accepted point to standard error.
    callback : callable, optional
        Called with each accepted :class:`~mappy_ds.ContinuationStep`.

    Returns
    -------
    list of ContinuationStep
        One step per accepted point, in order along the curve. ``y`` is the
        periodic point, ``params`` holds the parameter value, and ``info``
        holds ``eigvals``, ``eigvecs``, ``tangent`` (unit vector in the
        :math:`(\bm y, \lambda)` space), ``step``, ``newton_iterations`` and
        ``events``.

    Raises
    ------
    ValueError
        If the map does not depend on ``param_key``.

    See Also
    --------
    trace_cycle : Natural-parameter continuation at prescribed values.
    mappy_ds.continuation.arclength_continuation : The underlying algorithm.

    Examples
    --------
    The fixed point of ``y ↦ y + p - y²`` exists for ``p ≥ 0`` only; both
    fixed points ``±√p`` lie on one curve that folds at ``p = 0``. Starting
    on the upper branch with ``p`` decreasing, the continuation passes the
    fold and comes back along the lower branch:

    >>> import numpy as np
    >>> from mappy_ds import Diffeomorphism, mapping
    >>> from mappy_ds.root import trace_cycle_arclength
    >>> @mapping(1)
    ... def f(y, p):
    ...     return y + p["p"] - y**2
    >>> steps = trace_cycle_arclength(
    ...     Diffeomorphism(f), [0.5], {"p": 0.25}, "p", direction=-1,
    ...     step=0.05, step_max=0.1, bounds=(-1.0, 0.25),
    ... )
    >>> any("fold" in s.info["events"] for s in steps)
    True
    >>> steps[-1].y[0] < 0 and steps[-1].params["p"] > 0
    True
    """
    if param_key not in diffeo.param_keys:
        raise ValueError(f"the map does not depend on parameter {param_key!r}")
    col = diffeo.param_keys.index(param_key)
    _y0 = as_state(y0)
    dim = _y0.size
    base = dict(params)
    identity = np.eye(dim)
    results: dict[int, Any] = {}  # cache of MapResult keyed by id of u

    def system(u: np.ndarray) -> tuple[State, Jacobian]:
        p = {**base, param_key: float(u[dim])}
        res = diffeo.image_detail(
            u[:dim], p, period, jacobian=True, parameter_jacobian=True
        )
        assert res.jac is not None and res.jac_p is not None
        results.clear()
        results[0] = res
        return res.y - u[:dim], np.hstack([res.jac - identity, res.jac_p[:, [col]]])

    steps: list[ContinuationStep[State]] = []
    state: dict[str, Any] = {}

    def on_point(point: ArclengthPoint) -> None:
        # ``system`` was last evaluated at the accepted point by the corrector.
        res = results[0]
        assert res.jac is not None
        eig, eigvec = res.eigvals, res.eigvecs
        events: tuple[str, ...] = ()
        if steps:
            events = _classify_events(
                state["tangent"],
                point.tangent,
                state["jac"],
                res.jac,
                state["eig"],
                eig,
            )
        state.update(tangent=point.tangent, jac=res.jac, eig=eig)
        cs = ContinuationStep(
            point.u[:dim].copy(),
            {**base, param_key: float(point.u[dim])},
            {
                "eigvals": eig,
                "eigvecs": eigvec,
                "tangent": point.tangent,
                "step": point.step,
                "newton_iterations": point.newton_iterations,
                "events": events,
            },
        )
        steps.append(cs)
        if callback is not None:
            callback(cs)
        if show_progress:
            tag = f" {','.join(events)}" if events else ""
            print(
                f"ARCLENGTH {len(steps):5d} {param_key}={cs.params[param_key]:+.10g} "
                f"h={point.step:.3g} y={cs.y}{tag}",
                end="\r",
                file=sys.stderr,
                flush=True,
            )

    try:
        arclength_continuation(
            system,
            np.append(_y0, float(base[param_key])),
            step=step,
            step_min=step_min,
            step_max=step_max,
            max_steps=max_steps,
            direction=direction,
            bounds=bounds,
            callback=on_point,
        )
    finally:
        if show_progress:
            print(file=sys.stderr)
    return steps
