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
import numpy.typing as npt

from ._typing import Params
from .exceptions import MappyError

__all__ = [
    "ArclengthPoint",
    "ContinuationStep",
    "StepFunctionResult",
    "arclength_continuation",
    "continuation",
]

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


@dataclass
class ArclengthPoint:
    """One accepted point of :func:`arclength_continuation`.

    Attributes
    ----------
    u : numpy.ndarray
        The point on the solution curve, shape ``(m + 1,)``.
    tangent : numpy.ndarray
        Unit tangent of the curve at ``u`` in the direction of travel.
    step : float
        Arclength step that led to this point (``0`` for the first point).
    newton_iterations : int
        Newton iterations the corrector needed.
    """

    u: np.ndarray
    tangent: np.ndarray
    step: float
    newton_iterations: int


#: ``fun(u) -> (F, J)`` with ``F`` of shape ``(m,)`` and ``J`` of shape
#: ``(m, m + 1)``: the system whose zero set is a curve in ``R^{m+1}``.
ArclengthSystem = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _tangent(
    jac: np.ndarray, previous: np.ndarray | None, direction: int
) -> np.ndarray:
    """Return the unit null vector of ``jac`` oriented like ``previous``.

    With a previous tangent the bordered system ``[J; previousᵀ] t = e_last``
    is solved, which keeps the orientation; without one the null vector comes
    from the SVD and is oriented so that its last component has the sign of
    ``direction``.
    """
    m = jac.shape[0]
    if previous is not None:
        bordered = np.vstack([jac, previous[np.newaxis, :]])
        rhs = np.zeros(m + 1)
        rhs[-1] = 1.0
        t = np.linalg.solve(bordered, rhs)
    else:
        _, _, vt = np.linalg.svd(jac)
        t = vt[-1]
        if t[-1] * direction < 0:
            t = -t
    return t / np.linalg.norm(t)


def arclength_continuation(
    fun: ArclengthSystem,
    u0: npt.ArrayLike,
    *,
    step: float = 0.01,
    step_min: float = 1e-6,
    step_max: float = 0.1,
    max_steps: int = 1000,
    direction: int = 1,
    bounds: tuple[float, float] | None = None,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 10,
    callback: Callable[[ArclengthPoint], None] | None = None,
) -> list[ArclengthPoint]:
    r"""Follow a solution curve of ``F(u) = 0`` by pseudo-arclength continuation.

    ``F : R^{m+1} → R^m`` has a one-dimensional zero set. Starting from
    ``u0`` (which is first corrected onto the curve with its last component
    held fixed), each step predicts ``u = u_k + h t_k`` along the unit
    tangent ``t_k`` and corrects it with Newton's method on the bordered
    system

    .. math::

       F(u) = 0, \qquad t_k \cdot (u - u_k) - h = 0,

    whose Jacobian ``[J; t_kᵀ]`` is regular even where ``J`` loses rank in
    the parameter direction, so folds (turning points) are passed without
    special treatment (Keller, 1977). The step ``h`` grows by 1.5 when Newton
    converges in at most three iterations, is halved when six or more are
    needed or when it fails, and the orientation is kept by requiring
    ``t_k · t_{k+1} > 0``.

    Parameters
    ----------
    fun : callable
        ``fun(u) -> (F, J)`` returning the residual ``(m,)`` and its Jacobian
        ``(m, m + 1)`` at ``u``. It may raise :class:`~mappy_ds.MappyError`
        or ``numpy.linalg.LinAlgError`` to signal that ``u`` is not
        evaluable; the step is then retried with a smaller ``h``.
    u0 : array_like
        Starting point, shape ``(m + 1,)``; its last component is the
        parameter-like coordinate used by ``direction`` and ``bounds``.
    step : float, optional
        Initial arclength step (Euclidean distance in the ``u`` space), by
        default ``0.01``.
    step_min, step_max : float, optional
        Bounds of the adaptive step, by default ``1e-6`` and ``0.1``. The
        continuation stops when a step below ``step_min`` would be needed.
    max_steps : int, optional
        Maximum number of accepted points after the first, by default
        ``1000``.
    direction : int, optional
        ``+1`` to start with the last component of ``u`` increasing, ``-1``
        decreasing, by default ``1``.
    bounds : tuple of float, optional
        Stop when the last component of ``u`` would leave ``[lo, hi]``; the
        step is shortened so that the last accepted point lies close to the
        boundary.
    newton_tol : float, optional
        Convergence tolerance of the corrector (on the Newton update relative
        to ``1 + |u|`` and on ``|F|``), by default ``1e-10``.
    newton_max_iter : int, optional
        Maximum Newton iterations per step, by default ``10``.
    callback : callable, optional
        Called with each accepted :class:`ArclengthPoint`.

    Returns
    -------
    list of ArclengthPoint
        The accepted points in order, starting with the corrected ``u0``.
        The list ends when ``bounds`` are left, ``max_steps`` is reached, the
        step cannot be reduced further, or the curve closes on itself.

    Notes
    -----
    The arclength is measured in the Euclidean metric of ``u``; if the state
    and the parameter have very different scales, rescale the parameter in
    ``fun`` (or choose ``step`` accordingly) so that progress along the
    parameter is not starved.

    Examples
    --------
    The unit circle ``u0² + u1² = 1`` traced from ``(1, 0)`` with ``u1``
    increasing first; it is closed, so the continuation stops after one turn.

    >>> import numpy as np
    >>> from mappy_ds.continuation import arclength_continuation
    >>> def circle(u):
    ...     return np.array([u @ u - 1.0]), 2 * u[np.newaxis, :]
    >>> pts = arclength_continuation(circle, [1.0, 0.0], step=0.3, step_max=0.3)
    >>> all(abs(p.u @ p.u - 1) < 1e-9 for p in pts)
    True
    >>> pts[1].u[1] > 0 and 15 < len(pts) < 40
    True
    """
    u = np.array(u0, dtype=np.float64)
    if u.ndim != 1 or u.size < 2:
        raise ValueError("u0 must be a vector of length m + 1 >= 2")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    if not (0 < step_min <= step <= step_max):
        raise ValueError("need 0 < step_min <= step <= step_max")
    m = u.size - 1

    def corrector(
        u_pred: np.ndarray, base: np.ndarray, t: np.ndarray, h: float
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Newton on the bordered system; ``None`` if it does not converge."""
        x = u_pred.copy()
        for k in range(1, newton_max_iter + 1):
            res, jac = fun(x)
            res = np.asarray(res, dtype=np.float64).reshape(m)
            jac = np.asarray(jac, dtype=np.float64).reshape(m, m + 1)
            g = np.append(res, t @ (x - base) - h)
            bordered = np.vstack([jac, t[np.newaxis, :]])
            dx = np.linalg.solve(bordered, -g)
            x = x + dx
            if np.linalg.norm(dx) <= newton_tol * (1.0 + np.linalg.norm(x)):
                res, jac = fun(x)
                res = np.asarray(res, dtype=np.float64).reshape(m)
                jac = np.asarray(jac, dtype=np.float64).reshape(m, m + 1)
                if np.linalg.norm(res) <= 100 * newton_tol * (1.0 + np.linalg.norm(x)):
                    return x, jac, k
                return None
        return None

    # Correct the starting point with its parameter held fixed.
    e_last = np.zeros(m + 1)
    e_last[-1] = 1.0
    try:
        start = corrector(u, u, e_last, 0.0)
    except (MappyError, np.linalg.LinAlgError):
        start = None
    if start is None:
        raise RuntimeError("the starting point could not be corrected onto the curve")
    u, jac, iters = start
    t = _tangent(jac, None, direction)
    points = [ArclengthPoint(u, t, 0.0, iters)]
    if callback is not None:
        callback(points[0])

    h = float(step)
    u_start = u.copy()
    for _ in range(max_steps):
        accepted = None
        while accepted is None:
            if h < step_min:
                return points
            try:
                accepted = corrector(u + h * t, u, t, h)
            except (MappyError, np.linalg.LinAlgError):
                accepted = None
            if accepted is None:
                h *= 0.5
        u_new, jac_new, iters = accepted
        if bounds is not None and not (bounds[0] <= u_new[-1] <= bounds[1]):
            # Shrink the step so that the next point lands close to the
            # boundary; give up after a few attempts (e.g. strong curvature).
            landed = False
            for _ in range(4):
                target = bounds[0] if u_new[-1] < bounds[0] else bounds[1]
                frac = (target - u[-1]) / (u_new[-1] - u[-1])
                h = max(0.999 * frac * h, step_min)
                if h <= step_min:
                    break
                try:
                    accepted = corrector(u + h * t, u, t, h)
                except (MappyError, np.linalg.LinAlgError):
                    accepted = None
                if accepted is None:
                    break
                u_new, jac_new, iters = accepted
                if bounds[0] <= u_new[-1] <= bounds[1]:
                    landed = True
                    break
            if not landed:
                return points
            bounds_hit = True
        else:
            bounds_hit = False
        try:
            t_new = _tangent(jac_new, t, direction)
        except np.linalg.LinAlgError:
            t_new = _tangent(jac_new, None, 1 if t[-1] >= 0 else -1)
        point = ArclengthPoint(u_new, t_new, h, iters)
        points.append(point)
        if callback is not None:
            callback(point)
        if bounds_hit:
            return points
        # Closed curve: back at the start after having moved away from it.
        if len(points) > 3 and np.linalg.norm(u_new - u_start) < 0.5 * h:
            return points
        u, t = u_new, t_new
        if iters <= 3:
            h = min(1.5 * h, step_max)
        elif iters >= 6:
            h = max(0.5 * h, step_min)
    return points
