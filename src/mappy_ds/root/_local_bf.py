"""Local bifurcations of periodic points and their continuation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import root

from .._typing import Params, State, StateLike, as_state
from ..continuation import (
    ArclengthPoint,
    ContinuationStep,
    StepFunctionResult,
    arclength_continuation,
    continuation,
)
from ..maps import Map, MapResult
from ._cycle import CycleResult

__all__ = [
    "LocalBfResult",
    "find_local_bf",
    "trace_local_bf",
    "trace_local_bf_arclength",
]


@dataclass
class LocalBfResult(CycleResult):
    r"""Result of :func:`find_local_bf`.

    Attributes
    ----------
    success, y, eigvals, eigvecs, nfev, message
        As in :class:`CycleResult`, at the bifurcation point.
    residual : numpy.ndarray
        Residual of the full system at the last iterate: :math:`T^n(y) - y`
        followed by the real and imaginary parts of
        :math:`\det(DT^n(y) - e^{i\theta} I)`.
    params : dict or None
        Parameters with the bifurcation parameter at its solved value, or
        ``None`` if the search failed.
    theta : float or None
        Argument of the critical multiplier :math:`e^{i\theta}`, or ``None``
        if the search failed.
    """

    params: dict[str, Any] | None
    theta: float | None


def _cofactor(a: np.ndarray) -> np.ndarray:
    """Cofactor matrix ``C`` of ``a`` (so that ``d det(a) = Σ C_ij da_ij``).

    Computed from minors, which stays well defined where ``a`` is singular —
    exactly the situation at a bifurcation point, where ``det(a) = 0`` and
    the identity ``C = det(a) a⁻ᵀ`` breaks down.
    """
    d = a.shape[0]
    if d == 1:
        return np.ones((1, 1), dtype=a.dtype)
    c = np.empty_like(a)
    for i in range(d):
        rows = np.delete(np.arange(d), i)
        for j in range(d):
            cols = np.delete(np.arange(d), j)
            c[i, j] = (-1) ** (i + j) * np.linalg.det(a[np.ix_(rows, cols)])
    return c


def _bifurcation_system(
    diffeo: Map,
    y: State,
    params: dict[str, Any],
    theta: float,
    period: int,
    diff_keys: Sequence[str],
) -> tuple[State, np.ndarray, np.ndarray, np.ndarray, MapResult]:
    r"""Residual of the local-bifurcation conditions and its derivatives.

    The residual is ``(Tⁿ(y) - y, Re det A, Im det A)`` with
    ``A = DTⁿ(y) - e^{iθ} I``. Returned are the residual, its derivative with
    respect to ``y`` (``(dim+2) × dim``), with respect to each parameter in
    ``diff_keys`` (``(dim+2) × len(diff_keys)``) and with respect to ``θ``
    (``(dim+2,)``), all analytic from the map's Jacobian, Hessian, parameter
    derivative and mixed derivative, plus the :class:`~mappy_ds.MapResult`.
    """
    dim = y.size
    res = diffeo.image_detail(y, params, period, parameter_hessian=True)
    assert res.jac is not None and res.hes is not None
    assert res.jac_p is not None and res.hes_p is not None
    jac, hes = res.jac, res.hes
    mu = np.exp(1j * theta)
    a = jac - mu * np.eye(dim)
    det = np.linalg.det(a)
    cof = _cofactor(a)

    out = np.empty(dim + 2)
    out[:dim] = res.y - y
    out[dim] = det.real
    out[dim + 1] = det.imag

    ddet_dy = np.einsum("ij,ijk->k", cof, hes)  # ∂A_ij/∂y_k = hes[i, j, k]
    d_dy = np.zeros((dim + 2, dim))
    d_dy[:dim] = jac - np.eye(dim)
    d_dy[dim] = ddet_dy.real
    d_dy[dim + 1] = ddet_dy.imag

    d_dp = np.zeros((dim + 2, len(diff_keys)))
    for m, key in enumerate(diff_keys):
        if key not in res.param_keys:  # the map does not depend on it
            continue
        col = res.param_keys.index(key)
        ddet = np.sum(cof * res.hes_p[:, :, col])
        d_dp[:dim, m] = res.jac_p[:, col]
        d_dp[dim, m] = ddet.real
        d_dp[dim + 1, m] = ddet.imag

    ddet_dtheta = -1j * mu * np.trace(cof)  # ∂A/∂θ = -i e^{iθ} I
    d_dtheta = np.zeros(dim + 2)
    d_dtheta[dim] = ddet_dtheta.real
    d_dtheta[dim + 1] = ddet_dtheta.imag
    return out, d_dy, d_dp, d_dtheta, res


def _local_bf_system(
    diffeo: Map,
    var: State,
    params: dict[str, Any],
    param_key: str,
    period: int,
) -> tuple[State, np.ndarray]:
    """Residual and square Jacobian for unknowns ``var = (y, λ, θ)``."""
    dim = var.size - 2
    p = {**params, param_key: float(var[dim])}
    out, d_dy, d_dp, d_dtheta, _ = _bifurcation_system(
        diffeo, var[:dim], p, float(var[dim + 1]), period, (param_key,)
    )
    return out, np.column_stack([d_dy, d_dp, d_dtheta])


def find_local_bf(
    diffeo: Map,
    y0: StateLike,
    params: Params,
    param_key: str,
    theta: float,
    *,
    period: int = 1,
    **root_options: Any,
) -> LocalBfResult:
    r"""Find a local bifurcation point of a periodic point.

    A periodic point :math:`y` of period :math:`n` undergoes a local
    bifurcation when one of its characteristic multipliers crosses the unit
    circle at :math:`\mu = e^{i\theta}`. The unknowns
    :math:`(y, p, \theta)`, with :math:`p` = ``params[param_key]``, are
    solved from

    .. math::

        T^n(y) - y = 0, \qquad \det(DT^n(y) - e^{i\theta} I) = 0,

    the determinant being split into its real and imaginary parts. The
    initial ``theta`` selects the kind of bifurcation:

    * ``theta = 0`` (:math:`\mu = 1`): tangent (saddle-node) bifurcation;
    * ``theta = pi`` (:math:`\mu = -1`): period-doubling bifurcation;
    * any other value: Neimark–Sacker bifurcation, a complex-conjugate pair of
      multipliers :math:`e^{\pm i\theta}` on the unit circle.

    The Jacobian of this system is supplied analytically from the map's
    Jacobian, Hessian, parameter derivative (:attr:`MapResult.jac_p
    <mappy_ds.MapResult.jac_p>`) and mixed derivative
    (:attr:`MapResult.hes_p <mappy_ds.MapResult.hes_p>`).

    Parameters
    ----------
    diffeo : Map
        The map.
    y0 : array_like
        Initial guess of the periodic point.
    params : mapping
        Parameters; ``params[param_key]`` is the initial guess of the
        bifurcation value. Not modified.
    param_key : str
        Key of the bifurcation parameter.
    theta : float
        Initial guess of the argument of the critical multiplier.
    period : int, optional
        Period of the cycle, by default ``1``.
    **root_options
        Forwarded to :func:`scipy.optimize.root`.

    Returns
    -------
    LocalBfResult

    See Also
    --------
    trace_local_bf : Continue a local bifurcation along a second parameter.
    find_cycle
    """
    _y0 = as_state(y0)
    dim = _y0.size
    base = dict(params)
    var0 = np.concatenate([_y0, [float(base[param_key]), float(theta)]])

    def objective(var: State) -> tuple[State, np.ndarray]:
        return _local_bf_system(diffeo, var, base, param_key, period)

    sol = root(objective, var0, jac=True, **root_options)
    success = bool(sol.success)
    y1: State | None = None
    p1: dict[str, Any] | None = None
    theta1: float | None = None
    eigvals = eigvecs = None
    if success:
        x = np.asarray(sol.x, dtype=np.float64)
        y1 = x[:dim].copy()
        p1 = dict(base)
        p1[param_key] = float(x[dim])
        theta1 = float(x[dim + 1])
        # Same request as inside the objective, so that the multipliers are
        # consistent with the system that was solved (the variational
        # equations change the step sequence of the ODE solver).
        detail = diffeo.image_detail(y1, p1, period, parameter_hessian=True)
        eigvals, eigvecs = detail.eigvals, detail.eigvecs
    return LocalBfResult(
        success=success,
        y=y1,
        eigvals=eigvals,
        eigvecs=eigvecs,
        nfev=int(sol.get("nfev", 0)),
        residual=np.asarray(sol.fun, dtype=np.float64),
        message=str(sol.message),
        params=p1,
        theta=theta1,
    )


def trace_local_bf(
    diffeo: Map,
    y0: StateLike,
    params: Params,
    bf_param_key: str,
    theta: float,
    cnt_param_key: str,
    end_value: float,
    *,
    period: int = 1,
    resolution: int = 100,
    show_progress: bool = False,
    callback: Callable[[ContinuationStep[State]], None] | None = None,
    **root_options: Any,
) -> list[ContinuationStep[State]]:
    """Continue a local bifurcation along a second parameter.

    ``params[cnt_param_key]`` is swept from its current value to
    ``end_value``; at each point :func:`find_local_bf` solves for the periodic
    point, ``params[bf_param_key]`` and ``theta``, starting from the previous
    solution. The result traces a bifurcation curve in the
    (``cnt_param_key``, ``bf_param_key``) plane.

    Parameters
    ----------
    diffeo : Map
        The map.
    y0 : array_like
        Initial guess of the periodic point, ideally at a bifurcation point
        found with :func:`find_local_bf`.
    params : mapping
        Initial parameters; not modified.
    bf_param_key : str
        Key of the bifurcation parameter (solved at each step).
    theta : float
        Initial argument of the critical multiplier; see
        :func:`find_local_bf`.
    cnt_param_key : str
        Key of the continued parameter.
    end_value : float
        Final value of ``params[cnt_param_key]``.
    period : int, optional
        Period of the cycle, by default ``1``.
    resolution : int, optional
        Number of continuation points including both ends, by default ``100``.
    show_progress : bool, optional
        Print progress to standard error, by default ``False``.
    callback : callable, optional
        Called with each successful step (as returned in the list).
    **root_options
        Forwarded to :func:`scipy.optimize.root`.

    Returns
    -------
    list of ContinuationStep
        One step per bifurcation point found. ``y`` is the periodic point,
        ``params`` holds both the continued and the solved bifurcation
        parameter, and ``info`` holds ``theta``, ``eigvals``, ``eigvecs``,
        ``nfev`` and ``residual``.

    Raises
    ------
    ValueError
        If ``bf_param_key`` equals ``cnt_param_key``.

    See Also
    --------
    find_local_bf, mappy_ds.continuation.continuation
    """
    if bf_param_key == cnt_param_key:
        raise ValueError("bf_param_key and cnt_param_key must differ")

    _y0 = as_state(y0)
    dim = _y0.size

    def step(var: State, p: dict[str, Any]) -> StepFunctionResult[State]:
        res = find_local_bf(
            diffeo,
            var[:dim],
            p,
            bf_param_key,
            float(var[dim]),
            period=period,
            **root_options,
        )
        if not res.success or res.y is None or res.theta is None:
            return StepFunctionResult(False)
        return StepFunctionResult(
            True,
            np.append(res.y, res.theta),
            res.params,
            {
                "theta": res.theta,
                "eigvals": res.eigvals,
                "eigvecs": res.eigvecs,
                "nfev": res.nfev,
                "residual": res.residual,
            },
        )

    def to_public(s: ContinuationStep[State]) -> ContinuationStep[State]:
        return ContinuationStep(s.y[:dim].copy(), s.params, s.info)

    inner_callback = None
    if callback is not None:
        cb = callback

        def inner_callback(s: ContinuationStep[State]) -> None:
            cb(to_public(s))

    steps = continuation(
        step,
        np.append(_y0, float(theta)),
        params,
        cnt_param_key,
        end_value,
        resolution=resolution,
        show_progress=show_progress,
        callback=inner_callback,
    )
    return [to_public(s) for s in steps]


#: Resonances of a Neimark–Sacker curve flagged while ``θ`` varies: the test
#: function changes sign where the critical pair ``e^{±iθ}`` passes a root of
#: unity (1:1 at θ=0, 1:2 at θ=π, 1:3 at 2π/3, 1:4 at π/2).
def _resonance_events(theta_prev: float, theta: float) -> list[str]:
    events: list[str] = []
    if theta_prev == theta:
        return events
    s0, s1 = np.sin(theta_prev), np.sin(theta)
    c0, c1 = np.cos(theta_prev), np.cos(theta)
    if s0 * s1 < 0:
        events.append("resonance_1_1" if c1 > 0 else "resonance_1_2")
    if (c0 + 0.5) * (c1 + 0.5) < 0:
        events.append("resonance_1_3")
    if c0 * c1 < 0:
        events.append("resonance_1_4")
    return events


def trace_local_bf_arclength(
    diffeo: Map,
    y0: StateLike,
    params: Params,
    bf_param_key: str,
    theta: float,
    cnt_param_key: str,
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
    r"""Follow a local-bifurcation curve by pseudo-arclength continuation.

    The bifurcation conditions of :func:`find_local_bf`,

    .. math::

       T^n(\bm y) - \bm y = 0, \qquad
       \det\parens{DT^n(\bm y) - e^{i\theta} I} = 0 \;(\text{real and imaginary parts}),

    are :math:`d + 2` equations in the :math:`d + 3` unknowns
    :math:`u = (\bm y, \lambda_{\mathrm{bf}}, \theta, \lambda_{\mathrm{cnt}})`,
    so their solutions form a curve: the bifurcation curve in the
    :math:`(\lambda_{\mathrm{cnt}}, \lambda_{\mathrm{bf}})` plane together
    with the periodic point and the argument of the critical multiplier. The
    curve is followed with :func:`~mappy_ds.continuation.arclength_continuation`,
    so it may turn back in :math:`\lambda_{\mathrm{cnt}}` (as a fold curve
    does at a cusp) without stopping. The Jacobian of the system is fully
    analytic (:attr:`~mappy_ds.MapResult.jac`, :attr:`~mappy_ds.MapResult.hes`,
    :attr:`~mappy_ds.MapResult.jac_p`, :attr:`~mappy_ds.MapResult.hes_p`).

    Events between consecutive points are recorded in ``info["events"]`` of
    the later point:

    * ``"fold"`` — the curve turns back in :math:`\lambda_{\mathrm{cnt}}`
      (for a fold curve this happens at a cusp);
    * ``"resonance_1_1"``, ``"resonance_1_2"``, ``"resonance_1_3"``,
      ``"resonance_1_4"`` — on a Neimark–Sacker curve, :math:`\theta` passes
      :math:`0`, :math:`\pi`, :math:`2\pi/3` or :math:`\pi/2` (strong
      resonances);
    * ``"second_multiplier_crossing"`` — the number of multipliers outside
      the unit circle changes, i.e. another bifurcation curve is crossed
      (fold–flip, flip–Neimark–Sacker, ... points).

    Parameters
    ----------
    diffeo : Map
        The map.
    y0 : array_like
        Periodic point at the starting bifurcation point (e.g. from
        :func:`find_local_bf`); it is corrected first with both parameters
        held fixed.
    params : mapping
        Parameters at the start; ``params[bf_param_key]`` is the bifurcation
        value, ``params[cnt_param_key]`` the start of the continuation. Not
        modified.
    bf_param_key, cnt_param_key : str
        Keys of the bifurcation parameter (solved along the curve) and of the
        continuation parameter (the ``direction``/``bounds`` coordinate).
    theta : float
        Argument of the critical multiplier at the start (``0`` fold, ``pi``
        period doubling, otherwise Neimark–Sacker).
    period : int, optional
        Period of the cycle, by default ``1``.
    direction : int, optional
        ``+1`` to start with :math:`\lambda_{\mathrm{cnt}}` increasing, ``-1``
        decreasing; by default ``1``.
    bounds : tuple of float, optional
        Stop when :math:`\lambda_{\mathrm{cnt}}` leaves ``[lo, hi]``.
    step, step_min, step_max : float, optional
        Arclength step in the :math:`u` space; see
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
        One step per accepted point along the curve. ``y`` is the periodic
        point, ``params`` holds both parameters at their values on the curve,
        and ``info`` holds ``theta``, ``eigvals``, ``eigvecs``, ``tangent``
        (unit vector in the :math:`u` space), ``step``, ``newton_iterations``
        and ``events``.

    Raises
    ------
    ValueError
        If the two keys coincide or the map depends on neither.

    See Also
    --------
    trace_local_bf : Natural-parameter continuation of the same curve.
    trace_cycle_arclength : The same method for branches of periodic points.

    Examples
    --------
    The cusp normal form ``x ↦ x + λ₁ + λ₂ x - x³`` has the fold curve
    ``λ₂ = 3x², λ₁ = -2x³``, which turns back in ``λ₂`` at the cusp
    ``(0, 0)``. Starting on the upper branch and decreasing ``λ₂``, the
    continuation passes the cusp onto the lower branch:

    >>> import numpy as np
    >>> from mappy_ds import Diffeomorphism, mapping
    >>> from mappy_ds.root import trace_local_bf_arclength
    >>> @mapping(1)
    ... def cusp(x, p):
    ...     return x + p["l1"] + p["l2"] * x - x**3
    >>> steps = trace_local_bf_arclength(
    ...     Diffeomorphism(cusp), [0.5], {"l1": -0.25, "l2": 0.75}, "l1", 0.0,
    ...     "l2", direction=-1, step=0.1, step_max=0.2, bounds=(-1.0, 0.75),
    ... )
    >>> any("fold" in s.info["events"] for s in steps)
    True
    >>> steps[-1].y[0] < 0 and steps[-1].params["l1"] > 0
    True
    """
    if bf_param_key == cnt_param_key:
        raise ValueError("bf_param_key and cnt_param_key must differ")
    keys = (bf_param_key, cnt_param_key)
    if not any(k in diffeo.param_keys for k in keys):
        raise ValueError(f"the map depends on neither of {keys}")
    _y0 = as_state(y0)
    dim = _y0.size
    base = dict(params)
    cache: dict[int, MapResult] = {}

    def system(u: np.ndarray) -> tuple[State, np.ndarray]:
        p = {**base, bf_param_key: float(u[dim]), cnt_param_key: float(u[dim + 2])}
        out, d_dy, d_dp, d_dtheta, res = _bifurcation_system(
            diffeo, u[:dim], p, float(u[dim + 1]), period, keys
        )
        cache[0] = res
        # columns: y, λ_bf, θ, λ_cnt
        return out, np.column_stack([d_dy, d_dp[:, 0], d_dtheta, d_dp[:, 1]])

    steps: list[ContinuationStep[State]] = []
    state: dict[str, Any] = {}

    def on_point(point: ArclengthPoint) -> None:
        res = cache[0]
        eig, eigvec = res.eigvals, res.eigvecs
        th = float(point.u[dim + 1])
        outside = int(np.sum(np.abs(eig) > 1.0 + 1e-6))
        events: list[str] = []
        if steps:
            if state["tangent"][-1] * point.tangent[-1] < 0:
                events.append("fold")
            events.extend(_resonance_events(state["theta"], th))
            if outside != state["outside"]:
                events.append("second_multiplier_crossing")
        state.update(tangent=point.tangent, theta=th, outside=outside)
        cs = ContinuationStep(
            point.u[:dim].copy(),
            {
                **base,
                bf_param_key: float(point.u[dim]),
                cnt_param_key: float(point.u[dim + 2]),
            },
            {
                "theta": th,
                "eigvals": eig,
                "eigvecs": eigvec,
                "tangent": point.tangent,
                "step": point.step,
                "newton_iterations": point.newton_iterations,
                "events": tuple(events),
            },
        )
        steps.append(cs)
        if callback is not None:
            callback(cs)
        if show_progress:
            tag = f" {','.join(events)}" if events else ""
            lam_c, lam_b = cs.params[cnt_param_key], cs.params[bf_param_key]
            print(
                f"ARCLENGTH {len(steps):5d} {cnt_param_key}={lam_c:+.10g} "
                f"{bf_param_key}={lam_b:+.10g} θ={th:+.6f} h={point.step:.3g}{tag}",
                end="\r",
                file=sys.stderr,
                flush=True,
            )

    u0 = np.concatenate(
        [_y0, [float(base[bf_param_key]), float(theta), float(base[cnt_param_key])]]
    )
    try:
        arclength_continuation(
            system,
            u0,
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
