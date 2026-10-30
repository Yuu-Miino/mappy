"""Local bifurcations of periodic points and their continuation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import root

from .._typing import Params, State, StateLike, as_state
from ..continuation import ContinuationStep, StepFunctionResult, continuation
from ..maps import Map
from ._cycle import CycleResult

__all__ = ["LocalBfResult", "find_local_bf", "trace_local_bf"]


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


def _local_bf_system(
    diffeo: Map,
    var: State,
    params: dict[str, Any],
    param_key: str,
    period: int,
) -> tuple[State, np.ndarray]:
    r"""Residual and Jacobian of the local-bifurcation system.

    Unknowns are ``var = (y, λ, θ)`` with ``λ = params[param_key]``. The
    residual is ``(T(y) - y, Re det A, Im det A)`` with
    ``A = DT(y) - e^{iθ} I``. Its Jacobian uses the map's Jacobian, Hessian
    and parameter derivative; only ``∂(DT)/∂λ``, which enters the determinant
    rows, is approximated by a central difference (two extra map
    evaluations), because second derivatives with respect to parameters are
    not available.
    """
    dim = var.size - 2
    y = var[:dim]
    lam = float(var[dim])
    theta = float(var[dim + 1])
    p = dict(params)
    p[param_key] = lam
    res = diffeo.image_detail(
        y, p, period, jacobian=True, hessian=True, parameter_jacobian=True
    )
    assert res.jac is not None and res.hes is not None and res.jac_p is not None
    jac, hes = res.jac, res.hes
    mu = np.exp(1j * theta)
    a = jac - mu * np.eye(dim)
    det = np.linalg.det(a)
    cof = _cofactor(a)

    out = np.empty(dim + 2)
    out[:dim] = res.y - y
    out[dim] = det.real
    out[dim + 1] = det.imag

    # ∂(DT)/∂λ by a central difference
    h = 1e-6 * max(1.0, abs(lam))
    jac_plus = diffeo.image_detail(y, {**p, param_key: lam + h}, period).jac
    jac_minus = diffeo.image_detail(y, {**p, param_key: lam - h}, period).jac
    assert jac_plus is not None and jac_minus is not None
    djac_dlam = (jac_plus - jac_minus) / (2 * h)

    ddet_dy = np.einsum("ij,ijk->k", cof, hes)  # ∂A_ij/∂y_k = hes[i, j, k]
    ddet_dlam = np.sum(cof * djac_dlam)
    ddet_dtheta = -1j * mu * np.trace(cof)  # ∂A/∂θ = -i e^{iθ} I

    jacobian = np.zeros((dim + 2, dim + 2))
    jacobian[:dim, :dim] = jac - np.eye(dim)
    if param_key in res.param_keys:
        jacobian[:dim, dim] = res.jac_p[:, res.param_keys.index(param_key)]
    jacobian[dim, :dim] = ddet_dy.real
    jacobian[dim + 1, :dim] = ddet_dy.imag
    jacobian[dim, dim] = ddet_dlam.real
    jacobian[dim + 1, dim] = ddet_dlam.imag
    jacobian[dim, dim + 1] = ddet_dtheta.real
    jacobian[dim + 1, dim + 1] = ddet_dtheta.imag
    return out, jacobian


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
    Jacobian, Hessian and parameter derivative (:attr:`MapResult.jac_p
    <mappy_ds.MapResult.jac_p>`); only the derivative of the determinant
    with respect to the parameter uses a central difference of ``DTⁿ``.

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
        detail = diffeo.image_detail(
            y1, p1, period, jacobian=True, hessian=True, parameter_jacobian=True
        )
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
