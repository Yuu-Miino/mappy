"""Continuous and discrete modes of a hybrid dynamical system.

A *mode* is a primitive, smooth piece of a hybrid system together with the
rule that terminates it. Both kinds of modes expose the same operation,
:meth:`Mode.step`, which maps an initial state to the state at the end of the
mode and, on request, the Jacobian matrix and Hessian tensor of that map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from ._typing import Hessian, Jacobian, Params, State, StateLike, as_state
from .exceptions import ModeNotTerminatedError
from .functions import SystemFunction
from .trajectory import TrajectorySegment

__all__ = ["ContinuousMode", "DiscreteMode", "Mode", "StepResult"]


@dataclass(frozen=True)
class StepResult:
    """Result of :meth:`Mode.step`.

    Attributes
    ----------
    y : numpy.ndarray
        State at the end of the mode, shape ``(cod_dim,)``.
    jac : numpy.ndarray or None
        Jacobian matrix of the mode map, ``jac[i, j] = ∂y_i/∂y0_j``, shape
        ``(cod_dim, dom_dim)``. ``None`` if not requested.
    hes : numpy.ndarray or None
        Hessian tensor of the mode map, ``hes[i, j, k] = ∂²y_i/(∂y0_j ∂y0_k)``,
        shape ``(cod_dim, dom_dim, dom_dim)``. ``None`` if not requested.
    border : int or None
        Index of the border that terminated a continuous mode, ``None`` for
        discrete modes and for continuous modes without borders.
    t : float or None
        Time spent in a continuous mode, ``None`` for discrete modes.
    segment : TrajectorySegment or None
        Sampled trajectory, if requested and the mode is ``in_trajectory``.
    """

    y: State
    jac: Jacobian | None = None
    hes: Hessian | None = None
    border: int | None = None
    t: float | None = None
    segment: TrajectorySegment | None = None


class Mode(ABC):
    """Abstract base class of all modes.

    Parameters
    ----------
    name : str
        Unique name of the mode, used in transition rules.
    fun : SystemFunction
        Function governing the evolution of the state in the mode.
    in_trajectory : bool
        Whether the mode contributes a segment to computed trajectories.
    """

    def __init__(self, name: str, fun: SystemFunction, in_trajectory: bool) -> None:
        if not isinstance(fun, SystemFunction):
            raise TypeError(
                "fun must be a SystemFunction; decorate it with "
                "@vector_field(...) or @mapping(...)"
            )
        self.name = str(name)
        self.fun = fun
        self.in_trajectory = bool(in_trajectory)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.fun.__name__})"

    @property
    def dom_dim(self) -> int:
        """Dimension of the state space the mode starts in."""
        return self.fun.dom_dim

    @property
    def cod_dim(self) -> int:
        """Dimension of the state space the mode ends in."""
        return self.fun.cod_dim

    @abstractmethod
    def step(
        self,
        y0: StateLike,
        params: Params | None = None,
        *,
        jacobian: bool = False,
        hessian: bool = False,
        trajectory: bool = False,
    ) -> StepResult:
        """Evolve ``y0`` through the mode.

        Parameters
        ----------
        y0 : array_like
            Initial state of length ``dom_dim``.
        params : mapping, optional
            Parameters of the system.
        jacobian : bool, optional
            Compute the Jacobian matrix of the mode map.
        hessian : bool, optional
            Compute the Hessian tensor of the mode map (implies ``jacobian``).
        trajectory : bool, optional
            Record the trajectory within the mode.

        Returns
        -------
        StepResult
        """


class DiscreteMode(Mode):
    """A mode whose evolution is a single application of a map.

    Parameters
    ----------
    name : str
        Unique name of the mode.
    fun : SystemFunction
        The map, declared with :func:`mappy_ds.mapping`.
    in_trajectory : bool, optional
        Whether the jump contributes a (two-point) trajectory segment, by
        default ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> from mappy_ds import DiscreteMode, mapping
    >>> @mapping(2)
    ... def fire(y, p):
    ...     return y + np.array([p["c"] - 30, p["d"]])
    >>> jump = DiscreteMode("jump", fire)
    >>> jump.step([30.0, -1.0], {"c": -50.0, "d": 2.0}, jacobian=True).jac
    array([[1., 0.],
           [0., 1.]])
    """

    def __init__(
        self, name: str, fun: SystemFunction, in_trajectory: bool = False
    ) -> None:
        super().__init__(name, fun, in_trajectory)

    def step(
        self,
        y0: StateLike,
        params: Params | None = None,
        *,
        jacobian: bool = False,
        hessian: bool = False,
        trajectory: bool = False,
    ) -> StepResult:
        """Evolve ``y0`` through the mode; see :meth:`Mode.step`."""
        _y0 = as_state(y0)
        p = self.fun.params_array(params)
        y1 = self.fun.eval(_y0, p)
        jac = self.fun.eval_jacobian(_y0, p) if (jacobian or hessian) else None
        hes = self.fun.eval_hessian(_y0, p) if hessian else None
        segment = None
        if trajectory and self.in_trajectory and self.dom_dim == self.cod_dim:
            segment = TrajectorySegment(
                self.name, "discrete", np.column_stack([_y0, y1])
            )
        return StepResult(y1, jac, hes, segment=segment)


class ContinuousMode(Mode):
    r"""A mode whose evolution follows an autonomous ODE until a border is hit.

    Parameters
    ----------
    name : str
        Unique name of the mode.
    fun : SystemFunction
        Right-hand side of the ODE, declared with :func:`mappy_ds.vector_field`.
    borders : sequence of SystemFunction, optional
        Border functions declared with :func:`mappy_ds.border`. Integration
        stops at the first zero crossing of any border (in the direction the
        border declares). With no borders the mode is the time-``max_interval``
        flow map.
    max_interval : float, optional
        Upper bound of the integration time, by default ``20.0``.
    in_trajectory : bool, optional
        Whether the mode contributes a trajectory segment, by default ``True``.
    trajectory_dt : float, optional
        Sampling interval of recorded trajectories, by default ``0.01``.
    method, rtol, atol, **solver_options
        Forwarded to :func:`scipy.integrate.solve_ivp`.

    Notes
    -----
    When the Jacobian or Hessian is requested the variational equations

    .. math::

        \dot J = D f(\varphi)\, J, \qquad
        \dot H_{ijk} = \sum_{l,m} (D^2 f)_{ilm} J_{lj} J_{mk}
                        + \sum_l (Df)_{il} H_{ljk}

    are integrated along with the state, and the result is corrected for the
    dependence of the hitting time on the initial state (see the references).

    References
    ----------
    Y. Miino, "Mode concept to construct diffeomorphisms representing
    smooth/non-smooth dynamical systems", NOLTA, IEICE 15(2), 284–298, 2024.
    """

    def __init__(
        self,
        name: str,
        fun: SystemFunction,
        borders: Sequence[SystemFunction] = (),
        max_interval: float = 20.0,
        in_trajectory: bool = True,
        trajectory_dt: float = 1e-2,
        *,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **solver_options: Any,
    ) -> None:
        super().__init__(name, fun, in_trajectory)
        if fun.dom_dim != fun.cod_dim:
            raise ValueError("a vector field must map R^n to R^n")
        self.borders = tuple(borders)
        for b in self.borders:
            if not isinstance(b, SystemFunction) or b.cod_dim != 1:
                raise TypeError("borders must be declared with @border(...)")
            if b.dom_dim != fun.dom_dim:
                raise ValueError(
                    f"border {b.__name__!r} has dim {b.dom_dim}, expected {fun.dom_dim}"
                )
        if len({b.name for b in self.borders}) != len(self.borders):
            raise ValueError(
                f"borders of mode {name!r} must have distinct names; "
                "pass name=... to @border(...)"
            )
        self.max_interval = float(max_interval)
        self.trajectory_dt = float(trajectory_dt)
        self.solver_options: dict[str, Any] = {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            **solver_options,
        }

    @property
    def border_names(self) -> tuple[str, ...]:
        """Names of the borders, in the order of ``borders``."""
        return tuple(b.name for b in self.borders)

    # -- right-hand sides of the (variational) ODE ---------------------------

    def _rhs(self, p: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
        fun = self.fun

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            return fun.eval(y, p)

        return rhs

    def _rhs_jac(self, p: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
        fun, n = self.fun, self.dom_dim

        def rhs(_t: float, z: np.ndarray) -> np.ndarray:
            y = z[:n]
            jac = z[n:].reshape(n, n)
            return np.concatenate(
                [fun.eval(y, p), (fun.eval_jacobian(y, p) @ jac).ravel()]
            )

        return rhs

    def _rhs_hes(self, p: np.ndarray) -> Callable[[float, np.ndarray], np.ndarray]:
        fun, n = self.fun, self.dom_dim

        def rhs(_t: float, z: np.ndarray) -> np.ndarray:
            y = z[:n]
            jac = z[n : n + n * n].reshape(n, n)
            hes = z[n + n * n :].reshape(n, n, n)
            df = fun.eval_jacobian(y, p)
            d2f = fun.eval_hessian(y, p)
            dhes = np.einsum("ilm,lj,mk->ijk", d2f, jac, jac) + np.einsum(
                "il,ljk->ijk", df, hes
            )
            return np.concatenate([fun.eval(y, p), (df @ jac).ravel(), dhes.ravel()])

        return rhs

    def _events(self, params: Params | None) -> list[Callable[..., float]]:
        events = []
        for b in self.borders:
            pb = b.params_array(params)

            def event(
                _t: float, z: np.ndarray, b: SystemFunction = b, pb: np.ndarray = pb
            ) -> float:
                return float(b.eval(z[: self.dom_dim], pb)[0])

            event.terminal = True  # type: ignore[attr-defined]
            event.direction = b.direction  # type: ignore[attr-defined]
            events.append(event)
        return events

    def step(
        self,
        y0: StateLike,
        params: Params | None = None,
        *,
        jacobian: bool = False,
        hessian: bool = False,
        trajectory: bool = False,
    ) -> StepResult:
        """Evolve ``y0`` through the mode; see :meth:`Mode.step`."""
        n = self.dom_dim
        _y0 = as_state(y0)
        if _y0.shape != (n,):
            raise ValueError(f"expected a state of length {n}, got {_y0.shape}")
        jacobian = jacobian or hessian
        p = self.fun.params_array(params)

        if hessian:
            rhs = self._rhs_hes(p)
            z0 = np.concatenate([_y0, np.eye(n).ravel(), np.zeros(n**3)])
        elif jacobian:
            rhs = self._rhs_jac(p)
            z0 = np.concatenate([_y0, np.eye(n).ravel()])
        else:
            rhs = self._rhs(p)
            z0 = _y0

        record = trajectory and self.in_trajectory
        sol = solve_ivp(
            rhs,
            (0.0, self.max_interval),
            z0,
            events=self._events(params) or None,
            dense_output=record,
            **self.solver_options,
        )
        if sol.status == -1:
            raise RuntimeError(f"solve_ivp failed in mode {self.name!r}: {sol.message}")
        if sol.status == 0 and self.borders:
            raise ModeNotTerminatedError(self.name, self.max_interval)

        z1 = sol.y[:, -1]
        y1 = z1[:n].copy()
        t1 = float(sol.t[-1])
        border = None
        if sol.status == 1:
            border = next(i for i, te in enumerate(sol.t_events) if len(te) > 0)

        jac: Jacobian | None = None
        hes: Hessian | None = None
        if jacobian:
            jac = z1[n : n + n * n].reshape(n, n).copy()
            if hessian:
                hes = z1[n + n * n :].reshape(n, n, n).copy()
            if border is not None:
                jac, hes = self._border_correction(y1, p, params, border, jac, hes)

        segment = None
        if record:
            n_pts = max(int(np.ceil(t1 / self.trajectory_dt)), 1) + 1
            ts = np.linspace(0.0, t1, n_pts)
            assert sol.sol is not None
            segment = TrajectorySegment(self.name, "continuous", sol.sol(ts)[:n], ts)

        return StepResult(y1, jac, hes, border=border, t=t1, segment=segment)

    def _border_correction(
        self,
        y1: State,
        p: np.ndarray,
        params: Params | None,
        border: int,
        jac_flow: Jacobian,
        hes_flow: Hessian | None,
    ) -> tuple[Jacobian, Hessian | None]:
        """Account for the dependence of the hitting time on the initial state.

        With ``f`` the vector field, ``q`` the border function, ``J`` and
        ``H`` the derivatives of the flow at the hitting time ``τ``, the mode
        map ``y0 ↦ φ(τ(y0), y0)`` has Jacobian ``B J`` where
        ``B = I - f ⊗ ∇q / (∇q · f)``. The Hessian follows by differentiating
        once more, taking ``∂τ/∂y0 = -(∇q J) / (∇q · f)`` into account.
        """
        b = self.borders[border]
        pb = b.params_array(params)
        f = self.fun.eval(y1, p)
        q = b.eval_jacobian(y1, pb)[0]
        s = float(q @ f)
        if s == 0.0:
            raise ZeroDivisionError(
                f"border {b.__name__!r} of mode {self.name!r} is tangent to the flow"
            )
        bmat = np.eye(self.dom_dim) - np.outer(f, q) / s
        jac = bmat @ jac_flow
        if hes_flow is None:
            return jac, None

        df = self.fun.eval_jacobian(y1, p)
        hq = b.eval_hessian(y1, pb)[0]
        ds = hq @ f + q @ df  # ∂s/∂y
        # ∂B[a,d]/∂y_e
        dbmat = (
            -(np.einsum("ae,d->ade", df, q) + np.einsum("a,de->ade", f, hq)) / s
            + np.einsum("a,d,e->ade", f, q, ds) / s**2
        )
        tau = -(q @ jac_flow) / s  # ∂τ/∂y0
        djac_flow = hes_flow + np.einsum("db,c->dbc", df @ jac_flow, tau)
        hes = np.einsum("ade,ec,db->abc", dbmat, jac, jac_flow) + np.einsum(
            "ad,dbc->abc", bmat, djac_flow
        )
        return jac, hes
