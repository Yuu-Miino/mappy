"""Diffeomorphisms built from modes: discrete maps and Poincaré maps."""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from ._typing import Hessian, Jacobian, Params, State, StateLike, as_state
from .exceptions import TransitionError, UnknownModeError
from .functions import SystemFunction
from .modes import ContinuousMode, DiscreteMode, Mode
from .trajectory import Trajectory

__all__ = ["Diffeomorphism", "Map", "MapResult", "PoincareMap", "Transitions"]

#: Transition rules: mode name → next mode name, or one next mode per border
#: given as a list (in ``borders`` order) or as ``{border_name: next_mode}``.
Transitions = Mapping[str, str | Sequence[str] | Mapping[str, str]]


@dataclass
class MapResult:
    """Image of a point under a :class:`Map` with derivatives.

    Attributes
    ----------
    y : numpy.ndarray
        Image of the initial state.
    jac : numpy.ndarray or None
        Jacobian matrix ``jac[i, j] = ∂y_i/∂y0_j`` if requested.
    hes : numpy.ndarray or None
        Hessian tensor ``hes[i, j, k] = ∂²y_i/(∂y0_j ∂y0_k)`` if requested.
    mode_history : list of str
        Names of the modes visited, starting with the initial mode and ending
        with the final one.
    trajectory : Trajectory or None
        Recorded trajectory if requested.
    """

    y: State
    jac: Jacobian | None = None
    hes: Hessian | None = None
    mode_history: list[str] = field(default_factory=list)
    trajectory: Trajectory | None = None

    @property
    def end_mode(self) -> str:
        """Name of the mode the image belongs to."""
        return self.mode_history[-1]

    @cached_property
    def _eig(self) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        if self.jac is None:
            raise ValueError("Jacobian was not computed; pass jacobian=True")
        if self.jac.shape[0] != self.jac.shape[1]:
            raise ValueError(
                "eigenvalues need a square Jacobian (same domain and codomain)"
            )
        vals, vecs = np.linalg.eig(self.jac)
        return vals.astype(np.complex128), vecs.astype(np.complex128)

    @property
    def eigvals(self) -> npt.NDArray[np.complex128]:
        """Eigenvalues (characteristic multipliers) of the Jacobian."""
        return self._eig[0]

    @property
    def eigvecs(self) -> npt.NDArray[np.complex128]:
        """Eigenvectors of the Jacobian, as columns."""
        return self._eig[1]


class Map(ABC):
    """Abstract base class of maps built from modes.

    Subclasses implement :meth:`image_detail`; :meth:`image` and
    :meth:`trajectory` are convenience wrappers around it.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the domain (and codomain) of the map."""

    @abstractmethod
    def image_detail(
        self,
        y0: StateLike,
        params: Params | None = None,
        n: int = 1,
        *,
        jacobian: bool = True,
        hessian: bool = False,
        trajectory: bool = False,
    ) -> MapResult:
        """Compute the ``n``-th image of ``y0`` with derivatives.

        Parameters
        ----------
        y0 : array_like
            Initial state.
        params : mapping, optional
            Parameters of the system.
        n : int, optional
            Number of iterations of the map, by default ``1``.
        jacobian : bool, optional
            Compute the Jacobian matrix, by default ``True``.
        hessian : bool, optional
            Compute the Hessian tensor (implies ``jacobian``), by default
            ``False``.
        trajectory : bool, optional
            Record the trajectory, by default ``False``.

        Returns
        -------
        MapResult

        Notes
        -----
        Requesting derivatives integrates the variational equations together
        with the state, which changes the adaptive step sizes of the ODE
        solver. The image therefore differs slightly (on the order of the
        solver tolerance) from the one obtained with ``jacobian=False``.
        """

    def image(self, y0: StateLike, params: Params | None = None, n: int = 1) -> State:
        """Compute the ``n``-th image of ``y0``."""
        return self.image_detail(y0, params, n, jacobian=False).y

    def trajectory(
        self, y0: StateLike, params: Params | None = None, n: int = 1
    ) -> Trajectory:
        """Compute the trajectory from ``y0`` over ``n`` iterations."""
        result = self.image_detail(y0, params, n, jacobian=False, trajectory=True)
        assert result.trajectory is not None
        return result.trajectory


def _compose(
    jac: Jacobian | None,
    hes: Hessian | None,
    jac_step: Jacobian | None,
    hes_step: Hessian | None,
) -> tuple[Jacobian | None, Hessian | None]:
    """Chain rule for ``g ∘ f`` given derivatives of ``f`` (so far) and ``g``."""
    if jac_step is None:
        return None, None
    if jac is None:
        return jac_step, hes_step
    if hes is not None and hes_step is not None:
        hes_new = np.einsum("ilm,lj,mk->ijk", hes_step, jac, jac) + np.einsum(
            "il,ljk->ijk", jac_step, hes
        )
    else:
        hes_new = None
    return jac_step @ jac, hes_new


class PoincareMap(Map):
    """Poincaré map of a hybrid system described by modes and transitions.

    Starting in ``initial_mode``, the state is evolved mode by mode following
    ``transitions`` until one of the ``end_modes`` is reached; that is one
    iteration of the map. The Jacobian and Hessian are composed with the chain
    rule along the way.

    Parameters
    ----------
    modes : sequence of Mode
        All modes of the system, with distinct names.
    transitions : mapping
        Successor of each mode. The value is

        * a mode name (``str``) for a discrete mode, a borderless continuous
          mode (which ends at ``max_interval``) or, as a shorthand, a
          continuous mode with a single border;
        * a list of mode names, one per border in the order of ``borders``;
        * a mapping ``{border_name: mode_name}`` covering every border of the
          mode (see the ``name`` argument of :func:`~mappy_ds.border`).

        Only modes reachable from ``initial_mode`` need a rule.
    initial_mode : str
        Mode the map starts in (the Poincaré section).
    end_modes : str or sequence of str, optional
        Modes that terminate one iteration, by default ``initial_mode``.
    max_transitions : int, optional
        Safety bound on the number of mode transitions per iteration, by
        default ``1000``.

    Raises
    ------
    UnknownModeError
        If a mode name in ``transitions``, ``initial_mode`` or ``end_modes``
        is not among ``modes``.
    TransitionError
        If a rule does not match its mode (wrong number of entries, unknown
        or missing border name), if a mode reachable from ``initial_mode``
        has no rule, if the codomain of a mode and the domain of its
        successor have different dimensions, or if no end mode can be
        reached from ``initial_mode``.

    Examples
    --------
    See :mod:`mappy_ds` for a complete example.
    """

    def __init__(
        self,
        modes: Sequence[Mode],
        transitions: Transitions,
        initial_mode: str,
        end_modes: str | Sequence[str] | None = None,
        *,
        max_transitions: int = 1000,
    ) -> None:
        self.modes: dict[str, Mode] = {}
        for m in modes:
            if m.name in self.modes:
                raise ValueError(f"duplicate mode name {m.name!r}")
            self.modes[m.name] = m
        #: Successors of each mode, one per border (a single entry for modes
        #: without borders).
        self.transitions: dict[str, tuple[str, ...]] = {}
        for src, rule in transitions.items():
            if src not in self.modes:
                raise UnknownModeError(src)
            self.transitions[src] = self._normalize_rule(self.modes[src], rule)
        if initial_mode not in self.modes:
            raise UnknownModeError(initial_mode)
        self.initial_mode = initial_mode
        if end_modes is None:
            ends: tuple[str, ...] = (initial_mode,)
        elif isinstance(end_modes, str):
            ends = (end_modes,)
        else:
            ends = tuple(end_modes)
        for e in ends:
            if e not in self.modes:
                raise UnknownModeError(e)
        self.end_modes = ends
        self.max_transitions = int(max_transitions)
        self._check_graph()

    def _normalize_rule(
        self, mode: Mode, rule: str | Sequence[str] | Mapping[str, str]
    ) -> tuple[str, ...]:
        """Turn a transition rule into one successor per border."""
        names = mode.border_names if isinstance(mode, ContinuousMode) else ()
        n_exits = max(len(names), 1)
        if isinstance(rule, str):
            if n_exits != 1:
                raise TransitionError(
                    mode.name,
                    reason=f"the mode has {n_exits} borders but the rule names "
                    "a single successor",
                )
            targets: tuple[str, ...] = (rule,)
        elif isinstance(rule, Mapping):
            if not names:
                raise TransitionError(
                    mode.name, reason="the mode has no borders to name in a mapping"
                )
            unknown = set(rule) - set(names)
            missing = set(names) - set(rule)
            if unknown or missing:
                reason = []
                if unknown:
                    reason.append(f"unknown border(s) {sorted(unknown)}")
                if missing:
                    reason.append(f"no successor for border(s) {sorted(missing)}")
                raise TransitionError(mode.name, reason="; ".join(reason))
            targets = tuple(rule[name] for name in names)
        else:
            targets = tuple(rule)
            if len(targets) != n_exits:
                raise TransitionError(
                    mode.name,
                    reason=f"expected {n_exits} successor(s) for "
                    f"{len(names)} border(s), got {len(targets)}",
                )
        for t in targets:
            if not isinstance(t, str):
                raise TypeError(f"mode names must be strings, got {t!r}")
            if t not in self.modes:
                raise UnknownModeError(t)
            if mode.cod_dim != self.modes[t].dom_dim:
                raise TransitionError(
                    mode.name,
                    reason=f"it ends in dimension {mode.cod_dim} but "
                    f"{t!r} starts in dimension {self.modes[t].dom_dim}",
                )
        return targets

    def _check_graph(self) -> None:
        """Check that every reachable mode has a rule and an end mode is reachable."""
        seen: set[str] = set()
        frontier = [self.initial_mode]
        reachable_after_start: set[str] = set()
        while frontier:
            name = frontier.pop()
            if name in seen:
                continue
            seen.add(name)
            if name not in self.transitions:
                raise TransitionError(name, reason="no rule for a reachable mode")
            for nxt in self.transitions[name]:
                reachable_after_start.add(nxt)
                if nxt not in seen:
                    frontier.append(nxt)
        if not reachable_after_start & set(self.end_modes):
            raise TransitionError(
                self.initial_mode,
                reason=f"no end mode {list(self.end_modes)} is reachable",
            )

    def __repr__(self) -> str:
        return (
            f"PoincareMap({list(self.modes)}, initial_mode={self.initial_mode!r}, "
            f"end_modes={list(self.end_modes)})"
        )

    @property
    def dim(self) -> int:  # noqa: D102
        return self.modes[self.initial_mode].dom_dim

    def _next_mode(self, mode: Mode, border: int | None) -> str:
        try:
            rule = self.transitions[mode.name]
        except KeyError:  # pragma: no cover - ruled out by _check_graph
            raise TransitionError(mode.name) from None
        index = 0 if border is None else border
        if index >= len(rule):  # pragma: no cover - ruled out by _normalize_rule
            raise TransitionError(mode.name, border)
        return rule[index]

    def _run(
        self,
        y0: State,
        params: Params | None,
        mode: str,
        n: int,
        jacobian: bool,
        hessian: bool,
        trajectory: bool,
    ) -> Iterator[tuple[State, Jacobian | None, Hessian | None, list[str], Trajectory]]:
        """Yield the state after each iteration of the map."""
        y = y0
        jac: Jacobian | None = None
        hes: Hessian | None = None
        history = [mode]
        traj = Trajectory()
        current = self.modes[mode]
        for _ in range(n):
            for _ in range(self.max_transitions):
                res = current.step(
                    y, params, jacobian=jacobian, hessian=hessian, trajectory=trajectory
                )
                y = res.y
                if jacobian:
                    jac, hes = _compose(jac, hes, res.jac, res.hes)
                if res.segment is not None:
                    traj.segments.append(res.segment)
                nxt = self._next_mode(current, res.border)
                history.append(nxt)
                current = self.modes[nxt]
                if nxt in self.end_modes:
                    break
            else:
                raise RuntimeError(
                    f"no end mode reached after {self.max_transitions} transitions"
                )
            yield y, jac, hes, history, traj

    def image_detail(
        self,
        y0: StateLike,
        params: Params | None = None,
        n: int = 1,
        *,
        jacobian: bool = True,
        hessian: bool = False,
        trajectory: bool = False,
        mode: str | None = None,
    ) -> MapResult:
        """Compute the ``n``-th image of ``y0`` with derivatives.

        Parameters
        ----------
        y0, params, n, jacobian, hessian, trajectory
            See :meth:`Map.image_detail`.
        mode : str, optional
            Mode to start in instead of ``initial_mode``; useful when a
            previous image ended in a different end mode.

        Returns
        -------
        MapResult
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        start = self.initial_mode if mode is None else mode
        if start not in self.modes:
            raise UnknownModeError(start)
        _y0 = as_state(y0)
        if _y0.shape != (self.modes[start].dom_dim,):
            raise ValueError(
                f"mode {start!r} expects a state of length "
                f"{self.modes[start].dom_dim}, got {_y0.shape}"
            )
        runs = self._run(
            _y0, params, start, n, jacobian or hessian, hessian, trajectory
        )
        y, jac, hes, history, traj = collections.deque(runs, maxlen=1).pop()
        return MapResult(y, jac, hes, history, traj if trajectory else None)

    def image(
        self,
        y0: StateLike,
        params: Params | None = None,
        n: int = 1,
        *,
        mode: str | None = None,
    ) -> State:
        """Compute the ``n``-th image of ``y0``."""
        return self.image_detail(y0, params, n, jacobian=False, mode=mode).y

    def trajectory(
        self,
        y0: StateLike,
        params: Params | None = None,
        n: int = 1,
        *,
        mode: str | None = None,
    ) -> Trajectory:
        """Compute the trajectory from ``y0`` over ``n`` iterations."""
        result = self.image_detail(
            y0, params, n, jacobian=False, trajectory=True, mode=mode
        )
        assert result.trajectory is not None
        return result.trajectory


class Diffeomorphism(Map):
    r"""A discrete map :math:`y \mapsto f(y, p)` on :math:`\mathbb{R}^n`.

    Parameters
    ----------
    fun : SystemFunction
        The map, declared with :func:`mappy_ds.mapping`; its domain and
        codomain must have the same dimension.
    name : str, optional
        Name of the underlying mode, by default ``"map"``.

    Examples
    --------
    >>> import numpy as np
    >>> from mappy_ds import Diffeomorphism, mapping
    >>> @mapping(2)
    ... def henon(y, p):
    ...     x, u = y
    ...     return np.array([1 - p["a"] * x**2 + u, p["b"] * x])
    >>> henon_map = Diffeomorphism(henon)
    >>> henon_map.image([0.0, 0.0], {"a": 1.4, "b": 0.3}, n=2)
    array([-0.4,  0.3])
    """

    def __init__(self, fun: SystemFunction, name: str = "map") -> None:
        if fun.dom_dim != fun.cod_dim:
            raise ValueError("a diffeomorphism must map R^n to R^n")
        self.mode = DiscreteMode(name, fun, in_trajectory=True)
        self._pmap = PoincareMap([self.mode], {name: name}, name)

    def __repr__(self) -> str:
        return f"Diffeomorphism({self.mode.fun.__name__})"

    @property
    def fun(self) -> SystemFunction:
        """The underlying system function."""
        return self.mode.fun

    @property
    def dim(self) -> int:  # noqa: D102
        return self.mode.dom_dim

    def image_detail(
        self,
        y0: StateLike,
        params: Params | None = None,
        n: int = 1,
        *,
        jacobian: bool = True,
        hessian: bool = False,
        trajectory: bool = False,
    ) -> MapResult:
        """Compute the ``n``-th image of ``y0``; see :meth:`Map.image_detail`."""
        return self._pmap.image_detail(
            y0, params, n, jacobian=jacobian, hessian=hessian, trajectory=trajectory
        )
