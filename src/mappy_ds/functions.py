"""System functions with automatically derived Jacobians and Hessians.

A *system function* is a user-written Python function ``f(y, params)`` that
describes either the right-hand side of an ODE or a discrete map.  Wrapping it
in :class:`SystemFunction` (usually through the :func:`vector_field`,
:func:`mapping` and :func:`border` decorators) traces it once with SymPy
symbols, so that the function itself, its Jacobian and its Hessian are all
available as fast NumPy callables.

Because the function is traced symbolically it must be *expressible with
SymPy*: use ``sympy.cos`` rather than ``numpy.cos`` and avoid branching on the
values of ``y``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import sympy

from ._typing import Hessian, Jacobian, Params, State, StateLike, as_state
from .exceptions import SymbolicEvaluationError

__all__ = ["SystemFunction", "border", "mapping", "vector_field"]

#: Signature of a user-defined system function.
PyFunction = Callable[[Any, Any], Any]


class SystemFunction:
    """A vector-valued function :math:`f(y, p)` with symbolic derivatives.

    Parameters
    ----------
    fun : callable
        Python function ``fun(y, params)``. ``y`` is a one-dimensional array of
        length ``dom_dim`` and ``params`` a mapping holding at least the keys in
        ``param_keys`` (it is ``{}`` when ``param_keys`` is empty). The return
        value must be convertible to a vector of length ``cod_dim``.
    dom_dim : int
        Dimension of the domain (length of ``y``).
    cod_dim : int
        Dimension of the codomain (length of the returned vector).
    param_keys : iterable of str, optional
        Names of the parameters ``fun`` reads from ``params``.
    direction : int, optional
        Only meaningful for border functions: direction of the zero crossing
        that terminates a continuous mode (``1``, ``-1`` or ``0`` for both),
        forwarded to :func:`scipy.integrate.solve_ivp` events.

    Notes
    -----
    The function is traced with SymPy on first use. The Jacobian follows the
    convention ``jac[i, j] = ∂f_i/∂y_j`` and the Hessian
    ``hes[i, j, k] = ∂²f_i/(∂y_j ∂y_k)``.
    """

    def __init__(
        self,
        fun: PyFunction,
        dom_dim: int,
        cod_dim: int,
        param_keys: Iterable[str] = (),
        direction: int = 0,
    ) -> None:
        if dom_dim < 1 or cod_dim < 1:
            raise ValueError("dom_dim and cod_dim must be positive integers")
        self.pyfunc = fun
        self.dom_dim = int(dom_dim)
        self.cod_dim = int(cod_dim)
        self.param_keys: tuple[str, ...] = tuple(param_keys)
        self.direction = int(direction)
        self.__name__ = getattr(fun, "__name__", type(self).__name__)
        self.__doc__ = getattr(fun, "__doc__", None)
        self.__wrapped__ = fun

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.__name__}, dom_dim={self.dom_dim}, "
            f"cod_dim={self.cod_dim}, param_keys={list(self.param_keys)})"
        )

    # -- symbolic tracing -------------------------------------------------

    @cached_property
    def _symbols(self) -> tuple[list[sympy.Symbol], list[sympy.Symbol]]:
        y = list(sympy.symbols(f"y_0:{self.dom_dim}", real=True))
        p = [sympy.Symbol(k, real=True) for k in self.param_keys]
        return y, p

    @cached_property
    def expression(self) -> sympy.Matrix:
        """Symbolic expression of the function as a ``cod_dim × 1`` matrix."""
        y, p = self._symbols
        y_arr = np.array(y, dtype=object)
        p_map = dict(zip(self.param_keys, p, strict=True))
        try:
            raw = self.pyfunc(y_arr, p_map)
            expr = sympy.Matrix(np.atleast_1d(np.asarray(raw, dtype=object)).ravel())
        except SymbolicEvaluationError:
            raise
        except Exception as exc:
            raise SymbolicEvaluationError(
                f"cannot trace {self.__name__!r} symbolically: {exc}"
            ) from exc
        if expr.shape != (self.cod_dim, 1):
            raise ValueError(
                f"{self.__name__!r} returned {expr.shape[0]} components, "
                f"expected cod_dim={self.cod_dim}"
            )
        return expr

    @cached_property
    def jacobian_expression(self) -> sympy.Matrix:
        """Symbolic Jacobian (``cod_dim × dom_dim``)."""
        y, _ = self._symbols
        return self.expression.jacobian(y)

    @cached_property
    def hessian_expression(self) -> sympy.Array:
        """Symbolic Hessian (``cod_dim × dom_dim × dom_dim``)."""
        y, _ = self._symbols
        f = self.expression
        return sympy.Array(
            [
                [[sympy.diff(f[i], yj, yk) for yk in y] for yj in y]
                for i in range(self.cod_dim)
            ]
        )

    def _compile(self, expr: Any) -> Callable[..., Any]:
        y, p = self._symbols
        compiled: Callable[..., Any] = sympy.lambdify(
            [y, p], expr, modules="numpy", cse=True
        )
        return compiled

    @cached_property
    def _f(self) -> Callable[..., Any]:
        return self._compile(self.expression)

    @cached_property
    def _jac(self) -> Callable[..., Any]:
        return self._compile(self.jacobian_expression)

    @cached_property
    def _hes(self) -> Callable[..., Any]:
        return self._compile(self.hessian_expression)

    # -- numeric evaluation ------------------------------------------------

    def params_array(self, params: Params | None) -> np.ndarray:
        """Extract ``param_keys`` from ``params`` as a float array."""
        if not self.param_keys:
            return np.empty(0)
        if params is None:
            raise ValueError(
                f"{self.__name__!r} needs parameters {list(self.param_keys)}"
            )
        try:
            return np.array([params[k] for k in self.param_keys], dtype=np.float64)
        except KeyError as exc:
            raise KeyError(
                f"{self.__name__!r} needs parameter {exc.args[0]!r}"
            ) from exc

    def __call__(self, y: StateLike, params: Params | None = None) -> State:
        """Evaluate the function numerically."""
        return self.eval(as_state(y), self.params_array(params))

    def eval(self, y: State, p: np.ndarray) -> State:
        """Evaluate with a pre-extracted parameter array (hot path)."""
        return np.asarray(self._f(y, p), dtype=np.float64).reshape(self.cod_dim)

    def jacobian(self, y: StateLike, params: Params | None = None) -> Jacobian:
        """Evaluate the Jacobian matrix at ``y``."""
        return self.eval_jacobian(as_state(y), self.params_array(params))

    def eval_jacobian(self, y: State, p: np.ndarray) -> Jacobian:
        """Jacobian with a pre-extracted parameter array (hot path)."""
        return np.asarray(self._jac(y, p), dtype=np.float64).reshape(
            self.cod_dim, self.dom_dim
        )

    def hessian(self, y: StateLike, params: Params | None = None) -> Hessian:
        """Evaluate the Hessian tensor at ``y``."""
        return self.eval_hessian(as_state(y), self.params_array(params))

    def eval_hessian(self, y: State, p: np.ndarray) -> Hessian:
        """Hessian with a pre-extracted parameter array (hot path)."""
        return np.asarray(self._hes(y, p), dtype=np.float64).reshape(
            self.cod_dim, self.dom_dim, self.dom_dim
        )


def _as_keys(param_keys: Iterable[str] | None) -> tuple[str, ...]:
    if param_keys is None:
        return ()
    if isinstance(param_keys, str):
        raise TypeError("param_keys must be a sequence of names, not a string")
    return tuple(param_keys)


def vector_field(
    dim: int, param_keys: Sequence[str] | None = None
) -> Callable[[PyFunction], SystemFunction]:
    r"""Declare the right-hand side of an ODE :math:`\dot y = f(y, p)`.

    Parameters
    ----------
    dim : int
        Dimension of the state space.
    param_keys : sequence of str, optional
        Names of the parameters used by the function.

    Examples
    --------
    >>> import numpy as np
    >>> from mappy_ds import vector_field
    >>> @vector_field(dim=2, param_keys=["a", "b", "I"])
    ... def izhikevich(y, p):
    ...     v, u = y
    ...     return np.array([0.04 * v**2 + 5 * v + 140 - u + p["I"],
    ...                      p["a"] * (p["b"] * v - u)])
    >>> izhikevich.jacobian([0.0, 0.0], {"a": 0.2, "b": 0.2, "I": 10.0})
    array([[ 5.  , -1.  ],
           [ 0.04, -0.2 ]])
    """

    def decorator(fun: PyFunction) -> SystemFunction:
        return SystemFunction(fun, dim, dim, _as_keys(param_keys))

    return decorator


def mapping(
    dom_dim: int, cod_dim: int | None = None, param_keys: Sequence[str] | None = None
) -> Callable[[PyFunction], SystemFunction]:
    r"""Declare a discrete map :math:`y \mapsto f(y, p)`.

    Parameters
    ----------
    dom_dim : int
        Dimension of the domain.
    cod_dim : int, optional
        Dimension of the codomain, by default equal to ``dom_dim``.
    param_keys : sequence of str, optional
        Names of the parameters used by the function.

    Examples
    --------
    >>> import numpy as np
    >>> from mappy_ds import mapping
    >>> @mapping(2, param_keys=["a", "b"])
    ... def henon(y, p):
    ...     x, u = y
    ...     return np.array([1 - p["a"] * x**2 + u, p["b"] * x])
    >>> henon([0.0, 0.0], {"a": 1.4, "b": 0.3})
    array([1., 0.])
    """

    def decorator(fun: PyFunction) -> SystemFunction:
        return SystemFunction(
            fun, dom_dim, dom_dim if cod_dim is None else cod_dim, _as_keys(param_keys)
        )

    return decorator


def border(
    dim: int, param_keys: Sequence[str] | None = None, direction: int = 1
) -> Callable[[PyFunction], SystemFunction]:
    """Declare a border (section) :math:`b(y, p) = 0` of a continuous mode.

    Parameters
    ----------
    dim : int
        Dimension of the state space of the mode the border belongs to.
    param_keys : sequence of str, optional
        Names of the parameters used by the function.
    direction : int, optional
        Direction of the zero crossing that triggers the border: ``1`` for
        negative-to-positive, ``-1`` for positive-to-negative and ``0`` for
        both. By default ``1``.

    Examples
    --------
    >>> from mappy_ds import border
    >>> @border(dim=2)
    ... def fire(y, _):
    ...     return y[0] - 30.0
    """

    def decorator(fun: PyFunction) -> SystemFunction:
        return SystemFunction(fun, dim, 1, _as_keys(param_keys), direction=direction)

    return decorator
