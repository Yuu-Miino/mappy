"""Shared fixtures: the systems used throughout the test suite.

* Izhikevich neuron (NOLTA 2024 paper example), as in the ``mappy_ds`` docstring.
* Hénon map as a :class:`~mappy_ds.Diffeomorphism`.
* Forced Duffing oscillator (``examples/duffing.py`` ported to the new API).
* A piecewise-linear "relay" oscillator under periodic sampling with two
  continuous modes and two end modes (forced-Alpazur-like structure), whose
  Poincaré map is known in closed form.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import sympy

from mappy_ds import (
    ContinuousMode,
    Diffeomorphism,
    DiscreteMode,
    PoincareMap,
    border,
    mapping,
    vector_field,
)

# ---------------------------------------------------------------------------
# Finite-difference helpers
# ---------------------------------------------------------------------------


def fd_jacobian(
    f: Callable[[np.ndarray], np.ndarray], y0: np.ndarray, h: float = 1e-4
) -> np.ndarray:
    """Central-difference Jacobian ``J[i, j] = ∂f_i/∂y_j``."""
    y0 = np.asarray(y0, dtype=float)
    cols = []
    for j in range(y0.size):
        e = np.zeros_like(y0)
        e[j] = h
        cols.append((f(y0 + e) - f(y0 - e)) / (2 * h))
    return np.stack(cols, axis=-1)


def fd_hessian(
    f: Callable[[np.ndarray], np.ndarray], y0: np.ndarray, h: float = 1e-4
) -> np.ndarray:
    """Second-order central-difference Hessian ``H[i, j, k] = ∂²f_i/∂y_j∂y_k``."""
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    m = np.asarray(f(y0)).size
    hes = np.zeros((m, n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = h
        for k in range(n):
            ek = np.zeros(n)
            ek[k] = h
            hes[:, j, k] = (
                f(y0 + ej + ek) - f(y0 + ej - ek) - f(y0 - ej + ek) + f(y0 - ej - ek)
            ) / (4 * h * h)
    return hes


# ---------------------------------------------------------------------------
# Izhikevich neuron
# ---------------------------------------------------------------------------


@vector_field(dim=2, param_keys=["a", "b", "I"])
def izhikevich(y, p):
    v, u = y
    return np.array([0.04 * v**2 + 5 * v + 140 - u + p["I"], p["a"] * (p["b"] * v - u)])


@border(dim=2)
def fire(y, _):
    return y[0] - 30


@mapping(2, param_keys=["c", "d"])
def firing(y, p):
    return y + np.array([p["c"] - 30, p["d"]])


@mapping(2, 1)
def reduce(y, _):
    return np.array([0, 1]) @ y


@mapping(1, 2)
def revert(u, _):
    return u * np.array([0, 1]) + np.array([30, 0])


IZHIKEVICH_TRANSITIONS = {"A": ["P"], "B": "A", "P": "Pinv", "Pinv": "B"}


def make_izhikevich_map(**solver_options) -> PoincareMap:
    return PoincareMap(
        [
            ContinuousMode("A", izhikevich, borders=[fire], **solver_options),
            DiscreteMode("B", firing),
            DiscreteMode("P", reduce),
            DiscreteMode("Pinv", revert),
        ],
        IZHIKEVICH_TRANSITIONS,
        initial_mode="Pinv",
    )


@pytest.fixture
def izh_params() -> dict[str, float]:
    return {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}


@pytest.fixture
def izh_mode() -> ContinuousMode:
    return ContinuousMode("A", izhikevich, borders=[fire])


@pytest.fixture
def izh_mode_tight() -> ContinuousMode:
    return ContinuousMode("A", izhikevich, borders=[fire], rtol=1e-10, atol=1e-12)


@pytest.fixture
def izh_map() -> PoincareMap:
    return make_izhikevich_map()


@pytest.fixture
def izh_map_tight() -> PoincareMap:
    return make_izhikevich_map(rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Hénon map
# ---------------------------------------------------------------------------


@mapping(2, param_keys=["a", "b"])
def henon(y, p):
    x, u = y
    return np.array([1 - p["a"] * x**2 + u, p["b"] * x])


@pytest.fixture
def henon_params() -> dict[str, float]:
    return {"a": 1.4, "b": 0.3}


@pytest.fixture
def henon_fun():
    return henon


@pytest.fixture
def henon_map() -> Diffeomorphism:
    return Diffeomorphism(henon)


# ---------------------------------------------------------------------------
# Forced Duffing oscillator (examples/duffing.py)
# ---------------------------------------------------------------------------


@vector_field(3, param_keys=["k", "B0", "B"])
def duffing(y, p):
    x, v, t = y
    return np.array([v, -p["k"] * v - x**3 + p["B0"] + p["B"] * sympy.cos(t), 1])


@border(3)
def duffing_section(y, _):
    return y[2] - 2 * sympy.pi


@mapping(3)
def duffing_t_reset(y, _):
    return y - np.array([0, 0, 2 * sympy.pi])


@mapping(2, 3)
def append_time(y, _):
    return np.append(y, 0)


@mapping(3, 2)
def drop_time(y, _):
    return y[0:2]


def make_duffing_map(**solver_options) -> PoincareMap:
    return PoincareMap(
        [
            ContinuousMode(
                "duffing", duffing, borders=[duffing_section], **solver_options
            ),
            DiscreteMode("t_reset", duffing_t_reset),
            DiscreteMode("p", append_time),
            DiscreteMode("pinv", drop_time),
        ],
        {"p": "duffing", "duffing": ["t_reset"], "t_reset": "pinv", "pinv": "p"},
        "p",
    )


@pytest.fixture
def duffing_params() -> dict[str, float]:
    return {"k": 0.2, "B0": 0.1, "B": 0.1}


@pytest.fixture
def duffing_y0() -> np.ndarray:
    """Approximate period-1 point of the Duffing Poincaré map (examples)."""
    return np.array([0.232147751858, 0.075591687407])


@pytest.fixture
def duffing_map() -> PoincareMap:
    return make_duffing_map()


# ---------------------------------------------------------------------------
# Relay oscillator sampled with period T: two continuous modes, two end modes
# ---------------------------------------------------------------------------
#
# State (x, t). Mode "up": x' = 1 until x = 1 (-> "down"); mode "down":
# x' = -1 until x = -1 (-> "up"); both stop at t = T (-> sampling section).
# With T = 2.2, starting in "up" at x0 ∈ (-0.2, 1) the image is
# -0.2 - x0 in mode "down"; starting in "down" at x0 ∈ (-1, 0.2) it is
# 0.2 - x0 in mode "up".


@vector_field(2)
def relay_up(y, _):
    return np.array([1, 1])


@vector_field(2)
def relay_down(y, _):
    return np.array([-1, 1])


@border(2, direction=1)
def upper(y, _):
    return y[0] - 1


@border(2, direction=-1)
def lower(y, _):
    return y[0] + 1


@border(2, param_keys=["T"])
def sample(y, p):
    return y[1] - p["T"]


@mapping(2)
def time_reset(y, _):
    return y * np.array([1, 0])


@mapping(1, 2)
def add_time(y, _):
    return np.append(y, 0)


@mapping(2, 1)
def remove_time(y, _):
    return y[0:1]


@pytest.fixture
def relay_params() -> dict[str, float]:
    return {"T": 2.2}


@pytest.fixture
def relay_map() -> PoincareMap:
    modes = [
        ContinuousMode("up", relay_up, borders=[upper, sample]),
        ContinuousMode("down", relay_down, borders=[lower, sample]),
        DiscreteMode("r_up", time_reset),
        DiscreteMode("r_down", time_reset),
        DiscreteMode("p_up", add_time),
        DiscreteMode("p_down", add_time),
        DiscreteMode("pinv_up", remove_time),
        DiscreteMode("pinv_down", remove_time),
    ]
    transitions = {
        "p_up": "up",
        "p_down": "down",
        "up": ["down", "r_up"],
        "down": ["up", "r_down"],
        "r_up": "pinv_up",
        "r_down": "pinv_down",
        "pinv_up": "p_up",
        "pinv_down": "p_down",
    }
    return PoincareMap(modes, transitions, "p_up", end_modes=["p_up", "p_down"])
