"""Poincaré maps, periodic points and bifurcations of hybrid dynamical systems.

``mappy_ds`` implements the *mode concept*: a (possibly non-smooth) dynamical
system is split into smooth primitive pieces, *modes*, connected by
transition rules. From the modes the package builds the Poincaré map of the
system as a composite diffeomorphism and provides its Jacobian matrix and
Hessian tensor, which in turn feed periodic-point search, local bifurcation
analysis and parameter continuation.

Examples
--------
The Izhikevich neuron model with a firing threshold at ``v = 30``:

>>> import numpy as np
>>> from mappy_ds import (ContinuousMode, DiscreteMode, PoincareMap,
...                       border, mapping, vector_field)
>>> @vector_field(dim=2)
... def izhikevich(y, p):
...     v, u = y
...     return np.array([0.04 * v**2 + 5 * v + 140 - u + p["I"],
...                      p["a"] * (p["b"] * v - u)])
>>> @border(dim=2)
... def fire(y, _):
...     return y[0] - 30
>>> @mapping(2)
... def firing(y, p):
...     return y + np.array([p["c"] - 30, p["d"]])
>>> @mapping(2, 1)
... def reduce(y, _):
...     return np.array([0, 1]) @ y
>>> @mapping(1, 2)
... def revert(u, _):
...     return u * np.array([0, 1]) + np.array([30, 0])
>>> pmap = PoincareMap(
...     [ContinuousMode("A", izhikevich, borders=[fire]),
...      DiscreteMode("B", firing), DiscreteMode("P", reduce),
...      DiscreteMode("Pinv", revert)],
...     {"A": ["P"], "B": "A", "P": "Pinv", "Pinv": "B"},
...     initial_mode="Pinv",
... )
>>> params = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}
>>> res = pmap.image_detail(-3.0, params, hessian=True)
>>> print(f"{res.y[0]:.5f} {res.jac[0, 0]:.5f} {res.hes[0, 0, 0]:.5f}")
-3.47838 0.27080 -0.19326
>>> res.mode_history
['Pinv', 'B', 'A', 'P', 'Pinv']
"""

from importlib.metadata import PackageNotFoundError, version

from .exceptions import (
    MappyError,
    ModeNotTerminatedError,
    SymbolicEvaluationError,
    TransitionError,
    UnknownModeError,
)
from .functions import SystemFunction, border, mapping, vector_field
from .maps import Diffeomorphism, Map, MapResult, PoincareMap, Transitions
from .modes import ContinuousMode, DiscreteMode, Mode, StepResult
from .trajectory import Trajectory, TrajectorySegment

try:
    __version__ = version("mappy-ds")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "ContinuousMode",
    "Diffeomorphism",
    "DiscreteMode",
    "Map",
    "MapResult",
    "MappyError",
    "Mode",
    "ModeNotTerminatedError",
    "PoincareMap",
    "StepResult",
    "SymbolicEvaluationError",
    "SystemFunction",
    "Trajectory",
    "TrajectorySegment",
    "TransitionError",
    "Transitions",
    "UnknownModeError",
    "__version__",
    "border",
    "mapping",
    "vector_field",
]
