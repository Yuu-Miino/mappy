"""Type aliases shared across the package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

#: Parameters of a system: a mapping from parameter name to value.
Params: TypeAlias = Mapping[str, Any]

#: Anything accepted as a state vector (scalar, sequence or array).
StateLike: TypeAlias = npt.ArrayLike

#: A one-dimensional float array holding a state vector.
State: TypeAlias = npt.NDArray[np.float64]

#: A Jacobian matrix (``cod_dim``, ``dom_dim``).
Jacobian: TypeAlias = npt.NDArray[np.float64]

#: A Hessian tensor (``cod_dim``, ``dom_dim``, ``dom_dim``).
Hessian: TypeAlias = npt.NDArray[np.float64]


def as_state(y: StateLike) -> State:
    """Convert ``y`` to a one-dimensional float64 array.

    Scalars become arrays of shape ``(1,)``; array inputs are copied so the
    caller's data is never aliased.
    """
    arr = np.array(y, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"state must be one-dimensional, got shape {arr.shape}")
    return arr
