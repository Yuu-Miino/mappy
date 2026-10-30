"""Containers for computed trajectories."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["Trajectory", "TrajectorySegment"]


@dataclass(frozen=True)
class TrajectorySegment:
    """The part of a trajectory spent in a single mode.

    Attributes
    ----------
    mode : str
        Name of the mode.
    kind : {"continuous", "discrete"}
        Whether the segment comes from an ODE flow or a discrete map.
    y : numpy.ndarray
        Sampled states, shape ``(dim, n_points)``. A discrete segment holds the
        two points before and after the map.
    t : numpy.ndarray or None
        Sampling times relative to the start of the segment, shape
        ``(n_points,)``. ``None`` for discrete segments.
    """

    mode: str
    kind: Literal["continuous", "discrete"]
    y: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64] | None = None

    @property
    def dim(self) -> int:
        """Dimension of the state space of the segment."""
        return int(self.y.shape[0])


@dataclass
class Trajectory:
    """A trajectory as a sequence of :class:`TrajectorySegment`.

    Attributes
    ----------
    segments : list of TrajectorySegment
        Segments in chronological order. Modes with ``in_trajectory=False``
        do not contribute a segment.
    """

    segments: list[TrajectorySegment] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[TrajectorySegment]:
        return iter(self.segments)

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """All sampled states concatenated, shape ``(dim, n_points)``.

        Raises
        ------
        ValueError
            If the segments do not share the same dimension.
        """
        if not self.segments:
            return np.empty((0, 0))
        dims = {s.dim for s in self.segments}
        if len(dims) != 1:
            raise ValueError(f"segments have different dimensions: {sorted(dims)}")
        return np.concatenate([s.y for s in self.segments], axis=1)
