"""Periodic points, local bifurcations and their continuation.

* :func:`find_cycle` / :func:`trace_cycle` locate a periodic point of a map
  and follow it along a parameter; :func:`trace_cycle_arclength` follows it
  by pseudo-arclength continuation, through folds, and flags bifurcations.
* :func:`find_local_bf` / :func:`trace_local_bf` locate a tangent,
  period-doubling or Neimark–Sacker bifurcation of a periodic point and
  follow it along a second parameter.
"""

from ..continuation import ContinuationStep
from ._cycle import CycleResult, find_cycle, trace_cycle, trace_cycle_arclength
from ._local_bf import LocalBfResult, find_local_bf, trace_local_bf

__all__ = [
    "ContinuationStep",
    "CycleResult",
    "LocalBfResult",
    "find_cycle",
    "find_local_bf",
    "trace_cycle",
    "trace_cycle_arclength",
    "trace_local_bf",
]
