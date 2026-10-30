.. _root-and-continuation:

============================================
Periodic points, bifurcations, continuation
============================================

:mod:`mappy_ds.root` locates periodic points and local bifurcations of any
:class:`~mappy_ds.Map` and follows them along parameters;
:mod:`mappy_ds.continuation` provides the generic continuation loop they are
built on. The examples on this page continue the Izhikevich map ``pmap`` and
``params`` of the :doc:`tutorial <../getting_started>`.

Periodic points: ``find_cycle``
===============================

A point :math:`\bm y` is :math:`n`-periodic if :math:`T^n(\bm y) = \bm y`.
:func:`~mappy_ds.root.find_cycle` solves

.. math::

   F(\bm y) = T^n(\bm y) - \bm y = 0

with :func:`scipy.optimize.root`, passing the analytic Jacobian
:math:`DF = DT^n(\bm y) - I` obtained from
:meth:`~mappy_ds.Map.image_detail` (see :doc:`derivatives`). Convergence is
therefore quadratic and costs few map evaluations.

.. code-block:: python

   >>> from mappy_ds.root import find_cycle
   >>> cycle = find_cycle(pmap, -3.0, params)            # period=1 by default
   >>> cycle.success, cycle.y, cycle.eigvals, cycle.nfev
   (True, array([-3.71591625]), array([0.38565236+0.j]), 8)

The returned :class:`~mappy_ds.root.CycleResult` holds the periodic point
``y``, its characteristic multipliers ``eigvals`` (eigenvalues of
:math:`DT^n`) and ``eigvecs``, the residual and the solver message. Extra
keyword arguments go to :func:`scipy.optimize.root`, e.g.
``find_cycle(pmap, y0, params, period=2, method="lm", tol=1e-12)``.

The periodic point is stable if all multipliers lie inside the unit circle.
As a parameter varies, a multiplier leaving the unit circle signals a *local
bifurcation*:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Multiplier
     - ``theta``
     - Bifurcation
   * - :math:`\mu = 1`
     - ``0``
     - tangent (saddle-node)
   * - :math:`\mu = -1`
     - ``np.pi``
     - period doubling
   * - :math:`\mu = e^{\pm i\theta}`
     - :math:`0 < \theta < \pi`
     - Neimark–Sacker (needs dimension :math:`\ge 2`)

Following a cycle: ``trace_cycle``
==================================

:func:`~mappy_ds.root.trace_cycle` sweeps ``params[param_key]`` linearly from
its current value to ``end_value`` in ``resolution`` points (both ends
included) and calls :func:`~mappy_ds.root.find_cycle` at each value, starting
from the previous periodic point. It stops at the first failure and returns
the list of successful steps:

.. code-block:: python

   >>> from mappy_ds.root import trace_cycle
   >>> steps = trace_cycle(pmap, cycle.y, params, "I", 4.6, resolution=55)
   >>> last = steps[-1]
   >>> last.params["I"], last.y, last.info["eigvals"]
   (4.6, array([-6.78367947]), array([-0.9849115+0.j]))

The multiplier approaches :math:`-1`: a period-doubling bifurcation is near.

.. tip::

   Choose ``resolution`` so that consecutive parameter values are close: the
   previous solution must be a good initial guess. With too coarse a step the
   root finder may wander to states where, e.g., a continuous mode never
   reaches its border, and the resulting
   :class:`~mappy_ds.ModeNotTerminatedError` propagates out of
   :func:`~mappy_ds.root.trace_cycle`. Pass ``show_progress=True`` to print
   one status line per step, or a ``callback`` to receive each step as soon
   as it is found.

Local bifurcations: ``find_local_bf``
=====================================

:func:`~mappy_ds.root.find_local_bf` solves for the periodic point
:math:`\bm y`, the bifurcation parameter :math:`p` (``params[param_key]``)
and the argument :math:`\theta` of the critical multiplier from

.. math::

   T^n(\bm y) - \bm y = 0, \qquad
   \Re \det\parens{DT^n(\bm y) - e^{i\theta} I} = 0, \qquad
   \Im \det\parens{DT^n(\bm y) - e^{i\theta} I} = 0,

a system of :math:`\dim + 2` equations in :math:`\dim + 2` unknowns. The
initial ``theta`` selects the kind of bifurcation (table above). The
Jacobian of this system is supplied analytically: the fixed-point rows use
:math:`DT^n - I` and the parameter derivative :attr:`MapResult.jac_p
<mappy_ds.MapResult.jac_p>`, the determinant rows use Jacobi's formula with
the Hessian of the map (for :math:`\bm y`) and the cofactor matrix (for
:math:`\theta`); only :math:`\partial (DT^n)/\partial p` is replaced by a
central difference.

.. code-block:: python

   >>> import numpy as np
   >>> from mappy_ds.root import find_local_bf
   >>> bf = find_local_bf(pmap, last.y, last.params, "I", np.pi)
   >>> bf.success, bf.y, bf.params["I"], bf.theta, bf.eigvals
   (True, array([-6.79440734]), 4.586870384786432, 3.141592653589793, array([-1.+0.j]))

The result, a :class:`~mappy_ds.root.LocalBfResult`, adds ``params`` (all
parameters with the solved bifurcation value) and ``theta`` to the fields of
:class:`~mappy_ds.root.CycleResult`. Here the period-doubling bifurcation is
at :math:`I \approx 4.5869` for :math:`a = 0.2`.

Bifurcation curves: ``trace_local_bf``
======================================

:func:`~mappy_ds.root.trace_local_bf` continues a bifurcation point in a
second parameter ``cnt_param_key``, solving for ``bf_param_key`` (and the
periodic point and ``theta``) at each step. The result is a bifurcation
curve in the (``cnt_param_key``, ``bf_param_key``) plane:

.. code-block:: python

   >>> from mappy_ds.root import trace_local_bf
   >>> curve = trace_local_bf(pmap, bf.y, bf.params, "I", bf.theta,
   ...                        "a", 0.25, resolution=6)
   >>> for s in curve:
   ...     print(f'a = {s.params["a"]:.2f}  I = {s.params["I"]:.5f}  '
   ...           f'u* = {s.y[0]:.5f}  theta = {s.info["theta"]:.5f}')
   a = 0.20  I = 4.58687  u* = -6.79441  theta = 3.14159
   a = 0.21  I = 4.48482  u* = -6.87018  theta = 3.14159
   a = 0.22  I = 4.39125  u* = -6.93696  theta = 3.14159
   a = 0.23  I = 4.30499  u* = -6.99591  theta = 3.14159
   a = 0.24  I = 4.22505  u* = -7.04801  theta = 3.14159
   a = 0.25  I = 4.15060  u* = -7.09411  theta = 3.14159

Each step's ``params`` holds both the continued and the solved parameter;
``y`` is the periodic point (without ``theta``); ``info`` holds ``theta``,
``eigvals``, ``eigvecs``, ``nfev`` and ``residual``.

Reading ``ContinuationStep``
============================

All continuations return a list of
:class:`~mappy_ds.continuation.ContinuationStep`, one per successful
parameter value:

``y``
   The value of the continued variable — a periodic point for
   :func:`~mappy_ds.root.trace_cycle` and
   :func:`~mappy_ds.root.trace_local_bf`.
``params``
   A dictionary of the parameters at that step (a private copy).
``info``
   Extra data returned by the step function: for
   :func:`~mappy_ds.root.trace_cycle` ``eigvals``, ``eigvecs``, ``nfev`` and
   ``residual``; :func:`~mappy_ds.root.trace_local_bf` adds ``theta``.

A list of steps converts readily into arrays for plotting:

.. code-block:: python

   a = np.array([s.params["a"] for s in curve])
   I = np.array([s.params["I"] for s in curve])
   plt.plot(a, I)

Custom continuation
===================

:func:`~mappy_ds.continuation.continuation` runs the same loop with a step
function of your own. The step function receives the previous value ``y``
and a dictionary of parameters (with the continued parameter already set) and
returns a :class:`~mappy_ds.continuation.StepFunctionResult` with

* ``success`` — ``False`` stops the continuation;
* ``y`` — the new value, which seeds the next step;
* ``params`` — optionally, modified parameters (``None`` keeps them), e.g.
  when the step solves for a second parameter;
* ``info`` — optionally, a dictionary stored in ``ContinuationStep.info``.

For example, follow the fixed point while recording its multiplier and
stability, and stop gracefully instead of raising if the map cannot be
evaluated:

.. code-block:: python

   >>> from mappy_ds import MappyError
   >>> from mappy_ds.continuation import StepFunctionResult, continuation
   >>> def step(y, p):
   ...     try:
   ...         res = find_cycle(pmap, y, p)
   ...     except MappyError:
   ...         return StepFunctionResult(False)
   ...     if not res.success:
   ...         return StepFunctionResult(False)
   ...     mu = res.eigvals[0].real
   ...     return StepFunctionResult(True, res.y, info={"mu": mu,
   ...                                                  "stable": abs(mu) < 1})
   >>> out = continuation(step, cycle.y, params, "I", 8.0, resolution=5)
   >>> [(s.params["I"], round(float(s.info["mu"]), 5)) for s in out]
   [(10.0, 0.38565), (9.5, 0.35363), (9.0, 0.31666), (8.5, 0.27341), (8.0, 0.22196)]

The continued variable may be anything (a float, an array, a tuple); the
continuation only passes it from one step to the next.
:func:`~mappy_ds.root.trace_local_bf`, for instance, continues the
concatenation of the periodic point and ``theta``.
