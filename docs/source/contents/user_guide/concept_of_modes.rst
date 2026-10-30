.. _concept-of-modes:

================
Concept of modes
================

Many systems of practical interest are *piecewise* smooth: an ODE whose
right-hand side switches between regions, an ODE with impacts or resets, a
periodically forced system observed once per period. Their Poincaré maps are
not given by a single smooth formula, which makes Jacobians — and therefore
Newton's method, stability analysis and bifurcation analysis — awkward to
obtain.

The *mode concept* (Miino, NOLTA 2024) splits such a system into primitive
smooth pieces, the **modes**, and represents the Poincaré map as a
composition of the maps they induce,

.. math::

   T = T_{m-1} \circ T_{m-2} \circ \cdots \circ T_1 \circ T_0, \qquad
   T_i : M_i \to M_{i+1},

where every :math:`T_i` is a smooth diffeomorphism between (possibly
different-dimensional) spaces :math:`M_i`. Derivatives of :math:`T` then
follow from the derivatives of the :math:`T_i` by the chain rule (see
:doc:`derivatives`).

Modes
=====

A mode is a named, smooth step from an initial state to a final state. Both
kinds of modes implement :meth:`Mode.step <mappy_ds.Mode.step>`, which
returns a :class:`~mappy_ds.StepResult` with the final state ``y`` and, on
request, the Jacobian ``jac`` and Hessian ``hes`` of the mode map.

Continuous modes
----------------

A :class:`~mappy_ds.ContinuousMode` integrates an autonomous ODE

.. math::

   \deriv{\bm x}{t} = \bm f(\bm x, p), \qquad \bm x \in \R^n,

declared with :func:`~mappy_ds.vector_field`, from :math:`\bm x_0` until the
trajectory :math:`\bm\varphi(t, \bm x_0)` reaches one of its *borders*. The
mode map is

.. math::

   T_i(\bm x_0) = \bm\varphi(\tau(\bm x_0), \bm x_0),

where :math:`\tau(\bm x_0)` is the hitting time. The integration uses
:func:`scipy.integrate.solve_ivp` (``method``, ``rtol``, ``atol`` and other
options are forwarded). The integration time is bounded by ``max_interval``
(default ``20.0``): if no border is reached by then,
:class:`~mappy_ds.ModeNotTerminatedError` is raised. A continuous mode
*without* borders is the time-``max_interval`` flow map.

Non-autonomous systems are made autonomous by adding time as a state
variable, :math:`\dot t = 1`, with a border :math:`t - T_{\mathrm{period}} = 0`
and a discrete mode that resets :math:`t` afterwards. Use SymPy functions for
the forcing (``sympy.cos(t)``) — see :doc:`derivatives`.

Discrete modes
--------------

A :class:`~mappy_ds.DiscreteMode` applies a map :math:`\bm x \mapsto
\bm g(\bm x, p)` once. Maps are declared with
``mapping(dom_dim, cod_dim)`` (:func:`~mappy_ds.mapping`); the dimensions may
differ, which is how a state is projected onto a lower-dimensional Poincaré
section and embedded back. Typical discrete modes are

* state jumps and resets (impact laws, firing resets, ``t ← t - T``);
* projections onto the section coordinates (:math:`\R^n \to \R^{n-1}`) and
  their inverses (:math:`\R^{n-1} \to \R^n`);
* the map itself, for a system that is discrete from the start (see
  :class:`~mappy_ds.Diffeomorphism`).

Trajectory recording
--------------------

Each mode has an ``in_trajectory`` flag: continuous modes contribute a
sampled :class:`~mappy_ds.TrajectorySegment` by default (sampling interval
``trajectory_dt``), discrete modes do not, unless ``in_trajectory=True``, in
which case they contribute a two-point segment (only when the domain and
codomain dimensions agree).

Borders
=======

A border is a scalar function :math:`q(\bm x, p)` declared with
:func:`~mappy_ds.border`; the continuous mode stops where :math:`q = 0`. The
``direction`` argument selects which zero crossings count:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``direction``
     - Terminates the mode when :math:`q` …
   * - ``1`` (default)
     - crosses zero from negative to positive;
   * - ``-1``
     - crosses zero from positive to negative;
   * - ``0``
     - crosses zero in either direction.

A continuous mode may have several borders, ``borders=[q0, q1, ...]``; the
first one reached terminates the mode and its index is reported as
:attr:`StepResult.border <mappy_ds.StepResult>`. That index selects the next
mode (below).

.. caution::

   A trajectory that *starts* exactly on a border, moving in the border's
   direction, terminates immediately with :math:`\tau = 0`. Make sure that a
   continuous mode starts off its borders — usually the preceding discrete
   mode (a reset or a jump) takes care of that.

The derivatives of a continuous mode map include a correction for the
dependence of :math:`\tau` on :math:`\bm x_0`. It requires the flow to be
transversal to the border, :math:`\nabla q \cdot \bm f \neq 0`: an exactly
tangential hit raises :class:`ZeroDivisionError`, and a nearly tangential
(grazing) hit makes the derivatives ill-conditioned.

Transitions
===========

The transitions of a :class:`~mappy_ds.PoincareMap` (type
:data:`~mappy_ds.Transitions`) map each mode name to its successor(s):

* a **string** — the next mode, regardless of how the mode ended. This is the
  form for discrete modes, for continuous modes without borders and, as a
  shorthand, for continuous modes with a single border;
* a **list** — one next mode per border, in the order of the ``borders`` of a
  continuous mode: ``{"flow": ["after_q0", "after_q1"]}``.

Unknown mode names raise :class:`~mappy_ds.UnknownModeError` when the map is
constructed. A mode without a transition rule, or a border index beyond the
end of the list, raises :class:`~mappy_ds.TransitionError` when it is reached.

Initial and end modes
=====================

One iteration of a :class:`~mappy_ds.PoincareMap` starts in ``initial_mode``
— the mode whose domain is the Poincaré section — and follows the
transitions until it enters one of the ``end_modes`` (by default just
``initial_mode``). The image is the state *entering* that end mode, so for
the default the map is a self-map of the section.

Several end modes describe a section made of several pieces, e.g. one per
region of a switched system. :attr:`MapResult.end_mode
<mappy_ds.MapResult.end_mode>` tells which piece the image is on, and
``image_detail(..., mode=name)`` starts an iteration from a mode other than
``initial_mode``. When ``n > 1`` iterations are requested, each iteration
starts from the end mode reached by the previous one.

``max_transitions`` (default ``1000``) bounds the number of mode transitions
per iteration and guards against transition loops that never reach an end
mode.

The mode graph
==============

Modes and transitions form a directed graph. The Poincaré map is a closed
walk in it from the initial mode back to an end mode; non-smooth features of
the system appear as *branches* of the graph, one per border of a continuous
mode.

**Izhikevich neuron** (the :doc:`tutorial <../getting_started>`). A single
loop; the section is the firing threshold:

.. code-block:: text

   Pinv ──▶ B ──▶ A ──[border 0: v = 30, rising]──▶ P ──▶ Pinv   (end)

.. list-table::
   :header-rows: 1

   * - Mode
     - Kind
     - Map
     - Next
   * - ``Pinv`` (initial/end)
     - discrete, :math:`\R \to \R^2`
     - :math:`u \mapsto (30, u)`
     - ``B``
   * - ``B``
     - discrete, :math:`\R^2 \to \R^2`
     - :math:`(v, u) \mapsto (c, u + d)`
     - ``A``
   * - ``A``
     - continuous on :math:`\R^2`
     - Izhikevich flow until :math:`v - 30 = 0`
     - ``["P"]``
   * - ``P``
     - discrete, :math:`\R^2 \to \R`
     - :math:`(v, u) \mapsto u`
     - ``Pinv``

**Switched system with hysteresis** (forced Alpazur oscillator, see
:doc:`working_examples`). Two continuous modes, each with a switching border
and the stroboscopic section border :math:`t = 2\pi/\omega`:

.. code-block:: text

   p1    ──▶ alp_1
   alp_1 ──[border 0: v2 = -1, falling]───▶ alp_2
   alp_1 ──[border 1: t = 2π/ω]───────────▶ pr1 ──▶ pinv1 ──▶ p1   (end)
   p2    ──▶ alp_2
   alp_2 ──[border 0: v2 = -0.1, rising]──▶ alp_1
   alp_2 ──[border 1: t = 2π/ω]───────────▶ pr2 ──▶ pinv2 ──▶ p2   (end)

.. list-table::
   :header-rows: 1

   * - Mode
     - Kind
     - Transitions
   * - ``alp_1``
     - continuous, borders ``[v2 + 1 (dir. -1), t - 2π/ω]``
     - ``["alp_2", "pr1"]``
   * - ``alp_2``
     - continuous, borders ``[v2 + 0.1 (dir. +1), t - 2π/ω]``
     - ``["alp_1", "pr2"]``
   * - ``pr1``, ``pr2``
     - discrete, :math:`t \leftarrow t - 2\pi/\omega`
     - ``"pinv1"``, ``"pinv2"``
   * - ``pinv1``, ``pinv2``
     - discrete, :math:`\R^3 \to \R^2`
     - ``"p1"``, ``"p2"``
   * - ``p1``, ``p2``
     - discrete, :math:`\R^2 \to \R^3`
     - ``"alp_1"``, ``"alp_2"``

Here the section consists of two pieces (which region the state is in when
the forcing period elapses), so the map is built with ``initial_mode="p1"``
and ``end_modes=["p1", "p2"]``. The two switching borders at different
levels of :math:`v_2` make the switching hysteretic.

Mode history
============

:attr:`MapResult.mode_history <mappy_ds.MapResult>` lists the names of the
modes visited, starting with the initial mode and ending with the end mode
reached; consecutive iterations are concatenated. For the Izhikevich map:

.. code-block:: python

   >>> pmap.image_detail(-3.0, params).mode_history
   ['Pinv', 'B', 'A', 'P', 'Pinv']
   >>> pmap.image_detail(-3.0, params, n=2).mode_history
   ['Pinv', 'B', 'A', 'P', 'Pinv', 'B', 'A', 'P', 'Pinv']

The history identifies *which branch* of a non-smooth system an orbit took —
for example how many switchings occurred within one forcing period. Since the
map is smooth only as long as the sequence of modes does not change, a
periodic point and its derivatives are always associated with a particular
mode history. A change of the history under parameter variation signals a
border-collision (grazing) event, which is not a local bifurcation in the
sense of :func:`~mappy_ds.root.find_local_bf`.

Discrete-time systems
=====================

A system that is a map from the start needs no section: wrap the map in a
:class:`~mappy_ds.Diffeomorphism`, which is a one-mode Poincaré map whose
mode is called ``"map"``:

.. code-block:: python

   >>> from mappy_ds import Diffeomorphism, mapping
   >>> @mapping(2, param_keys=["a", "b"])
   ... def henon(y, p):
   ...     x, u = y
   ...     return np.array([1 - p["a"] * x**2 + u, p["b"] * x])
   >>> henon_map = Diffeomorphism(henon)
   >>> henon_map.image([0.0, 0.0], {"a": 1.4, "b": 0.3}, n=2)
   array([-0.4,  0.3])

Every :class:`~mappy_ds.Map` — :class:`~mappy_ds.PoincareMap` or
:class:`~mappy_ds.Diffeomorphism` — offers ``image``, ``image_detail`` and
``trajectory`` and can be passed to everything in :mod:`mappy_ds.root` and
:mod:`mappy_ds.plot`.
