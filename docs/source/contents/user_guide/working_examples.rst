.. _working-examples:

================
Working examples
================

The ``examples/`` directory of the `repository
<https://github.com/Yuu-Miino/mappy/tree/master/examples>`_ contains one
script per model below. They are not part of the installed package; run them
from a clone of the repository:

.. code-block:: sh

   git clone https://github.com/Yuu-Miino/mappy.git
   cd mappy
   uv sync --all-extras          # or: pip install -e ".[plot]"
   uv run python examples/henon.py --help

Each script takes a ``--mode`` option selecting what to do; ``--help`` lists
the choices. Typically:

* an interactive mode (the default) opens :func:`~mappy_ds.plot.plot2d`
  (needs a display);
* a non-interactive mode iterates the map and saves a figure (works
  headless, e.g. with ``MPLBACKEND=Agg``);
* ``trace`` runs the numerical analysis — periodic points, their
  continuation, local bifurcations and bifurcation curves — and prints the
  results.

For example:

.. code-block:: sh

   uv run python examples/duffing.py --mode trace

The mode graphs below are written as in :doc:`concept_of_modes`; the mode
names used inside the scripts may differ.

Hénon map
=========

``examples/henon.py``. The classic two-dimensional map

.. math::

   x_{k+1} = 1 - a x_k^2 + y_k, \qquad y_{k+1} = b x_k,

with :math:`a = 1.4`, :math:`b = 0.3` (the Hénon attractor). It is
discrete from the start, so the whole system is one discrete mode wrapped in
a :class:`~mappy_ds.Diffeomorphism`.

Mode graph:

.. code-block:: text

   map ──▶ map

The interactive plot shows only the image points (``only_map=True``); ``a``
and ``b`` can be changed with the keyboard. Two further modes sweep
:math:`a \in [1, 1.42]`: ``--mode bifurcation`` draws the one-parameter
bifurcation diagram of :math:`x`, and ``--mode mle`` the maximum Lyapunov
exponent by Benettin's method — a unit tangent vector is pushed forward by
``res.jac`` at every iteration and renormalised — which is positive exactly
in the chaotic windows of the diagram.

Izhikevich neuron
=================

``examples/izhikevich.py``. The spiking neuron of the
:doc:`tutorial <../getting_started>`:

.. math::

   \dot v = 0.04 v^2 + 5 v + 140 - u + I, \qquad \dot u = a (b v - u),
   \qquad
   v = 30 \;\Rightarrow\; (v, u) \leftarrow (c, u + d),

with :math:`a = 0.2`, :math:`b = 0.2`, :math:`c = -50`, :math:`d = 2`,
:math:`I = 10`. The Poincaré section is the firing threshold
:math:`v = 30`, parametrised by :math:`u`, so the Poincaré map is
one-dimensional.

Mode graph:

.. code-block:: text

   Pinv (R → R²) ──▶ B (jump) ──▶ A (flow) ──[v = 30]──▶ P (R² → R) ──▶ Pinv

The analysis follows the fixed point as :math:`I` decreases, finds the
period-doubling bifurcation (:math:`I \approx 4.5869` at :math:`a = 0.2`)
and traces it in the :math:`(a, I)` plane — see
:doc:`root_and_continuation`.

Forced Duffing oscillator
=========================

``examples/duffing.py``. A periodically forced Duffing equation

.. math::

   \dot x = y, \qquad
   \dot y = -k y - x^3 + B_0 + B \cos t,

with :math:`k = 0.2`, :math:`B_0 = 0.1`, :math:`B = 0.1`. Time is added as a
third state variable, :math:`\dot t = 1`, so that the system is autonomous;
the stroboscopic Poincaré map samples :math:`(x, y)` once per forcing period
:math:`2\pi`. The forcing must be written with ``sympy.cos`` (see
:doc:`derivatives`).

Mode graph:

.. code-block:: text

   p (R² → R³) ──▶ duffing ──[t = 2π]──▶ t_reset ──▶ pinv (R³ → R²) ──▶ p

* ``p`` lifts :math:`(x, y)` to :math:`(x, y, 0)`;
* ``duffing`` integrates until the border :math:`t - 2\pi = 0`;
* ``t_reset`` subtracts :math:`2\pi` from :math:`t`;
* ``pinv`` projects back onto :math:`(x, y)`.

The ``trace`` mode follows the fixed point along :math:`B` and locates
period-doubling bifurcations (the first at :math:`B \approx 0.17169` for
:math:`k = 0.2`, :math:`B_0 = 0.1`), then continues one of them in the
:math:`(B, B_0)` plane.
``--mode bifurcation`` draws the one-parameter bifurcation diagram of
:math:`x` on the section over :math:`B \in [0.1, 0.375]`; the 1-periodic
point is unstable between the two period doublings.

Forced Alpazur oscillator
=========================

``examples/forced_alpazur.py``. A piecewise-smooth oscillator whose
nonlinear element switches between two characteristics with hysteresis,
under periodic forcing:

.. math::

   \dot v_1 = -k v_1 - v_2 + V_0 + V \cos \omega t, \qquad
   \dot v_2 = v_1 + (1 - g_i) v_2 - \frac{v_2^3}{3} + B_i, \qquad
   \dot t = 1,

where :math:`i \in \{1, 2\}` is the active region. The system switches from
region 1 to region 2 when :math:`v_2` falls through :math:`-1`, and from
region 2 back to region 1 when :math:`v_2` rises through :math:`-0.1`. The
example uses :math:`k = 0.1`, :math:`g_1 = 0.2`, :math:`g_2 = 2`,
:math:`B_1 = 1.3`, :math:`B_2 = 5`, :math:`V_0 = 0`, :math:`V = 2.4`,
:math:`\omega = 1.26`.

Mode graph (two continuous modes with two borders each — a switching border
and the stroboscopic border :math:`t = 2\pi/\omega`):

.. code-block:: text

   p1    ──▶ alp_1
   alp_1 ──[border 0: v2 = -1, falling]───▶ alp_2
   alp_1 ──[border 1: t = 2π/ω]───────────▶ pr1 ──▶ pinv1 ──▶ p1   (end)
   p2    ──▶ alp_2
   alp_2 ──[border 0: v2 = -0.1, rising]──▶ alp_1
   alp_2 ──[border 1: t = 2π/ω]───────────▶ pr2 ──▶ pinv2 ──▶ p2   (end)

Because the state on the section may be in either region, the section has
two pieces and the map has two end modes; ``mode_history`` records the
switchings that happened within each period. Trajectories are coloured by
region with per-mode :class:`~mappy_ds.plot.LineStyle` dictionaries (see
:doc:`plotting`). See :doc:`concept_of_modes` for a detailed description of
this mode graph.

Three-dimensional coupled logistic map
======================================

``examples/logistic_3d.py``. Three cyclically coupled logistic maps

.. math::

   \begin{aligned}
   x_{k+1} &= \alpha x_k (1 - x_k) + \beta y_k^2 x_k + \gamma z_k^3, \\
   y_{k+1} &= \alpha y_k (1 - y_k) + \beta z_k^2 y_k + \gamma x_k^3, \\
   z_{k+1} &= \alpha z_k (1 - z_k) + \beta x_k^2 z_k + \gamma y_k^3,
   \end{aligned}

a :class:`~mappy_ds.Diffeomorphism` on :math:`\R^3` (mode graph
``map ──▶ map``). The example starts at :math:`\alpha = 2.7`,
:math:`\beta = -1`, :math:`\gamma = 0.0146`, finds a fixed point, continues
it in :math:`\alpha`, locates a period-doubling bifurcation
(``theta = np.pi``) and traces it in the :math:`(\beta, \alpha)` plane. The
interactive mode plots a two-dimensional projection of the orbit.
