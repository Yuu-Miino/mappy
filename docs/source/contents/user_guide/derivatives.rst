.. _derivatives:

===========================
Jacobian and Hessian tensor
===========================

Every map in mappy-ds can return, besides the image of a point, its Jacobian
matrix and its Hessian tensor with respect to the initial state. They are
*exact* up to the accuracy of the ODE solver — no finite differences are
involved. This page explains where they come from.

Index conventions
=================

For a map :math:`\bm y = T(\bm y_0)` from :math:`\R^n` to :math:`\R^m`:

.. math::

   \mathtt{jac}[i, j] = \pderiv{y_i}{y_{0,j}}, \qquad
   \mathtt{hes}[i, j, k] = \frac{\partial^2 y_i}{\partial y_{0,j}\,\partial y_{0,k}},

so ``jac`` has shape ``(m, n)`` and ``hes`` has shape ``(m, n, n)``. The same
convention is used by :class:`~mappy_ds.SystemFunction` (derivatives of
:math:`\bm f` with respect to :math:`\bm y`), :class:`~mappy_ds.StepResult`
(one mode) and :class:`~mappy_ds.MapResult` (a whole iteration).

.. code-block:: python

   res = pmap.image_detail(y0, params)                # Jacobian only (default)
   res = pmap.image_detail(y0, params, hessian=True)  # Jacobian and Hessian
   res = pmap.image_detail(y0, params, jacobian=False)  # image only

Requesting the Hessian implies the Jacobian. The eigenvalues of a square
Jacobian are available as :attr:`MapResult.eigvals
<mappy_ds.MapResult.eigvals>` (characteristic multipliers) and
:attr:`MapResult.eigvecs <mappy_ds.MapResult.eigvecs>`.

Symbolic tracing of system functions
====================================

The decorators :func:`~mappy_ds.vector_field`, :func:`~mappy_ds.mapping` and
:func:`~mappy_ds.border` wrap a Python function ``f(y, p)`` into a
:class:`~mappy_ds.SystemFunction`. On first use it calls ``f`` *once* with
SymPy symbols ``y_0, y_1, ...`` for the state and a parameter mapping that
hands out one symbol per key it is asked for — every ``p["a"]`` in the body
records ``"a"`` in :attr:`~mappy_ds.SystemFunction.param_keys`, so parameters
need no declaration (but must be read as ``p[key]``: ``p.get``, ``in`` and
iteration are rejected) — and differentiates the resulting expression
symbolically. The function,
its Jacobian and its Hessian are then compiled with
:func:`sympy.lambdify` into NumPy callables, which are what the integrator
evaluates.

.. code-block:: python

   >>> import numpy as np, sympy
   >>> from mappy_ds import vector_field
   >>> @vector_field(dim=2)
   ... def pendulum(y, p):
   ...     return np.array([y[1], -p["k"] * y[1] - sympy.cos(y[0])])
   >>> pendulum.expression
   Matrix([
   [              y_1],
   [-k*y_1 - cos(y_0)]])
   >>> pendulum.jacobian_expression
   Matrix([
   [       0,  1],
   [sin(y_0), -k]])
   >>> pendulum.jacobian([0.0, 0.0], {"k": 0.1})
   array([[ 0. ,  1. ],
          [ 0. , -0.1]])

Because of the tracing, a system function must be expressible in SymPy:

* use ``sympy.cos``, ``sympy.exp``, ... instead of ``numpy.cos``,
  ``numpy.exp``, ... — NumPy ufuncs cannot be applied to SymPy symbols;
* arithmetic operators, ``**``, indexing, ``numpy.array`` construction,
  ``@`` with constant arrays and ``numpy.append`` all work, since they only
  combine the symbols;
* do not branch on the *values* of ``y`` (``if y[0] > 0``): a symbol has no
  truth value. Model the branches as separate modes and borders instead —
  that is precisely what the mode concept is for.

Violations raise :class:`~mappy_ds.SymbolicEvaluationError` the first time
the function is evaluated:

.. code-block:: python

   >>> @vector_field(dim=2)
   ... def bad(y, p):
   ...     return np.array([y[1], -p["k"] * y[1] - np.cos(y[0])])
   >>> bad([0.0, 0.0], {"k": 0.1})
   Traceback (most recent call last):
     ...
   mappy_ds.exceptions.SymbolicEvaluationError: cannot trace 'bad' symbolically:
   loop of ufunc does not support argument 0 of type Symbol which has no callable
   cos method

Discrete modes
==============

For a discrete mode :math:`\bm y = \bm g(\bm y_0)`, the Jacobian and Hessian
are those of :math:`\bm g`, evaluated from the compiled symbolic derivatives.

Continuous modes
================

Variational equations
---------------------

Let :math:`\bm\varphi(t, \bm x_0)` be the solution of
:math:`\dot{\bm x} = \bm f(\bm x)` with :math:`\bm\varphi(0, \bm x_0) =
\bm x_0`, and write :math:`J(t) = \pderiv{\bm\varphi}{\bm x_0}` and
:math:`H(t) = \pderiv{J}{\bm x_0}`. Differentiating the ODE with respect to
the initial state gives the first and second variational equations

.. math::

   \deriv{J}{t} = Df(\bm\varphi)\, J, \qquad J(0) = I,

.. math::

   \deriv{H_{ijk}}{t} = \sum_{l,m} \parens{D^2 f(\bm\varphi)}_{ilm} J_{lj} J_{mk}
                      + \sum_l \parens{Df(\bm\varphi)}_{il} H_{ljk},
   \qquad H(0) = 0,

where :math:`Df` and :math:`D^2 f` are the symbolic Jacobian and Hessian of
the vector field. When derivatives are requested, a continuous mode
integrates these :math:`n + n^2` (or :math:`n + n^2 + n^3`) equations together
with the state.

Border correction
-----------------

The mode does not stop at a fixed time but at the hitting time
:math:`\tau(\bm x_0)` of the border :math:`q(\bm x) = 0`. With
:math:`\bm x_1 = \bm\varphi(\tau, \bm x_0)`, the mode map is
:math:`T(\bm x_0) = \bm\varphi(\tau(\bm x_0), \bm x_0)`. Differentiating
:math:`q(T(\bm x_0)) = 0` yields

.. math::

   \pderiv{\tau}{\bm x_0} = -\frac{\nabla q(\bm x_1)\, J(\tau)}
                                  {\nabla q(\bm x_1) \cdot \bm f(\bm x_1)},

and therefore

.. math::

   DT = B(\bm x_1)\, J(\tau), \qquad
   B(\bm x) = I - \frac{\bm f(\bm x) \otimes \nabla q(\bm x)}
                       {\nabla q(\bm x) \cdot \bm f(\bm x)}.

:math:`B` is the projection along the flow onto the tangent space of the
border. Differentiating once more,

.. math::

   D^2 T = \parens{\pderiv{B}{\bm x}(\bm x_1)\, DT}\, J(\tau)
         + B(\bm x_1) \brackets{H(\tau)
         + \parens{Df(\bm x_1)\, J(\tau)} \otimes \pderiv{\tau}{\bm x_0}},

with, writing :math:`s(\bm x) = \nabla q(\bm x) \cdot \bm f(\bm x)`,

.. math::

   \pderiv{B_{ad}}{x_e} =
   -\frac{(Df)_{ae}\, (\nabla q)_d + f_a\, (\nabla^2 q)_{de}}{s}
   + \frac{f_a\, (\nabla q)_d}{s^2}\pderiv{s}{x_e}, \qquad
   \pderiv{s}{\bm x} = \nabla^2 q\, \bm f + \nabla q\, Df.

The correction requires the flow to cross the border transversally,
:math:`s(\bm x_1) \ne 0`. A continuous mode without borders is a fixed-time
flow map and needs no correction (:math:`DT = J`, :math:`D^2 T = H`).

These are the formulas of the reference paper (Miino 2024, eqs. (13)–(15) for
the Jacobian and (18)–(22) for the Hessian), written with the index
convention above.

Parameter derivatives
=====================

Pass ``parameter_jacobian=True`` to :meth:`~mappy_ds.Map.image_detail` to
obtain :attr:`MapResult.jac_p <mappy_ds.MapResult.jac_p>`, the derivative of
the image with respect to the parameters: ``jac_p[i, m] = ∂y_i/∂p_m``, with
``m`` indexing :attr:`Map.param_keys <mappy_ds.Map.param_keys>` — every
parameter read by any mode or border of the map, in order of appearance. The
derivative with respect to the system parameters is what parameter
continuation and the detection of bifurcations need; the sections above give
derivatives with respect to the state only.

For a discrete mode the parameter Jacobian :math:`\partial_p f` comes
straight from SymPy. For a continuous mode the sensitivity
:math:`P = \partial\varphi/\partial p` of the flow obeys

.. math::

   \deriv{P}{t} = \pderiv{\bm f}{\bm x}\parens{\varphi}\, P
                  + \pderiv{\bm f}{p}\parens{\varphi}, \qquad P(0) = 0,

which is integrated together with the state. At a border
:math:`q(\bm x, p) = 0` the hitting time depends on the parameters too, both
through the trajectory and through the border itself, and the mode map
:math:`\bm x_0 \mapsto \varphi(\tau(\bm x_0, p), \bm x_0, p)` has

.. math::

   \pderiv{T}{p} = B\, P(\tau) - \frac{\bm f(\bm x_1)}{\nabla q \cdot \bm f(\bm x_1)}
                   \otimes \pderiv{q}{p}(\bm x_1, p),

with the same :math:`B` as in the border correction above. Compositions
follow the chain rule :math:`\partial_p (g \circ f) = Dg\, \partial_p f +
\partial_p g`.

.. code-block:: python

   >>> res = pmap.image_detail(-3.0, params, parameter_jacobian=True)
   >>> res.param_keys
   ('I', 'a', 'b', 'c', 'd')
   >>> res.jac_p.shape
   (1, 5)

Mixed derivatives
=================

``parameter_hessian=True`` additionally returns :attr:`MapResult.hes_p
<mappy_ds.MapResult.hes_p>`, the mixed second derivative
``hes_p[i, j, m] = ∂²y_i/(∂y0_j ∂p_m)`` — the derivative of the Jacobian with
respect to the parameters. It is what the Jacobian of a bifurcation condition
needs (the conditions involve :math:`DT^n`, so their parameter derivative
involves :math:`\partial_p DT^n`), and what makes
:func:`~mappy_ds.root.find_local_bf` fully analytic.

For a continuous mode the sensitivity of the flow Jacobian,
:math:`M = \partial^2\varphi/\partial\bm x_0\,\partial p`, obeys

.. math::

   \deriv{M_{ijm}}{t} = \sum_{l,r} (D^2 f)_{ilr} J_{lj} P_{rm}
                      + \sum_l (\partial_p Df)_{ilm} J_{lj}
                      + \sum_l (Df)_{il} M_{ljm}, \qquad M(0) = 0,

integrated with the state, and at a border the parameter derivative
:math:`\partial_p T = B P + G`, :math:`G = -\bm f \otimes \partial_p q / s`,
is differentiated once more with respect to :math:`\bm x_0`. Every quantity
evaluated at the hitting point :math:`\bm x_1` depends on :math:`\bm x_0`
through the mode Jacobian :math:`J_T = B J`, and :math:`P(\tau)` also through
:math:`\tau`:

.. math::

   \pderiv{(\partial_p T)}{\bm x_0}
   = \parens{\pderiv{B}{\bm x} J_T} P
   + B \brackets{ M + \parens{Df\, P + \partial_p \bm f} \otimes \pderiv{\tau}{\bm x_0} }
   + \parens{\pderiv{G}{\bm x} J_T},
   \qquad
   \pderiv{G}{\bm x} = -\frac{Df \otimes \partial_p q + \bm f \otimes \partial^2_{\bm x p} q}{s}
                       + \frac{(\bm f \otimes \partial_p q) \otimes \partial_{\bm x} s}{s^2},

with :math:`\partial B/\partial\bm x` as in the border correction above. The
last term vanishes for borders that do not depend on the parameters.
Compositions follow :math:`\partial_p D(g \circ f) = D^2 g\,[Df, \partial_p f]
+ (\partial_p Dg)\, Df + Dg\, \partial_p Df`.

:func:`~mappy_ds.root.find_local_bf` uses ``jac``, ``hes``, ``jac_p`` and
``hes_p`` to supply the root finder with a fully analytic Jacobian of the
bifurcation conditions.

Composition
===========

One iteration of a :class:`~mappy_ds.PoincareMap` is the composition
:math:`T = T_{m-1} \circ \cdots \circ T_0` of the mode maps visited. With
:math:`\bm x_{k+1} = T_k(\bm x_k)` and :math:`J_k`, :math:`H_k` the
derivatives of :math:`T_k \circ \cdots \circ T_0` with respect to
:math:`\bm x_0`, the chain rule gives the recurrences

.. math::

   J_k = DT_k(\bm x_k)\, J_{k-1},

.. math::

   (H_k)_{ijk'} = \sum_{l,m} \parens{D^2 T_k(\bm x_k)}_{ilm} (J_{k-1})_{lj} (J_{k-1})_{mk'}
                + \sum_l \parens{DT_k(\bm x_k)}_{il} (H_{k-1})_{ljk'},

starting from :math:`J_0 = DT_0`, :math:`H_0 = D^2 T_0`. The dimensions of
the intermediate spaces need not agree — a projection :math:`\R^2 \to \R`
simply has a :math:`1 \times 2` Jacobian — so :math:`J_k` is in general a
rectangular matrix; only the Jacobian of a complete iteration (from the
section back to the section) is square. In NumPy terms the update is

.. code-block:: python

   hes = (np.einsum("ilm,lj,mk->ijk", hes_step, jac, jac)
          + np.einsum("il,ljk->ijk", jac_step, hes))
   jac = jac_step @ jac

For ``n`` iterations of the map the same recurrence simply continues across
iterations.

Accuracy
========

The derivatives of continuous modes are as accurate as the integration. The
solver tolerances can be set per mode, e.g.
``ContinuousMode("A", f, borders=[q], rtol=1e-10, atol=1e-12)``. Note that the
state integrated alone and the state integrated together with the variational
equations follow slightly different step sequences; ``pmap.image(y0)`` and
``pmap.image_detail(y0).y`` therefore agree only to the solver tolerance.
