.. _api-continuation:

==============================================
Parameter continuation (``continuation``)
==============================================

.. module:: mappy_ds.continuation

.. currentmodule:: mappy_ds.continuation

Natural-parameter continuation driven by a user-supplied step function, and
pseudo-arclength continuation of a curve ``F(u) = 0`` given ``F`` and its
Jacobian. The specialised continuations :func:`mappy_ds.root.trace_cycle`,
:func:`mappy_ds.root.trace_local_bf` and
:func:`mappy_ds.root.trace_cycle_arclength` are built on them.

Natural parameter
-----------------

.. autosummary::
   :toctree: generated/

   continuation
   StepFunctionResult
   ContinuationStep

Pseudo-arclength
----------------

.. autosummary::
   :toctree: generated/

   arclength_continuation
   ArclengthPoint
