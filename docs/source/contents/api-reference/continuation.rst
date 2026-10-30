.. _api-continuation:

==============================================
Parameter continuation (``continuation``)
==============================================

.. module:: mappy_ds.continuation

.. currentmodule:: mappy_ds.continuation

Natural-parameter continuation driven by a user-supplied step function. The
specialised continuations :func:`mappy_ds.root.trace_cycle` and
:func:`mappy_ds.root.trace_local_bf` are built on it.

.. autosummary::
   :toctree: generated/

   continuation
   StepFunctionResult
   ContinuationStep
