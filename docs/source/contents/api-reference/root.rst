.. _api-root:

=============================================
Periodic points and bifurcations (``root``)
=============================================

.. module:: mappy_ds.root

.. currentmodule:: mappy_ds.root

Root finding for periodic points and local bifurcations of any
:class:`~mappy_ds.Map`, and their continuation. See
:doc:`/contents/user_guide/root_and_continuation` for a walkthrough.

Periodic points
---------------

.. autosummary::
   :toctree: generated/

   find_cycle
   trace_cycle
   trace_cycle_arclength
   CycleResult

Local bifurcations
------------------

.. autosummary::
   :toctree: generated/

   find_local_bf
   trace_local_bf
   LocalBfResult
