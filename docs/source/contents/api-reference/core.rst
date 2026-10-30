.. _api-core:

======================
Core (``mappy_ds``)
======================

.. module:: mappy_ds

.. currentmodule:: mappy_ds

The top-level package. All names below are importable directly from
``mappy_ds``.

.. data:: __version__
   :type: str

   Version string of the installed ``mappy-ds`` distribution (``"0+unknown"``
   when running from a source tree that is not installed).

System functions
----------------

Decorators that turn a plain Python function ``f(y, params)`` into a
:class:`SystemFunction` whose Jacobian and Hessian are derived symbolically.
See :doc:`/contents/user_guide/derivatives`.

.. autosummary::
   :toctree: generated/

   vector_field
   mapping
   border
   SystemFunction

Modes
-----

The smooth building blocks of a hybrid system.
See :doc:`/contents/user_guide/concept_of_modes`.

.. autosummary::
   :toctree: generated/

   Mode
   ContinuousMode
   DiscreteMode
   StepResult

Maps
----

.. autosummary::
   :toctree: generated/

   Map
   PoincareMap
   Diffeomorphism
   MapResult

.. data:: Transitions

   Type of the transition rules accepted by :class:`PoincareMap`:
   ``Mapping[str, str | Sequence[str]]``. Each key is a mode name; the value is
   the name of the next mode, or for a continuous mode one next mode per
   border (in the order of :attr:`ContinuousMode.borders`).

Trajectories
------------

.. autosummary::
   :toctree: generated/

   Trajectory
   TrajectorySegment

Exceptions
----------

All exceptions raised deliberately by the package derive from
:class:`MappyError`.

.. autosummary::
   :toctree: generated/

   MappyError
   SymbolicEvaluationError
   UnknownModeError
   TransitionError
   ModeNotTerminatedError
