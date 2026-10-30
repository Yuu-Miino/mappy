.. _api-reference:

*************
API reference
*************

The public API of ``mappy_ds`` is split into four modules. Everything needed
to *describe* a system and evaluate its map lives in the top-level package;
periodic points and bifurcations are in :mod:`mappy_ds.root`, generic
parameter continuation in :mod:`mappy_ds.continuation` and plotting (optional,
needs matplotlib) in :mod:`mappy_ds.plot`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Contents
   * - :doc:`core`
     - System-function decorators, modes, maps, results, trajectories and
       exceptions.
   * - :doc:`root`
     - :func:`~mappy_ds.root.find_cycle`, :func:`~mappy_ds.root.trace_cycle`,
       :func:`~mappy_ds.root.find_local_bf`,
       :func:`~mappy_ds.root.trace_local_bf`.
   * - :doc:`continuation`
     - :func:`~mappy_ds.continuation.continuation` with a user-defined step
       function.
   * - :doc:`plot`
     - :func:`~mappy_ds.plot.plot2d` (interactive) and
       :func:`~mappy_ds.plot.plot_trajectory` (static).

.. toctree::
   :maxdepth: 2
   :hidden:

   core
   root
   continuation
   plot
