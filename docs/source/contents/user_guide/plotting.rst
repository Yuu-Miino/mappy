.. _plotting:

========
Plotting
========

:mod:`mappy_ds.plot` contains two helpers built on matplotlib: a static
:func:`~mappy_ds.plot.plot_trajectory` and an interactive
:func:`~mappy_ds.plot.plot2d`. Matplotlib is an optional dependency:

.. code-block:: sh

   pip install "mappy-ds[plot]"

``import mappy_ds.plot`` works without matplotlib; calling a plotting function
then raises :class:`ImportError` with the install hint.

Trajectories
============

:meth:`Map.trajectory <mappy_ds.Map.trajectory>` (or ``image_detail(...,
trajectory=True)``) returns a :class:`~mappy_ds.Trajectory`: a list of
:class:`~mappy_ds.TrajectorySegment`, one per visited mode whose
``in_trajectory`` flag is set. A segment knows its ``mode`` name, its
``kind`` (``"continuous"`` or ``"discrete"``), the sampled states ``y`` of
shape ``(dim, n_points)`` and, for continuous segments, the sampling times
``t`` relative to the start of the segment. ``Trajectory.y`` concatenates all
segments when they share the same dimension.

Static plots: ``plot_trajectory``
=================================

:func:`~mappy_ds.plot.plot_trajectory` draws the segments of a trajectory on
a matplotlib Axes (a new figure if ``ax`` is omitted) and returns the Axes.
``xkey`` and ``ykey`` select the state components; segments whose dimension
is too small (for instance on a one-dimensional section) are skipped.
``styles`` is either a single :class:`~mappy_ds.plot.LineStyle` or a
dictionary from mode name to style, with the optional key ``"_default"`` for
modes it does not list.

.. plot::
   :context: reset

   import numpy as np
   import matplotlib.pyplot as plt
   from mappy_ds import Diffeomorphism, mapping
   from mappy_ds.plot import LineStyle, plot_trajectory

   @mapping(2, param_keys=["a", "b"])
   def henon(y, p):
       x, u = y
       return np.array([1 - p["a"] * x**2 + u, p["b"] * x])

   henon_map = Diffeomorphism(henon)
   params = {"a": 1.4, "b": 0.3}

   fig, ax = plt.subplots()
   # Each iteration is a two-point segment of the mode "map".
   traj = henon_map.trajectory([0.1, 0.1], params, n=30)
   plot_trajectory(traj, ax, styles=LineStyle("tab:gray", 0.4))

   # The attractor itself: the points of a long orbit.
   y, points = np.array([0.1, 0.1]), []
   for _ in range(2000):
       y = henon_map.image(y, params)
       points.append(y)
   points = np.array(points)
   ax.plot(points[100:, 0], points[100:, 1], ".", markersize=1, color="tab:red")
   ax.set(xlabel="x", ylabel="y")

Interactive plots: ``plot2d``
=============================

:func:`~mappy_ds.plot.plot2d` iterates a map forever in a matplotlib
animation. Each frame computes one iteration, draws the trajectory segments
(unless ``only_map``) and the image point, then continues from the image:

.. code-block:: python

   from mappy_ds.plot import LineStyle, PlotConfig, plot2d

   config = PlotConfig(
       xlim=(-1.5, 1.5), ylim=(-0.5, 0.5), xlabel="x", ylabel="y",
       only_map=True,
       param_keys=["a", "b"], param_step=0.01,
       point_style=LineStyle("teal", 1.0),
   )
   anim = plot2d(henon_map, [0.1, 0.1], params, config)

It needs an interactive matplotlib backend. With ``show=True`` (the default)
it calls :func:`matplotlib.pyplot.show` and blocks; with ``show=False`` it
returns the :class:`~matplotlib.animation.FuncAnimation` immediately — keep a
reference to it, or the animation stops.

Controls
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Input
     - Action
   * - mouse click
     - Restart from the clicked point. For a two-dimensional (or larger)
       state, components ``xkey`` and ``ykey`` are set to the click
       coordinates. For a one-dimensional state, the coordinate on
       ``config.scalar_axis`` (``"x"`` or ``"y"``) becomes the state. A
       :class:`~mappy_ds.PoincareMap` restarts in its start mode.
   * - space or ``e``
     - Clear the axes.
   * - ``f``
     - Toggle ``only_map`` (draw only the image points, or the trajectories
       too). Matplotlib's full-screen binding of ``f`` is removed.
   * - ``p``
     - Select the next parameter of ``config.param_keys``; its name is printed.
   * - up / down
     - Increase / decrease the selected parameter by ``config.param_step``;
       the parameters are printed. The map is not reset.

If an iteration fails (for example a continuous mode does not reach its
border), a message is printed and the animation pauses until you click a new
initial point.

For a :class:`~mappy_ds.PoincareMap`, ``mode=`` starts in a mode other than
``initial_mode``; later iterations continue from the end mode reached by each
image, so maps with several end modes work as expected.

``PlotConfig``
--------------

:class:`~mappy_ds.plot.PlotConfig` is a dataclass; every field has a default.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Default
     - Meaning
   * - ``only_map``
     - ``False``
     - Draw only the image points.
   * - ``figsize``
     - ``(6, 6)``
     - Figure size in inches.
   * - ``xlabel``, ``ylabel``
     - ``"x"``, ``"y"``
     - Axis labels.
   * - ``xlim``, ``ylim``
     - ``(-3, 3)``
     - Axis limits.
   * - ``xkey``, ``ykey``
     - ``0``, ``1``
     - State components on the horizontal and vertical axes.
   * - ``linewidth``
     - ``1.0``
     - Width of trajectory lines.
   * - ``markersize``
     - ``3.0``
     - Size of point markers.
   * - ``param_keys``
     - ``[]``
     - Parameters adjustable with the keyboard.
   * - ``param_index``
     - ``0``
     - Index into ``param_keys`` of the parameter selected first.
   * - ``param_step``
     - ``0.1``
     - Increment of one up/down key press.
   * - ``scalar_axis``
     - ``"x"``
     - For one-dimensional states, the axis read on click.
   * - ``trajectory_style``
     - ``LineStyle()`` (black, 0.3)
     - Style of trajectory segments, one for all or one per mode.
   * - ``point_style``
     - ``LineStyle("red", 1.0)``
     - Style of image points, chosen by the mode of the last segment.
   * - ``click_style``
     - ``LineStyle("blue", 1.0)``
     - Style of the marker at a clicked initial point.

``LineStyle``
-------------

:class:`~mappy_ds.plot.LineStyle` is a frozen dataclass with a matplotlib
``color`` (default ``"black"``) and an ``alpha`` (default ``0.3``). Per-mode
styles are dictionaries; for example, to colour two regions of a switched
system differently:

.. code-block:: python

   styles = {
       "alp_1": LineStyle("orange", 0.6),
       "alp_2": LineStyle("teal", 0.6),
       "_default": LineStyle("gray", 0.3),
   }
   PlotConfig(trajectory_style=styles, point_style=styles)
