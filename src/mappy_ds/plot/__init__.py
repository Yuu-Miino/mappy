"""Plotting tools (requires the ``plot`` extra: ``pip install 'mappy-ds[plot]'``).

Matplotlib is imported lazily, so importing this module works without it;
calling a plotting function raises :class:`ImportError` if it is missing.
"""

from ._plot2d import LineStyle, PlotConfig, plot2d, plot_trajectory

__all__ = ["LineStyle", "PlotConfig", "plot2d", "plot_trajectory"]
