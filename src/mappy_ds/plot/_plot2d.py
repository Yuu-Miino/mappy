"""Interactive two-dimensional phase portraits of maps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._typing import Params, State, StateLike, as_state
from ..exceptions import MappyError
from ..maps import Map, MapResult, PoincareMap
from ..trajectory import Trajectory

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    from matplotlib.figure import Figure

__all__ = ["LineStyle", "PlotConfig", "plot2d", "plot_trajectory"]

#: Key of the fallback entry in a per-mode style dictionary.
DEFAULT_KEY = "_default"


@dataclass(frozen=True)
class LineStyle:
    """Color and opacity of plotted lines or markers.

    Attributes
    ----------
    color : str
        Any matplotlib color specification.
    alpha : float
        Opacity between ``0`` (transparent) and ``1`` (opaque).
    """

    color: str = "black"
    alpha: float = 0.3


#: A single style for every mode, or one style per mode name. A dictionary may
#: contain the key ``"_default"`` used for modes it does not list.
Styles = LineStyle | Mapping[str, LineStyle]


@dataclass
class PlotConfig:
    """Configuration of :func:`plot2d`.

    Attributes
    ----------
    only_map : bool
        Draw only the image points, not the trajectories between them.
    figsize : tuple of float
        Figure size in inches.
    xlabel, ylabel : str
        Axis labels.
    xlim, ylim : tuple of float
        Axis limits.
    xkey, ykey : int
        Indices of the state components on the horizontal and vertical axes.
    linewidth : float
        Width of trajectory lines.
    markersize : float
        Size of the point markers.
    param_keys : list of str
        Parameters that can be changed with the up/down keys.
    param_index : int
        Index into ``param_keys`` of the parameter changed first.
    param_step : float
        Increment applied by one up/down key press.
    scalar_axis : {"x", "y"}
        For one-dimensional maps, the axis whose coordinate becomes the state
        when the figure is clicked.
    trajectory_style : LineStyle or dict of str to LineStyle
        Style of trajectory segments, either one for all modes or one per mode
        (with optional ``"_default"`` fallback).
    point_style : LineStyle or dict of str to LineStyle
        Style of image points, chosen by the mode of the last trajectory
        segment.
    click_style : LineStyle
        Style of the marker placed at a clicked initial state.
    """

    only_map: bool = False
    figsize: tuple[float, float] = (6, 6)
    xlabel: str = "x"
    ylabel: str = "y"
    xlim: tuple[float, float] = (-3, 3)
    ylim: tuple[float, float] = (-3, 3)
    xkey: int = 0
    ykey: int = 1
    linewidth: float = 1.0
    markersize: float = 3.0
    param_keys: list[str] = field(default_factory=list)
    param_index: int = 0
    param_step: float = 0.1
    scalar_axis: Literal["x", "y"] = "x"
    trajectory_style: LineStyle | dict[str, LineStyle] = field(
        default_factory=LineStyle
    )
    point_style: LineStyle | dict[str, LineStyle] = field(
        default_factory=lambda: LineStyle("red", 1.0)
    )
    click_style: LineStyle = field(default_factory=lambda: LineStyle("blue", 1.0))


def _style_for(styles: Styles, mode: str | None, default: LineStyle) -> LineStyle:
    """Pick the style of ``mode`` from ``styles``.

    Parameters
    ----------
    styles : LineStyle or mapping of str to LineStyle
        A single style, or one style per mode name.
    mode : str or None
        Mode name to look up.
    default : LineStyle
        Style used when ``styles`` has neither ``mode`` nor ``"_default"``.

    Returns
    -------
    LineStyle
    """
    if isinstance(styles, LineStyle):
        return styles
    if mode is not None and mode in styles:
        return styles[mode]
    return styles.get(DEFAULT_KEY, default)


def _require_pyplot() -> Any:
    """Import :mod:`matplotlib.pyplot` or explain how to install it."""
    try:
        from matplotlib import pyplot
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting requires matplotlib; install it with "
            "`pip install 'mappy-ds[plot]'`"
        ) from exc
    return pyplot


def plot_trajectory(
    trajectory: Trajectory,
    ax: Axes | None = None,
    *,
    xkey: int = 0,
    ykey: int = 1,
    styles: Styles | None = None,
    linewidth: float = 1.0,
) -> Axes:
    """Draw a :class:`~mappy_ds.Trajectory` onto a matplotlib Axes.

    Segments whose dimension is too small to contain ``xkey`` and ``ykey``
    are skipped.

    Parameters
    ----------
    trajectory : Trajectory
        Trajectory to draw, e.g. from :meth:`mappy_ds.Map.trajectory`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    xkey, ykey : int, optional
        State components on the horizontal and vertical axes, by default
        ``0`` and ``1``.
    styles : LineStyle or mapping of str to LineStyle, optional
        One style for all segments or one per mode name (with optional
        ``"_default"`` fallback). By default black, fully opaque lines.
    linewidth : float, optional
        Line width, by default ``1.0``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on.
    """
    if ax is None:
        pyplot = _require_pyplot()
        _, ax = pyplot.subplots()
        assert ax is not None
    default = LineStyle("black", 1.0)
    if styles is None:
        styles = default
    for seg in trajectory:
        if seg.dim <= max(xkey, ykey):
            continue
        style = _style_for(styles, seg.mode, default)
        ax.plot(
            seg.y[xkey],
            seg.y[ykey],
            "-",
            linewidth=linewidth,
            color=style.color,
            alpha=style.alpha,
        )
    return ax


@dataclass
class _PlotState:
    """Mutable state of an interactive :func:`plot2d` session."""

    y: State
    mode: str | None
    params: dict[str, Any]
    only_map: bool
    param_index: int
    clicked: bool = True  # mark the very first initial state
    stalled: bool = False


def _draw_axes(ax: Axes, config: PlotConfig) -> None:
    """Apply limits, labels and grid of ``config`` to ``ax``."""
    ax.set_xlim(config.xlim)
    ax.set_ylim(config.ylim)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.grid(color="gainsboro", zorder=9)


def _clear(ax: Axes, config: PlotConfig) -> None:
    """Remove every drawn line from ``ax`` and redraw the axes decoration."""
    for line in list(ax.lines):
        line.remove()
    _draw_axes(ax, config)


def _setup_plot2d(
    diffeo: Map,
    y0: StateLike,
    params: Params | None,
    config: PlotConfig,
    mode: str | None,
) -> tuple[Figure, Callable[[int], None], _PlotState]:
    """Create the figure, connect the event handlers and build the updater.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    update : callable
        Animation callback computing and drawing one iteration.
    state : _PlotState
        Mutable session state shared by the callbacks.
    """
    pyplot = _require_pyplot()
    from matplotlib import rcParams

    is_pmap = isinstance(diffeo, PoincareMap)
    start_mode: str | None = None
    if isinstance(diffeo, PoincareMap):
        start_mode = diffeo.initial_mode if mode is None else mode
    elif mode is not None:
        raise ValueError("mode is only supported for PoincareMap")

    state = _PlotState(
        y=as_state(y0),
        mode=start_mode,
        params=dict(params) if params is not None else {},
        only_map=config.only_map,
        param_index=config.param_index,
    )

    fig, ax = pyplot.subplots(figsize=config.figsize)
    if "f" in rcParams["keymap.fullscreen"]:
        rcParams["keymap.fullscreen"].remove("f")
    _draw_axes(ax, config)

    def iterate() -> MapResult:
        if isinstance(diffeo, PoincareMap):
            return diffeo.image_detail(
                state.y, state.params, jacobian=False, trajectory=True, mode=state.mode
            )
        return diffeo.image_detail(
            state.y, state.params, jacobian=False, trajectory=True
        )

    def update(_frame: int) -> None:
        if state.stalled:
            return
        clicked = state.clicked
        state.clicked = False
        try:
            result = iterate()
        except (MappyError, RuntimeError, ValueError) as exc:
            print(f"iteration failed: {exc}; click to choose a new initial state")
            state.stalled = True
            return
        state.y = result.y
        if is_pmap:
            state.mode = result.end_mode

        segments = result.trajectory.segments if result.trajectory else []
        segments = [s for s in segments if s.dim > max(config.xkey, config.ykey)]
        msize = config.markersize
        if clicked and segments:
            first = segments[0].y
            cs = config.click_style
            ax.plot(
                first[config.xkey, 0],
                first[config.ykey, 0],
                ".",
                markersize=msize,
                color=cs.color,
                alpha=cs.alpha,
            )
        if not state.only_map:
            plot_trajectory(
                Trajectory(segments),
                ax,
                xkey=config.xkey,
                ykey=config.ykey,
                styles=_with_default(config.trajectory_style, LineStyle()),
                linewidth=config.linewidth,
            )
        if segments:
            last = segments[-1]
            px, py = last.y[config.xkey, -1], last.y[config.ykey, -1]
            ps = _style_for(config.point_style, last.mode, LineStyle("red", 1.0))
        elif result.y.size > max(config.xkey, config.ykey):
            px, py = result.y[config.xkey], result.y[config.ykey]
            ps = _style_for(config.point_style, state.mode, LineStyle("red", 1.0))
        else:
            return
        ax.plot(px, py, ".", markersize=msize, color=ps.color, alpha=ps.alpha)

    def on_key(event: KeyEvent) -> None:
        match event.key:
            case " " | "e":
                _clear(ax, config)
            case "f":
                state.only_map = not state.only_map
                _clear(ax, config)
            case "p":
                if not config.param_keys:
                    return
                state.param_index = (state.param_index + 1) % len(config.param_keys)
                print(f"adjustable parameter: {config.param_keys[state.param_index]}")
            case "up" | "down":
                if not config.param_keys:
                    return
                key = config.param_keys[state.param_index % len(config.param_keys)]
                step = config.param_step * (-1 if event.key == "down" else 1)
                state.params[key] = round(state.params[key] + step, 10)
                print({k: state.params[k] for k in config.param_keys})

    def on_click(event: MouseEvent) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        y = state.y.copy()
        if start_mode is not None and isinstance(diffeo, PoincareMap):
            # Restart from the section the user chose; its dimension is known.
            state.mode = start_mode
            dim = diffeo.modes[start_mode].dom_dim
            if y.shape != (dim,):
                y = np.zeros(dim)
        if y.size == 1:
            y[0] = event.xdata if config.scalar_axis == "x" else event.ydata
        else:
            y[config.xkey] = event.xdata
            y[config.ykey] = event.ydata
        print(y)
        state.y = y
        state.clicked = True
        state.stalled = False

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    return fig, update, state


def _with_default(styles: Styles, default: LineStyle) -> Styles:
    """Return ``styles`` with a ``"_default"`` entry, without mutating it."""
    if isinstance(styles, LineStyle) or DEFAULT_KEY in styles:
        return styles
    return {**styles, DEFAULT_KEY: default}


def plot2d(
    diffeo: Map,
    y0: StateLike,
    params: Params | None = None,
    config: PlotConfig | None = None,
    *,
    mode: str | None = None,
    show: bool = True,
) -> FuncAnimation:
    """Iterate a map forever and draw its orbit interactively.

    Each animation frame computes one iteration of ``diffeo`` and draws the
    trajectory segments (unless ``only_map``) and the image point.

    Interaction:

    - mouse click: restart from the clicked coordinates (for a
      one-dimensional state the coordinate on ``config.scalar_axis``);
    - space or ``e``: clear the axes;
    - ``f``: toggle drawing the trajectories (``only_map``);
    - ``p``: select the next parameter of ``config.param_keys``;
    - up/down: change the selected parameter by ``config.param_step``.

    Parameters
    ----------
    diffeo : Map
        The map to iterate, e.g. a :class:`~mappy_ds.Diffeomorphism` or a
        :class:`~mappy_ds.PoincareMap`.
    y0 : array_like
        Initial state.
    params : mapping, optional
        Parameters of the system; a copy is modified by the up/down keys.
    config : PlotConfig, optional
        Plot configuration; it is not modified.
    mode : str, optional
        For a :class:`~mappy_ds.PoincareMap`, the mode to start in instead of
        its ``initial_mode``. Later iterations continue from the mode each
        image ends in.
    show : bool, optional
        Call :func:`matplotlib.pyplot.show` (blocking), by default ``True``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The running animation; keep a reference to it while it runs.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``mode`` is given for a map other than a PoincareMap.
    """
    pyplot = _require_pyplot()
    from matplotlib.animation import FuncAnimation

    fig, update, _ = _setup_plot2d(
        diffeo, y0, params, config if config is not None else PlotConfig(), mode
    )
    anim = FuncAnimation(fig, update, interval=1, repeat=False, cache_frame_data=False)
    if show:
        pyplot.show()
    return anim
