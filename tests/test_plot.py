"""Tests of :mod:`mappy_ds.plot` (non-interactive, Agg backend)."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.colors import to_rgba
from numpy.testing import assert_allclose

from conftest import make_izhikevich_map
from mappy_ds import Trajectory, TrajectorySegment
from mappy_ds.plot import LineStyle, PlotConfig, plot2d, plot_trajectory
from mappy_ds.plot._plot2d import _setup_plot2d

HENON_PARAMS = {"a": 1.4, "b": 0.3}
_IZH = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0}


@pytest.fixture(autouse=True)
def _close_figures():
    # _setup_plot2d edits rcParams["keymap.fullscreen"]; restore it afterwards.
    with matplotlib.rc_context():
        yield
    plt.close("all")


def press(fig, key):
    event = KeyEvent("key_press_event", fig.canvas, key=key)
    fig.canvas.callbacks.process("key_press_event", event)


def click(fig, ax, x, y):
    px, py = ax.transData.transform((x, y))
    event = MouseEvent("button_press_event", fig.canvas, px, py, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


# -- plot_trajectory ---------------------------------------------------------------


def _segments():
    return Trajectory(
        [
            TrajectorySegment(
                "flow", "continuous", np.array([[0, 1, 2], [0, 1, 0.0]]), np.arange(3.0)
            ),
            TrajectorySegment("proj", "discrete", np.array([[1.0, 2.0]])),  # dim 1
            TrajectorySegment("jump", "discrete", np.array([[2.0, 0.0], [0.0, 1.0]])),
        ]
    )


def test_plot_trajectory_lines_and_skipping():
    ax = plot_trajectory(_segments())
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.lines) == 2  # the one-dimensional segment is skipped
    assert_allclose(ax.lines[0].get_xdata(), [0, 1, 2])
    assert_allclose(ax.lines[0].get_ydata(), [0, 1, 0])
    assert_allclose(ax.lines[1].get_xdata(), [2, 0])
    assert to_rgba(ax.lines[0].get_color()) == to_rgba("black")


def test_plot_trajectory_on_given_axes_with_keys():
    _, ax = plt.subplots()
    traj = Trajectory(
        [TrajectorySegment("m", "discrete", np.array([[0.0, 1], [2, 3], [4, 5]]))]
    )
    out = plot_trajectory(traj, ax, xkey=2, ykey=0, linewidth=2.5)
    assert out is ax
    assert_allclose(ax.lines[0].get_xdata(), [4, 5])
    assert_allclose(ax.lines[0].get_ydata(), [0, 1])
    assert ax.lines[0].get_linewidth() == 2.5
    # ykey beyond every segment: nothing is drawn.
    plot_trajectory(traj, ax, ykey=3)
    assert len(ax.lines) == 1


def test_plot_trajectory_per_mode_styles():
    styles = {
        "flow": LineStyle("orange", 0.5),
        "_default": LineStyle("teal", 0.8),
    }
    ax = plot_trajectory(_segments(), styles=styles)
    flow, jump = ax.lines
    assert to_rgba(flow.get_color()) == to_rgba("orange")
    assert flow.get_alpha() == 0.5
    assert to_rgba(jump.get_color()) == to_rgba("teal")
    assert jump.get_alpha() == 0.8


def test_plot_trajectory_mode_without_style_or_default():
    ax = plot_trajectory(_segments(), styles={"flow": LineStyle("orange", 0.5)})
    assert to_rgba(ax.lines[1].get_color()) == to_rgba("black")


def test_plot_trajectory_single_style():
    ax = plot_trajectory(_segments(), styles=LineStyle("green", 0.4))
    assert all(to_rgba(line.get_color()) == to_rgba("green") for line in ax.lines)


def test_plot_trajectory_of_izhikevich_map():
    traj = make_izhikevich_map().trajectory(-3.0, {**_IZH, "I": 10.0}, n=2)
    ax = plot_trajectory(traj, styles={"A": LineStyle("orange", 1.0)})
    assert len(ax.lines) == 2


# -- plot2d ------------------------------------------------------------------------


def test_plot2d_returns_animation(henon_map):
    anim = plot2d(
        henon_map, [0.1, 0.1], HENON_PARAMS, PlotConfig(only_map=True), show=False
    )
    assert isinstance(anim, FuncAnimation)
    fig = anim._fig
    ax = fig.axes[0]
    # Drawing starts the animation, which renders the first frame.
    fig.canvas.draw()
    assert len(ax.lines) >= 1


def test_plot2d_show_under_agg(henon_map):
    # Agg cannot show figures; some matplotlib versions warn about it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
        anim = plot2d(henon_map, [0.1, 0.1], HENON_PARAMS, show=True)
    assert isinstance(anim, FuncAnimation)
    anim._fig.canvas.draw()


def test_plot2d_mode_requires_poincare_map(henon_map):
    with pytest.raises(ValueError, match="PoincareMap"):
        plot2d(henon_map, [0.1, 0.1], HENON_PARAMS, mode="map", show=False)


def test_update_only_map(henon_map):
    config = PlotConfig(only_map=True)
    fig, update, state = _setup_plot2d(
        henon_map, [0.1, 0.1], HENON_PARAMS, config, None
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 0
    assert state.clicked
    y = np.array([0.1, 0.1])
    for _ in range(3):
        update(0)
        y = henon_map.image(y, HENON_PARAMS)
    assert_allclose(state.y, y)
    assert not state.clicked
    # One marker for the clicked initial state plus one point per iteration.
    assert len(ax.lines) == 4
    assert_allclose(ax.lines[0].get_xydata(), [[0.1, 0.1]])
    assert_allclose(ax.lines[-1].get_xydata(), [y])


def test_update_with_trajectories(henon_map):
    style = LineStyle("purple", 0.7)
    config = PlotConfig(trajectory_style=style, point_style=LineStyle("green", 1.0))
    fig, update, _ = _setup_plot2d(henon_map, [0.1, 0.1], HENON_PARAMS, config, None)
    ax = fig.axes[0]
    update(0)
    # click marker, trajectory segment, image point
    assert len(ax.lines) == 3
    assert to_rgba(ax.lines[1].get_color()) == to_rgba("purple")
    assert ax.lines[1].get_alpha() == 0.7
    assert to_rgba(ax.lines[2].get_color())[:3] == to_rgba("green")[:3]


def test_key_presses(henon_map, capsys):
    config = PlotConfig(only_map=True, param_keys=["a", "b"], param_step=0.1)
    fig, update, state = _setup_plot2d(
        henon_map, [0.1, 0.1], HENON_PARAMS, config, None
    )
    ax = fig.axes[0]
    for _ in range(3):
        update(0)
    assert len(ax.lines) > 0

    press(fig, "f")
    assert state.only_map is False
    assert len(ax.lines) == 0  # cleared
    press(fig, "f")
    assert state.only_map is True

    update(0)
    assert len(ax.lines) > 0
    press(fig, " ")
    assert len(ax.lines) == 0
    update(0)
    press(fig, "e")
    assert len(ax.lines) == 0

    assert state.param_index == 0
    press(fig, "up")
    assert state.params["a"] == pytest.approx(1.5)
    press(fig, "down")
    press(fig, "down")
    assert state.params["a"] == pytest.approx(1.3)
    press(fig, "p")
    assert state.param_index == 1
    press(fig, "up")
    assert state.params["b"] == pytest.approx(0.4)
    assert state.params["a"] == pytest.approx(1.3)
    press(fig, "p")
    assert state.param_index == 0  # wraps around
    out = capsys.readouterr().out
    assert "adjustable parameter: b" in out
    # The caller's params are not modified.
    assert HENON_PARAMS == {"a": 1.4, "b": 0.3}


def test_param_keys_ignored_without_config(henon_map):
    fig, _, state = _setup_plot2d(
        henon_map, [0.1, 0.1], HENON_PARAMS, PlotConfig(), None
    )
    press(fig, "p")
    press(fig, "up")
    assert state.param_index == 0
    assert state.params == HENON_PARAMS


def test_f_key_does_not_toggle_fullscreen(henon_map):
    _setup_plot2d(henon_map, [0.1, 0.1], HENON_PARAMS, PlotConfig(), None)
    assert "f" not in matplotlib.rcParams["keymap.fullscreen"]


def test_click_resets_state(henon_map, capsys):
    config = PlotConfig(only_map=True)
    fig, update, state = _setup_plot2d(
        henon_map, [0.1, 0.1], HENON_PARAMS, config, None
    )
    ax = fig.axes[0]
    update(0)
    update(0)
    assert not state.clicked
    click(fig, ax, 0.5, -0.5)
    assert_allclose(state.y, [0.5, -0.5])
    assert state.clicked
    assert not state.stalled
    n_lines = len(ax.lines)
    update(0)
    # A new click marker at the clicked state plus the image point.
    assert len(ax.lines) == n_lines + 2
    assert_allclose(ax.lines[-2].get_xydata(), [[0.5, -0.5]])
    assert_allclose(state.y, henon_map.image([0.5, -0.5], HENON_PARAMS))
    capsys.readouterr()


def test_click_outside_axes_is_ignored(henon_map):
    fig, _, state = _setup_plot2d(
        henon_map, [0.1, 0.1], HENON_PARAMS, PlotConfig(), None
    )
    state.clicked = False
    event = MouseEvent("button_press_event", fig.canvas, 0, 0, button=1)
    fig.canvas.callbacks.process("button_press_event", event)
    assert not state.clicked
    assert_allclose(state.y, [0.1, 0.1])


# -- plot2d with a PoincareMap --------------------------------------------------------


def test_poincare_map_scalar_state_and_mode(relay_map, relay_params):
    config = PlotConfig(xlim=(-2, 2), ylim=(-1, 3), scalar_axis="y")
    fig, update, state = _setup_plot2d(relay_map, [0.0], relay_params, config, None)
    ax = fig.axes[0]
    assert state.mode == "p_up"
    update(0)
    assert state.mode == "p_down"
    assert_allclose(state.y, [-0.2], atol=1e-9)
    assert len(ax.lines) > 0
    # Clicking restarts from the initial mode with the y coordinate.
    click(fig, ax, 1.0, 0.25)
    assert state.mode == "p_up"
    assert_allclose(state.y, [0.25])


def test_poincare_map_mode_argument(relay_map, relay_params):
    fig, update, state = _setup_plot2d(
        relay_map, [-0.2], relay_params, PlotConfig(), "p_down"
    )
    assert state.mode == "p_down"
    update(0)
    assert state.mode == "p_up"
    assert_allclose(state.y, [0.4], atol=1e-9)
    click(fig, fig.axes[0], 0.1, 0.0)
    assert state.mode == "p_down"
    assert_allclose(state.y, [0.1])


def test_failing_iteration_stalls(capsys):
    pmap = make_izhikevich_map()
    params = {**_IZH, "I": -200.0}  # never fires: ModeNotTerminatedError
    fig, update, state = _setup_plot2d(pmap, [-3.0], params, PlotConfig(), None)
    update(0)
    assert state.stalled
    assert "iteration failed" in capsys.readouterr().out
    n_lines = len(fig.axes[0].lines)
    update(0)  # no-op while stalled
    assert len(fig.axes[0].lines) == n_lines
    click(fig, fig.axes[0], -1.0, 0.0)
    assert not state.stalled
    assert state.mode == "Pinv"
