"""Hénon map: the simplest use of a discrete map with mappy_ds.

The map ``(x, y) -> (1 - a x^2 + y, b x)`` is declared with ``@mapping`` and
wrapped in a :class:`~mappy_ds.Diffeomorphism`. With the classical values
``a = 1.4, b = 0.3`` the orbit settles on the Hénon strange attractor.

Modes (``--mode``):

``plot`` (default)
    Interactive iteration: image points in teal accumulate on the attractor.
    Click to restart from a new initial point, press ``f`` to toggle the
    jumps between successive points (orange), ``p`` to switch between ``a``
    and ``b`` and the up/down keys to change it.
``orbit``
    Non-interactive: iterate the map ``--frames`` times and save the points
    to ``--output`` (PNG). Works without a display.

Run from the repository root, e.g.::

    uv run python examples/henon.py --mode orbit --frames 5000
"""

import argparse
import time

import numpy as np

from mappy_ds import Diffeomorphism, mapping
from mappy_ds.plot import LineStyle, PlotConfig, plot2d

Y0 = np.array([0.1, 0.1])
PARAMS = {"a": 1.4, "b": 0.3}

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "pgf", "ps", "svg", "cairo", "template")


@mapping(2, 2)
def henon(iny, param):
    x, y = iny
    a = param["a"]
    b = param["b"]

    return np.array([1 - a * x**2 + y, b * x])


def plot_config() -> PlotConfig:
    return PlotConfig(
        xlim=(-1.5, 1.5),
        ylim=(-0.5, 0.5),
        param_keys=["a", "b"],
        param_index=0,
        trajectory_style=LineStyle("orange"),
        point_style=LineStyle("teal"),
        only_map=True,
    )


def run_plot(diff: Diffeomorphism) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    anim = plot2d(diff, Y0, PARAMS, plot_config(), show=False)
    if matplotlib.get_backend().lower() in NON_INTERACTIVE_BACKENDS:
        plt.gcf().canvas.draw()  # starts the animation: draws one iteration
        print("non-interactive backend: figure built; try --mode orbit")
    else:
        plt.show()
    del anim


def run_orbit(diff: Diffeomorphism, frames: int, output: str) -> None:
    import matplotlib.pyplot as plt

    config = plot_config()
    y, points = Y0, np.empty((frames, 2))
    for i in range(frames):
        y = diff.image(y, PARAMS)
        points[i] = y
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.plot(points[:, 0], points[:, 1], ".", markersize=1, color="teal")
    ax.set(xlim=config.xlim, ylim=config.ylim, xlabel="x", ylabel="y")
    fig.savefig(output)
    print(f"last point: {points[-1]}")
    print(f"saved {frames} points to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["plot", "orbit"], default="plot")
    parser.add_argument("--frames", type=int, default=2000, help="points (orbit)")
    parser.add_argument("--output", default="henon.png", help="PNG file (orbit)")
    args = parser.parse_args()

    diff = Diffeomorphism(henon, "henon")
    start = time.perf_counter()
    if args.mode == "plot":
        run_plot(diff)
    else:
        run_orbit(diff, args.frames, args.output)
        print(f"elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
