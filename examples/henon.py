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
``bifurcation``
    One-parameter bifurcation diagram of ``x`` over ``a`` in ``[1, 1.42]``
    (``--resolution`` parameter values, ``--frames`` iterations each after a
    transient), saved to ``--output``. Cf. Miino (2024), Fig. 10 (top).
``mle``
    Maximum Lyapunov exponent over the same range of ``a`` by Benettin's
    method, using the Jacobian of the map, saved to ``--output``. Positive
    values mark the chaotic windows of the bifurcation diagram. Cf. Miino
    (2024), Fig. 10 (bottom).

Run from the repository root, e.g.::

    uv run python examples/henon.py --mode orbit --frames 5000
    uv run python examples/henon.py --mode mle --resolution 100
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


A_RANGE = (1.0, 1.42)
Y0_SWEEP = np.array([-1.06770401965, 0.368194183333])  # on the attractor at a=1.42


def run_bifurcation(
    diff: Diffeomorphism, resolution: int, frames: int, output: str
) -> None:
    import matplotlib.pyplot as plt

    transient = frames // 4
    a_values = np.linspace(*A_RANGE, resolution)
    y = Y0_SWEEP
    xs = np.empty((resolution, frames - transient))
    for k, a in enumerate(a_values):
        params = {**PARAMS, "a": a}
        for i in range(frames):
            y = diff.image(y, params)
            if i >= transient:
                xs[k, i - transient] = y[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        np.repeat(a_values, frames - transient), xs.ravel(), ",", color="tab:orange"
    )
    ax.set(xlabel="a", ylabel="x", xlim=A_RANGE)
    fig.savefig(output, dpi=150)
    print(
        f"saved bifurcation diagram ({resolution} x {frames - transient}) to {output}"
    )


def run_mle(diff: Diffeomorphism, resolution: int, frames: int, output: str) -> None:
    """Maximum Lyapunov exponent by Benettin's method.

    A unit tangent vector is pushed forward by the Jacobian of the map at each
    iteration and renormalised; the exponent is the mean of the logarithmic
    growth factors. The state is carried over from one ``a`` to the next.
    """
    import matplotlib.pyplot as plt

    a_values = np.linspace(*A_RANGE, resolution)
    mle = np.empty(resolution)
    y = Y0_SWEEP
    for k, a in enumerate(a_values):
        params = {**PARAMS, "a": a}
        unit = np.array([1.0, 0.0])
        total = 0.0
        for _ in range(frames):
            res = diff.image_detail(y, params)
            y = res.y
            unit = res.jac @ unit
            norm = np.linalg.norm(unit)
            total += np.log(norm)
            unit /= norm
        mle[k] = total / frames
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.plot(a_values, mle, color="red", linewidth=1)
    ax.set(xlabel="a", ylabel="MLE", xlim=A_RANGE)
    fig.savefig(output, dpi=150)
    print(f"MLE at a={a_values[-1]:.3f}: {mle[-1]:+.4f} ({frames} iterations)")
    print(f"saved MLE curve ({resolution} points) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode", choices=["plot", "orbit", "bifurcation", "mle"], default="plot"
    )
    parser.add_argument(
        "--frames", type=int, default=2000, help="iterations per parameter value"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=200,
        help="parameter values (bifurcation, mle)",
    )
    parser.add_argument("--output", default="henon.png", help="PNG file")
    args = parser.parse_args()

    diff = Diffeomorphism(henon, "henon")
    start = time.perf_counter()
    if args.mode == "plot":
        run_plot(diff)
    elif args.mode == "orbit":
        run_orbit(diff, args.frames, args.output)
    elif args.mode == "bifurcation":
        run_bifurcation(diff, args.resolution, args.frames, args.output)
    else:
        run_mle(diff, args.resolution, args.frames, args.output)
    if args.mode != "plot":
        print(f"elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
