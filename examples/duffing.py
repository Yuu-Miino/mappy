"""Periodically forced Duffing oscillator: Poincaré map and period doubling.

The non-autonomous Duffing equation

    x' = y,  y' = -k y - x^3 + B0 + B cos(t)

is made autonomous by adding time as a third state variable ``t``. The
stroboscopic Poincaré map ``(x, y) -> (x, y)`` after one forcing period is
built from four modes (cf. Miino, NOLTA 2024):

* ``p``       (2 -> 3): lift the section point to ``(x, y, 0)``;
* ``duffing`` (3 -> 3): integrate until the border ``t = 2 pi``;
* ``t_reset`` (3 -> 3): reset ``t`` to ``0``;
* ``pinv``    (3 -> 2): project back onto ``(x, y)``.

Modes (``--mode``):

``plot`` (default)
    Interactive iteration of the Poincaré map: trajectories in grey, image
    points in red. Click to restart from a new initial point, press ``p`` to
    switch between ``B0`` and ``B`` and the up/down keys to change it.
``orbit``
    Non-interactive: iterate the map ``--frames`` times and save the
    trajectory to ``--output`` (PNG). Works without a display.
``trace``
    Find the 1-periodic point at ``B = 0.1``, continue it along ``B``, locate
    its two period-doubling bifurcations (expected at ``B ≈ 0.171687`` and
    ``B ≈ 0.368655`` for ``k = 0.2, B0 = 0.1``) and continue the first one in
    the ``(B, B0)`` plane up to ``B = 0.35``. Prints the results.
``bifurcation``
    One-parameter bifurcation diagram of ``x`` on the section over ``B`` in
    ``[0.1, 0.375]`` (``--resolution`` values of ``B``, ``--frames`` map
    iterations each after a transient), saved to ``--output``. The two
    period doublings found by ``trace`` bound the window where the 1-periodic
    point is unstable. Cf. Miino (2024), Fig. 8 (top). Takes a while: every
    point is one period of the ODE.

Run from the repository root, e.g.::

    uv run python examples/duffing.py --mode trace
    uv run python examples/duffing.py --mode bifurcation --resolution 60
"""

import argparse
import time

import numpy as np
from sympy import cos

from mappy_ds import (
    ContinuousMode,
    DiscreteMode,
    PoincareMap,
    border,
    mapping,
    vector_field,
)
from mappy_ds.plot import LineStyle, PlotConfig, plot2d, plot_trajectory
from mappy_ds.root import find_cycle, find_local_bf, trace_cycle, trace_local_bf

Y0 = np.array([0.232147751858, 0.075591687407])
PARAMS = {"k": 0.2, "B0": 0.1, "B": 0.1}

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "pgf", "ps", "svg", "cairo", "template")

# Period-doubling bifurcation values of B for k = 0.2, B0 = 0.1.
REFERENCE_PD = (0.171687, 0.368655)


@vector_field(3)
def duffing(iny, param):
    x, y, t = iny
    k = param["k"]
    B = param["B"]
    B0 = param["B0"]

    return np.array([y, -k * y - x**3 + B0 + B * cos(t), 1])


@border(3, direction=1)
def poin_sec(iny, _):
    return iny[2] - 2 * np.pi


@mapping(3, 3)
def t_reset(iny, _):
    return iny - np.array([0, 0, 2 * np.pi])


@mapping(2, 3)
def p(iny, _):
    return np.append(iny, 0)


@mapping(3, 2)
def pinv(iny, _):
    return iny[0:2]


def build_map() -> PoincareMap:
    modes = [
        ContinuousMode("duffing", duffing, borders=[poin_sec]),
        DiscreteMode("t_reset", t_reset),
        DiscreteMode("p", p),
        DiscreteMode("pinv", pinv),
    ]
    transitions = {
        "p": "duffing",
        "duffing": ["t_reset"],
        "t_reset": "pinv",
        "pinv": "p",
    }
    return PoincareMap(modes, transitions, initial_mode="p")


def plot_config() -> PlotConfig:
    return PlotConfig(
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        param_keys=["B0", "B"],
        param_index=0,
    )


def run_plot(pmap: PoincareMap) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    anim = plot2d(pmap, Y0, PARAMS, plot_config(), show=False)
    if matplotlib.get_backend().lower() in NON_INTERACTIVE_BACKENDS:
        plt.gcf().canvas.draw()  # starts the animation: draws one iteration
        print("non-interactive backend: figure built; try --mode orbit")
    else:
        plt.show()
    del anim


def run_orbit(pmap: PoincareMap, frames: int, output: str) -> None:
    import matplotlib.pyplot as plt

    config = plot_config()
    fig, ax = plt.subplots(figsize=config.figsize)
    y, points = Y0, []
    for _ in range(frames):
        res = pmap.image_detail(y, PARAMS, jacobian=False, trajectory=True)
        plot_trajectory(res.trajectory, ax, styles=LineStyle("black", 0.3))
        y = res.y
        points.append(y)
    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], ".", color="red")
    ax.set(xlim=config.xlim, ylim=config.ylim, xlabel="x", ylabel="y")
    fig.savefig(output)
    print(f"saved {frames} iterations to {output}")


def run_bifurcation(
    pmap: PoincareMap, resolution: int, frames: int, output: str
) -> None:
    import matplotlib.pyplot as plt

    transient = frames // 4
    b_values = np.linspace(0.1, 0.375, resolution)
    y = Y0
    xs = np.empty((resolution, frames - transient))
    for k, b in enumerate(b_values):
        params = {**PARAMS, "B": b}
        for i in range(frames):
            y = pmap.image(y, params)
            if i >= transient:
                xs[k, i - transient] = y[0]
        print(f"B = {b:.4f}", end="\r", flush=True)
    print()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        np.repeat(b_values, frames - transient), xs.ravel(), ".", ms=1, color="gray"
    )
    ax.set(xlabel="B", ylabel="x", xlim=(0.1, 0.375))
    fig.savefig(output, dpi=150)
    print(
        f"saved bifurcation diagram ({resolution} x {frames - transient}) to {output}"
    )


def run_trace(pmap: PoincareMap) -> None:
    res = pmap.image_detail(Y0, PARAMS)
    print(f"T(y0) = {res.y}, eigenvalues {res.eigvals}")

    cycle = find_cycle(pmap, Y0, PARAMS)
    if not cycle.success or cycle.y is None:
        raise RuntimeError(f"find_cycle failed: {cycle.message}")
    print(f"1-periodic point at B = {PARAMS['B']}: {cycle.y}")

    # Continue the fixed point towards the first period doubling ...
    steps = trace_cycle(pmap, cycle.y, PARAMS, "B", 0.17, resolution=10)
    last = steps[-1]
    print(f"traced {len(steps)} steps to B = {last.params['B']:.6f}")
    print(f"  eigenvalues {last.info['eigvals']}")

    found = []
    bf1 = find_local_bf(pmap, last.y, last.params, "B", np.pi)
    if not bf1.success or bf1.params is None or bf1.theta is None:
        raise RuntimeError(f"find_local_bf failed: {bf1.message}")
    found.append(bf1.params["B"])
    print(f"period doubling #1 at B = {bf1.params['B']:.6f}, y = {bf1.y}")

    # ... and past it (the point is now a saddle) towards the second one.
    steps2 = trace_cycle(pmap, last.y, last.params, "B", 0.36, resolution=20)
    last2 = steps2[-1]
    print(f"traced {len(steps2)} steps to B = {last2.params['B']:.6f}")
    print(f"  eigenvalues {last2.info['eigvals']}")
    bf2 = find_local_bf(pmap, last2.y, last2.params, "B", np.pi)
    if bf2.success and bf2.params is not None:
        found.append(bf2.params["B"])
        print(f"period doubling #2 at B = {bf2.params['B']:.6f}, y = {bf2.y}")

    for got, ref in zip(found, REFERENCE_PD, strict=False):
        status = "OK" if abs(got - ref) < 1e-5 else "MISMATCH"
        print(f"check: B = {got:.6f} vs reference {ref} -> {status}")

    # Continue the first period-doubling point in the (B, B0) plane.
    curve = trace_local_bf(
        pmap, bf1.y, bf1.params, "B0", bf1.theta, "B", 0.35, resolution=10
    )
    print(f"period-doubling curve ({len(curve)} points):")
    print("        B        B0     theta")
    for s in curve:
        print(f"  {s.params['B']:.5f}  {s.params['B0']:.5f}  {s.info['theta']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode", choices=["plot", "orbit", "trace", "bifurcation"], default="plot"
    )
    parser.add_argument(
        "--frames", type=int, default=50, help="map iterations (orbit, bifurcation)"
    )
    parser.add_argument(
        "--resolution", type=int, default=120, help="values of B (bifurcation)"
    )
    parser.add_argument("--output", default="duffing.png", help="PNG file")
    args = parser.parse_args()

    pmap = build_map()
    start = time.perf_counter()
    if args.mode == "plot":
        run_plot(pmap)
    elif args.mode == "orbit":
        run_orbit(pmap, args.frames, args.output)
    elif args.mode == "bifurcation":
        run_bifurcation(pmap, args.resolution, args.frames, args.output)
    else:
        run_trace(pmap)
    if args.mode != "plot":
        print(f"elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
