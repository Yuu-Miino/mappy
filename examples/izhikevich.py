"""Izhikevich neuron model: a hybrid system with a firing reset.

The membrane potential ``v`` and recovery variable ``u`` follow

    v' = 0.04 v^2 + 5 v + 140 - u + I,  u' = a (b v - u)

until ``v`` reaches 30 (firing), when ``(v, u)`` jumps to ``(c, u + d)``.
Sampling ``u`` right after each firing gives a one-dimensional Poincaré map
built from four modes (cf. Miino, NOLTA 2024):

* ``m0`` (1 -> 2): lift ``u`` to the reset point ``(c, u)``;
* ``m1`` (2 -> 2): integrate the ODE until the border ``v = 30``;
* ``m2`` (2 -> 2): firing jump ``(v, u) -> (v - 30 + c, u + d)``;
* ``m3`` (2 -> 1): keep ``u``.

Modes (``--mode``):

``plot`` (default)
    Interactive iteration of the map: the continuous flow in teal, the
    firing jumps in orange. Click to restart from the clicked ``u`` (the
    vertical coordinate), press ``p`` to switch between ``a`` and ``I`` and
    the up/down keys to change it.
``orbit``
    Non-interactive: iterate the map ``--frames`` times and save the
    trajectory to ``--output`` (PNG). Works without a display.
``trace``
    Find the fixed point of the map, continue it along ``I`` down to 4.5,
    locate a period-doubling bifurcation in ``I`` and continue it along
    ``a`` up to 0.3. Prints the results.

Run from the repository root, e.g.::

    uv run python examples/izhikevich.py --mode trace
"""

import argparse
import time

import numpy as np

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

Y0 = -3.0
PARAMS = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "pgf", "ps", "svg", "cairo", "template")


## Izhikevich neuron model
@vector_field(2, ["a", "b", "I"])
def izhikevich(y, param):
    v, u = y
    a = param["a"]
    b = param["b"]
    I = param["I"]

    return np.array([0.04 * (v**2) + 5.0 * v + 140.0 - u + I, a * (b * v - u)])


## Firing border
@border(2, direction=1)
def fire_border(y, _):
    return y[0] - 30.0


## Firing jump
@mapping(2, 2, ["c", "d"])
def jump(y, param):
    C = np.array([-30 + param["c"], param["d"]])
    return y + C


## Dimension conversion (p: 1 -> 2, pinv: 2 -> 1)
@mapping(1, 2, ["c"])
def p(y, param):
    return np.array([0, 1]) * y + np.array([param["c"], 0])


@mapping(2, 1)
def pinv(y, _):
    return np.array([0, 1]) @ y


def build_map() -> PoincareMap:
    modes = [
        DiscreteMode("m0", p),
        ContinuousMode("m1", izhikevich, borders=[fire_border]),
        DiscreteMode("m2", jump, in_trajectory=True),
        DiscreteMode("m3", pinv),
    ]
    transitions = {"m0": "m1", "m1": ["m2"], "m2": "m3", "m3": "m0"}
    return PoincareMap(modes, transitions, initial_mode="m0")


def plot_config() -> PlotConfig:
    return PlotConfig(
        xlim=(-80, 35),
        ylim=(-10, 2),
        xlabel="v",
        ylabel="u",
        param_keys=["a", "I"],
        param_index=0,
        scalar_axis="y",
        trajectory_style={"m1": LineStyle("teal"), "m2": LineStyle("orange")},
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
    res = pmap.image_detail(Y0, PARAMS, n=frames, jacobian=False, trajectory=True)
    plot_trajectory(res.trajectory, ax, styles=config.trajectory_style)
    ax.set(xlim=config.xlim, ylim=config.ylim, xlabel="v", ylabel="u")
    fig.savefig(output)
    print(f"u after {frames} firings: {res.y[0]:.6f}")
    print(f"saved {frames} iterations to {output}")


def run_trace(pmap: PoincareMap) -> None:
    res = pmap.image_detail(Y0, PARAMS)
    print(f"T(u0) = {res.y}, derivative {res.jac[0, 0]:.6f}")
    print(f"mode history: {res.mode_history}")

    cycle = find_cycle(pmap, Y0, PARAMS)
    if not cycle.success or cycle.y is None:
        raise RuntimeError(f"find_cycle failed: {cycle.message}")
    print(f"fixed point at I = {PARAMS['I']}: u = {cycle.y[0]:.6f}")

    steps = trace_cycle(pmap, cycle.y, PARAMS, "I", 4.5, resolution=20)
    last = steps[-1]
    print(
        f"traced {len(steps)} steps to I = {last.params['I']:.6f}: "
        f"u = {last.y[0]:.6f}, multiplier {last.info['eigvals'][0].real:.6f}"
    )

    bf = find_local_bf(pmap, last.y, last.params, "I", np.pi)
    if not bf.success or bf.params is None or bf.theta is None:
        raise RuntimeError(f"find_local_bf failed: {bf.message}")
    print(f"period doubling at I = {bf.params['I']:.6f}, u = {bf.y[0]:.6f}")

    curve = trace_local_bf(
        pmap, bf.y, bf.params, "I", bf.theta, "a", 0.3, resolution=10
    )
    print(f"period-doubling curve ({len(curve)} points):")
    print("        a         I     theta")
    for s in curve:
        print(f"  {s.params['a']:.5f}  {s.params['I']:.5f}  {s.info['theta']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["plot", "orbit", "trace"], default="plot")
    parser.add_argument("--frames", type=int, default=30, help="iterations (orbit)")
    parser.add_argument("--output", default="izhikevich.png", help="PNG (orbit)")
    args = parser.parse_args()

    pmap = build_map()
    start = time.perf_counter()
    if args.mode == "plot":
        run_plot(pmap)
    elif args.mode == "orbit":
        run_orbit(pmap, args.frames, args.output)
    else:
        run_trace(pmap)
    if args.mode != "plot":
        print(f"elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
