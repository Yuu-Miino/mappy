"""Periodically forced Alpazur oscillator: a piecewise-smooth system.

The circuit switches between two smooth vector fields, ``alp_1`` and
``alp_2`` (different conductance ``g`` and bias ``B``), with hysteresis:
``alp_1`` hands over to ``alp_2`` when ``v2`` falls below ``-1`` and
``alp_2`` hands back when ``v2`` rises above ``-0.1``. Both are forced by
``V cos(omega t)``; time is the third state variable. The stroboscopic
Poincaré map ``(v1, v2) -> (v1, v2)`` after one forcing period is built from
eight modes (cf. Miino, NOLTA 2024): the two vector fields, a time reset and
the projections ``p``/``p_inv`` for each of them. One iteration can end in
``p1`` or ``p2``, depending on the vector field active at the section, and
the next iteration starts from there.

Modes (``--mode``):

``plot`` (default)
    Interactive iteration of the map: ``alp_1`` in orange, ``alp_2`` in
    teal. Click to restart from a new initial point (in ``alp_1``), press
    ``p`` to switch between ``V`` and ``B1`` and the up/down keys to change
    it.
``orbit``
    Non-interactive: iterate the map ``--frames`` times and save the
    trajectory to ``--output`` (PNG). Works without a display.
``trace``
    Find the fixed point of the map at ``B1 = 1.3``, continue it along
    ``B1`` up to 9.0, locate its period-doubling bifurcation in ``B1``
    (``B1 ≈ 9.0777``) and continue it along ``V`` up to 3.0. Prints the
    results.

Run from the repository root, e.g.::

    uv run python examples/forced_alpazur.py --mode trace
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

Y0 = np.array([-7.739219543281, -0.188608176591])
PARAMS = {
    "k": 0.1,
    "g1": 0.2,
    "g2": 2.0,
    "B1": 1.3,
    "B2": 5,
    "V0": 0,
    "V": 2.4,
    "omega": 1.26,
}

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "pgf", "ps", "svg", "cairo", "template")


@vector_field(3, ["k", "V0", "V", "omega", "g1", "B1"])
def alp_1(y, param):
    v1, v2, t = y
    k = param["k"]
    V0 = param["V0"]
    V = param["V"]
    omega = param["omega"]
    g1 = param["g1"]
    B1 = param["B1"]

    return np.array(
        [
            -k * v1 - v2 + V0 + V * cos(omega * t),
            v1 + (1 - g1) * v2 - (v2**3) / 3 + B1,
            1,
        ]
    )


@vector_field(3, ["k", "V0", "V", "omega", "g2", "B2"])
def alp_2(y, param):
    v1, v2, t = y
    k = param["k"]
    V0 = param["V0"]
    V = param["V"]
    omega = param["omega"]
    g2 = param["g2"]
    B2 = param["B2"]

    return np.array(
        [
            -k * v1 - v2 + V0 + V * cos(omega * t),
            v1 + (1 - g2) * v2 - (v2**3) / 3 + B2,
            1,
        ]
    )


@border(3, direction=-1)
def border_1(y, _):
    return y[1] + 1.0


@border(3, direction=1)
def border_2(y, _):
    return y[1] + 0.1


@border(3, ["omega"])
def poin_sec(y, param):
    return y[2] - 2 * np.pi / param["omega"]


@mapping(3, 3, ["omega"])
def poin_sec_reset(y, param):
    return y - np.array([0, 0, 2 * np.pi / param["omega"]])


@mapping(2, 3)
def p(y, _):
    return np.append(y, 0)


@mapping(3, 2)
def p_inv(y, _):
    return y[0:2]


def build_map() -> PoincareMap:
    modes = [
        ContinuousMode("alp_1", alp_1, borders=[border_1, poin_sec]),
        ContinuousMode("alp_2", alp_2, borders=[border_2, poin_sec]),
        DiscreteMode("pr1", poin_sec_reset),
        DiscreteMode("pr2", poin_sec_reset),
        DiscreteMode("p1", p),
        DiscreteMode("pinv1", p_inv),
        DiscreteMode("p2", p),
        DiscreteMode("pinv2", p_inv),
    ]
    transitions = {
        "p1": "alp_1",
        "p2": "alp_2",
        "alp_1": ["alp_2", "pr1"],
        "alp_2": ["alp_1", "pr2"],
        "pr1": "pinv1",
        "pr2": "pinv2",
        "pinv1": "p1",
        "pinv2": "p2",
    }
    return PoincareMap(modes, transitions, initial_mode="p1", end_modes=["p1", "p2"])


def plot_config() -> PlotConfig:
    styles = {"alp_1": LineStyle("orange"), "alp_2": LineStyle("teal")}
    return PlotConfig(
        xlim=(-10, 4),
        ylim=(-1.5, 4),
        xlabel="v1",
        ylabel="v2",
        param_keys=["V", "B1"],
        param_index=1,
        trajectory_style=styles,
        point_style=styles,
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
    ax.set(xlim=config.xlim, ylim=config.ylim, xlabel="v1", ylabel="v2")
    fig.savefig(output)
    print(f"state after {frames} iterations: {res.y} (in {res.end_mode})")
    print(f"saved {frames} iterations to {output}")


def run_trace(pmap: PoincareMap) -> None:
    res = pmap.image_detail(Y0, PARAMS)
    print(f"T(y0) = {res.y}, eigenvalues {res.eigvals}")
    print(f"mode history: {res.mode_history}")

    # Y0 is not periodic at these parameters; the search converges to the
    # stable fixed point (which switches vector fields four times per period).
    cycle = find_cycle(pmap, Y0, PARAMS)
    if not cycle.success or cycle.y is None:
        raise RuntimeError(f"find_cycle failed: {cycle.message}")
    print(f"fixed point at B1 = {PARAMS['B1']}: {cycle.y}")

    # The multipliers stay small up to B1 ≈ 8.3, so continue up to 9.0 where
    # one of them approaches -1.
    steps = trace_cycle(pmap, cycle.y, PARAMS, "B1", 9.0, resolution=20)
    last = steps[-1]
    print(f"traced {len(steps)} steps to B1 = {last.params['B1']:.6f}")
    print(f"  eigenvalues {last.info['eigvals']}")

    bf = find_local_bf(pmap, last.y, last.params, "B1", np.pi)
    if not bf.success or bf.params is None or bf.theta is None:
        raise RuntimeError(f"find_local_bf failed: {bf.message}")
    print(f"period doubling at B1 = {bf.params['B1']:.6f}, y = {bf.y}")

    curve = trace_local_bf(
        pmap, bf.y, bf.params, "B1", bf.theta, "V", 3.0, resolution=10
    )
    print(f"period-doubling curve ({len(curve)} points):")
    print("        V        B1     theta")
    for s in curve:
        print(f"  {s.params['V']:.5f}  {s.params['B1']:.5f}  {s.info['theta']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["plot", "orbit", "trace"], default="plot")
    parser.add_argument("--frames", type=int, default=20, help="iterations (orbit)")
    parser.add_argument("--output", default="forced_alpazur.png", help="PNG (orbit)")
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
