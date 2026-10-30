"""Three coupled logistic maps: fixed-point continuation and bifurcations.

The map on R^3

    x -> alpha x (1 - x) + beta y^2 x + gamma z^3   (and cyclically in y, z)

is declared with ``@mapping`` and wrapped in a
:class:`~mappy_ds.Diffeomorphism`. Two parameter sets are provided:
``PARAMS`` (used by default) and ``PARAMS_ALT`` (``--alt-params``).

Modes (``--mode``):

``plot`` (default)
    Interactive iteration projected on the ``(x, z)`` plane: image points in
    teal. Click to restart, ``f`` toggles the jumps between points, ``p``
    cycles through ``alpha``, ``beta`` and ``gamma`` and the up/down keys
    change the selected one by 0.05.
``orbit``
    Non-interactive: iterate the map ``--frames`` times and save the points
    to ``--output`` (PNG). Works without a display.
``trace``
    Find a fixed point and continue it along ``alpha`` up to 3.35. Shows
    (and saves to ``--output``) the signed moduli and arguments of its
    eigenvalues against ``alpha``.
``bf``
    As ``trace``, then locate a period-doubling bifurcation in ``alpha`` and
    continue it along ``beta`` down to -2. Shows (and saves to
    ``--output``) the bifurcation curve in the ``(alpha, beta)`` plane and
    the eigenvalues along it.

Run from the repository root, e.g.::

    uv run python examples/logistic_3d.py --mode bf --output bf.png
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from mappy_ds import Diffeomorphism, mapping
from mappy_ds.continuation import ContinuationStep
from mappy_ds.plot import LineStyle, PlotConfig, plot2d
from mappy_ds.root import find_cycle, find_local_bf, trace_cycle, trace_local_bf

Y0 = np.array([0.75, 0.72, 0.74])
PARAMS = {"alpha": 2.7, "beta": -1, "gamma": 0.0146}
PARAMS_ALT = {"alpha": 3.6324, "beta": 0.0193, "gamma": 0.0146}

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "pgf", "ps", "svg", "cairo", "template")


@mapping(3, 3)
def logistic_3d(iny, param):
    x, y, z = iny
    alpha = param["alpha"]
    beta = param["beta"]
    gamma = param["gamma"]

    return np.array(
        [
            alpha * x * (1 - x) + beta * (y**2) * x + gamma * (z**3),
            alpha * y * (1 - y) + beta * (z**2) * y + gamma * (x**3),
            alpha * z * (1 - z) + beta * (x**2) * z + gamma * (y**3),
        ]
    )


def plot_config() -> PlotConfig:
    return PlotConfig(
        xlim=(0, 1.2),
        ylim=(0, 1.2),
        ykey=2,
        xlabel="x",
        ylabel="z",
        param_keys=["alpha", "beta", "gamma"],
        param_index=0,
        trajectory_style=LineStyle("orange"),
        point_style=LineStyle("teal"),
        only_map=True,
        param_step=0.05,
    )


def signed_moduli(steps: list[ContinuationStep]) -> np.ndarray:
    """``sign(Re mu) |mu|`` of the eigenvalues ``mu`` at each step."""
    eigs = np.array([s.info["eigvals"] for s in steps])
    return np.sign(eigs.real) * np.abs(eigs)


def finish_figure(output: str | None) -> None:
    if output:
        plt.savefig(output)
        print(f"saved figure to {output}")
    plt.show()


def run_plot(dmap: Diffeomorphism, params: dict) -> None:
    import matplotlib

    anim = plot2d(dmap, Y0, params, plot_config(), show=False)
    if matplotlib.get_backend().lower() in NON_INTERACTIVE_BACKENDS:
        plt.gcf().canvas.draw()  # starts the animation: draws one iteration
        print("non-interactive backend: figure built; try --mode orbit")
    else:
        plt.show()
    del anim


def run_orbit(dmap: Diffeomorphism, params: dict, frames: int, output: str) -> None:
    config = plot_config()
    y, points = Y0, np.empty((frames, 3))
    for i in range(frames):
        y = dmap.image(y, params)
        points[i] = y
    fig, ax = plt.subplots(figsize=config.figsize)
    ax.plot(points[:, 0], points[:, 2], ".", markersize=2, color="teal")
    ax.set(xlim=config.xlim, ylim=config.ylim, xlabel="x", ylabel="z")
    fig.savefig(output)
    print(f"last point: {points[-1]}")
    print(f"saved {frames} points to {output}")


def trace_fixed_point(
    dmap: Diffeomorphism, params: dict
) -> list[ContinuationStep[np.ndarray]]:
    ret = find_cycle(dmap, Y0, params)
    if not ret.success or ret.y is None:
        raise RuntimeError(f"failed to find a cycle: {ret.message}")
    print(f"fixed point at alpha = {params['alpha']}: {ret.y}")

    tr = trace_cycle(dmap, ret.y, params, "alpha", 3.35, show_progress=True)
    last = tr[-1]
    print(f"traced {len(tr)} steps to alpha = {last.params['alpha']:.6f}")
    print(f"  y = {last.y}, eigenvalues {last.info['eigvals']}")
    return tr


def run_trace(dmap: Diffeomorphism, params: dict, output: str | None) -> None:
    tr = trace_fixed_point(dmap, params)
    alphas = [s.params["alpha"] for s in tr]
    thetas = [np.angle(s.info["eigvals"]) for s in tr]

    eg = plt.subplot(211)
    eg.grid()
    eg.plot(alphas, signed_moduli(tr))
    eg.set_ylabel("sign * |eigvals|")
    eg.set_xticklabels([])
    th = plt.subplot(212)
    th.grid()
    th.plot(alphas, thetas)
    th.set_xlabel("alpha")
    th.set_ylabel("theta")
    finish_figure(output)


def run_bf(dmap: Diffeomorphism, params: dict, output: str | None) -> None:
    tr = trace_fixed_point(dmap, params)
    y1, p1 = tr[-1].y, tr[-1].params

    bf = find_local_bf(dmap, y1, p1, "alpha", np.pi)
    if not bf.success or bf.y is None or bf.params is None or bf.theta is None:
        raise RuntimeError(f"failed to find a local bifurcation: {bf.message}")
    print(f"period doubling at alpha = {bf.params['alpha']:.6f}, y = {bf.y}")

    tr_bf = trace_local_bf(
        dmap,
        bf.y,
        bf.params,
        "alpha",
        bf.theta,
        "beta",
        -2,
        resolution=300,
        show_progress=True,
    )
    last = tr_bf[-1]
    print(
        f"traced {len(tr_bf)} bifurcation points to beta = "
        f"{last.params['beta']:.6f}, alpha = {last.params['alpha']:.6f}"
    )

    alphas = [s.params["alpha"] for s in tr_bf]
    betas = [s.params["beta"] for s in tr_bf]

    bf2 = plt.subplot(211)
    bf2.grid()
    bf2.plot(alphas, betas)
    bf2.set_xticklabels([])
    bf2.set_ylabel("beta")
    eg = plt.subplot(212)
    eg.grid()
    eg.plot(alphas, signed_moduli(tr_bf))
    eg.set_xlabel("alpha")
    eg.set_ylabel("sign * |eigvals|")
    finish_figure(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode", choices=["plot", "orbit", "trace", "bf"], default="plot"
    )
    parser.add_argument(
        "--alt-params", action="store_true", help="use PARAMS_ALT instead of PARAMS"
    )
    parser.add_argument("--frames", type=int, default=2000, help="points (orbit)")
    parser.add_argument("--output", default=None, help="PNG file (required for orbit)")
    args = parser.parse_args()

    params = PARAMS_ALT if args.alt_params else PARAMS
    dmap = Diffeomorphism(logistic_3d, "logistic_3d")
    start = time.perf_counter()
    if args.mode == "plot":
        run_plot(dmap, params)
        return
    if args.mode == "orbit":
        run_orbit(dmap, params, args.frames, args.output or "logistic_3d.png")
    elif args.mode == "trace":
        run_trace(dmap, params, args.output)
    else:
        run_bf(dmap, params, args.output)
    print(f"elapsed: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
