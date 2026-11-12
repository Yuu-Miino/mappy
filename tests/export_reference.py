"""Export reference values of the test systems for the Julia port (MappyDS.jl).

Run from the repository root::

    uv run python tests/export_reference.py > reference.json

The continuous systems are integrated with DOP853 at ``rtol=1e-12`` and
``atol=1e-14`` so that the values are converged to ~1e-10 and the Julia test
suite can compare against them at ``1e-6`` with its own solver. Copy the
output to ``test/fixtures/reference.json`` of ``MappyDS.jl``; do not edit it
by hand.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import mappy_ds
from conftest import (
    add_time,
    henon,
    lower,
    make_duffing_map,
    make_izhikevich_map,
    relay_down,
    relay_up,
    remove_time,
    sample,
    time_reset,
    upper,
)
from mappy_ds import (
    ContinuousMode,
    Diffeomorphism,
    DiscreteMode,
    PoincareMap,
    border,
    vector_field,
)
from mappy_ds.root import find_cycle

TIGHT = {"method": "DOP853", "rtol": 1e-12, "atol": 1e-14}


def arr(x: np.ndarray | None) -> list | None:
    return None if x is None else np.asarray(x, dtype=float).tolist()


def carr(x: np.ndarray) -> list[list[float]]:
    """Complex array as ``[[re, im], ...]``."""
    return [[float(z.real), float(z.imag)] for z in np.asarray(x).ravel()]


def detail(pmap, y0, params, **kwargs) -> dict:
    res = pmap.image_detail(y0, params, hessian=True, **kwargs)
    return {
        "y0": arr(np.atleast_1d(y0)),
        "params": dict(params),
        "y": arr(res.y),
        "jac": arr(res.jac),
        "hes": arr(res.hes),
        "mode_history": list(res.mode_history),
    }


def make_relay_map() -> PoincareMap:
    modes = [
        ContinuousMode("up", relay_up, borders=[upper, sample], **TIGHT),
        ContinuousMode("down", relay_down, borders=[lower, sample], **TIGHT),
        DiscreteMode("r_up", time_reset),
        DiscreteMode("r_down", time_reset),
        DiscreteMode("p_up", add_time),
        DiscreteMode("p_down", add_time),
        DiscreteMode("pinv_up", remove_time),
        DiscreteMode("pinv_down", remove_time),
    ]
    transitions = {
        "p_up": "up",
        "p_down": "down",
        "up": ["down", "r_up"],
        "down": ["up", "r_down"],
        "r_up": "pinv_up",
        "r_down": "pinv_down",
        "pinv_up": "p_up",
        "pinv_down": "p_down",
    }
    return PoincareMap(modes, transitions, "p_up", end_modes=["p_up", "p_down"])


def main() -> None:
    out: dict = {
        "generator": {
            "package": "mappy-ds",
            "version": mappy_ds.__version__,
            "solver": TIGHT,
            "script": "tests/export_reference.py",
        }
    }

    # -- Izhikevich -----------------------------------------------------------
    izh = make_izhikevich_map(**TIGHT)
    p10 = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}
    p8 = {**p10, "I": 8.0}
    fp = find_cycle(izh, [-4.6], p8, tol=1e-12)
    assert fp.success
    izh_mode = ContinuousMode(
        "A", izh.modes["A"].fun, borders=izh.modes["A"].borders, **TIGHT
    )
    step = izh_mode.step([-50.0, -1.7], p10, hessian=True)
    out["izhikevich"] = {
        "paper": {
            "I10": {
                "y0": -3.0,
                "y": -3.4783849356,
                "jac": 0.27080365164,
                "hes": -0.193258205836,
            },
            "I8": {"u": -4.6454201715, "jac": 0.2219605834, "hes": -0.2604668216},
        },
        "image": [
            detail(izh, [-3.0], p10),
            detail(izh, [-4.0], {**p10, "I": 9.0}),
            detail(izh, [-3.0], p10, n=2),
        ],
        "fixed_point_I8": {
            "params": p8,
            "y": arr(fp.y),
            "eigvals": carr(fp.eigvals),
            **{k: v for k, v in detail(izh, fp.y, p8).items() if k in ("jac", "hes")},
        },
        "mode_step": {
            "y0": [-50.0, -1.7],
            "params": p10,
            "y": arr(step.y),
            "t": float(step.t),
            "border": int(step.border),
            "jac": arr(step.jac),
            "hes": arr(step.hes),
        },
    }

    # -- Hénon -----------------------------------------------------------------
    hmap = Diffeomorphism(henon)
    hp = {"a": 1.4, "b": 0.3}
    fp1 = find_cycle(hmap, [0.6, 0.2], hp, tol=1e-12)
    fp2 = find_cycle(hmap, [1.0, -0.1], hp, period=2, tol=1e-12)
    assert fp1.success and fp2.success
    out["henon"] = {
        "params": hp,
        "image": [detail(hmap, [0.1, -0.2], hp, n=3), detail(hmap, [0.0, 0.0], hp)],
        "fixed_point": {"y": arr(fp1.y), "eigvals": carr(fp1.eigvals)},
        "period_two": {"y": arr(fp2.y), "eigvals": carr(fp2.eigvals)},
    }

    # -- Duffing ---------------------------------------------------------------
    dmap = make_duffing_map(**TIGHT)
    dp = {"k": 0.2, "B0": 0.1, "B": 0.1}
    dy0 = [0.232147751858, 0.075591687407]
    dfp = find_cycle(dmap, dy0, dp, tol=1e-12)
    assert dfp.success
    out["duffing"] = {
        "params": dp,
        "image": [detail(dmap, dy0, dp)],
        "fixed_point": {
            "y": arr(dfp.y),
            "eigvals": carr(dfp.eigvals),
            **{k: v for k, v in detail(dmap, dfp.y, dp).items() if k in ("jac", "hes")},
        },
    }

    # -- Relay oscillator (closed form) -------------------------------------
    rmap = make_relay_map()
    rp = {"T": 2.2}
    out["relay"] = {
        "params": rp,
        "image": [
            detail(rmap, [0.0], rp),
            detail(rmap, [-0.2], rp, mode="p_down"),
            detail(rmap, [0.0], rp, n=2),
        ],
    }

    # -- Rotation with a parametrised circular border -------------------------
    @vector_field(2)
    def rot(y, p):
        return np.array([-p["w"] * y[1] + 0.5 * y[0], p["w"] * y[0]])

    @border(2)
    def circle(y, p):
        return y[0] ** 2 + y[1] ** 2 - p["r"] ** 2

    mode = ContinuousMode("rot", rot, borders=[circle], **TIGHT)
    rpar = {"w": 1.0, "r": 1.5}
    rstep = mode.step([0.6, -0.3], rpar, hessian=True)
    out["rotation_mode_step"] = {
        "y0": [0.6, -0.3],
        "params": rpar,
        "y": arr(rstep.y),
        "t": float(rstep.t),
        "border": int(rstep.border),
        "jac": arr(rstep.jac),
        "hes": arr(rstep.hes),
    }

    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
