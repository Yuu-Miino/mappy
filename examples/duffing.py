import numpy as np
from mappy import ContinuousMode as CM, DiscreteMode as DM, PoincareMap
from mappy.root import *
from sympy import cos
from mappy.tools import mplot2d, Plot2dConfig


@CM.function(dimension=3, param_keys=["k", "B0", "B"])
def duffing(iny, param):
    x, y, t = iny
    k = param["k"]
    B = param["B"]
    B0 = param["B0"]

    return np.array([y, -k * y - x**3 + B0 + B * cos(t), 1])


@CM.border(1)
def poin_sec(iny, _):
    return iny[2] - 2 * np.pi


@DM.function(3, 3)
def t_reset(iny, _):
    return iny - np.array([0, 0, 2 * np.pi])


@DM.function(2, 3)
def p(iny, _):
    return np.append(iny, 0)


@DM.function(3, 2)
def pinv(iny, _):
    return iny[0:2]


def main():
    y0 = np.array([0.232147751858, 0.075591687407])
    param = {"k": 0.2, "B0": 0.1, "B": 0.1}

    all_modes = (
        CM("duffing", duffing, borders=[poin_sec]),
        DM("t_reset", t_reset),
        DM("p", p),
        DM("pinv", pinv),
    )

    trans = {"p": "duffing", "duffing": ["t_reset"], "t_reset": "pinv", "pinv": "p"}

    pmap = PoincareMap(all_modes, trans, "p")

    config = Plot2dConfig(
        xrange=(-1.5, 1.5),
        yrange=(-1.5, 1.5),
        param_keys=["B0", "B"],
        param_idx=0,
    )

    mplot2d(pmap, y0, "p", param, config=config)

    """ print(pmap.image_detail(y0, "p", params=param))

    ret = trace_cycle(
        poincare_map=pmap,
        y0=y0,
        params=param,
        period=1,
        cnt_param_idx="B",
        end_val=0.17,
        show_progress=True,
        resolution=10,
    )

    yb, paramb = ret[-1][0:2]
    ret_lbf = trace_local_bf(
        poincare_map=pmap,
        y0=yb,
        params=paramb,
        bf_param_idx="B0",
        theta=np.pi,
        cnt_param_idx="B",
        end_val=0.35,
        period=1,
        resolution=100,
        show_progress=True,
    )
    print(len(ret_lbf)) """


if __name__ == "__main__":
    main()
