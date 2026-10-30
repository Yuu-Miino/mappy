import numpy as np
from mappy import ContinuousMode as CM, DiscreteMode as DM, PoincareMap
from mappy.root import *
from mappy.tools import mplot2d, Plot2dConfig, ColorAlpha


## Izhikevich neuron model
@CM.function(dimension=2, param_keys=["a", "b", "I"])
def izhikevich(y: np.ndarray, param: dict | None) -> np.ndarray:
    v, u = y
    if param is None:
        raise ValueError
    a = param["a"]
    b = param["b"]
    I = param["I"]

    ret = np.array([0.04 * (v**2) + 5.0 * v + 140.0 - u + I, a * (b * v - u)])
    return ret


## Firing border
@CM.border(direction=1)
def fire_border(y, _):
    return y[0] - 30.0


## Firing jump
@DM.function(domain_dimension=2, codomain_dimension=2, param_keys=["c", "d"])
def jump(y, param):
    C = np.array([-30 + param["c"], param["d"]])
    return y + C


## Dimension conversion (p: 1 -> 2, pinv: 2 -> 1)
@DM.function(domain_dimension=1, codomain_dimension=2, param_keys=["c"])
def p(y, param):
    return np.array([0, 1]) * y + np.array([param["c"], 0])


@DM.function(domain_dimension=2, codomain_dimension=1)
def pinv(y, _):
    return np.array([0, 1]) @ y


## Main
def main():
    y0 = -3
    param = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}

    all_modes = (
        DM("m0", p),
        CM("m1", izhikevich, borders=[fire_border]),
        DM("m2", jump, inTraj=True),
        DM("m3", pinv),
    )

    transitions = {"m0": "m1", "m1": ["m2"], "m2": "m3", "m3": "m0"}

    pmap = PoincareMap(all_modes, transitions, "m0")

    mode = "trace"

    if mode == "traj":
        config = Plot2dConfig(
            xrange=(-80, 35),
            yrange=(-10, 2),
            xlabel="v",
            ylabel="u",
            param_keys=["a", "I"],
            param_idx=0,
            float_mouse_xy="y",
            traj_color={
                "m1": ColorAlpha("teal"),
                "m2": ColorAlpha("orange"),
            },
        )

        mplot2d(pmap, y0, "m0", param, config=config)

    elif mode == "trace":
        print(pmap.image_detail(y0, "m0", params=param))

        res = trace_cycle(pmap, y0, param, "I", 4.5, show_progress=True, m0="m0")

        y1, p1 = res[-1][0:2]

        ret = trace_local_bf(
            pmap,
            y0=y1,
            params=p1,
            bf_param_key="I",
            theta=3.14,
            cnt_param_key="a",
            end_val=0.3,
            show_progress=True,
            m0="m0",
        )
        print(len(ret))


if __name__ == "__main__":
    main()
