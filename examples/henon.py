import numpy as np
from mappy import DiscreteMode as DM, Diffeomorphism
from mappy.tools import plot2d, Plot2dConfig, ColorAlpha


@DM.function(2, 2, ["a", "b"])
def henon(iny, param):
    x, y = iny
    a = param["a"]
    b = param["b"]

    return np.array([1 - a * x**2 + y, b * x])


if __name__ == "__main__":
    vecx = np.array([0.1, 0.1])
    param = {"a": 1.4, "b": 0.3}

    diff = Diffeomorphism("henon", henon)

    config = Plot2dConfig(
        xrange=(-1.5, 1.5),
        yrange=(-0.5, 0.5),
        param_keys=["a", "b"],
        param_idx=0,
        traj_color=ColorAlpha("orange"),
        point_color=ColorAlpha("teal"),
        only_map=True,
    )

    plot2d(diff, vecx, param, config)
