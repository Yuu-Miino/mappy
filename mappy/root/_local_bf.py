"""Local bifurcation set
"""
from typing import TypeVar
from mappy import PoincareMap
import numpy
from scipy.optimize import root

Y = TypeVar('Y', numpy.ndarray, float)
P = TypeVar('P', numpy.ndarray, float)
V_LBF = TypeVar('V_LBF', bound=numpy.ndarray)

__all__ = [
    "find_local_bf"
]

def cond_local_bf (
    pmap: PoincareMap,
    var: V_LBF,
    params: P,
    param_idx: int = 0,
    period: int = 1
) -> V_LBF:
    y0 = var[0:pmap.dimension]
    param, theta = var[pmap.dimension:-1]
    inparams = params.copy()
    inparams[param_idx] = param

    res = pmap.image_detail(y0, inparams, iterations=period)
    det = numpy.linalg.det(res.jac - numpy.exp(1j*theta))

    ret = numpy.empty(pmap.dimension + 2, )
    ret[0:pmap.dimension]  = res.y - y0
    ret[pmap.dimension:-1] = numpy.real(det), numpy.imag(det)
    return ret

def find_local_bf (
    poincare_map: PoincareMap,
    y0: Y,
    params: P,
    param_idx: int,
    theta: float,
    period: int = 1,
):
    objective_fun = lambda y: cond_local_bf(
        pmap=poincare_map,
        var=y,
        params=params,
        param_idx=param_idx,
        period=period
    )

    var = numpy.array(y0).squeeze()
    var = numpy.append(var, [params[param_idx], theta])
    rt = root(objective_fun, var)
    print(rt)

