"""Tests of :mod:`mappy_ds.root` (periodic points and local bifurcations)."""

from __future__ import annotations

import doctest
import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conftest import make_izhikevich_map
from mappy_ds import Diffeomorphism, mapping
from mappy_ds.root import (
    CycleResult,
    LocalBfResult,
    find_cycle,
    find_local_bf,
    trace_cycle,
    trace_cycle_arclength,
    trace_local_bf,
    trace_local_bf_arclength,
)

HENON_FIXED_POINT = np.array([0.631354477, 0.189406343])
HENON_MULTIPLIERS = np.array([-1.923738858, 0.155946322])
IZH_PARAMS_I8 = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 8.0}


def henon_pd_value(b):
    """Exact period-doubling value of ``a`` for the Hénon fixed point."""
    return 3 * (1 - b) ** 2 / 4


@mapping(1)
def saddle_node(x, p):
    return x + p["p"] - x**2


@mapping(1)
def no_fixed_point(x, _):
    return x + 1 + x**2


@pytest.fixture(scope="module")
def izh_map_module():
    return make_izhikevich_map()


@pytest.fixture(scope="module")
def izh_trace(izh_map_module):
    start = find_cycle(izh_map_module, [-4.6], IZH_PARAMS_I8)
    assert start.success
    return trace_cycle(izh_map_module, start.y, IZH_PARAMS_I8, "I", 4.5, resolution=10)


# -- find_cycle ------------------------------------------------------------------


def test_find_cycle_henon_fixed_point(henon_map, henon_params):
    res = find_cycle(henon_map, [0.6, 0.2], henon_params)
    assert isinstance(res, CycleResult)
    assert res.success
    assert res.y.shape == (2,)
    assert_allclose(res.y, HENON_FIXED_POINT, rtol=1e-8)
    assert_allclose(np.sort(res.eigvals.real), HENON_MULTIPLIERS, rtol=1e-8)
    assert res.eigvecs.shape == (2, 2)
    assert res.nfev > 0
    assert_allclose(res.residual, 0.0, atol=1e-10)
    assert isinstance(res.message, str)


def test_find_cycle_henon_period_two(henon_map, henon_params):
    res = find_cycle(henon_map, [1.0, -0.1], henon_params, period=2)
    assert res.success
    y = res.y
    assert_allclose(henon_map.image(y, henon_params, n=2), y, atol=1e-10)
    assert np.linalg.norm(henon_map.image(y, henon_params) - y) > 0.1
    assert_allclose(y, [0.97580005, -0.14274002], rtol=1e-7)
    # The multipliers are those of the second iterate.
    jac2 = henon_map.image_detail(y, henon_params, n=2).jac
    assert_allclose(np.sort(res.eigvals), np.sort(np.linalg.eigvals(jac2)))


def test_find_cycle_period_two_from_attractor(henon_map, henon_params):
    """An initial guess taken from the attractor also finds a 2-cycle."""
    y = np.array([0.1, 0.1])
    y = henon_map.image(y, henon_params, n=1002)
    res = find_cycle(henon_map, y, henon_params, period=2)
    assert res.success
    assert_allclose(henon_map.image(res.y, henon_params, n=2), res.y, atol=1e-10)


def test_find_cycle_izhikevich(izh_map_module):
    res = find_cycle(izh_map_module, [-4.6], IZH_PARAMS_I8)
    assert res.success
    assert_allclose(res.y, [-4.645418], rtol=1e-6)
    assert_allclose(res.eigvals, [0.221961], rtol=1e-5)


def test_find_cycle_failure_no_fixed_point():
    res = find_cycle(Diffeomorphism(no_fixed_point), [0.0])
    assert not res.success
    assert res.y is None
    assert res.eigvals is None
    assert res.eigvecs is None
    assert res.residual.shape == (1,)


def test_find_cycle_failure_maxfev(henon_map, henon_params):
    res = find_cycle(henon_map, [0.0, 0.0], henon_params, options={"maxfev": 2})
    assert not res.success
    assert res.y is None
    assert "maxfev" in res.message


def test_find_cycle_root_options_reach_scipy(henon_map, henon_params):
    res = find_cycle(henon_map, [0.6, 0.2], henon_params, method="lm")
    assert res.success
    assert_allclose(res.y, HENON_FIXED_POINT, rtol=1e-8)
    with pytest.raises(ValueError, match="method"):
        find_cycle(henon_map, [0.6, 0.2], henon_params, method="no-such-method")


def test_find_cycle_does_not_modify_params(henon_map, henon_params):
    snapshot = dict(henon_params)
    find_cycle(henon_map, [0.6, 0.2], henon_params)
    assert henon_params == snapshot


# -- trace_cycle -------------------------------------------------------------------


def test_trace_cycle_izhikevich(izh_trace):
    assert len(izh_trace) == 10
    currents = [s.params["I"] for s in izh_trace]
    assert currents[0] == 8.0
    assert currents[-1] == 4.5
    assert np.all(np.diff(currents) < 0)
    for s in izh_trace:
        assert {"eigvals", "eigvecs", "nfev", "residual"} <= set(s.info)
        assert s.y.shape == (1,)
    mu = np.array([s.info["eigvals"][0].real for s in izh_trace])
    assert mu[0] == pytest.approx(0.221961, rel=1e-5)
    assert np.all(np.diff(mu) < 0)
    assert mu[-1] < -1  # period doubling crossed


def test_trace_cycle_callback_and_progress(henon_map, henon_params, capsys):
    seen = []
    steps = trace_cycle(
        henon_map,
        HENON_FIXED_POINT,
        henon_params,
        "a",
        1.2,
        resolution=3,
        callback=seen.append,
        show_progress=True,
    )
    assert seen == steps
    assert len(steps) == 3
    assert "SUCCESS" in capsys.readouterr().err
    for s in steps:
        a, b = s.params["a"], s.params["b"]
        x = (-(1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
        assert_allclose(s.y, [x, b * x], rtol=1e-8)


def test_trace_cycle_stops_on_failure():
    # x -> x + p - x^2 has fixed points only for p >= 0.
    steps = trace_cycle(
        Diffeomorphism(saddle_node), [0.5], {"p": 0.25}, "p", -0.25, resolution=3
    )
    # p = 0 still has the (degenerate) fixed point 0; p = -0.25 has none.
    assert [s.params["p"] for s in steps] == [0.25, 0.0]
    assert_allclose(steps[0].y, [0.5])
    assert_allclose(steps[1].y, [0.0], atol=1e-8)


# -- find_local_bf ------------------------------------------------------------------


def test_find_local_bf_izhikevich_period_doubling(izh_map_module, izh_trace):
    mu = np.array([s.info["eigvals"][0].real for s in izh_trace])
    k = int(np.argmin(np.abs(mu + 1)))
    j = int(np.argmax(mu < -1))  # first trace point past mu = -1
    assert j > 0
    lo, hi = izh_trace[j].params["I"], izh_trace[j - 1].params["I"]
    start = izh_trace[k]
    res = find_local_bf(izh_map_module, start.y, start.params, "I", np.pi)
    assert isinstance(res, LocalBfResult)
    assert res.success
    assert abs(res.eigvals[0] + 1) < 1e-6
    assert res.theta == pytest.approx(np.pi)
    # The bifurcation value lies between the trace points bracketing mu = -1.
    assert lo < res.params["I"] < hi
    assert res.residual.shape == (1 + 2,)
    assert res.y.shape == (1,)
    assert res.params["a"] == IZH_PARAMS_I8["a"]


def test_find_local_bf_saddle_node():
    smap = Diffeomorphism(saddle_node)
    params = {"p": 0.1}
    res = find_local_bf(smap, [0.3], params, "p", 0.0)
    assert res.success
    assert_allclose(res.y, [0.0], atol=1e-6)
    assert res.params["p"] == pytest.approx(0.0, abs=1e-8)
    assert np.angle(np.exp(1j * res.theta)) == pytest.approx(0.0, abs=1e-6)
    assert_allclose(res.eigvals, [1.0], atol=1e-6)
    assert params == {"p": 0.1}  # not modified


def test_find_local_bf_henon_period_doubling(henon_map):
    params = {"a": 0.35, "b": 0.3}
    start = find_cycle(henon_map, [0.9, 0.3], params)
    assert start.success
    res = find_local_bf(henon_map, start.y, params, "a", np.pi)
    assert res.success
    assert res.params["a"] == pytest.approx(henon_pd_value(0.3), abs=1e-8)
    assert res.params["b"] == 0.3
    assert res.residual.shape == (2 + 2,)
    assert_allclose(np.sort(res.eigvals.real), [-1.0, 0.3], atol=1e-8)
    assert params == {"a": 0.35, "b": 0.3}


def test_find_local_bf_failure(henon_map, henon_params):
    res = find_local_bf(
        henon_map, [0.6, 0.2], henon_params, "a", np.pi, options={"maxfev": 2}
    )
    assert not res.success
    assert res.y is None
    assert res.params is None
    assert res.theta is None
    assert res.eigvals is None


# -- trace_local_bf ------------------------------------------------------------------


def test_trace_local_bf_same_keys_raise(henon_map, henon_params):
    with pytest.raises(ValueError, match="differ"):
        trace_local_bf(henon_map, [0.6, 0.2], henon_params, "a", np.pi, "a", 1.0)


def test_trace_local_bf_henon_period_doubling(henon_map):
    params = {"a": 0.35, "b": 0.3}
    start = find_cycle(henon_map, [0.9, 0.3], params)
    bf = find_local_bf(henon_map, start.y, params, "a", np.pi)
    assert bf.success
    seen = []
    steps = trace_local_bf(
        henon_map,
        bf.y,
        bf.params,
        "a",
        bf.theta,
        "b",
        0.2,
        resolution=6,
        callback=seen.append,
    )
    assert len(steps) == 6
    assert_allclose([s.params["b"] for s in steps], np.linspace(0.3, 0.2, 6))
    for s in steps:
        b = s.params["b"]
        assert s.params["a"] == pytest.approx(henon_pd_value(b), abs=1e-6)
        assert s.y.shape == (2,)  # public steps exclude theta
        x = (1 - b) / (2 * s.params["a"])
        assert_allclose(s.y, [x, b * x], atol=1e-6)
        assert s.info["theta"] == pytest.approx(np.pi)
        assert s.info["residual"].shape == (2 + 2,)
        assert np.min(np.abs(s.info["eigvals"] + 1)) < 1e-6
    assert len(seen) == len(steps)
    for a, b in zip(seen, steps, strict=True):
        assert a.y.shape == (2,)
        assert_allclose(a.y, b.y)
        assert a.params == b.params


def test_trace_local_bf_stops_on_failure(henon_map):
    params = {"a": 0.3675, "b": 0.3}
    steps = trace_local_bf(
        henon_map,
        [0.95238095, 0.28571429],
        params,
        "a",
        np.pi,
        "b",
        0.2,
        resolution=3,
        options={"maxfev": 1},
    )
    # The first point is already a bifurcation point; moving b makes the
    # (starved) solver fail, which ends the trace.
    assert len(steps) == 1
    assert steps[0].params["b"] == 0.3


# -- doctests ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", ["mappy_ds.root", "mappy_ds.root._cycle", "mappy_ds.root._local_bf"]
)
def test_doctests(module):
    result = doctest.testmod(
        importlib.import_module(module), optionflags=doctest.NORMALIZE_WHITESPACE
    )
    assert result.failed == 0


# -- analytic Jacobian of the local-bifurcation system -----------------------------


def _check_system_jacobian(diffeo, var, params, key, period=1, h=1e-6, **tol):
    from mappy_ds.root._local_bf import _local_bf_system

    f0, jac = _local_bf_system(diffeo, var, params, key, period)
    fd = np.zeros_like(jac)
    for k in range(var.size):
        e = np.zeros_like(var)
        e[k] = h
        fp = _local_bf_system(diffeo, var + e, params, key, period)[0]
        fm = _local_bf_system(diffeo, var - e, params, key, period)[0]
        fd[:, k] = (fp - fm) / (2 * h)
    assert f0.shape == (var.size,)
    assert_allclose(jac, fd, **tol)


def test_local_bf_system_jacobian_henon(henon_map):
    params = {"a": 0.35, "b": 0.3}
    fp = find_cycle(henon_map, [0.5, 0.15], params).y
    var = np.concatenate([fp, [params["a"], np.pi]])
    _check_system_jacobian(henon_map, var, params, "a", rtol=1e-6, atol=1e-8)


def test_local_bf_system_jacobian_izhikevich(izh_trace):
    mu = np.array([s.info["eigvals"][0].real for s in izh_trace])
    start = izh_trace[int(np.argmin(np.abs(mu + 1)))]
    pmap = make_izhikevich_map(rtol=1e-10, atol=1e-12)
    var = np.concatenate([start.y, [start.params["I"], np.pi]])
    # fully analytic now (no finite difference left in the system Jacobian)
    _check_system_jacobian(pmap, var, start.params, "I", h=1e-5, rtol=1e-5, atol=1e-6)


def test_cofactor_matches_determinant_derivative():
    from mappy_ds.root._local_bf import _cofactor

    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    cof = _cofactor(a)
    # Jacobi's formula: d det(a) = sum(C * da); also C = det(a) a^{-T}
    assert_allclose(cof, np.linalg.det(a) * np.linalg.inv(a).T, rtol=1e-10)
    assert_allclose(_cofactor(np.array([[5.0]])), [[1.0]])


# -- trace_cycle_arclength -----------------------------------------------------------


def test_trace_cycle_arclength_saddle_node_fold():
    steps = trace_cycle_arclength(
        Diffeomorphism(saddle_node),
        [0.5],
        {"p": 0.25},
        "p",
        direction=-1,
        step=0.05,
        step_max=0.1,
        bounds=(-1.0, 0.3),
    )
    ps = np.array([s.params["p"] for s in steps])
    ys = np.array([s.y[0] for s in steps])
    assert_allclose(ps, ys**2, atol=1e-9)
    assert ps.min() < 0.01 and ys[-1] < -0.3  # through the fold, onto -sqrt(p)
    folds = [i for i, s in enumerate(steps) if "fold" in s.info["events"]]
    assert len(folds) == 1
    assert ps[folds[0] - 1] < 0.05 and ps[folds[0]] < 0.05
    for s in steps:
        assert {
            "eigvals",
            "eigvecs",
            "tangent",
            "step",
            "newton_iterations",
            "events",
        } <= set(s.info)
        assert s.info["tangent"].shape == (2,)


def test_trace_cycle_arclength_henon_period_doubling(henon_map):
    params = {"a": 0.1, "b": 0.3}
    fp = find_cycle(henon_map, [0.7, 0.2], params).y
    steps = trace_cycle_arclength(
        henon_map, fp, params, "a", bounds=(0.0, 1.0), step=0.02, step_max=0.1
    )
    a_values = np.array([s.params["a"] for s in steps])
    assert a_values[0] == pytest.approx(0.1) and a_values[-1] > 0.9
    assert np.all(np.diff(a_values) > 0)  # no fold on this branch
    for s in steps:
        assert_allclose(henon_map.image(s.y, s.params), s.y, atol=1e-9)
    pd = [i for i, s in enumerate(steps) if "period_doubling" in s.info["events"]]
    assert len(pd) == 1
    a_pd = 3 * (1 - 0.3) ** 2 / 4  # exact: 0.3675
    assert steps[pd[0] - 1].params["a"] < a_pd < steps[pd[0]].params["a"]
    refined = find_local_bf(henon_map, steps[pd[0]].y, steps[pd[0]].params, "a", np.pi)
    assert refined.success and refined.params["a"] == pytest.approx(a_pd, abs=1e-8)
    assert not any("fold" in s.info["events"] for s in steps)


def test_trace_cycle_arclength_izhikevich(izh_map_module):
    params = dict(IZH_PARAMS_I8)
    fp = find_cycle(izh_map_module, [-4.6], params).y
    steps = trace_cycle_arclength(
        izh_map_module,
        fp,
        params,
        "I",
        direction=-1,
        bounds=(4.5, 8.0),
        step=0.1,
        step_max=0.5,
    )
    currents = np.array([s.params["I"] for s in steps])
    assert currents[0] == pytest.approx(8.0)
    assert np.all(np.diff(currents) < 0) and currents[-1] < 4.7
    pd = [i for i, s in enumerate(steps) if "period_doubling" in s.info["events"]]
    assert len(pd) == 1
    assert steps[pd[0]].params["I"] < 4.5869 < steps[pd[0] - 1].params["I"]


def test_trace_cycle_arclength_requires_dependence(henon_map, henon_params):
    with pytest.raises(ValueError, match="does not depend"):
        trace_cycle_arclength(henon_map, [0.6, 0.2], henon_params, "zzz")


def test_trace_cycle_arclength_callback_and_progress(capsys):
    seen = []
    steps = trace_cycle_arclength(
        Diffeomorphism(saddle_node),
        [0.5],
        {"p": 0.25},
        "p",
        direction=-1,
        step=0.1,
        step_max=0.1,
        bounds=(-1.0, 0.3),
        max_steps=5,
        show_progress=True,
        callback=seen.append,
    )
    assert len(steps) == 6 and len(seen) == 6
    err = capsys.readouterr().err
    assert "ARCLENGTH" in err and err.endswith("\n")


# -- trace_local_bf_arclength -------------------------------------------------------


@mapping(1)
def cusp(x, p):
    return x + p["l1"] + p["l2"] * x - x**3


def test_trace_local_bf_arclength_cusp_fold_curve():
    # fixed points: l1 + l2 x - x^3 = 0; fold where l2 = 3 x^2, i.e. l1 = -2 x^3
    steps = trace_local_bf_arclength(
        Diffeomorphism(cusp),
        [0.5],
        {"l1": -0.25, "l2": 0.75},
        "l1",
        0.0,
        "l2",
        direction=-1,
        step=0.1,
        step_max=0.2,
        bounds=(-1.0, 0.75),
    )
    xs = np.array([s.y[0] for s in steps])
    l1 = np.array([s.params["l1"] for s in steps])
    l2 = np.array([s.params["l2"] for s in steps])
    assert_allclose(l2, 3 * xs**2, atol=1e-9)
    assert_allclose(l1, -2 * xs**3, atol=1e-9)
    assert l2.min() < 0.1  # reached the neighbourhood of the cusp ...
    assert xs[-1] < -0.3 and l1[-1] > 0  # ... and continued on the other branch
    folds = [i for i, s in enumerate(steps) if "fold" in s.info["events"]]
    assert len(folds) == 1 and abs(xs[folds[0]]) < 0.2
    for s in steps:
        assert s.info["theta"] == pytest.approx(0.0, abs=1e-9)
        assert abs(s.info["eigvals"][0] - 1) < 1e-8  # multiplier +1 along a fold curve
        assert {
            "theta",
            "eigvals",
            "eigvecs",
            "tangent",
            "step",
            "newton_iterations",
            "events",
        } <= set(s.info)
        assert s.info["tangent"].shape == (4,)


def test_trace_local_bf_arclength_henon_period_doubling_curve(henon_map):
    params = {"a": 0.3675, "b": 0.3}
    fp = find_cycle(henon_map, [0.5, 0.15], params).y
    steps = trace_local_bf_arclength(
        henon_map,
        fp,
        params,
        "a",
        np.pi,
        "b",
        direction=-1,
        bounds=(-0.5, 0.3),
        step=0.05,
        step_max=0.2,
    )
    bs = np.array([s.params["b"] for s in steps])
    assert bs[0] == pytest.approx(0.3) and bs[-1] < -0.4
    for s in steps:
        assert s.params["a"] == pytest.approx(
            3 * (1 - s.params["b"]) ** 2 / 4, abs=1e-9
        )
        assert s.info["theta"] == pytest.approx(np.pi, abs=1e-9)
        assert not s.info["events"]  # smooth curve, no fold, no resonance


def test_trace_local_bf_arclength_izhikevich(izh_map_module):
    params = {**IZH_PARAMS_I8, "I": 4.5869}
    bf = find_local_bf(izh_map_module, [-6.79], params, "I", np.pi)
    assert bf.success
    steps = trace_local_bf_arclength(
        izh_map_module,
        bf.y,
        bf.params,
        "I",
        np.pi,
        "a",
        bounds=(0.2, 0.3),
        step=0.02,
        step_max=0.1,
    )
    a_values = np.array([s.params["a"] for s in steps])
    assert a_values[0] == pytest.approx(0.2) and a_values[-1] > 0.29
    assert np.all(np.diff(a_values) > 0)
    for s in steps:
        assert abs(s.info["eigvals"][0] + 1) < 1e-6
    # agrees with the natural-parameter continuation at the final a
    nat = trace_local_bf(
        izh_map_module,
        bf.y,
        bf.params,
        "I",
        np.pi,
        "a",
        float(a_values[-1]),
        resolution=6,
    )
    assert nat[-1].params["I"] == pytest.approx(steps[-1].params["I"], abs=1e-6)


def test_trace_local_bf_arclength_argument_errors(henon_map, henon_params):
    with pytest.raises(ValueError, match="differ"):
        trace_local_bf_arclength(henon_map, [0.6, 0.2], henon_params, "a", 0.0, "a")
    with pytest.raises(ValueError, match="neither"):
        trace_local_bf_arclength(henon_map, [0.6, 0.2], henon_params, "x", 0.0, "y")


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (0.1, -0.1, ["resonance_1_1"]),
        (np.pi - 0.1, np.pi + 0.1, ["resonance_1_2"]),
        (2 * np.pi / 3 - 0.1, 2 * np.pi / 3 + 0.1, ["resonance_1_3"]),
        (np.pi / 2 - 0.1, np.pi / 2 + 0.1, ["resonance_1_4"]),
        (0.3, 0.4, []),
        (np.pi, np.pi, []),
    ],
)
def test_resonance_events(before, after, expected):
    from mappy_ds.root._local_bf import _resonance_events

    assert _resonance_events(before, after) == expected
