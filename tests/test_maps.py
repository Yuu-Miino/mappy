"""Tests of :mod:`mappy_ds.maps` and :mod:`mappy_ds.trajectory`."""

from __future__ import annotations

import re

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import root

from conftest import fd_hessian, fd_jacobian
from mappy_ds import (
    ContinuousMode,
    Diffeomorphism,
    DiscreteMode,
    MapResult,
    PoincareMap,
    Trajectory,
    TrajectorySegment,
    TransitionError,
    UnknownModeError,
    border,
    mapping,
    vector_field,
)

# Reference values of the Izhikevich example (Miino, NOLTA 2024).
IZH_I10_Y0 = -3.0
IZH_I10 = {"y": -3.4783849356, "jac": 0.27080365164, "hes": -0.193258205836}
IZH_I8 = {"u": -4.6454201715, "jac": 0.2219605834, "hes": -0.2604668216}

# The same quantities computed here with DOP853 at rtol=1e-13, atol=1e-14
# (converged to ~1e-11). The I=10 Jacobian/Hessian reference values above
# deviate from these by ~1e-6 / ~1e-5 relative.
IZH_I10_CONVERGED = {
    "y": -3.478384992562513,
    "jac": 0.27080389832382407,
    "hes": -0.19326009887888657,
}
IZH_I8_CONVERGED = {
    "u": -4.645418482929349,
    "jac": 0.22196057194767838,
    "hes": -0.2604668251615617,
}

HENON_FIXED_POINT = np.array([0.631354477, 0.189406343])
HENON_MULTIPLIERS = np.array([-1.923738858, 0.155946322])


def _izh_fixed_point(pmap, params):
    sol = root(lambda u: pmap.image(u, params) - u, [-4.6])
    assert sol.success
    return sol.x


# -- Izhikevich: paper values ---------------------------------------------------


def test_izhikevich_paper_values(izh_map, izh_params):
    res = izh_map.image_detail(IZH_I10_Y0, izh_params, hessian=True)
    assert res.y.shape == (1,)
    assert res.jac.shape == (1, 1)
    assert res.hes.shape == (1, 1, 1)
    assert_allclose(res.y[0], IZH_I10["y"], rtol=1e-5)
    assert_allclose(res.jac[0, 0], IZH_I10["jac"], rtol=1e-5)
    assert_allclose(res.hes[0, 0, 0], IZH_I10["hes"], rtol=1e-5)


def test_izhikevich_fixed_point_paper_values(izh_map, izh_params):
    params = {**izh_params, "I": 8.0}
    u = _izh_fixed_point(izh_map, params)
    res = izh_map.image_detail(u, params, hessian=True)
    assert_allclose(u[0], IZH_I8["u"], rtol=1e-5)
    assert_allclose(res.jac[0, 0], IZH_I8["jac"], rtol=1e-5)
    assert_allclose(res.hes[0, 0, 0], IZH_I8["hes"], rtol=1e-5)
    assert_allclose(res.eigvals, [IZH_I8["jac"]], rtol=1e-5)


def test_izhikevich_tight_tolerance(izh_map_tight, izh_params):
    res = izh_map_tight.image_detail(IZH_I10_Y0, izh_params, hessian=True)
    assert_allclose(res.y[0], IZH_I10_CONVERGED["y"], rtol=1e-8)
    assert_allclose(res.jac[0, 0], IZH_I10_CONVERGED["jac"], rtol=1e-7)
    assert_allclose(res.hes[0, 0, 0], IZH_I10_CONVERGED["hes"], rtol=1e-7)
    # Against the published values: y agrees to ~1e-8, jac/hes to ~1e-6/1e-5.
    assert_allclose(res.y[0], IZH_I10["y"], rtol=1e-7)
    assert_allclose(res.jac[0, 0], IZH_I10["jac"], rtol=2e-6)
    assert_allclose(res.hes[0, 0, 0], IZH_I10["hes"], rtol=2e-5)


def test_izhikevich_fixed_point_tight_tolerance(izh_map_tight, izh_params):
    params = {**izh_params, "I": 8.0}
    u = _izh_fixed_point(izh_map_tight, params)
    res = izh_map_tight.image_detail(u, params, hessian=True)
    assert_allclose(u[0], IZH_I8_CONVERGED["u"], rtol=1e-8)
    assert_allclose(res.jac[0, 0], IZH_I8_CONVERGED["jac"], rtol=1e-7)
    assert_allclose(res.hes[0, 0, 0], IZH_I8_CONVERGED["hes"], rtol=1e-7)
    assert_allclose(u[0], IZH_I8["u"], rtol=1e-6)
    assert_allclose(res.jac[0, 0], IZH_I8["jac"], rtol=1e-7)
    assert_allclose(res.hes[0, 0, 0], IZH_I8["hes"], rtol=1e-7)


def test_izhikevich_map_derivatives_finite_difference(izh_map_tight, izh_params):
    u0 = np.array([-3.0])
    res = izh_map_tight.image_detail(u0, izh_params, hessian=True)

    def f(u):
        return izh_map_tight.image(u, izh_params)

    assert_allclose(res.jac, fd_jacobian(f, u0), atol=1e-6)
    assert_allclose(res.hes, fd_hessian(f, u0), atol=1e-4)


def test_izhikevich_mode_history(izh_map, izh_params):
    res = izh_map.image_detail(IZH_I10_Y0, izh_params, jacobian=False)
    assert res.mode_history == ["Pinv", "B", "A", "P", "Pinv"]
    assert res.end_mode == "Pinv"
    res3 = izh_map.image_detail(IZH_I10_Y0, izh_params, n=3, jacobian=False)
    assert len(res3.mode_history) == 1 + 3 * 4
    assert res3.mode_history[:5] == res.mode_history
    assert res3.mode_history.count("A") == 3


def test_image_equals_image_detail(izh_map, izh_params):
    y = izh_map.image(IZH_I10_Y0, izh_params, n=2)
    detail = izh_map.image_detail(IZH_I10_Y0, izh_params, n=2, jacobian=False)
    assert_allclose(y, detail.y, rtol=0, atol=0)
    # With derivatives the augmented (variational) system is integrated, so
    # step-size control differs and the image agrees only to solver tolerance.
    with_jac = izh_map.image_detail(IZH_I10_Y0, izh_params, n=2)
    assert_allclose(y, with_jac.y, rtol=1e-5)
    # n iterations compose single iterations.
    y1 = izh_map.image(IZH_I10_Y0, izh_params)
    assert_allclose(y, izh_map.image(y1, izh_params), rtol=1e-6)


def test_image_detail_flags(izh_map, izh_params):
    res = izh_map.image_detail(IZH_I10_Y0, izh_params)
    assert res.jac is not None  # jacobian defaults to True
    assert res.hes is None
    assert res.trajectory is None
    res = izh_map.image_detail(IZH_I10_Y0, izh_params, jacobian=False)
    assert res.jac is None
    assert res.hes is None


def test_poincare_map_dim(izh_map, duffing_map):
    assert izh_map.dim == 1
    assert duffing_map.dim == 2


# -- MapResult -------------------------------------------------------------------


def test_eig_consistent_with_numpy(henon_map, henon_params):
    res = henon_map.image_detail([0.1, 0.2], henon_params, n=2)
    vals, vecs = np.linalg.eig(res.jac)
    assert res.eigvals.dtype == np.complex128
    assert res.eigvecs.dtype == np.complex128
    assert_allclose(res.eigvals, vals)
    assert_allclose(res.eigvecs, vecs)
    for lam, v in zip(res.eigvals, res.eigvecs.T, strict=True):
        assert_allclose(res.jac @ v, lam * v, atol=1e-12)


def test_eig_complex_multipliers(duffing_map, duffing_params, duffing_y0):
    res = duffing_map.image_detail(duffing_y0, duffing_params)
    vals = np.linalg.eigvals(res.jac)
    assert_allclose(np.sort_complex(res.eigvals), np.sort_complex(vals))
    assert np.all(np.abs(res.eigvals.imag) > 0)  # a stable focus
    assert np.all(np.abs(res.eigvals) < 1)


def test_eigvals_require_jacobian(izh_map, izh_params):
    res = izh_map.image_detail(IZH_I10_Y0, izh_params, jacobian=False)
    with pytest.raises(ValueError, match="Jacobian"):
        _ = res.eigvals
    with pytest.raises(ValueError):
        _ = res.eigvecs


def test_eigvals_require_square_jacobian():
    res = MapResult(np.zeros(2), jac=np.ones((2, 3)))
    with pytest.raises(ValueError, match="square"):
        _ = res.eigvals


# -- Hénon diffeomorphism ------------------------------------------------------


def test_henon_iterates(henon_map, henon_fun, henon_params):
    y0 = np.array([0.1, -0.2])
    manual = y0
    for _ in range(3):
        manual = henon_fun(manual, henon_params)
    assert_allclose(henon_map.image(y0, henon_params, n=3), manual, rtol=1e-14)


def test_henon_third_iterate_derivatives(henon_map, henon_fun, henon_params):
    y0 = np.array([0.1, -0.2])
    res = henon_map.image_detail(y0, henon_params, n=3, hessian=True)

    def f(y):
        return henon_map.image(y, henon_params, n=3)

    assert_allclose(res.jac, fd_jacobian(f, y0, h=1e-5), atol=1e-8)
    assert_allclose(res.hes, fd_hessian(f, y0, h=1e-4), atol=1e-6)

    # Chain rule by hand.
    y1 = henon_fun(y0, henon_params)
    y2 = henon_fun(y1, henon_params)
    jac = (
        henon_fun.jacobian(y2, henon_params)
        @ henon_fun.jacobian(y1, henon_params)
        @ henon_fun.jacobian(y0, henon_params)
    )
    assert_allclose(res.jac, jac, rtol=1e-13)


def test_henon_fixed_point(henon_map, henon_params):
    sol = root(lambda y: henon_map.image(y, henon_params) - y, [0.5, 0.2])
    assert sol.success
    assert_allclose(sol.x, HENON_FIXED_POINT, rtol=1e-8)
    res = henon_map.image_detail(sol.x, henon_params)
    assert_allclose(np.sort(res.eigvals.real), HENON_MULTIPLIERS, rtol=1e-8)
    assert_allclose(res.eigvals.imag, 0.0)


def test_henon_mode_history_and_trajectory(henon_map, henon_params):
    res = henon_map.image_detail([0.0, 0.0], henon_params, n=3, trajectory=True)
    assert res.mode_history == ["map"] * 4
    traj = res.trajectory
    assert len(traj) == 3
    assert all(s.kind == "discrete" for s in traj)
    assert traj.y.shape == (2, 6)
    assert_allclose(traj.segments[-1].y[:, -1], res.y)


def test_diffeomorphism_attributes(henon_map, henon_fun):
    assert henon_map.fun is henon_fun
    assert henon_map.dim == 2
    assert "henon" in repr(henon_map)
    named = Diffeomorphism(henon_fun, name="H")
    assert named.image_detail([0.0, 0.0], {"a": 1.4, "b": 0.3}).mode_history == [
        "H",
        "H",
    ]


def test_diffeomorphism_requires_square_map():
    @mapping(2, 1)
    def proj(y, _):
        return y[0]

    with pytest.raises(ValueError):
        Diffeomorphism(proj)


# -- Duffing (examples/duffing.py) ---------------------------------------------


def test_duffing_period_one_point(duffing_map, duffing_params, duffing_y0):
    res = duffing_map.image_detail(duffing_y0, duffing_params)
    assert res.mode_history == ["p", "duffing", "t_reset", "pinv", "p"]
    assert res.y.shape == (2,)
    assert res.jac.shape == (2, 2)
    assert_allclose(res.y, duffing_y0, atol=1e-5)


def test_duffing_derivatives_finite_difference(duffing_params, duffing_y0):
    from conftest import make_duffing_map

    pmap = make_duffing_map(rtol=1e-10, atol=1e-12)
    res = pmap.image_detail(duffing_y0, duffing_params, hessian=True)

    def f(y):
        return pmap.image(y, duffing_params)

    assert_allclose(res.jac, fd_jacobian(f, duffing_y0), atol=1e-6)
    assert_allclose(res.hes, fd_hessian(f, duffing_y0), atol=1e-4)


# -- two end modes (forced-Alpazur-like relay oscillator) ------------------------


def test_end_mode_differs_from_initial_mode(relay_map, relay_params):
    res = relay_map.image_detail([0.0], relay_params, hessian=True)
    assert res.mode_history == ["p_up", "up", "down", "r_down", "pinv_down", "p_down"]
    assert res.end_mode == "p_down"
    assert_allclose(res.y, [-0.2], atol=1e-9)
    assert_allclose(res.jac, [[-1.0]], atol=1e-9)
    assert_allclose(res.hes, [[[0.0]]], atol=1e-9)


def test_mode_override(relay_map, relay_params):
    res = relay_map.image_detail([-0.2], relay_params, mode="p_down")
    assert res.mode_history[0] == "p_down"
    assert res.end_mode == "p_up"
    assert_allclose(res.y, [0.4], atol=1e-9)
    assert_allclose(res.jac, [[-1.0]], atol=1e-9)
    assert_allclose(relay_map.image([-0.2], relay_params, mode="p_down"), res.y)


def test_iterations_continue_from_end_mode(relay_map, relay_params):
    res = relay_map.image_detail([0.0], relay_params, n=2)
    assert res.end_mode == "p_up"
    assert res.mode_history.count("p_down") == 1
    assert_allclose(res.y, [0.4], atol=1e-9)
    assert_allclose(res.jac, [[1.0]], atol=1e-9)


def test_mode_override_unknown(relay_map, relay_params):
    with pytest.raises(UnknownModeError):
        relay_map.image_detail([0.0], relay_params, mode="nope")


def test_wrong_initial_state_length(izh_map, izh_params):
    with pytest.raises(ValueError):
        izh_map.image_detail([1.0, 2.0], izh_params)


# -- trajectories --------------------------------------------------------------


def test_trajectory_only_in_trajectory_modes(izh_map, izh_params):
    traj = izh_map.trajectory(IZH_I10_Y0, izh_params, n=2)
    assert isinstance(traj, Trajectory)
    assert len(traj) == 2
    assert [s.mode for s in traj] == ["A", "A"]
    assert all(s.kind == "continuous" for s in traj)
    y = traj.y
    assert y.shape == (2, sum(s.y.shape[1] for s in traj))
    assert_allclose(y[:, : traj.segments[0].y.shape[1]], traj.segments[0].y)
    # Each segment starts right after the firing reset.
    for seg in traj:
        assert seg.y[0, 0] == pytest.approx(-50.0)
        assert seg.y[0, -1] == pytest.approx(30.0, abs=1e-6)


def test_image_detail_with_trajectory(izh_map, izh_params):
    res = izh_map.image_detail(IZH_I10_Y0, izh_params, trajectory=True)
    assert isinstance(res.trajectory, Trajectory)
    assert len(res.trajectory) == 1
    assert res.trajectory.segments[0].y[1, -1] == pytest.approx(res.y[0])


def test_trajectory_mixed_dimensions_raise():
    traj = Trajectory(
        [
            TrajectorySegment("a", "discrete", np.zeros((2, 2))),
            TrajectorySegment("b", "continuous", np.zeros((3, 4)), np.arange(4.0)),
        ]
    )
    assert [s.dim for s in traj] == [2, 3]
    with pytest.raises(ValueError, match="dimensions"):
        _ = traj.y


def test_trajectory_concatenation_by_hand():
    s1 = TrajectorySegment("a", "continuous", np.ones((2, 3)), np.arange(3.0))
    s2 = TrajectorySegment("b", "discrete", np.zeros((2, 2)))
    traj = Trajectory([s1, s2])
    assert_allclose(traj.y, np.hstack([np.ones((2, 3)), np.zeros((2, 2))]))


def test_empty_trajectory():
    assert Trajectory().y.shape == (0, 0)
    assert len(Trajectory()) == 0


@pytest.mark.parametrize("n", [0, -1])
def test_nonpositive_n_raises(izh_map, henon_map, izh_params, henon_params, n):
    with pytest.raises(ValueError):
        izh_map.image(IZH_I10_Y0, izh_params, n=n)
    with pytest.raises(ValueError):
        henon_map.image([0.0, 0.0], henon_params, n=n)


# -- construction and run-time errors ------------------------------------------


@mapping(1)
def ident(y, _):
    return y


@pytest.fixture
def ab_modes():
    return [DiscreteMode("a", ident), DiscreteMode("b", ident)]


def test_duplicate_mode_names():
    with pytest.raises(ValueError, match="duplicate"):
        PoincareMap(
            [DiscreteMode("a", ident), DiscreteMode("a", ident)], {"a": "a"}, "a"
        )


@pytest.mark.parametrize(
    ("transitions", "initial", "end"),
    [
        ({"x": "a"}, "a", None),  # unknown source
        ({"a": "x"}, "a", None),  # unknown target
        ({"a": ["x"]}, "a", None),  # unknown target in a list
        ({"a": "b"}, "x", None),  # unknown initial mode
        ({"a": "b"}, "a", "x"),  # unknown end mode
        ({"a": "b"}, "a", ["b", "x"]),  # unknown end mode in a list
    ],
)
def test_unknown_modes(ab_modes, transitions, initial, end):
    with pytest.raises(UnknownModeError) as excinfo:
        PoincareMap(ab_modes, transitions, initial, end_modes=end)
    assert excinfo.value.name == "x"
    assert isinstance(excinfo.value, KeyError)
    assert "'x'" in str(excinfo.value)


def test_missing_rule_for_reachable_mode(ab_modes):
    with pytest.raises(TransitionError) as excinfo:
        PoincareMap(ab_modes, {"a": "b"}, "a")
    assert excinfo.value.mode == "b"
    assert "reachable" in str(excinfo.value)


def test_unreachable_mode_needs_no_rule(ab_modes):
    pmap = PoincareMap([*ab_modes, DiscreteMode("c", ident)], {"a": "a"}, "a")
    assert pmap.image([1.0]) == [1.0]


def test_rule_with_wrong_number_of_entries(izh_params):
    from conftest import fire, firing, izhikevich

    modes = [ContinuousMode("A", izhikevich, borders=[fire]), DiscreteMode("B", firing)]
    with pytest.raises(TransitionError) as excinfo:
        PoincareMap(modes, {"B": "A", "A": []}, "B")
    assert excinfo.value.mode == "A"
    assert "expected 1" in str(excinfo.value)
    with pytest.raises(TransitionError, match="single successor"):
        PoincareMap(
            [ContinuousMode("A", izhikevich, borders=[fire, fire2]), *modes[1:]],
            {"B": "A", "A": "B"},
            "B",
        )


@border(2, name="fire2")
def fire2(y, _):
    return y[0] - 40.0


def test_named_border_rules(izh_params):
    from conftest import fire, firing, izhikevich

    modes = [
        ContinuousMode("A", izhikevich, borders=[fire, fire2]),
        DiscreteMode("B", firing),
        DiscreteMode("C", firing),
    ]
    by_name = PoincareMap(
        modes, {"B": "A", "C": "A", "A": {"fire2": "C", "fire": "B"}}, "B"
    )
    by_list = PoincareMap(modes, {"B": "A", "C": "A", "A": ["B", "C"]}, "B")
    assert (
        by_name.transitions
        == by_list.transitions
        == {"B": ("A",), "C": ("A",), "A": ("B", "C")}
    )
    assert by_name.image_detail([-50.0, -1.7], izh_params).mode_history == [
        "B",
        "A",
        "B",
    ]


@pytest.mark.parametrize(
    ("rule", "fragment"),
    [
        ({"fire": "B"}, "no successor for border(s) ['fire2']"),
        ({"fire": "B", "fire2": "C", "nope": "B"}, "unknown border(s) ['nope']"),
    ],
)
def test_named_border_rules_must_match_borders(rule, fragment):
    from conftest import fire, firing, izhikevich

    modes = [
        ContinuousMode("A", izhikevich, borders=[fire, fire2]),
        DiscreteMode("B", firing),
        DiscreteMode("C", firing),
    ]
    with pytest.raises(TransitionError, match=re.escape(fragment)):
        PoincareMap(modes, {"B": "A", "C": "A", "A": rule}, "B")


def test_mapping_rule_rejected_for_discrete_mode(ab_modes):
    with pytest.raises(TransitionError, match="no borders"):
        PoincareMap(ab_modes, {"a": {"x": "b"}, "b": "a"}, "a")


def test_duplicate_border_names_rejected():
    from conftest import fire, izhikevich

    with pytest.raises(ValueError, match="distinct names"):
        ContinuousMode("A", izhikevich, borders=[fire, fire])


def test_dimension_mismatch_rejected():
    @mapping(2, 1)
    def proj(y, _):
        return y[0]

    with pytest.raises(
        TransitionError, match="dimension 1 but 'p' starts in dimension 2"
    ):
        PoincareMap([DiscreteMode("p", proj)], {"p": "p"}, "p")


def test_unreachable_end_mode_rejected(ab_modes):
    modes = [*ab_modes, DiscreteMode("c", ident)]
    with pytest.raises(TransitionError, match="no end mode"):
        PoincareMap(modes, {"a": "b", "b": "a"}, "a", end_modes="c")


def test_borderless_mode_with_list_transition():
    @vector_field(1)
    def decay(y, _):
        return -y

    with pytest.raises(TransitionError, match="expected 1"):
        PoincareMap(
            [ContinuousMode("flow", decay, max_interval=1.0)],
            {"flow": ["flow", "flow"]},
            "flow",
        )


def test_borderless_flow_as_poincare_map():
    @vector_field(1)
    def decay(y, _):
        return -y

    pmap = PoincareMap(
        [ContinuousMode("flow", decay, max_interval=1.0, rtol=1e-10, atol=1e-12)],
        {"flow": "flow"},
        "flow",
    )
    res = pmap.image_detail([2.0], n=2)
    assert_allclose(res.y, [2.0 * np.exp(-2.0)], rtol=1e-8)
    assert_allclose(res.jac, [[np.exp(-2.0)]], rtol=1e-8)


def test_max_transitions_exhausted(ab_modes):
    @vector_field(1)
    def grow(y, _):
        return y

    @border(1, name="near")
    def near(y, _):
        return y[0] - 2.0

    @border(1, name="far")
    def far(y, _):
        return y[0] - 1e6

    @mapping(1)
    def halve(y, _):
        return y / 2

    modes = [
        ContinuousMode("flow", grow, borders=[near, far], max_interval=100.0),
        DiscreteMode("reset", halve),
        DiscreteMode("end", ident),
    ]
    pmap = PoincareMap(
        modes,
        {"flow": {"near": "reset", "far": "end"}, "reset": "flow", "end": "end"},
        "flow",
        end_modes="end",
        max_transitions=10,
    )
    assert pmap.max_transitions == 10
    with pytest.raises(RuntimeError, match="10 transitions"):
        pmap.image([1.0])


def test_repr(relay_map):
    text = repr(relay_map)
    assert "p_up" in text
    assert "p_down" in text


def test_diffeomorphism_trajectory(henon_map, henon_fun, henon_params):
    y0 = np.array([0.1, -0.2])
    traj = henon_map.trajectory(y0, henon_params, n=2)
    assert len(traj) == 2
    y1 = henon_fun(y0, henon_params)
    y2 = henon_fun(y1, henon_params)
    assert_allclose(traj.y, np.column_stack([y0, y1, y1, y2]))
