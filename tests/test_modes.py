"""Tests of :mod:`mappy_ds.modes`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from conftest import fd_hessian, fd_jacobian, fire, izhikevich
from mappy_ds import (
    ContinuousMode,
    DiscreteMode,
    MappyError,
    ModeNotTerminatedError,
    StepResult,
    TrajectorySegment,
    border,
    mapping,
    vector_field,
)

IZH_Y0 = np.array([-50.0, -1.7])

# -- DiscreteMode -----------------------------------------------------------


@pytest.fixture
def henon_mode(henon_fun):
    return DiscreteMode("H", henon_fun)


def test_discrete_step_plain(henon_mode, henon_params):
    res = henon_mode.step([0.5, 0.1], henon_params)
    assert isinstance(res, StepResult)
    assert_allclose(res.y, [1 - 1.4 * 0.25 + 0.1, 0.15])
    assert res.jac is None
    assert res.hes is None
    assert res.border is None
    assert res.t is None
    assert res.segment is None


def test_discrete_step_jacobian_only(henon_mode, henon_params):
    res = henon_mode.step([0.5, 0.1], henon_params, jacobian=True)
    assert_allclose(res.jac, [[-1.4, 1.0], [0.3, 0.0]])
    assert res.hes is None


def test_discrete_step_hessian_implies_jacobian(henon_mode, henon_params):
    res = henon_mode.step([0.5, 0.1], henon_params, hessian=True)
    assert res.jac is not None
    assert res.hes is not None
    assert res.hes.shape == (2, 2, 2)
    assert res.hes[0, 0, 0] == pytest.approx(-2.8)


def test_discrete_segment_requires_both_flags(henon_fun, henon_params):
    y0 = [0.5, 0.1]
    off = DiscreteMode("H", henon_fun)
    on = DiscreteMode("H", henon_fun, in_trajectory=True)
    assert off.step(y0, henon_params, trajectory=True).segment is None
    assert on.step(y0, henon_params).segment is None
    seg = on.step(y0, henon_params, trajectory=True).segment
    assert isinstance(seg, TrajectorySegment)
    assert seg.kind == "discrete"
    assert seg.mode == "H"
    assert seg.t is None
    assert seg.y.shape == (2, 2)
    assert_allclose(seg.y[:, 0], y0)
    assert_allclose(seg.y[:, 1], henon_fun(y0, henon_params))


def test_discrete_no_segment_when_dimension_changes():
    @mapping(2, 1)
    def proj(y, _):
        return y[0]

    mode = DiscreteMode("P", proj, in_trajectory=True)
    res = mode.step([1.0, 2.0], trajectory=True)
    assert res.segment is None
    assert_allclose(res.y, [1.0])


def test_mode_requires_system_function():
    with pytest.raises(TypeError):
        DiscreteMode("bad", lambda y, p: y)
    with pytest.raises(TypeError):
        ContinuousMode("bad", lambda y, p: y)


def test_mode_dims_and_repr(henon_mode):
    assert (henon_mode.dom_dim, henon_mode.cod_dim) == (2, 2)
    assert "H" in repr(henon_mode)


# -- ContinuousMode: Izhikevich, through the firing border ---------------------


def test_continuous_step_hits_border(izh_mode, izh_params):
    res = izh_mode.step(IZH_Y0, izh_params)
    assert res.border == 0
    assert res.t > 0
    assert res.y[0] == pytest.approx(30.0, abs=1e-6)
    assert res.jac is None
    assert res.hes is None
    assert res.segment is None


def test_continuous_jacobian_and_hessian_shapes(izh_mode, izh_params):
    res = izh_mode.step(IZH_Y0, izh_params, hessian=True)
    assert res.jac.shape == (2, 2)
    assert res.hes.shape == (2, 2, 2)
    # The image lies on the border v = 30, so the first row vanishes.
    assert_allclose(res.jac[0], 0.0, atol=1e-12)
    assert_allclose(res.hes[0], 0.0, atol=1e-12)
    # Hessian is symmetric in its last two indices.
    assert_allclose(res.hes, res.hes.transpose(0, 2, 1), atol=1e-12)


def test_continuous_derivatives_do_not_change_state(izh_mode, izh_params):
    plain = izh_mode.step(IZH_Y0, izh_params)
    full = izh_mode.step(IZH_Y0, izh_params, hessian=True)
    assert_allclose(full.y, plain.y, rtol=1e-6)
    assert full.t == pytest.approx(plain.t, rel=1e-6)


def test_continuous_jacobian_finite_difference(izh_mode_tight, izh_params):
    res = izh_mode_tight.step(IZH_Y0, izh_params, jacobian=True)

    def f(y):
        return izh_mode_tight.step(y, izh_params).y

    assert_allclose(res.jac, fd_jacobian(f, IZH_Y0, h=1e-4), atol=1e-6)


def test_continuous_hessian_finite_difference(izh_mode_tight, izh_params):
    res = izh_mode_tight.step(IZH_Y0, izh_params, hessian=True)

    def f(y):
        return izh_mode_tight.step(y, izh_params).y

    fd = fd_hessian(f, IZH_Y0, h=1e-4)
    # Sanity check that the test is not vacuous.
    assert np.abs(res.hes).max() > 1e-2
    assert_allclose(res.hes, fd, atol=1e-4)


def test_parametrised_border_derivatives_finite_difference():
    """Border depending on the state through a non-trivial gradient."""

    @vector_field(2)
    def rot(y, p):
        return np.array([-p["w"] * y[1] + 0.5 * y[0], p["w"] * y[0]])

    @border(2)
    def circle(y, p):
        return y[0] ** 2 + y[1] ** 2 - p["r"] ** 2

    mode = ContinuousMode("rot", rot, borders=[circle], rtol=1e-10, atol=1e-12)
    params = {"w": 1.0, "r": 1.5}
    y0 = np.array([0.6, -0.3])
    res = mode.step(y0, params, hessian=True)
    assert res.border == 0
    assert np.linalg.norm(res.y) == pytest.approx(1.5, abs=1e-8)

    def f(y):
        return mode.step(y, params).y

    assert_allclose(res.jac, fd_jacobian(f, y0), atol=1e-6)
    assert_allclose(res.hes, fd_hessian(f, y0), atol=1e-4)


def test_continuous_trajectory_segment(izh_mode, izh_params):
    res = izh_mode.step(IZH_Y0, izh_params, trajectory=True)
    seg = res.segment
    assert isinstance(seg, TrajectorySegment)
    assert seg.kind == "continuous"
    assert seg.mode == "A"
    assert seg.t[0] == 0.0
    assert seg.t[-1] == pytest.approx(res.t)
    assert seg.y.ndim == 2
    assert seg.y.shape[0] == 2
    assert seg.y.shape[1] == seg.t.size
    assert seg.dim == 2
    assert np.all(np.diff(seg.t) > 0)
    assert_allclose(seg.y[:, 0], IZH_Y0)
    assert_allclose(seg.y[:, -1], res.y, rtol=1e-8)


def test_continuous_no_segment_when_not_in_trajectory(izh_params):
    mode = ContinuousMode("A", izhikevich, borders=[fire], in_trajectory=False)
    assert mode.step(IZH_Y0, izh_params, trajectory=True).segment is None


def test_continuous_wrong_state_length(izh_mode, izh_params):
    with pytest.raises(ValueError):
        izh_mode.step([1.0, 2.0, 3.0], izh_params)
    with pytest.raises(ValueError):
        izh_mode.step(1.0, izh_params)


def test_border_must_be_border_function():
    with pytest.raises(TypeError):
        ContinuousMode("A", izhikevich, borders=[izhikevich])
    with pytest.raises(TypeError):
        ContinuousMode("A", izhikevich, borders=[lambda y, p: y[0]])


def test_border_dimension_mismatch():
    @border(3)
    def b3(y, _):
        return y[2]

    with pytest.raises(ValueError):
        ContinuousMode("A", izhikevich, borders=[b3])


def test_vector_field_must_be_square():
    @mapping(2, 1)
    def proj(y, _):
        return y[0]

    with pytest.raises(ValueError):
        ContinuousMode("A", proj)


def test_mode_not_terminated(izh_params):
    @border(2)
    def far(y, _):
        return y[0] - 1e6

    @vector_field(2)
    def slow(y, _):
        return np.array([1, 0])

    mode = ContinuousMode("slow", slow, borders=[far], max_interval=1.0)
    with pytest.raises(ModeNotTerminatedError) as excinfo:
        mode.step([0.0, 0.0])
    assert isinstance(excinfo.value, MappyError)
    assert excinfo.value.mode == "slow"
    assert excinfo.value.max_interval == 1.0
    assert "slow" in str(excinfo.value)


def test_border_direction_is_respected():
    """A border with direction -1 ignores the upward crossing."""

    @vector_field(1)
    def up(y, _):
        return np.array([1])

    @border(1, direction=-1)
    def at_one(y, _):
        return y[0] - 1

    @border(1, direction=1)
    def at_two(y, _):
        return y[0] - 2

    mode = ContinuousMode("up", up, borders=[at_one, at_two])
    res = mode.step([0.0])
    assert res.border == 1
    assert res.y[0] == pytest.approx(2.0)
    assert res.t == pytest.approx(2.0)


# -- ContinuousMode without borders: time-max_interval flow ----------------------

A_LIN = np.array([[0.1, 1.0], [-2.0, -0.3]])


@vector_field(2)
def linear(y, _):
    return A_LIN @ y


def test_borderless_mode_is_flow_map():
    t_end = 1.5
    mode = ContinuousMode("lin", linear, max_interval=t_end, rtol=1e-10, atol=1e-12)
    y0 = np.array([1.0, 0.5])
    res = mode.step(y0, hessian=True)
    assert res.border is None
    assert res.t == pytest.approx(t_end)
    flow = expm(A_LIN * t_end)
    assert_allclose(res.y, flow @ y0, rtol=1e-8)
    assert_allclose(res.jac, flow, rtol=1e-8)
    assert_allclose(res.hes, 0.0, atol=1e-12)
    fd = fd_jacobian(lambda y: mode.step(y).y, y0)
    assert_allclose(res.jac, fd, atol=1e-6)


def test_borderless_mode_default_tolerance():
    mode = ContinuousMode("lin", linear, max_interval=1.0)
    y0 = np.array([1.0, 0.5])
    res = mode.step(y0, jacobian=True)
    assert_allclose(res.jac, expm(A_LIN), atol=1e-5)
