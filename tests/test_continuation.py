"""Tests of :mod:`mappy_ds.continuation`."""

from __future__ import annotations

import doctest

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mappy_ds.continuation as continuation_module
from mappy_ds.continuation import (
    ArclengthPoint,
    ContinuationStep,
    StepFunctionResult,
    arclength_continuation,
    continuation,
)


def sqrt_step(y, p):
    return StepFunctionResult(True, p["p"] ** 0.5)


def test_linspace_sweep():
    steps = continuation(sqrt_step, 1.0, {"p": 1.0}, "p", 4.0, resolution=7)
    assert len(steps) == 7
    values = [s.params["p"] for s in steps]
    assert_allclose(values, np.linspace(1.0, 4.0, 7))
    assert values[0] == 1.0
    assert values[-1] == 4.0  # end value hit exactly
    assert all(isinstance(s, ContinuationStep) for s in steps)
    assert all(s.info == {} for s in steps)
    assert_allclose([s.y for s in steps], np.sqrt(values))


def test_decreasing_sweep():
    steps = continuation(sqrt_step, 2.0, {"p": 4.0}, "p", 1.0, resolution=4)
    assert [s.params["p"] for s in steps] == [4.0, 3.0, 2.0, 1.0]


def test_previous_value_seeds_next_step():
    seen = []

    def step(y, p):
        seen.append(y)
        return StepFunctionResult(True, y + 1)

    steps = continuation(step, 0, {"p": 0.0}, "p", 1.0, resolution=4)
    assert seen == [0, 1, 2, 3]
    assert [s.y for s in steps] == [1, 2, 3, 4]


def test_stops_at_first_failure():
    def step(y, p):
        return StepFunctionResult(p["p"] < 2.5, y)

    calls = []

    def counted(y, p):
        calls.append(p["p"])
        return step(y, p)

    steps = continuation(counted, 0.0, {"p": 0.0}, "p", 5.0, resolution=6)
    assert [s.params["p"] for s in steps] == [0.0, 1.0, 2.0]
    assert calls == [0.0, 1.0, 2.0, 3.0]  # no call after the failure


def test_stops_when_y_is_none():
    def step(y, p):
        return StepFunctionResult(True, None if p["p"] >= 1.0 else y)

    steps = continuation(step, 0.0, {"p": 0.0}, "p", 2.0, resolution=5)
    assert [s.params["p"] for s in steps] == [0.0, 0.5]


def test_failure_at_first_step_returns_empty():
    steps = continuation(
        lambda y, p: StepFunctionResult(False), 0.0, {"p": 0.0}, "p", 1.0
    )
    assert steps == []


def test_returned_params_are_carried():
    """A parameter solved by the step function is seen by the next step."""
    received = []

    def step(y, p):
        received.append(dict(p))
        p["q"] = p["q"] + 10.0  # e.g. a solved bifurcation parameter
        return StepFunctionResult(True, y, p, {"q": p["q"]})

    steps = continuation(step, 0.0, {"p": 0.0, "q": 0.0}, "p", 2.0, resolution=3)
    assert [r["q"] for r in received] == [0.0, 10.0, 20.0]
    assert [s.params["q"] for s in steps] == [10.0, 20.0, 30.0]
    # The swept parameter still follows the linspace.
    assert [r["p"] for r in received] == [0.0, 1.0, 2.0]
    assert [s.info["q"] for s in steps] == [10.0, 20.0, 30.0]


def test_step_function_may_modify_its_dict():
    def step(y, p):
        p["p"] = 1e9
        p["junk"] = 1
        return StepFunctionResult(True, y)

    steps = continuation(step, 0.0, {"p": 0.0}, "p", 1.0, resolution=3)
    assert [s.params for s in steps] == [{"p": 0.0}, {"p": 0.5}, {"p": 1.0}]


def test_caller_params_untouched():
    params = {"p": 1.0, "q": 2.0}
    snapshot = dict(params)

    def step(y, p):
        p["q"] = -1.0
        return StepFunctionResult(True, y, p)

    steps = continuation(step, 0.0, params, "p", 3.0, resolution=3)
    assert params == snapshot
    # Steps own their params.
    steps[0].params["p"] = 99.0
    assert steps[1].params["p"] == 2.0


def test_callback_called_once_per_success():
    collected = []

    def step(y, p):
        return StepFunctionResult(p["p"] < 3.5, p["p"])

    steps = continuation(
        step, 0.0, {"p": 0.0}, "p", 5.0, resolution=6, callback=collected.append
    )
    assert len(collected) == len(steps) == 4
    assert collected == steps


@pytest.mark.parametrize("resolution", [1, 0, -3])
def test_resolution_too_small(resolution):
    with pytest.raises(ValueError, match="resolution"):
        continuation(sqrt_step, 1.0, {"p": 1.0}, "p", 2.0, resolution=resolution)


def test_show_progress(capsys):
    continuation(sqrt_step, 1.0, {"p": 1.0}, "p", 4.0, resolution=4, show_progress=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("SUCCESS") == 4
    assert "4/4" in captured.err
    assert "p=+4" in captured.err
    assert captured.err.endswith("\n")


def test_show_progress_newline_even_on_error(capsys):
    def step(y, p):
        if p["p"] > 1.5:
            raise ZeroDivisionError
        return StepFunctionResult(True, y)

    with pytest.raises(ZeroDivisionError):
        continuation(step, 0.0, {"p": 1.0}, "p", 2.0, resolution=3, show_progress=True)
    assert capsys.readouterr().err.endswith("\n")


def test_no_output_without_progress(capsys):
    continuation(sqrt_step, 1.0, {"p": 1.0}, "p", 4.0, resolution=4)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_doctest():
    result = doctest.testmod(
        continuation_module, optionflags=doctest.NORMALIZE_WHITESPACE
    )
    assert result.attempted > 0
    assert result.failed == 0


# -- pseudo-arclength continuation ---------------------------------------------


def circle(u):
    return np.array([u @ u - 1.0]), 2.0 * u[np.newaxis, :]


def test_arclength_circle_is_closed_and_passes_folds():
    pts = arclength_continuation(circle, [1.0, 0.0], step=0.3, step_max=0.3)
    assert 15 < len(pts) < 40
    for p in pts:
        assert abs(p.u @ p.u - 1.0) < 1e-9
        assert abs(np.linalg.norm(p.tangent) - 1.0) < 1e-12
        assert abs(p.tangent @ p.u) < 1e-8  # tangent to the circle
    assert pts[1].u[1] > 0  # direction=+1: last component increases first
    # the parameter component of the tangent changes sign twice (two folds)
    signs = np.sign([p.tangent[1] for p in pts])
    assert np.sum(signs[1:] != signs[:-1]) == 2
    # stopped because the curve closed on itself, not because of max_steps
    assert np.linalg.norm(pts[-1].u - pts[0].u) < 0.5


def test_arclength_direction_and_bounds():
    pts = arclength_continuation(
        circle, [1.0, 0.0], step=0.1, step_max=0.1, direction=-1, bounds=(-0.5, 1.0)
    )
    assert pts[1].u[1] < 0
    assert all(p.u[1] >= -0.5 for p in pts)
    assert pts[-1].u[1] < -0.3  # stopped at the lower bound, not earlier


def test_arclength_corrects_starting_point_with_parameter_fixed():
    pts = arclength_continuation(circle, [0.7, 0.3], step=0.1, max_steps=0)
    assert len(pts) == 1
    assert pts[0].u[1] == pytest.approx(0.3)
    assert pts[0].u[0] == pytest.approx(np.sqrt(1 - 0.09))
    assert pts[0].step == 0.0


def test_arclength_passes_saddle_node_fold():
    # fixed points of y -> y + p - y^2: the curve p = y^2 folds at (0, 0)
    def system(u):
        y, p = u
        return np.array([p - y * y]), np.array([[-2.0 * y, 1.0]])

    pts = arclength_continuation(
        system, [0.5, 0.25], step=0.05, step_max=0.1, direction=-1, bounds=(-1, 0.3)
    )
    ys = np.array([p.u[0] for p in pts])
    ps = np.array([p.u[1] for p in pts])
    assert_allclose(ps, ys**2, atol=1e-9)
    assert ps.min() < 0.01  # reached the fold
    assert ys[-1] < -0.3 and ps[-1] > 0.1  # and continued on the other branch


def test_arclength_step_adapts_and_retries():
    calls = {"n": 0}

    def system(u):
        calls["n"] += 1
        if u[1] > 0.6 and calls["n"] % 7 == 0:
            raise np.linalg.LinAlgError("pretend the map failed here")
        return circle(u)

    pts = arclength_continuation(
        system, [1.0, 0.0], step=0.05, step_min=1e-4, step_max=0.4, bounds=(-2, 0.9)
    )
    steps = [p.step for p in pts[1:]]
    assert max(steps) > 0.05  # grew
    assert min(steps) > 0  # never below step_min
    for p in pts:
        assert abs(p.u @ p.u - 1.0) < 1e-9


def test_arclength_invalid_arguments():
    with pytest.raises(ValueError):
        arclength_continuation(circle, [1.0], step=0.1)
    with pytest.raises(ValueError):
        arclength_continuation(circle, [1.0, 0.0], direction=2)
    with pytest.raises(ValueError):
        arclength_continuation(circle, [1.0, 0.0], step=1.0, step_max=0.1)


def test_arclength_callback_receives_every_point():
    seen: list[ArclengthPoint] = []
    pts = arclength_continuation(
        circle, [1.0, 0.0], step=0.3, step_max=0.3, callback=seen.append
    )
    assert len(seen) == len(pts)
    assert all(a is b for a, b in zip(seen, pts, strict=True))


def test_arclength_uncorrectable_start_raises():
    def nowhere(u):
        return np.array([u @ u + 1.0]), 2.0 * u[np.newaxis, :]

    with pytest.raises(RuntimeError, match="starting point"):
        arclength_continuation(nowhere, [1.0, 0.0])
