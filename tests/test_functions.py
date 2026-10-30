"""Tests of :mod:`mappy_ds.functions`."""

from __future__ import annotations

import numpy as np
import pytest
import sympy
from numpy.testing import assert_allclose

from mappy_ds import (
    DiscreteMode,
    SymbolicEvaluationError,
    SystemFunction,
    border,
    mapping,
    vector_field,
)

# -- decorators ---------------------------------------------------------------


def test_vector_field_attributes():
    @vector_field(3, ["k", "B"])
    def f(y, p):
        return y

    assert isinstance(f, SystemFunction)
    assert (f.dom_dim, f.cod_dim) == (3, 3)
    assert f.param_keys == ("k", "B")
    assert f.direction == 0
    assert f.__name__ == "f"


def test_mapping_attributes():
    @mapping(2)
    def square(y, _):
        return y

    @mapping(2, 1, param_keys=["c"])
    def proj(y, p):
        return y[0] * p["c"]

    assert (square.dom_dim, square.cod_dim, square.param_keys) == (2, 2, ())
    assert (proj.dom_dim, proj.cod_dim, proj.param_keys) == (2, 1, ("c",))


@pytest.mark.parametrize("direction", [1, -1, 0])
def test_border_attributes(direction):
    @border(2, ["T"], direction=direction)
    def b(y, p):
        return y[1] - p["T"]

    assert (b.dom_dim, b.cod_dim) == (2, 1)
    assert b.param_keys == ("T",)
    assert b.direction == direction


def test_border_default_direction_is_positive():
    @border(2)
    def b(y, _):
        return y[0]

    assert b.direction == 1


@pytest.mark.parametrize(
    "decorator",
    [
        lambda: vector_field(2, "abc"),
        lambda: mapping(2, param_keys="abc"),
        lambda: border(2, "abc"),
    ],
)
def test_string_param_keys_rejected(decorator):
    def f(y, p):
        return y[0]

    with pytest.raises(TypeError, match="string"):
        decorator()(f)


def test_nonpositive_dimension_rejected():
    with pytest.raises(ValueError):
        SystemFunction(lambda y, p: y, 0, 1)


# -- numerical evaluation ----------------------------------------------------


def test_evaluation_dtype_and_shape(henon_fun, henon_params):
    out = henon_fun([0.5, 0.1], henon_params)
    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert_allclose(out, [1 - 1.4 * 0.25 + 0.1, 0.3 * 0.5])


def test_integer_input_gives_float_output():
    @mapping(1)
    def half(y, _):
        return y / 2

    out = half(np.array([3]))
    assert out.dtype == np.float64
    assert_allclose(out, [1.5])


def test_scalar_input_accepted():
    @mapping(1, 2, param_keys=["s"])
    def lift(u, p):
        return u * np.array([0, 1]) + np.array([p["s"], 0])

    out = lift(-3.0, {"s": 30.0})
    assert out.shape == (2,)
    assert_allclose(out, [30.0, -3.0])
    assert lift.jacobian(-3.0, {"s": 30.0}).shape == (2, 1)


def test_dom_dim_differs_from_cod_dim():
    @mapping(2, 1)
    def norm2(y, _):
        return y[0] ** 2 + y[1] ** 2

    @mapping(1, 2)
    def circle(u, _):
        return np.array([sympy.cos(u[0]), sympy.sin(u[0])])

    y = np.array([1.0, 2.0])
    assert norm2(y).shape == (1,)
    assert_allclose(norm2(y), [5.0])
    assert norm2.jacobian(y).shape == (1, 2)
    assert_allclose(norm2.jacobian(y), [[2.0, 4.0]])
    assert norm2.hessian(y).shape == (1, 2, 2)
    assert_allclose(norm2.hessian(y), [[[2.0, 0.0], [0.0, 2.0]]])

    th = 0.3
    assert circle([th]).shape == (2,)
    assert_allclose(circle([th]), [np.cos(th), np.sin(th)])
    assert circle.jacobian([th]).shape == (2, 1)
    assert_allclose(circle.jacobian([th]), [[-np.sin(th)], [np.cos(th)]])
    assert circle.hessian([th]).shape == (2, 1, 1)
    assert_allclose(circle.hessian([th]).ravel(), [-np.cos(th), -np.sin(th)])


def test_henon_jacobian_and_hessian(henon_fun, henon_params):
    a, b = henon_params["a"], henon_params["b"]
    x, u = 0.37, -0.12
    jac = henon_fun.jacobian([x, u], henon_params)
    assert jac.shape == (2, 2)
    assert jac.dtype == np.float64
    assert_allclose(jac, [[-2 * a * x, 1.0], [b, 0.0]])

    hes = henon_fun.hessian([x, u], henon_params)
    expected = np.zeros((2, 2, 2))
    expected[0, 0, 0] = -2 * a
    assert hes.shape == (2, 2, 2)
    assert_allclose(hes, expected)


def test_symbolic_expressions(henon_fun):
    a, b = sympy.symbols("a b", real=True)
    y0, y1 = sympy.symbols("y_0 y_1", real=True)
    assert sympy.simplify(henon_fun.expression[0] - (1 - a * y0**2 + y1)) == 0
    assert sympy.simplify(henon_fun.expression[1] - b * y0) == 0
    assert henon_fun.jacobian_expression.shape == (2, 2)
    assert henon_fun.hessian_expression.shape == (2, 2, 2)


def test_constant_function():
    @vector_field(2)
    def const(y, _):
        return np.array([1, -2])

    assert_allclose(const([5.0, 7.0]), [1.0, -2.0])
    assert_allclose(const.jacobian([5.0, 7.0]), np.zeros((2, 2)))
    assert_allclose(const.hessian([5.0, 7.0]), np.zeros((2, 2, 2)))


def test_sympy_functions_are_supported():
    @vector_field(2, ["w"])
    def osc(y, p):
        return np.array([y[1], -sympy.sin(p["w"] * y[0])])

    y, w = np.array([0.4, 0.1]), 2.0
    assert_allclose(osc(y, {"w": w}), [0.1, -np.sin(w * 0.4)])
    assert_allclose(osc.jacobian(y, {"w": w}), [[0, 1], [-w * np.cos(w * 0.4), 0]])


def test_extra_parameters_are_ignored(henon_fun, henon_params):
    out = henon_fun([0.0, 0.0], {**henon_params, "unused": 3.0})
    assert_allclose(out, [1.0, 0.0])


# -- errors --------------------------------------------------------------------


def test_missing_parameter_raises_keyerror(henon_fun):
    with pytest.raises(KeyError, match="b"):
        henon_fun([0.0, 0.0], {"a": 1.4})


def test_params_none_when_needed_raises(henon_fun):
    with pytest.raises(ValueError):
        henon_fun([0.0, 0.0])


def test_wrong_number_of_components_raises_valueerror():
    @mapping(2, 3)
    def bad(y, _):
        return y

    with pytest.raises(ValueError, match="components"):
        bad([1.0, 2.0])


def test_numpy_cos_on_symbols_raises():
    @vector_field(2)
    def uses_numpy(y, _):
        return np.array([y[1], np.cos(y[0])])

    with pytest.raises(SymbolicEvaluationError):
        uses_numpy([0.0, 0.0])


def test_branching_on_state_raises():
    @mapping(1)
    def relu(y, _):
        return y if y[0] > 0 else 0 * y

    with pytest.raises(SymbolicEvaluationError):
        relu([1.0])


def test_two_dimensional_state_rejected(henon_fun, henon_params):
    with pytest.raises(ValueError):
        henon_fun(np.zeros((2, 1)), henon_params)


# -- sharing -----------------------------------------------------------------


def test_function_shared_by_two_modes(henon_fun, henon_params):
    m1 = DiscreteMode("m1", henon_fun)
    m2 = DiscreteMode("m2", henon_fun)
    assert m1.fun is m2.fun
    y0 = [0.1, 0.2]
    r1 = m1.step(y0, henon_params, hessian=True)
    r2 = m2.step(y0, henon_params, hessian=True)
    assert_allclose(r1.y, r2.y)
    assert_allclose(r1.jac, r2.jac)
    assert_allclose(r1.hes, r2.hes)
    # Composing through both modes equals applying the function twice.
    assert_allclose(m2.step(r1.y, henon_params).y, henon_fun(r1.y, henon_params))


def test_repr(henon_fun):
    assert "henon" in repr(henon_fun)
    assert "dom_dim=2" in repr(henon_fun)
