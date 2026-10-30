# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-30

A rewrite of the package. The mode concept and the numerics are those of
Miino, *NOLTA, IEICE* 15(2), 2024; the API is new and not compatible with the
unreleased `mappy` 0.0.x.

### Changed

- Renamed the distribution to `mappy-ds` and the package to `mappy_ds`; the
  PyPI name `mappy` belongs to the minimap2 bindings.
- Dropped Python 3.10; supports 3.11–3.13. NumPy 2 compatible.
- `src/` layout built with hatchling; `uv.lock` committed; ruff, mypy
  (`--strict`) and pytest configured in `pyproject.toml`.
- System functions are declared with `@vector_field(dim)`,
  `@mapping(dom_dim, cod_dim)` and `@border(dim, direction)`
  instead of `ContinuousMode.function` / `DiscreteMode.function` /
  `ContinuousMode.border`. They are traced once with SymPy and compiled to
  NumPy, so the function, its Jacobian and its Hessian are all fast; the same
  `SystemFunction` can be shared by several modes. The parameters a function
  reads (`p[key]`) are discovered during the trace, so there is no
  `param_keys` declaration.
- `ContinuousMode` and `DiscreteMode` expose one `step(y0, params, *,
  jacobian, hessian, trajectory)` returning a `StepResult` dataclass.
- `PoincareMap(modes, transitions, initial_mode, end_modes=None)` fixes the
  initial mode at construction; `Diffeomorphism(fun)` wraps a single discrete
  mode. Both derive from the `Map` ABC with `image`, `image_detail` and
  `trajectory`; `MapResult` carries `y`, `jac`, `hes`, `eigvals`, `eigvecs`,
  `mode_history` and `trajectory`.
- States are always float64 arrays of shape `(dim,)`; scalars and sequences
  are accepted as input.
- Derivative conventions: `jac[i, j] = ∂y_i/∂y0_j`,
  `hes[i, j, k] = ∂²y_i/(∂y0_j ∂y0_k)`; the variational equations, the border
  correction and the chain rule are written with `einsum`.
- `mappy_ds.root`: `find_cycle` uses the analytic Jacobian in
  `scipy.optimize.root`; `find_local_bf`, `trace_cycle` and `trace_local_bf`
  return dataclasses (`CycleResult`, `LocalBfResult`, `ContinuationStep`).
- `mappy_ds.continuation.continuation` sweeps the parameter with `linspace`
  and returns `ContinuationStep` objects.
- `mappy_ds.plot`: one `plot2d(diffeo, y0, params, config)` for every map
  (returns the `FuncAnimation`), `PlotConfig` / `LineStyle` dataclasses and a
  non-interactive `plot_trajectory`. matplotlib is the optional extra `plot`.
- Exceptions derive from `MappyError`: `UnknownModeError`, `TransitionError`,
  `ModeNotTerminatedError`, `SymbolicEvaluationError`.
- Examples moved to `examples/` with an `argparse` CLI; the LaTeX note moved
  to `docs/notes/`.
- Documentation rewritten for Sphinx 8 (tutorial, user guide, API reference).

### Added

- Test suite (164 tests) including finite-difference validation of Jacobians
  and Hessians through borders and the reference values of the paper.
- GitHub Actions: lint/type-check/test matrix, docs deployment, release to
  PyPI through trusted publishing on `v*` tags.
- `max_transitions` safety bound on `PoincareMap`; borderless continuous
  modes act as fixed-time flow maps.

### Removed

- `solve_ivbmp`, `solve_poincare_map`, `BasicResult`, the `mappy.typing`
  TypeVars and the empty `mappy.limit` subpackage.
- The import-time `numpy.set_printoptions` side effect.

[Unreleased]: https://github.com/Yuu-Miino/mappy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Yuu-Miino/mappy/releases/tag/v0.1.0
