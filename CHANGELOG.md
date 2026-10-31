# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-31

Mixed derivatives and pseudo-arclength continuation of bifurcation curves.

### Added

- Python 3.14 is tested in CI and declared in the classifiers; the 3.15
  pre-release runs as an experimental job.
- Dependabot configuration for `uv.lock` and the GitHub Actions (weekly,
  grouped updates).
- Mixed second derivatives: `image_detail(..., parameter_hessian=True)`
  returns `MapResult.hes_p` (`∂²y_i/∂y0_j∂p_m`) from a third variational
  system in continuous modes, its border correction (including borders that
  depend on parameters) and the chain rule across modes;
  `SystemFunction.mixed_hessian()`; `Mode.step(..., mixed_hessian=True)`.
- `trace_local_bf_arclength`: pseudo-arclength continuation of a
  bifurcation curve in the `(y, λ_bf, θ, λ_cnt)` space with the fully
  analytic Jacobian of the bifurcation conditions; passes cusps and flags
  `fold`, strong resonances `resonance_1_1` … `resonance_1_4` on
  Neimark–Sacker curves and `second_multiplier_crossing`.

### Changed

- `find_local_bf` is fully analytic: the parameter derivative of `DTⁿ` comes
  from `hes_p` instead of a central difference.

## [0.2.0] - 2026-10-30

Parameter derivatives of the Poincaré map and pseudo-arclength continuation.

### Added

- Derivatives with respect to the parameters: `image_detail(...,
  parameter_jacobian=True)` returns `MapResult.jac_p` (columns ordered as
  `Map.param_keys`), computed from the SymPy parameter Jacobian of each
  function, the sensitivity equations of continuous modes with the border
  correction (including borders that depend on parameters) and the chain
  rule across modes. `Mode.step` takes `param_keys=` and returns
  `StepResult.jac_p`; `SystemFunction` gains `parameter_jacobian()`.
- Pseudo-arclength continuation: `mappy_ds.continuation.arclength_continuation`
  (generic predictor–corrector on a bordered system, adaptive step, fold
  passing, closed-curve detection) and `mappy_ds.root.trace_cycle_arclength`,
  which follows a branch of periodic points through folds using `jac` and
  `jac_p` for the tangent, and flags `fold`, `branch_point`,
  `period_doubling` and `neimark_sacker` events from sign changes between
  consecutive points.
- `find_local_bf` solves with an analytic Jacobian of the bifurcation system
  (Jacobian, Hessian and `jac_p` of the map; only the parameter derivative of
  `DTⁿ` uses a central difference).
- `examples/henon.py --mode bifurcation|mle` (one-parameter bifurcation
  diagram and maximum Lyapunov exponent by Benettin's method) and
  `examples/duffing.py --mode bifurcation`, reproducing Fig. 8 and Fig. 10
  of Miino (2024); ported from the `develop` branch.
- The release workflow checks that the pushed tag matches the version in
  `pyproject.toml`.

### Changed

- The docs are deployed to `doc.mappy` with a deploy key (`DOC_DEPLOY_KEY`)
  instead of a personal access token, and `Deploy documents` can be run
  manually for any tag or branch.

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

- Test suite (175 tests) including finite-difference validation of Jacobians
  and Hessians through borders and the reference values of the paper.
- GitHub Actions: lint/type-check/test matrix, docs deployment, release to
  PyPI through trusted publishing on `v*` tags.
- `max_transitions` safety bound on `PoincareMap`; borderless continuous
  modes act as fixed-time flow maps.
- Transition rules of a continuous mode can name its borders,
  `{"flow": {"q0": "after_q0", "q1": "after_q1"}}`, using `@border(...,
  name=...)` or the function name; this form does not depend on the order
  of `borders`.
- `PoincareMap` validates its graph at construction: rule arity against the
  borders, border names, dimension consistency of every edge, a rule for
  every reachable mode and reachability of an end mode.

### Removed

- `solve_ivbmp`, `solve_poincare_map`, `BasicResult`, the `mappy.typing`
  TypeVars and the empty `mappy.limit` subpackage.
- The import-time `numpy.set_printoptions` side effect.

[Unreleased]: https://github.com/Yuu-Miino/mappy/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Yuu-Miino/mappy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Yuu-Miino/mappy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Yuu-Miino/mappy/releases/tag/v0.1.0
