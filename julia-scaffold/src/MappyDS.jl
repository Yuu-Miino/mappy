"""
    MappyDS

Poincaré maps, periodic points and bifurcations of hybrid (smooth / non-smooth)
dynamical systems: the Julia counterpart of the Python package
[`mappy-ds`](https://github.com/Yuu-Miino/mappy).

`MappyDS` implements the *mode concept* (Miino, NOLTA, IEICE 15(2), 2024): a
dynamical system is split into smooth primitive pieces, *modes*, connected by
transition rules. From the modes the package builds the Poincaré map of the
system as a composite diffeomorphism together with its Jacobian matrix and
Hessian tensor, which feed periodic-point search, local bifurcation analysis
and parameter continuation.

The implementation plan and the API mapping from the Python package are in
`docs/JULIA_RELEASE_PLAN.md` of the `mappy` repository.
"""
module MappyDS

using LinearAlgebra
using Printf
using StaticArrays
using ForwardDiff
using SciMLBase
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqVerner: Vern7, Vern9
using SimpleNonlinearSolve
using PrecompileTools: @setup_workload, @compile_workload

include("exceptions.jl")
include("plot_api.jl")

# Milestone 0.1.0 (see the release plan, §5):
# include("functions.jl")      # VectorField / Mapping / Border, ForwardDiff derivatives
# include("trajectory.jl")     # Trajectory / TrajectorySegment
# include("modes.jl")          # ContinuousMode / DiscreteMode / StepResult, evolve
# include("maps.jl")           # PoincareMap / Diffeomorphism / MapResult
# include("continuation.jl")   # continuation (0.1.0), arclength_continuation (0.2.0)
# include("root/cycle.jl")     # find_cycle, trace_cycle (0.1.0), trace_cycle_arclength (0.2.0)
# include("root/local_bf.jl")  # find_local_bf, trace_local_bf(_arclength) (0.3.0)

export MappyError, UnknownModeError, TransitionError, ModeNotTerminatedError,
    DerivativeError
export plot2d, plot_trajectory

# @setup_workload begin
#     # Izhikevich example of the Python docstring; filled in at 0.1.0.
#     @compile_workload begin
#     end
# end

end # module
