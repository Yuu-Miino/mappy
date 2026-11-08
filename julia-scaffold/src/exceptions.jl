"""
    MappyError <: Exception

Base type of all errors raised deliberately by `MappyDS`, so callers can catch
the whole family at once (Python: `mappy_ds.MappyError`).
"""
abstract type MappyError <: Exception end

"""
    UnknownModeError(name)

A mode name does not appear in the mode set of a Poincaré map.
"""
struct UnknownModeError <: MappyError
    name::Symbol
end
Base.showerror(io::IO, e::UnknownModeError) = print(io, "mode ", repr(e.name), " is not defined")

"""
    TransitionError(mode; border=nothing, reason=nothing)

The transition rules of a Poincaré map are inconsistent (missing rule, wrong
number of entries, unknown border name, mismatched dimensions, no reachable
end mode), or a mode ended through a border that has no successor.
"""
struct TransitionError <: MappyError
    mode::Symbol
    border::Union{Nothing, Int, Symbol}
    reason::Union{Nothing, String}
end
TransitionError(mode::Symbol; border=nothing, reason=nothing) = TransitionError(mode, border, reason)
function Base.showerror(io::IO, e::TransitionError)
    if e.reason !== nothing
        print(io, "transition from mode ", repr(e.mode), ": ", e.reason)
    elseif e.border === nothing
        print(io, "no transition is defined from mode ", repr(e.mode))
    else
        print(io, "no transition is defined from mode ", repr(e.mode), " via border ", repr(e.border))
    end
end

"""
    ModeNotTerminatedError(mode, max_interval)

A continuous mode reached `max_interval` without hitting a border.
"""
struct ModeNotTerminatedError <: MappyError
    mode::Symbol
    max_interval::Float64
end
Base.showerror(io::IO, e::ModeNotTerminatedError) = print(io,
    "continuous mode ", repr(e.mode), " did not reach any border within max_interval=", e.max_interval)

"""
    DerivativeError(fun, msg)

A system function could not be differentiated with ForwardDiff (Python:
`SymbolicEvaluationError`). Typical causes are element types fixed to
`Float64` in the user function or operations that are not differentiable.
"""
struct DerivativeError <: MappyError
    fun::Symbol
    msg::String
end
Base.showerror(io::IO, e::DerivativeError) = print(io, "cannot differentiate ", repr(e.fun), ": ", e.msg)
