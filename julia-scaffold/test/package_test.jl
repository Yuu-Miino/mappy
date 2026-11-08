@testitem "exception hierarchy" begin
    for T in (UnknownModeError, TransitionError, ModeNotTerminatedError, DerivativeError)
        @test T <: MappyError
    end
    @test sprint(showerror, UnknownModeError(:A)) == "mode :A is not defined"
    @test occursin("max_interval=20.0", sprint(showerror, ModeNotTerminatedError(:A, 20.0)))
end
