using MappyDS
using Test
using Aqua
using JET
using ExplicitImports
using ReTestItems

@testset "MappyDS.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MappyDS)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(MappyDS; target_defined_modules = true)
    end
    @testset "Explicit imports" begin
        @test check_no_implicit_imports(MappyDS) === nothing
        @test check_no_stale_explicit_imports(MappyDS) === nothing
    end
end

# Unit tests are `@testitem`s in test/*_test.jl (one file per Python test
# module: functions, modes, maps, continuation, root, package). The
# cross-check against the Python implementation (test/crosscheck_test.jl,
# PythonCall + mappy-ds) runs only when MAPPYDS_CROSSCHECK=1.
runtests(MappyDS)
