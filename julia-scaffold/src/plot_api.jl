# Plotting entry points. The methods live in the package extension
# `MappyDSMakieExt` (ext/MappyDSMakieExt.jl) and become available after
# `using Makie` (GLMakie for the interactive viewer, CairoMakie for static
# figures). Only the function stubs and the configuration types are defined
# here so that the core package does not depend on Makie.

"""
    plot2d(map, y0, p, config::PlotConfig)

Interactive two-dimensional phase portrait of a map (Python:
`mappy_ds.plot.plot2d`). Requires a Makie backend to be loaded.
"""
function plot2d end

"""
    plot_trajectory(map, y0, p; kwargs...)

Non-interactive plot of a trajectory (Python: `mappy_ds.plot.plot_trajectory`).
Requires a Makie backend to be loaded.
"""
function plot_trajectory end
