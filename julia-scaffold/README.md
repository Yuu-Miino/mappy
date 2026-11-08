# MappyDS.jl

[![CI](https://github.com/Yuu-Miino/MappyDS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Yuu-Miino/MappyDS.jl/actions/workflows/CI.yml)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://yuu-miino.github.io/doc.mappy/julia/stable/)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://yuu-miino.github.io/doc.mappy/julia/dev/)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Poincaré maps, periodic points and bifurcations of **hybrid (smooth / non-smooth)
dynamical systems**. The Julia counterpart of the Python package
[`mappy-ds`](https://github.com/Yuu-Miino/mappy).

`MappyDS` implements the *mode concept*: a dynamical system is split into smooth
primitive pieces — **modes** — connected by transition rules. From the modes the
package automatically builds the Poincaré map of the system as a composite
diffeomorphism together with its **Jacobian matrix** and **Hessian tensor**, which
feed periodic-point search, local bifurcation analysis and parameter continuation.

> Y. Miino, "Mode concept to construct diffeomorphisms representing smooth/non-smooth
> dynamical systems", *Nonlinear Theory and Its Applications, IEICE*, 15(2), 284–298, 2024.
> [doi:10.1587/nolta.15.284](https://doi.org/10.1587/nolta.15.284)

> **Status:** under construction. The package is being ported from `mappy-ds`
> 0.3.0 following the plan in
> [`docs/JULIA_RELEASE_PLAN.md`](https://github.com/Yuu-Miino/mappy/blob/master/docs/JULIA_RELEASE_PLAN.md)
> of the Python repository; it is not registered yet.

## Installation

Once registered in the General registry:

```julia
pkg> add MappyDS
```

Add a Makie backend (`GLMakie` for the interactive viewer, `CairoMakie` for
static figures) to enable `plot2d` and `plot_trajectory`.

## Documentation

<https://yuu-miino.github.io/doc.mappy/julia/stable/> (the Python documentation is
at <https://yuu-miino.github.io/doc.mappy/latest/>).

## Development

```julia
pkg> activate .
pkg> instantiate
pkg> test
```

Formatting follows [Runic](https://github.com/fredrikekre/Runic.jl); CI checks it.

## License

MIT — see [LICENSE](LICENSE).
