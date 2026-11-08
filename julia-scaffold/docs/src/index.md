```@meta
CurrentModule = MappyDS
```

# MappyDS.jl

Poincaré maps, periodic points and bifurcations of **hybrid (smooth /
non-smooth) dynamical systems**: the Julia counterpart of the Python package
[mappy-ds](https://yuu-miino.github.io/doc.mappy/latest/).

`MappyDS` implements the *mode concept* [miino2024](@cite): a dynamical system
is split into smooth primitive pieces, **modes**, connected by transition
rules. From the modes the package builds the Poincaré map of the system as a
composite diffeomorphism together with its **Jacobian matrix** and **Hessian
tensor**, which feed periodic-point search, local bifurcation analysis and
parameter continuation.

!!! note "Status"
    The package is being ported from `mappy-ds` 0.3.0. See
    [Python package](@ref) for the feature matrix and the API mapping.

## Index

```@index
```

```@autodocs
Modules = [MappyDS]
```
