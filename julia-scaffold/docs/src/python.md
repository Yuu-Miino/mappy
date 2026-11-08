# Python package

`MappyDS.jl` follows the Python package `mappy-ds`
([documentation](https://yuu-miino.github.io/doc.mappy/latest/)). The table
tracks which Python release each Julia release matches.

| MappyDS.jl | mappy-ds | Features |
| --- | --- | --- |
| 0.1 | 0.1.0 | modes, Poincaré maps, Jacobian and Hessian, `find_cycle`, `continuation` / `trace_cycle` |
| 0.2 | 0.2.0 | parameter derivatives `jac_p`, pseudo-arclength continuation of periodic points |
| 0.3 | 0.3.0 | mixed derivatives `hes_p`, local bifurcations and their continuation, Makie viewer |

## API mapping

| Python (`mappy_ds`) | Julia (`MappyDS`) |
| --- | --- |
| `@vector_field(dim)` | `VectorField(f, dim)` |
| `@mapping(dom, cod)` | `Mapping(f, dom, cod)` |
| `@border(dim, direction, name)` | `Border(q, dim; direction, name)` |
| `Mode.step(...)` | `evolve(mode, y0, p; ...)` |
| `PoincareMap`, `Diffeomorphism`, `image`, `image_detail`, `trajectory` | same names |
| `mappy_ds.root.find_cycle`, `find_local_bf`, `trace_*` | same names |
| `mappy_ds.plot.plot2d`, `plot_trajectory` | same names, via the Makie extension |
| `params` (`dict`) | `NamedTuple`; `param_keys` are all its keys |
