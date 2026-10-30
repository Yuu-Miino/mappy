# mappy-ds

[![CI](https://github.com/Yuu-Miino/mappy/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuu-Miino/mappy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mappy-ds.svg)](https://pypi.org/project/mappy-ds/)
[![Python](https://img.shields.io/pypi/pyversions/mappy-ds.svg)](https://pypi.org/project/mappy-ds/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Poincaré maps, periodic points and bifurcations of **hybrid (smooth / non-smooth) dynamical systems**.

`mappy_ds` implements the *mode concept*: a dynamical system is split into smooth
primitive pieces — **modes** — connected by transition rules. From the modes the
package automatically builds the Poincaré map of the system as a composite
diffeomorphism together with its **Jacobian matrix** and **Hessian tensor**, which
feed periodic-point search, local bifurcation analysis and parameter continuation.

> Y. Miino, "Mode concept to construct diffeomorphisms representing smooth/non-smooth
> dynamical systems", *Nonlinear Theory and Its Applications, IEICE*, 15(2), 284–298, 2024.
> [doi:10.1587/nolta.15.284](https://doi.org/10.1587/nolta.15.284)

## Installation

```bash
pip install mappy-ds
```

Add the `plot` extra for the interactive matplotlib viewer:

```bash
pip install "mappy-ds[plot]"
```

Requires Python 3.11 or newer.

## Quick start

The Izhikevich neuron model fires (jumps) when the membrane potential reaches
`v = 30`. Describe the ODE, the border, the jump and the Poincaré section as modes
and let the package build the map:

```python
import numpy as np
from mappy_ds import (
    ContinuousMode,
    DiscreteMode,
    PoincareMap,
    border,
    mapping,
    vector_field,
)


@vector_field(dim=2)
def izhikevich(y, p):
    v, u = y
    return np.array([0.04 * v**2 + 5 * v + 140 - u + p["I"], p["a"] * (p["b"] * v - u)])


@border(dim=2)
def fire(y, _):
    return y[0] - 30


@mapping(2)
def firing(y, p):
    return y + np.array([p["c"] - 30, p["d"]])


@mapping(2, 1)
def reduce(y, _):  # section (v = 30) -> u
    return np.array([0, 1]) @ y


@mapping(1, 2)
def revert(u, _):  # u -> section (v = 30)
    return u * np.array([0, 1]) + np.array([30, 0])


pmap = PoincareMap(
    [
        ContinuousMode("A", izhikevich, borders=[fire]),
        DiscreteMode("B", firing),
        DiscreteMode("P", reduce),
        DiscreteMode("Pinv", revert),
    ],
    {"A": ["P"], "B": "A", "P": "Pinv", "Pinv": "B"},
    initial_mode="Pinv",
)

params = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}
res = pmap.image_detail(-3.0, params, hessian=True)
print(res.y, res.jac, res.hes)  # [-3.47838] [[0.27080]] [[[-0.19326]]]
print(res.mode_history)  # ['Pinv', 'B', 'A', 'P', 'Pinv']
```

Find a fixed point and follow it while a parameter changes:

```python
from mappy_ds.root import find_cycle, trace_cycle

fp = find_cycle(pmap, -4.5, {**params, "I": 8.0})
print(fp.y, fp.eigvals)  # [-4.64542] [0.22196+0.j]

branch = trace_cycle(
    pmap, fp.y, {**params, "I": 8.0}, "I", end_value=4.5, resolution=50
)
```

Smooth systems are a special case: a discrete map is a single mode
(`Diffeomorphism`), and a periodically forced oscillator becomes autonomous by
adding time as a state and resetting it on a section — see [`examples/`](examples).

## Features

- **Modes**: `ContinuousMode` (ODE integrated with SciPy until a border is hit) and
  `DiscreteMode` (a map), freely composed through transition rules.
- **Derivatives for free**: system functions are traced once with SymPy; the
  Jacobian and Hessian of every mode map — including the correction for the
  state-dependent hitting time at a border — and of their composition are computed
  automatically.
- **`mappy_ds.root`**: periodic points (`find_cycle`), local bifurcation points —
  tangent, period-doubling and Neimark–Sacker — (`find_local_bf`), and their
  parameter continuation (`trace_cycle`, `trace_local_bf`).
- **`mappy_ds.plot`**: an interactive matplotlib viewer of trajectories and return
  maps with mouse / keyboard control of the initial state and parameters.

## Documentation

<https://yuu-miino.github.io/doc.mappy/latest/>

## Development

```bash
git clone https://github.com/Yuu-Miino/mappy.git
cd mappy
uv sync
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
