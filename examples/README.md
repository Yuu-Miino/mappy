# Examples

Example scripts for `mappy_ds`. The hybrid-system examples reproduce figures of
Y. Miino, "Mode concept to construct diffeomorphisms representing
smooth/non-smooth dynamical systems", NOLTA, IEICE 15(2), 2024.

| Script | What it shows | Modes |
| --- | --- | --- |
| `henon.py` | Hénon map as a `Diffeomorphism`; the strange attractor | `plot`, `orbit` |
| `logistic_3d.py` | Three coupled logistic maps: fixed-point continuation, period-doubling curve in (alpha, beta) | `plot`, `orbit`, `trace`, `bf` |
| `izhikevich.py` | Izhikevich neuron as a 1-D Poincaré map with a firing reset; period doubling in (a, I) | `plot`, `orbit`, `trace` |
| `duffing.py` | Forced Duffing oscillator, stroboscopic map; period doublings at B ≈ 0.171687 and 0.368655 | `plot`, `orbit`, `trace` |
| `forced_alpazur.py` | Forced Alpazur oscillator, piecewise-smooth with hysteresis; period doubling in (V, B1) | `plot`, `orbit`, `trace` |

Each script documents its modes in the module docstring and in `--help`.

## Running

From the repository root:

```sh
uv run python examples/duffing.py                 # interactive plot (default)
uv run python examples/duffing.py --mode trace    # continuation, prints results
uv run python examples/duffing.py --mode orbit --frames 100 --output duffing.png
```

- `plot` opens an interactive window (needs a display). Click to restart from
  a point, `p` selects a parameter, up/down change it, `f` toggles drawing the
  trajectories, space clears the axes. With a non-interactive backend such as
  `MPLBACKEND=Agg` it only builds the figure and exits.
- `orbit` iterates the map `--frames` times and saves a PNG to `--output`;
  it needs no display.
- `trace` / `bf` run periodic-point and bifurcation continuations and print
  the results (`logistic_3d.py` also plots them; pass `--output` to save the
  figure). Each finishes in a few seconds.
