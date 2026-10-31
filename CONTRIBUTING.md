# Contributing to mappy-ds

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Yuu-Miino/mappy.git
cd mappy
uv sync            # installs the package plus the `dev` group (test, lint, docs)
uvx pre-commit install      # optional: run ruff on every commit
```

## Everyday commands

| Task              | Command                                                  |
| ----------------- | -------------------------------------------------------- |
| Run tests         | `uv run pytest`                                          |
| Tests + coverage  | `uv run pytest --cov`                                    |
| Lint              | `uv run ruff check .`                                    |
| Format            | `uv run ruff format .`                                   |
| Type check        | `uv run mypy`                                            |
| Build docs        | `uv run sphinx-build docs/source docs/build/html`        |
| Run all hooks     | `uvx pre-commit run --all-files`                         |

CI (`.github/workflows/ci.yml`) runs ruff, `ruff format --check`, mypy and the
test suite on Python 3.11–3.14 (and, as an experimental job, the 3.15 pre-release); please make sure these pass locally before
opening a pull request. Docs are built in CI with `-W` (warnings are errors).

## Branches

- `develop` — integration branch; pushes deploy the `develop` docs.
- `master` — released code.

## Release procedure

1. Bump `version` in `pyproject.toml`.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading and update the links at the bottom.
3. Commit, merge to `master`, then tag and push:

   ```sh
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. The tag triggers:
   - `release.yml` — builds the sdist/wheel, publishes to PyPI via trusted
     publishing and creates a GitHub release with the dist files attached;
   - `deploy_docs.yaml` — builds the docs into `vX.Y.Z/` in
     [`Yuu-Miino/doc.mappy`](https://github.com/Yuu-Miino/doc.mappy) and
     re-points `latest` to the newest version.

See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for the one-time PyPI setup.
