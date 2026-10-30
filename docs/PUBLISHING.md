# Publishing `mappy-ds` to PyPI

This project publishes to PyPI with
[trusted publishing](https://docs.pypi.org/trusted-publishers/) (OpenID Connect).
No API tokens are stored anywhere: the `publish` job in
`.github/workflows/release.yml` requests a short-lived token from PyPI.

> **Why `mappy-ds`?** The name `mappy` on PyPI is already taken by an unrelated
> project (the Python binding of minimap2). The distribution is therefore
> named `mappy-ds` and the import name is `mappy_ds`.

The steps below are needed only once, before the first release.

## 1. Register a pending publisher on PyPI

Because `mappy-ds` does not exist on PyPI yet, create a *pending* publisher;
the project is created on the first successful upload.

1. Log in to <https://pypi.org> (enable 2FA if not already done).
2. Go to **Your account → Publishing**
   (<https://pypi.org/manage/account/publishing/>).
3. Under **Add a new pending publisher**, choose **GitHub** and fill in:

   | Field             | Value         |
   | ----------------- | ------------- |
   | PyPI Project Name | `mappy-ds`    |
   | Owner             | `Yuu-Miino`   |
   | Repository name   | `mappy`       |
   | Workflow name     | `release.yml` |
   | Environment name  | `pypi`        |

4. Click **Add**.

Note: a pending publisher does not reserve the name. Publish soon after
registering it.

## 2. Create the `pypi` environment on GitHub

1. In <https://github.com/Yuu-Miino/mappy>, open
   **Settings → Environments → New environment**, name it `pypi`.
2. Recommended protection rules:
   - **Deployment branches and tags** → *Selected branches and tags* →
     add a tag rule `v*`.
   - Optionally **Required reviewers** → yourself, so every upload needs a
     manual approval.

No secrets need to be added to this environment.

## 3. (Optional) Dry run on TestPyPI

To rehearse before the real upload:

1. Register a pending publisher on <https://test.pypi.org> exactly as in step 1,
   but with environment name `testpypi`.
2. Create a `testpypi` environment on GitHub (step 2).
3. Temporarily add a job to `release.yml` (or a separate workflow file — in
   that case register *that* file name as the workflow on TestPyPI):

   ```yaml
   publish-testpypi:
     needs: build
     runs-on: ubuntu-latest
     environment: testpypi
     permissions:
       id-token: write
     steps:
       - uses: actions/download-artifact@v4
         with:
           name: dist
           path: dist/
       - uses: pypa/gh-action-pypi-publish@release/v1
         with:
           repository-url: https://test.pypi.org/legacy/
   ```

4. Push a pre-release tag (e.g. `v0.1.0rc1`, with the matching `version` in
   `pyproject.toml`) and check
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mappy-ds`.
   Note that the docs workflow also runs on `v*` tags.

## 4. First release

Follow the release procedure in [`CONTRIBUTING.md`](../CONTRIBUTING.md):
bump the version, update `CHANGELOG.md`, push the tag `vX.Y.Z`. Once the
`publish` job succeeds, the pending publisher becomes a regular trusted
publisher of the new `mappy-ds` project.

## Troubleshooting

- **`invalid-publisher`** — owner, repository, workflow file name or
  environment name do not match what was registered on PyPI (they are
  case-sensitive to the workflow file name).
- **File already exists** — PyPI never accepts the same version twice; bump
  `version` and tag again.
- **Docs deployment fails** — `deploy_docs.yaml` pushes to
  `Yuu-Miino/doc.mappy` with a deploy key, stored as the `DOC_DEPLOY_KEY`
  repository secret of `mappy`. To (re)create it:

  ```sh
  ssh-keygen -t ed25519 -N "" -C "mappy docs deploy" -f doc_mappy_deploy_key
  ```

  Add `doc_mappy_deploy_key.pub` under **Settings → Deploy keys** of
  `doc.mappy` with *Allow write access*, add the private key
  `doc_mappy_deploy_key` as the `DOC_DEPLOY_KEY` secret of `mappy`, then
  delete both files. Deploy keys do not expire and are scoped to one
  repository, unlike a personal access token.
