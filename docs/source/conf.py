# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# The package is expected to be installed in the build environment
# (``uv sync --group docs``); no ``sys.path`` manipulation is needed.

import inspect
import os
from functools import cached_property
from importlib.metadata import version as _dist_version

# -- Project information -----------------------------------------------------

project = "mappy-ds"
copyright = "2023–2026, Yuu Miino"  # noqa: RUF001
author = "Yuu Miino"

version = _dist_version("mappy-ds")
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- autodoc / autosummary / numpydoc ---------------------------------------

autosummary_generate = True
autodoc_typehints = "none"  # types are documented in the numpydoc sections

numpydoc_show_class_members = False  # members are listed by the class template
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False


def _public_members() -> dict[str, dict[str, list[str]]]:
    """Collect the documented methods and properties of each public class.

    The autosummary class template (``_templates/autosummary/class.rst``) uses
    this table to list members defined in ``mappy_ds`` only, so that methods
    inherited from builtins (``Exception.add_note``, ...) and dataclass fields
    (already described in the numpydoc "Attributes" section) are not repeated.
    """
    import mappy_ds
    import mappy_ds.continuation
    import mappy_ds.plot
    import mappy_ds.root

    table: dict[str, dict[str, list[str]]] = {}
    for module in (mappy_ds, mappy_ds.continuation, mappy_ds.root, mappy_ds.plot):
        for name in module.__all__:
            cls = getattr(module, name)
            if not inspect.isclass(cls):
                continue
            methods: list[str] = []
            properties: list[str] = []
            for attr in sorted(dir(cls)):
                if attr.startswith("_") and attr != "__call__":
                    continue
                for klass in cls.__mro__:
                    if attr in vars(klass):
                        raw = vars(klass)[attr]
                        owner = klass
                        break
                else:  # pragma: no cover
                    continue
                if not owner.__module__.startswith("mappy_ds"):
                    continue
                if isinstance(raw, (property, cached_property)):
                    properties.append(attr)
                elif inspect.isfunction(raw):
                    methods.append(attr)
            table[f"{module.__name__}.{name}"] = {
                "methods": methods,
                "properties": properties,
            }
    return table


autosummary_context = {"mappy_members": _public_members()}

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
# Offline builds (no network access) cannot download the inventories, which
# is a warning and thus an error under ``-W``. Set MAPPY_DOCS_OFFLINE=1 to
# skip intersphinx in that case; cross-references then render as plain text.
if os.environ.get("MAPPY_DOCS_OFFLINE"):
    extensions.remove("sphinx.ext.intersphinx")

# -- matplotlib plot directive -----------------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 110)]
plot_rcparams = {"figure.figsize": (6, 4), "savefig.bbox": "tight"}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"mappy-ds {version}"
html_static_path = ["_static"]
html_css_files = ["styles/MapPy.css"]

html_theme_options = {
    "logo": {
        "text": "mappy-ds",
        "image_light": "https://miino.sytes.net/figs/089-traj.gif",
        "image_dark": "https://miino.sytes.net/figs/089-traj.gif",
    },
    "github_url": "https://github.com/Yuu-Miino/mappy",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mappy-ds/",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
    ],
    "switcher": {
        "version_match": version,
        "json_url": "https://yuu-miino.github.io/doc.mappy/latest/_static/switcher.json",
    },
    # The switcher JSON lives on the published site; do not fail the build
    # when it cannot be fetched (e.g. offline or before the first deploy).
    "check_switcher": False,
    "navbar_start": ["navbar-logo", "version-switcher"],
}

html_sidebars = {
    "contents/**": ["search-field.html", "sidebar-nav-bs.html"],
}

# -- MathJax -----------------------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "bm": [r"\boldsymbol{#1}", 1],
            "deriv": [r"\frac{d#1}{d#2}", 2],
            "pderiv": [r"\frac{\partial#1}{\partial#2}", 2],
            "parens": [r"\left(#1\right)", 1],
            "braces": [r"\left\{#1\right\}", 1],
            "brackets": [r"\left[#1\right]", 1],
            "set": [r"\braces{#1}", 1],
            "case": [r"\left\{#1\right.", 1],
            "R": r"\mathbb{R}",
        }
    }
}
