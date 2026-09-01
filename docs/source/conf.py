# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# docs/source/conf.py -> repo root is two levels up.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information ------------------------------------------------------

project = "pydex"
copyright = "2026, Kennedy Putra Kusumo, Salvador Garcia-Munoz, and contributors"
author = "Kennedy Putra Kusumo, Salvador Garcia-Munoz, and contributors"

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Read the version from pyproject.toml -- the file that actually declares it --
# rather than from the INSTALLED package metadata.
#
# Metadata was the previous source, and it is a trap with an editable install:
# `pip install -e .` writes the version ONCE, at install time, and nothing
# refreshes it when pyproject.toml is bumped. The code stays live (the .pth
# MetaPathFinder resolves imports to the repo) but the version does not, so
# bumping the version and rebuilding docs embedded the OLD number. That is how
# the published site said "pydex 0.2.1 documentation" across 0.3.0, 0.4.0 and
# 0.4.1. The documented workaround was to remember `pip install -e . --no-deps`
# before every docs build; reading the declaring file makes the mistake
# unrepresentable instead of relying on a checklist step (PROJECT_NOTES Open
# Item 22).
#
# Installed metadata remains the fallback for the case pyproject.toml is
# genuinely absent, e.g. a docs build from an unpacked sdist.
def _read_version():
    pyproject = os.path.join(os.path.dirname(__file__), "..", "..",
                             "pyproject.toml")
    pyproject = os.path.abspath(pyproject)
    if os.path.isfile(pyproject):
        try:
            import tomllib                       # py3.11+
        except ModuleNotFoundError:
            tomllib = None
        if tomllib is not None:
            with open(pyproject, "rb") as fh:
                found = tomllib.load(fh).get("project", {}).get("version")
            if found:
                return found
        else:                                    # py3.9/3.10: no tomllib
            import re
            with open(pyproject, "r", encoding="utf-8") as fh:
                src = fh.read()
            # first `version = "..."` inside the [project] table
            block = re.split(r"^\[", src, flags=re.M)
            for chunk in block:
                if chunk.startswith("project]"):
                    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', chunk,
                                  flags=re.M)
                    if m:
                        return m.group(1)
    try:
        return _pkg_version("pydex")
    except PackageNotFoundError:        # docs built without installing pydex
        return "0.0.0+unknown"


release = _read_version()
version = release

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

# autodoc pulls live objects from the installed package, so pydex (and its
# hard dependencies: numpy, scipy, pandas, matplotlib, numdifftools, pyomo,
# joblib, dill) must be importable in the environment sphinx-build runs in.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,      # keep the 49 undocumented private helpers
                                  # out of the rendered API page by default
    "private-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# -- Napoleon (Google + NumPy docstring support) ------------------------------
# designer.py is documented Google-style throughout, except for 14
# pre-existing NumPy-style docstrings. napoleon_numpy_docstring=True (the
# default) lets both coexist without converting the 14 by hand.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
# Composite type strings such as "array-like, shape (n_mp, n_mp)" are tokenised
# into adjacent <em> fragments either way. Checked in the built HTML: the
# fragments render contiguously and read correctly, so this is cosmetic only.
napoleon_preprocess_types = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Property setters: Sphinx documents these from the getter, so the 4 "missing"
# setter docstrings the HANDOFF flags are not a gap — this suppresses the
# otherwise-noisy "duplicate object description" warning some Sphinx versions
# emit for property setter/getter pairs.
autodoc_inherit_docstrings = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
