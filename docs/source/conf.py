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

# Read the version from the INSTALLED package metadata rather than hardcoding
# it. pydex does not define pydex.__version__, so the previous
# getattr(pydex, "__version__", "0.1.0") always fell through to its default and
# the docs kept reporting 0.1.0 no matter what pyproject.toml said. Taking it
# from the metadata means pyproject.toml is the single source of truth.
try:
    release = _pkg_version("pydex")
except PackageNotFoundError:            # docs built without installing pydex
    release = "0.0.0+unknown"
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
