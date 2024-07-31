# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0
#
#
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.append(os.path.abspath("extensions"))
sys.path.append(os.path.abspath("../../"))

from sphinx.ext.autodoc import between


def setup(app):
    app.connect("autodoc-process-docstring", between("-!-", None, False, True))


project = "Digital Pathology Lab Tools"
copyright = "2024 Sunnybrook Research Institute"
author = "Sunnybrook Research Institute"
release = "0.0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autodoc_mock_imports = ["torch", "torchvision", "openslide"]
extensions = ["sphinx.ext.autodoc", "sphinx_multiversion", "sphinx.ext.napoleon", "autoparam"]
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["_static"]
html_css_files = ["custom.css", "colors.css"]
html_sidebars = {
    "**": ["globaltoc.html", "sourcelink.html", "searchbox.html"],
    "using/windows": ["windowssidebar.html", "searchbox.html"],
}

# Autoclass configuration
autoclass_content = "init"
autodoc_member_order = "groupwise"
