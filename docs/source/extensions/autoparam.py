# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Autoparam extension for Sphinx.

This extenstion allows to use the following directive in RST files:

.. autoparam::
   :params: wsi_file, level_or_minsize
   :paths: dplabtools.slides.processing.mask.base.BaseMask, dplabtools.slides.processing.mask.base.BaseMask

which will be rendered as a single list of class parameters and their docstrings, which can be pulled out
from multiple classes:

    Parameters:
    - wsi_file (str) - WSI file name or path.
    - level_or_minsize (int) - Description here.

Additionally, to hide the class signature the following CSS rules must be present:

    dt.sig.sig-object.py{
        display: none;
    }

    dt[id^="dplabtools"].sig.sig-object.py{
        display: inline;
    }

History:

ver. 0.3:
- add supports for multi-line strings
- dedicated AutoParamCore class for easier unit testing

"""

from docutils.statemachine import StringList
from docutils.parsers.rst import Directive
from docutils import nodes
from sphinx.ext.autodoc import SUPPRESS

from autoparamcore import AutoParamCore


def setup(app):
    app.add_directive("autoparam", AutoParamDocumenter)

    return {
        "version": "0.3",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


def get_arg(arg):
    if arg in (None, True):
        return SUPPRESS
    return arg


class AutoParamDocumenter(Directive):
    option_spec = {"params": get_arg, "paths": get_arg}

    def run(self):
        params = []
        paths = []
        if "params" in self.options:
            params = [x.strip() for x in self.options.get("params").split(",") if x.strip()]
        if "paths" in self.options:
            paths = [x.strip() for x in self.options.get("paths").split(",") if x.strip()]

        autoparam = AutoParamCore(params, paths)
        docstring = autoparam.docstring
        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(StringList(docstring.split("\n")), 0, node)
        return node.children
