# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024-2026 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Package version management."""

import pathlib
import importlib.metadata

import tomli

pyproject_path = pathlib.Path(__file__).parent / "pyproject.toml"

if (pyproject_path).exists():
    with open(pyproject_path, "r") as pf:
        __version__ = tomli.load(pf)["project"]["version"]
else:
    __version__ = importlib.metadata.version("dplabtools")
