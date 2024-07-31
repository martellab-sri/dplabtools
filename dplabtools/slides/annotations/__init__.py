# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""All importable classes from slides.annotations namespace."""

from dplabtools.slides.annotations.readers.qupath import QuPathProjectReader
from dplabtools.slides.annotations.readers.sedeen import SedeenReader
from dplabtools.slides.annotations.readers.asap import AsapReader

__all__ = ["QuPathProjectReader", "SedeenReader", "AsapReader"]
