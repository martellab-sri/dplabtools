# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for reading ASAP annotation files."""

from dplabtools.slides.annotations.readers.base import BaseReader
from dplabtools.slides.annotations.parsers.asap import AsapParser
from dplabtools.slides.annotations.validators.asap import AsapValidator
from dplabtools.slides.annotations.mappers.asap import AsapMapper


class AsapReader(BaseReader):
    """Class for reading ASAP annotation files."""

    _parser_class = AsapParser
    _validator_class = AsapValidator
    _mapper_class = AsapMapper
