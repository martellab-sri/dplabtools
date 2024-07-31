# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for reading Sedeen annotation files."""

from dplabtools.slides.annotations.readers.base import BaseReader
from dplabtools.slides.annotations.parsers.sedeen import SedeenParser
from dplabtools.slides.annotations.validators.sedeen import SedeenValidator
from dplabtools.slides.annotations.mappers.sedeen import SedeenMapper


class SedeenReader(BaseReader):
    """Class for reading Sedeen annotation files."""

    _parser_class = SedeenParser
    _validator_class = SedeenValidator
    _mapper_class = SedeenMapper
