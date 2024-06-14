# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for parsing annotation files.

Parser class should read all useful information from the source file and filter out non-polygon annotations.
Returned data is a list of dictionaries, each dictionary representing one annotation.
"""

from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Base class for annotation parsers."""

    def __init__(self, *, raw_data_file):
        self._raw_data_file = raw_data_file
        self._annotations = []
        self._parse()

    @abstractmethod
    def _parse(self):
        pass

    @property
    def annotations(self):
        """Return parsed annotations as list."""
        return self._annotations
