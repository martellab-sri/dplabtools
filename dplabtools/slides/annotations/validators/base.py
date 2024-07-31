# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for validating annotations returned by parsers.

Validator class should check annotation data for common problems found in annotation files.
"""

from abc import ABC, abstractmethod


class BaseValidator(ABC):
    """Base class for annotation validators."""

    def __init__(self, *, annotations):
        self._annotations = annotations
        self._validate()

    @abstractmethod
    def _validate(self):
        pass
