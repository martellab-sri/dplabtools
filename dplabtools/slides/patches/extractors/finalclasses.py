# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Final high level classes for patch extraction.

Classes:
    DiskPatchExtractor
    MultiResDiskPatchExtractor
    MemPatchExtractor
    MultiResMemPatchExtractor

"""

from dplabtools.slides.patches.extractors.mixins import (
    DiskPatchExtractorMixin,
    MultiResDiskPatchExtractorMixin,
    MemPatchExtractorMixin,
    MultiResMemPatchExtractorMixin,
)
from dplabtools.slides.patches.extractors.base import BasePatchExtractor, MultiResBasePatchExtractor


class DiskPatchExtractor(DiskPatchExtractorMixin, BasePatchExtractor):
    """Class for extracting patches to disk."""

    def __init__(self, **kwargs):
        """Init parameters are defined in two parent classes."""
        super().__init__(**kwargs)


class MultiResDiskPatchExtractor(MultiResDiskPatchExtractorMixin, MultiResBasePatchExtractor):
    """Class for extracting multi resolution patches to disk."""

    def __init__(self, **kwargs):
        """Init parameters are defined in two parent classes."""
        super().__init__(**kwargs)


class MemPatchExtractor(MemPatchExtractorMixin, BasePatchExtractor):
    """Class for extracting in-memory patches."""

    def __init__(self, **kwargs):
        """Init parameters are defined in two parent classes."""
        super().__init__(**kwargs)


class MultiResMemPatchExtractor(MultiResMemPatchExtractorMixin, MultiResBasePatchExtractor):
    """Class for extracting in-memory multi resolution patches."""

    def __init__(self, **kwargs):
        """Init parameters are defined in two parent classes."""
        super().__init__(**kwargs)
