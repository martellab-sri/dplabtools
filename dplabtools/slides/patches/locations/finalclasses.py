# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Final high level classes for patch location computing.

Classes:
    WholeImageRandomPatches
    WholeImagePoissonDiskPatches
    WholeImageGridPatches
    WholeImageCustomPatches
    WholeImageInvertedRandomPatches
    WholeImageInvertedPoissonDiskPatches
    WholeImageInvertedGridPatches
    WholeImageInvertedCustomPatches
    PolygonRegionRandomPatches
    PolygonRegionPoissonDiskPatches
    PolygonRegionGridPatches
    PolygonRegionCustomPatches
"""

from dplabtools.slides.patches.locations.regions import (
    WholeImagePatches,
    WholeImageInvertedPatches,
    PolygonRegionPatches,
)
from dplabtools.slides.patches.locations.mixins import RandomMixin, PoissonDiskMixin, GridMixin, CustomPointsMixin


class WholeImageRandomPatches(RandomMixin, WholeImagePatches):
    """Class for calculating patch locations on whole images using random sampling."""

    pass


class WholeImagePoissonDiskPatches(PoissonDiskMixin, WholeImagePatches):
    """Class for calculating patch locations on whole images using Poisson disk sampling."""

    pass


class WholeImageGridPatches(GridMixin, WholeImagePatches):
    """Class for calculating patch locations on whole images using grid based sampling."""

    pass


class WholeImageCustomPatches(CustomPointsMixin, WholeImagePatches):
    """Class for calculating patch locations on whole images using custom points."""

    pass


class WholeImageInvertedRandomPatches(RandomMixin, WholeImageInvertedPatches):
    """Class for calculating patch locations on whole images with excluded polygon regions using random sampling."""

    pass


class WholeImageInvertedPoissonDiskPatches(PoissonDiskMixin, WholeImageInvertedPatches):
    """Class for calculating patch locations on whole images with excluded polygon regions using Poisson disk sampling."""

    pass


class WholeImageInvertedGridPatches(GridMixin, WholeImageInvertedPatches):
    """Class for calculating patch locations on whole images with excluded polygon regions using grid based sampling."""

    pass


class WholeImageInvertedCustomPatches(CustomPointsMixin, WholeImageInvertedPatches):
    """Class for calculating patch locations on whole images with excluded polygon regions using custom points."""

    pass


class PolygonRegionRandomPatches(RandomMixin, PolygonRegionPatches):
    """Class for calculating patch locations from polygon regions using random sampling."""

    pass


class PolygonRegionPoissonDiskPatches(PoissonDiskMixin, PolygonRegionPatches):
    """Class for calculating patch locations from polygon regions using Poisson disk sampling."""

    pass


class PolygonRegionGridPatches(GridMixin, PolygonRegionPatches):
    """Class for calculating patch locations from polygon regions using grid based sampling."""

    pass


class PolygonRegionCustomPatches(CustomPointsMixin, PolygonRegionPatches):
    """Class for calculating patch locations from polygon regions using custom points."""

    pass
