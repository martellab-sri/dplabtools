# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Mixin classes for patch location computing.

The purpose of those classes is to provide sampling methods for patches calculated in specified regions.
"""

from random import uniform

from dplabtools.common import roundfl
from dplabtools.slides.utils.mask import get_mask_pixel_step
from dplabtools.slides.patches.locations.poisson2d import PoissonDisk2DPoints
from dplabtools.slides.patches.locations.utils import expand_scalar_param


class RandomMixin:
    """Class providing random sampling for patch location computing."""

    # Stop searching for new patch locations after this number of failed attempts
    _break_factor = 100

    def __init__(self, *, num_patches, **kwargs):
        """Init random sampling.

        Parameters
        ----------
        num_patches : int
            Number of patch locations to calculate.
        """
        self._mixin_params = locals()
        self._num_patches = num_patches
        super().__init__(**kwargs)

    def _expand_params(self):
        self._num_patches = expand_scalar_param(self._num_patches, "num_patches", self._polygon_count)

    def _get_polygon_patches(self, shapely_polygon, counter):
        patches = []
        num_patches = self._num_patches[counter]
        patch_counter = 0
        run_counter = 0
        cumulative_break_factor = self._break_factor * num_patches
        x_min, y_min, x_max, y_max = self._get_shapely_bbox_points(shapely_polygon)
        while patch_counter < num_patches:
            run_counter += 1
            x1 = uniform(x_min, x_max)
            y1 = uniform(y_min, y_max)
            self._add_valid_patch(patches, (x1, y1), shapely_polygon, counter, (x_min, y_min, x_max, y_max))
            patch_counter = len(patches)
            if run_counter > cumulative_break_factor:
                raise ValueError("Could not find %d random patches for polygon: %s" % (num_patches, shapely_polygon))
        return patches


class GridMixin:
    """Class providing grid based sampling for patch location computing."""

    # Maximum allowed stride value
    _max_stride = 5

    def __init__(self, *, patch_stride=1, **kwargs):
        """Init grid sampling.

        Parameters
        ----------
        patch_stride : int or float, default=1
            Measure of overlapping between grid patches.
        """
        self._mixin_params = locals()
        self._patch_stride = patch_stride
        super().__init__(**kwargs)

    def _expand_params(self):
        self._patch_stride = expand_scalar_param(self._patch_stride, "patch_stride", self._polygon_count)

    def _get_polygon_patches(self, shapely_polygon, counter):
        patches = []
        patch_stride = self._patch_stride[counter]
        x_min, y_min, x_max, y_max = self._get_shapely_bbox_points(shapely_polygon)
        mask_pixel_step = get_mask_pixel_step(
            self._patch_size, patch_stride, self._relative_resample_factor, max_stride=self._max_stride
        )
        y1 = y_min
        step_y = 0
        while roundfl(y1) < y_max:
            y1 = y_min + step_y * mask_pixel_step
            step_y += 1
            x1 = x_min
            step_x = 0
            while roundfl(x1) < x_max:
                x1 = x_min + step_x * mask_pixel_step
                step_x += 1
                self._add_valid_patch(patches, (x1, y1), shapely_polygon, counter, (x_min, y_min, x_max, y_max))
        return patches


class PoissonDiskMixin:
    """Class providing Poisson disk based sampling for patch location computing."""

    def __init__(self, *, poisson_spacing=50, **kwargs):
        """Init Poisson disk sampling.

        Parameters
        ----------
        poisson_spacing : int, default=50
            Spacing between calculated patches, the optimal value should be determined experimentally.
        """
        self._mixin_params = locals()
        self._poisson_spacing = poisson_spacing
        super().__init__(**kwargs)

    def _expand_params(self):
        self._poisson_spacing = expand_scalar_param(self._poisson_spacing, "poisson_spacing", self._polygon_count)

    def _get_polygon_patches(self, shapely_polygon, counter):
        patches = []
        poisson_spacing = self._poisson_spacing[counter]
        x_min, y_min, x_max, y_max = self._get_shapely_bbox_points(shapely_polygon)
        width = x_max - x_min
        height = y_max - y_min
        poisson_points = PoissonDisk2DPoints(width=width, height=height, radius=poisson_spacing)
        for ppoint in poisson_points:
            x1 = ppoint[0] + x_min
            y1 = ppoint[1] + y_min
            self._add_valid_patch(patches, (x1, y1), shapely_polygon, counter, (x_min, y_min, x_max, y_max))
        return patches


class CustomPointsMixin:
    """Class accepting user provided points for patch location computing."""

    def __init__(self, *, points, **kwargs):
        """Init user provided points.

        Parameters
        ----------
        points : list of tuples
            List of (x,y) `int` values representing points calculated using WSI level 0 coordinates. Points should
            represent top-left corner of patches.
        """
        self._mixin_params = locals()
        self._points = points
        super().__init__(**kwargs)

    def _expand_params(self):
        pass

    def _get_polygon_patches(self, shapely_polygon, counter):
        patches = []
        x_min, y_min, x_max, y_max = self._get_shapely_bbox_points(shapely_polygon)
        for point in self._points:
            x1 = point[0] / self._mask_downsample_factor
            y1 = point[1] / self._mask_downsample_factor
            self._add_valid_patch(patches, (x1, y1), shapely_polygon, counter, (x_min, y_min, x_max, y_max))
        return patches
