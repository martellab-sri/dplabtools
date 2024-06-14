# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Mask created based on user-provided polygons, typically WSI annotations."""

import numpy as np
from skimage.draw import polygon as draw_polygon

from dplabtools.slides.processing.mask.base import BaseMask


class WSIPolygonMask(BaseMask):
    """Class for creating WSI masks based on predefined polygons."""

    def __init__(self, *, polygons, **kwargs):
        """Create a WSIPolygonMask object.

        Parameters
        ----------
        polygons : list of AnnotationPolygon objects
            Polygons representing desired image foreground.
        """
        self._polygons = polygons
        super().__init__(**kwargs)

    def _process_wsi_file(self, wsi_file):
        mask_size = self._wsi_slide.level_dimensions[self._level]
        self._mask_array = self._create_mask(mask_size, self._polygons)
        self._mask_array_transposed = np.transpose(self._mask_array)

    @staticmethod
    def _create_mask(mask_size, polygons):
        mask = np.zeros(mask_size, dtype=bool)
        mask_width, mask_height = mask_size
        for poly in polygons:
            poly_points = np.array(poly.points, dtype=int)
            row_coords = poly_points[:, 0]
            col_coords = poly_points[:, 1]
            if max(row_coords) > mask_width or max(col_coords) > mask_height:
                raise ValueError("Polygon coordinates exceed mask size %s" % str(mask_size))
            rows, cols = draw_polygon(row_coords, col_coords, mask.shape)
            mask[rows, cols] = True
        return mask
