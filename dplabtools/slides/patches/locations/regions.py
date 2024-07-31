# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Region classes for patch location computing.

The purpose of those classes is to provide areas of interest for which patches will be computed.
For whole images the entire slide is that area (as defined by mask), polygon based regions limit patch computing to
specific areas.
"""

from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

from dplabtools.slides.utils.mask import get_mask_bounding_box
from dplabtools.slides.patches.locations.base import BasePatches
from dplabtools.slides.utils import AnnotationPolygon, PolygonData


class WholeImagePatches(BasePatches):
    """Class providing whole image region as area of interest for patch location computing."""

    _full_overlap_check = False
    _check_polygons = False
    _filter_polygons = True
    _included_labels = []
    _excluded_labels = []

    def __init__(self, *, weak_label="", **kwargs):
        """Init whole image based region class.

        Parameters
        ----------
        weak_label : str, optional
            Label assigned to all calculated patches.
        """
        self._region_params = locals()
        self._weak_label = weak_label
        super().__init__(**kwargs)

    def _get_mask_polygons(self):
        mask_polygons = []
        mb = get_mask_bounding_box(self._mask_array)
        # see example at the end why adding 1 is necessary for creating polygons based on numpy generated bounding boxes
        mb = (mb[0], mb[1], mb[2] + 1, mb[3] + 1)
        polygon_points = [(mb[0], mb[1]), (mb[0], mb[3]), (mb[2], mb[3]), (mb[2], mb[1])]
        mask_polygon = AnnotationPolygon(points=polygon_points, label=self._weak_label)
        mask_polygons.append(mask_polygon)
        return mask_polygons


class WholeImageInvertedPatches(BasePatches):
    """Class providing inverted whole image region as area of interest for patch location computing."""

    _full_overlap_check = True
    _check_polygons = False
    _filter_polygons = False  # filtering will be applied here, not in the base class

    def __init__(self, *, polygon_data, weak_label="", included_labels=[], excluded_labels=[], **kwargs):
        """Init inverted whole image region based class.

        Parameters
        ----------
        polygon_data : list of AnnotationPolygon objects or JSON file/string with serialized AnnotationPolygon objects
            Polygons representing excluded regions.

        weak_label : str, optional
            Label assigned to all calculated patches.

        included_labels : list of str, optional
            Polygon labels included in calculations, all other labels will be ignored.

        excluded_labels : list of str, optional
            Polygon labels excluded from calculations.
        """
        self._region_params = locals()
        self._init_polygons = PolygonData(polygon_data=polygon_data).polygons
        self._weak_label = weak_label
        self._included_labels = included_labels
        self._excluded_labels = excluded_labels
        super().__init__(**kwargs)

    @staticmethod
    def _merge_annotation_polygons(polygons, wsi_name):
        merged_polygons = []
        if polygons:
            # polygon buffers are not used here, so default them to zero
            buffer_list = [0] * len(polygons)
            shapely_polygons = BasePatches._get_shapely_polygons(polygons, buffer_list, wsi_name)
            merged_shapely_polygons = unary_union(shapely_polygons)
            if isinstance(merged_shapely_polygons, Polygon):
                annotation_polygon = AnnotationPolygon(points=list(merged_shapely_polygons.exterior.coords), label="")
                merged_polygons.append(annotation_polygon)
            elif isinstance(merged_shapely_polygons, MultiPolygon):
                for geom in merged_shapely_polygons.geoms:
                    annotation_polygon = AnnotationPolygon(points=list(geom.exterior.coords), label="")
                    merged_polygons.append(annotation_polygon)
            else:
                raise TypeError(
                    "%s: Incorrect type after merging polygons [%s]" % (wsi_name, type(merged_shapely_polygons))
                )
        return merged_polygons

    def _get_mask_polygons(self):
        """Get mask level polygons.

        - Polygons are used as holes in this class
        - Since polygons may be overlapping and contained in other polygons, they must be merged before using
          them as holes
        - Filtering (excluding polygons) must be done before merging them
        - Finally one AnnotationPolygon object with holes is created
        """
        mask_polygons = []
        mdf = self._mask_downsample_factor
        mb = get_mask_bounding_box(self._mask_array)
        mb = (mb[0], mb[1], mb[2] + 1, mb[3] + 1)
        polygon_points = [(mb[0], mb[1]), (mb[0], mb[3]), (mb[2], mb[3]), (mb[2], mb[1])]
        holes = []

        filtered_init_polygons = []
        for polygon in self._init_polygons:
            if self._included_labels:
                if polygon.label not in self._included_labels:
                    continue
            elif self._excluded_labels:
                if polygon.label in self._excluded_labels:
                    continue
            filtered_init_polygons.append(polygon)

        merged_filtered_polygons = self._merge_annotation_polygons(filtered_init_polygons, self._wsi_name)
        for polygon in merged_filtered_polygons:
            points = []
            for x, y in polygon.points:
                points.append((x / mdf, y / mdf))
            holes.append(points)

        mask_polygon = AnnotationPolygon(points=polygon_points, label=self._weak_label, holes=holes)
        mask_polygons.append(mask_polygon)
        return mask_polygons


class PolygonRegionPatches(BasePatches):
    """Class providing specific regions as areas of interest for patch location computing."""

    _full_overlap_check = True
    _filter_polygons = True
    _polygons_overlap_threshold = 1

    def __init__(self, *, polygon_data, check_polygons=True, included_labels=[], excluded_labels=[], **kwargs):
        """Init polygon based region class.

        Parameters
        ----------
        polygon_data : list of AnnotationPolygon objects or JSON file/string with serialized AnnotationPolygon objects
            Polygons representing regions of interest.

        check_polygons : bool, default=True
            Enable checking for overlapping polygons and polygons contained in other polygons.

        included_labels : list of str, optional
            Polygon labels included in calculations, all other labels will be ignored.

        excluded_labels : list of str, optional
            Polygon labels excluded from calculations.
        """
        self._region_params = locals()
        self._init_polygons = PolygonData(polygon_data=polygon_data).polygons
        self._check_polygons = check_polygons
        self._included_labels = included_labels
        self._excluded_labels = excluded_labels
        super().__init__(**kwargs)

    def _get_mask_polygons(self):
        # mapping polygon points and holes to mask level
        mask_polygons = []
        mdf = self._mask_downsample_factor
        for polygon in self._init_polygons:
            points = []
            holes = []
            for x, y in polygon.points:
                points.append((x / mdf, y / mdf))
            for hole in polygon.holes:
                mask_hole = [(x / mdf, y / mdf) for x, y in hole]
                holes.append(mask_hole)
            mask_polygon = AnnotationPolygon(points=points, label=polygon.label, holes=holes)
            mask_polygons.append(mask_polygon)
        return mask_polygons

    @classmethod
    def set_polygons_overlap_threshold(cls, polygons_overlap_threshold):
        """Set threshold value for reporting overlapping polygons as log messages.

        Overlapping areas above the threshold value will be logged, area unit is one patch.
        """
        cls._polygons_overlap_threshold = polygons_overlap_threshold

    @property
    def polygons_overlap_threshold(self):
        """Return `polygons_overlap_threshold` value set by set_polygons_overlap_threshold (default=1)."""
        return self._polygons_overlap_threshold


"""
NumPy based bounding box will not correspond to properly computed shapely polygon:

from PIL import Image
import numpy as np
from shapely.geometry import Polygon

image = Image.new(mode = "RGB", size = (30, 50))
image_array = np.asarray(image)
image_bounding_box = ibb = (0, 0, 29, 49)
image_polygon = Polygon([(ibb[0], ibb[1]), (ibb[0], ibb[3]), (ibb[2], ibb[3]), (ibb[2], ibb[1]), (ibb[0], ibb[1])])
print("image area =", image.width * image.height)
print("polygon area =", image_polygon.area)
"""
