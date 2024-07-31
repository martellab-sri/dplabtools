# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for patch location computing."""

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict

import numpy as np
from PIL import Image
import cv2
from matplotlib import colors
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union

from dplabtools.common import print_out, roundfl
from dplabtools.slides.utils import MaskData
from dplabtools.slides.utils.wsi import (
    get_wsi_name,
    get_wsi_downsample_factor,
    find_wsi_level,
    get_level_and_mpp,
    get_level_or_level,
    get_wsi_level_array,
    get_target_resample_factor,
)
from dplabtools.slides import GenericSlide
from dplabtools.slides.patches.locations.utils import expand_scalar_param
from dplabtools.slides.patches.locations.polyfix import fix_polygon_type


class BasePatches(ABC):
    """Base class for whole image or polygon based patch locations computing.

    -!-
    Variable description
    --------------------
    - _patch_level, _patch_mpp: indicate level or resolution for patch location computing and further extraction
    - _patch_info: placeholder for simple patch related stats
    - _param_info: placeholder for parameters used to compute patches
    - _mask_level_patches: computed patch data using mask level coordinates
    - _level0_patches: computed patch data using level zero coordinates
    - _mask_downsample_factor: ratio between level zero and provided mask
    - _target_resample_factor: ratio between level zero and level/mpp of patches to be calculated
    - _relative_resample_factor: ratio between patches to be calculated and mask
    - _preview_resample_factor: ratio between mask and patch preview image
    -!-
    """

    def __init__(
        self,
        *,
        wsi_file,
        mask_data,
        patch_size=256,
        level_or_mpp=0,
        foreground_ratio=0.95,
        overlap_ratio=0.95,
        polygon_buffer=0,
    ):
        """Init method for BasePatches and derived classes.

        Parameters
        ----------
        wsi_file : str
            WSI file name or path.

        mask_data : str or object
            Mask file name or path, or NumPy array object, or Pillow image object.

        patch_size : int, default=256
            Size of calculated patches.

        level_or_mpp : int or float, default=0
            WSI level or MPP value of calculated patches.

        foreground_ratio : float, default=0.95
            Minimum percentage of tissue in each calculated patch.

        overlap_ratio : float, default=0.95
            Minimum percentage of overlapping between each calculated patch and the image or polygon region.

        polygon_buffer : int, default=0
            Additional buffer to increase or decrease the image or polygon region.
        """
        self._abstract_params = locals().copy()
        self._wsi_file = wsi_file
        self._mask_data = MaskData(mask_data=mask_data)
        self._patch_size = patch_size
        self._level_or_mpp = level_or_mpp
        self._foreground_ratio = foreground_ratio
        self._overlap_ratio = overlap_ratio
        self._polygon_buffer = polygon_buffer
        self._wsi_name = get_wsi_name(wsi_file)
        self._wsi_slide = GenericSlide(wsi_file=wsi_file)
        self._abstract_params["mask_data"] = str(self._mask_data)
        self._mask_array = self._mask_data.mask_array
        self._patch_level, self._patch_mpp = get_level_and_mpp(self._level_or_mpp)
        self._patch_info = {}
        self._param_info = {}
        self._mask_level_patches = []
        self._level0_patches = []
        self._mask_downsample_factor = None
        self._target_resample_factor = None
        self._relative_resample_factor = None
        self._preview_resample_factor = None
        self._drawn_labels = []
        self._drawn_patches_labels = []
        self._run()

    @abstractmethod
    def _get_mask_polygons(self):
        """Get list of polygon regions for patch computing.

        Get list of AnnotationPolygon objects where 'points' use mask level coordinates.
        """
        pass

    def _run(self):
        """Start all processing steps."""
        self._check_included_excluded_args()
        if self._patch_mpp:
            self._wsi_slide.check_mpp_data(self._wsi_slide.mpp_data)
            self._wsi_slide.check_mpp_range(self._patch_mpp)
        self._mask_downsample_factor = get_wsi_downsample_factor(self._wsi_slide, self._mask_array.shape)
        self._target_resample_factor = get_target_resample_factor(
            self._patch_level, self._patch_mpp, self._wsi_slide.level_downsamples, self._wsi_slide.level_mpp_values
        )
        self._relative_resample_factor = self._get_relative_resample_factor(
            self._mask_downsample_factor, self._target_resample_factor
        )
        self._mask_downsample_level = find_wsi_level(self._wsi_slide, self._mask_array.shape)
        self._mask_patch_size = self._get_mask_patch_size(self._patch_size, self._relative_resample_factor)
        self._polygons = self._get_mask_polygons()
        if self._filter_polygons:
            self._polygons = self._get_filtered_polygons()
        self._polygon_count = len(self._polygons)
        self._foreground_ratio = expand_scalar_param(self._foreground_ratio, "foreground_ratio", self._polygon_count)
        self._overlap_ratio = expand_scalar_param(self._overlap_ratio, "overlap_ratio", self._polygon_count)
        self._polygon_buffer = expand_scalar_param(self._polygon_buffer, "polygon_buffer", self._polygon_count)
        self._polygon_labels = self._get_polygon_labels(self._polygons)
        self._polygon_drawing_points = self._get_polygon_drawing_points(self._polygons)
        self._polygon_drawing_holes = self._get_polygon_drawing_holes(self._polygons)
        self._shapely_polygons = self._get_shapely_polygons(self._polygons, self._polygon_buffer, self._wsi_name)
        if self._check_polygons:
            self._run_polygons_check(
                self._shapely_polygons, self._mask_patch_size, self.polygons_overlap_threshold, self._wsi_name
            )
        self._bounding_boxes = self._get_bounding_boxes(self._shapely_polygons)
        self._expand_params()
        self._create_param_info()
        self._collect_all_patches()
        self._exclude_duplicate_patches(
            self._level0_patches, self._mask_level_patches, self._patch_info, self._wsi_name
        )

    def _check_included_excluded_args(self):
        """Check argument lists with included/excluded labels."""
        if self._included_labels and self._excluded_labels:
            raise ValueError("%s: Only one list of labels can be specified at a time" % self._wsi_name)

    @staticmethod
    def _get_relative_resample_factor(mask_downsample_factor, target_resample_factor):
        """Get relative_resample_factor."""
        relative_resample_factor = mask_downsample_factor / target_resample_factor
        return relative_resample_factor

    @staticmethod
    def _get_mask_patch_size(patch_size, relative_resample_factor):
        """Get mask patch size."""
        mask_patch_size = patch_size / relative_resample_factor
        return mask_patch_size

    def _get_filtered_polygons(self):
        """Apply included and excluded labels to polygon list."""
        polygons = []
        for polygon in self._polygons:
            if self._included_labels:
                if polygon.label not in self._included_labels:
                    continue
            elif self._excluded_labels:
                if polygon.label in self._excluded_labels:
                    continue
            polygons.append(polygon)
        return polygons

    @staticmethod
    def _get_polygon_labels(polygons):
        """Get list of polygon labels."""
        polygon_labels = [polygon.label for polygon in polygons]
        return polygon_labels

    @staticmethod
    def _get_polygon_drawing_points(polygons):
        """Get list of polygon points for drawing."""
        polygon_points = [polygon.points for polygon in polygons]
        return polygon_points

    @staticmethod
    def _get_polygon_drawing_holes(polygons):
        """Get list of polygon holes for drawing.

        If polygon has no holes, then empty list is added to maintain order.
        """
        polygon_holes = []
        for polygon in polygons:
            if polygon.holes:
                holes = list(polygon.holes)
                polygon_holes.append(holes)
            else:
                polygon_holes.append([])
        return polygon_holes

    @staticmethod
    def _get_shapely_polygons(polygons, polygon_buffer_list, wsi_name):
        """Get list of shapely polygons created using AnnotationPolygon class.

        Polygon buffer parameter may be positive or negative (increase or decrease polygon area).
        """
        if len(polygons) != len(polygon_buffer_list):
            raise IndexError("%s: Number of polygons should match number of buffer list items" % wsi_name)

        shapely_polygons = []
        for counter, polygon in enumerate(polygons):
            polygon_buffer = polygon_buffer_list[counter]
            shapely_polygon = BasePatches._create_valid_polygon(polygon, polygon_buffer, wsi_name)
            if shapely_polygon.area == 0:
                raise ValueError(
                    "%s: Mask level polygon is empty. Recommended actions: increase mask size, "
                    "increase polygon buffer value, or exclude polygon." % wsi_name
                )
            shapely_polygons.append(shapely_polygon)
        return shapely_polygons

    @staticmethod
    def _create_valid_polygon(polygon, polygon_buffer, wsi_name):
        """Return shapely polygon object based on annotation polygon provided.

        Returned object should be either Polygon or MultiPolygon.
        """
        new_polygon = Polygon(polygon.points, holes=polygon.holes)
        # Invalid polygons should be fixed, but during that operation their type may change
        if not new_polygon.is_valid:
            new_polygon = make_valid(new_polygon)
        # All polygons used in further calculations should be of one of those two types
        if not isinstance(new_polygon, (Polygon, MultiPolygon)):
            new_polygon = fix_polygon_type(new_polygon, wsi_name)
        # Apply buffer after obtaining proper polygon, polygon type may change again:
        # - after adding positive buffer, Multipolygon becomes Polygon
        # - after adding negative buffer, Multipolygon remains Multipolygon
        # - number of holes may get reduced after applying negative buffer
        if polygon_buffer != 0:
            new_polygon = new_polygon.buffer(polygon_buffer)
        if not new_polygon.is_valid:
            raise ValueError("%s: Operation resulted in invalid polygon." % wsi_name)
        if not isinstance(new_polygon, (Polygon, MultiPolygon)):
            raise TypeError("%s: Operation resulted in wrong polygon type [%s]." % (wsi_name, type(new_polygon)))
        return new_polygon

    @staticmethod
    def _run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, wsi_name):
        """Run checks to determine annotation/polygon overlapping and embedded polygons.

        If annotation/polygon contains other polygons, those contained polygons will be converted to holes.
        This requires creating new polygon objects and updating main polygon list with new shapely polygons.
        """
        contained_polygons, overlapping_polygons = BasePatches._get_contained_and_overlapping_polygons(shapely_polygons)
        if overlapping_polygons:
            for polygon_index, overlapping_list in overlapping_polygons.items():
                for subpolygon_index in overlapping_list:
                    overlapping_area = (
                        shapely_polygons[polygon_index].intersection(shapely_polygons[subpolygon_index]).area
                    )
                    overlapping_area_patch_units = roundfl(overlapping_area / (mask_patch_size * mask_patch_size))
                    if overlapping_area_patch_units >= polygons_overlap_threshold:
                        print_out(
                            "%s: Polygon %d overlaps with polygon %d, overlapping area: %f patches"
                            % (wsi_name, polygon_index, subpolygon_index, overlapping_area_patch_units)
                        )
        # replace polygons if contained polygons found
        if contained_polygons:
            BasePatches._replace_polygons(shapely_polygons, contained_polygons, wsi_name)

    @staticmethod
    def _get_contained_and_overlapping_polygons(shapely_polygons):
        """Find all contained and overlapping polygons.

        Loop through all shapely polygons and check which polygons contain other polygons and which
        overlap with other polygons. During those operations only polygon external boundaries are taken
        into account, hence the helper function below.
        """

        def get_boundary_polygon(polygon):
            boundary_polygon = polygon
            if isinstance(polygon, Polygon) and len(polygon.interiors):
                boundary_polygon = Polygon(polygon.exterior.coords)
            elif isinstance(polygon, MultiPolygon) and any(len(geometry.interiors) for geometry in polygon.geoms):
                polygon_list = [Polygon(geometry.exterior.coords) for geometry in polygon.geoms]
                boundary_polygon = MultiPolygon(polygon_list)
            return boundary_polygon

        contained_polygons = defaultdict(lambda: [])
        overlapping_polygons = defaultdict(lambda: [])
        if len(shapely_polygons) > 1:
            for index1, poly1 in enumerate(shapely_polygons):
                poly1 = get_boundary_polygon(poly1)
                for index2, poly2 in enumerate(shapely_polygons[index1 + 1 :], start=index1 + 1):
                    poly2 = get_boundary_polygon(poly2)
                    if poly1.contains(poly2):
                        contained_polygons[index1].append(index2)
                    elif poly2.contains(poly1):
                        contained_polygons[index2].append(index1)
                    elif poly1.overlaps(poly2):
                        overlapping_polygons[index1].append(index2)
        return (dict(contained_polygons), dict(overlapping_polygons))

    @staticmethod
    def _replace_polygons(shapely_polygons, contained_polygons, wsi_name):
        """Replace polygons which contain other polygons.

        Replacement here means creating new polygons with holes in place of contained polygons.
        Holes containing other holes and overlapping holes need special processing using "_get_holes_union_coords"
        [reason: if polygons/multipolygons have overlapping/embedded holes, shapely will calculate them
        as polygon area - which is incorrect in this context].
        """
        for polygon_index in contained_polygons:
            polygon = shapely_polygons[polygon_index]
            if isinstance(polygon, Polygon):
                # parent is Polygon
                new_poly = BasePatches._get_replacement_polygon(shapely_polygons, polygon_index, contained_polygons)
                if not isinstance(new_poly, Polygon):
                    raise TypeError("%s: Incorrect polygon replacement type [%s]" % (wsi_name, type(new_poly)))
            elif isinstance(polygon, MultiPolygon):
                # parent is MultiPolygon
                new_poly = BasePatches._get_replacement_multipolygon(
                    shapely_polygons, polygon_index, contained_polygons
                )
                if not isinstance(new_poly, MultiPolygon):
                    raise TypeError("%s: Incorrect multi polygon replacement type [%s]" % (wsi_name, type(new_poly)))
            if not new_poly.is_valid:
                new_poly = make_valid(new_poly)
                if not isinstance(new_poly, (Polygon, MultiPolygon)):
                    new_poly = fix_polygon_type(new_poly, wsi_name)
                    # if still incorrect object type then TypeError will be raised inside fix_polygon_type
            shapely_polygons[polygon_index] = new_poly

    @staticmethod
    def _get_replacement_polygon(shapely_polygons, polygon_index, contained_polygons):
        """Return polygon with holes created based on contained polygons."""
        contains_list = contained_polygons[polygon_index]
        polygon = shapely_polygons[polygon_index]
        # get existing holes
        holes_coords = [list(interior.coords) for interior in polygon.interiors]
        # this list keeps track of polygons which are included in other polygons and should be excluded
        # for performance purposes
        excluded_subpolygons = []
        for contained_poly_index in contains_list:
            if contained_poly_index in excluded_subpolygons:
                continue
            if contained_poly_index in contained_polygons:
                excluded_subpolygons.extend(contained_polygons[contained_poly_index])
            contained_poly = shapely_polygons[contained_poly_index]
            if isinstance(contained_poly, Polygon):
                holes_coords.append(list(contained_poly.exterior.coords))
            elif isinstance(contained_poly, MultiPolygon):
                for int_poly in contained_poly.geoms:
                    holes_coords.append(list(int_poly.exterior.coords))
        # special treatment for overlapping and embedded holes
        holes_coords = BasePatches._get_holes_union_coords(holes_coords)
        return Polygon(polygon.exterior.coords, holes=holes_coords)

    @staticmethod
    def _get_replacement_multipolygon(shapely_polygons, polygon_index, contained_polygons):
        """Return multi polygon with holes created based on contained polygons."""
        contains_list = contained_polygons[polygon_index]
        multipolygon = []
        for subpolygon in shapely_polygons[polygon_index].geoms:
            # this object is required because contains is used below and holes should be excluded from that operation
            subpolygon_without_holes = Polygon(subpolygon.exterior.coords)
            holes_coords = [list(interior.coords) for interior in subpolygon.interiors]
            excluded_subpolygons = []
            for contained_poly_index in contains_list:
                if contained_poly_index in excluded_subpolygons:
                    continue
                if contained_poly_index in contained_polygons:
                    excluded_subpolygons.extend(contained_polygons[contained_poly_index])
                contained_poly = shapely_polygons[contained_poly_index]
                # need to know which exact subpolygon contains given polygon
                if subpolygon_without_holes.contains(contained_poly):
                    if isinstance(contained_poly, Polygon):
                        holes_coords.append(list(contained_poly.exterior.coords))
                    elif isinstance(contained_poly, MultiPolygon):
                        for int_poly in contained_poly.geoms:
                            holes_coords.append(list(int_poly.exterior.coords))
            # special treatment for overlapping and embedded holes
            holes_coords = BasePatches._get_holes_union_coords(holes_coords)
            new_polygon = Polygon(subpolygon.exterior.coords, holes=holes_coords)
            multipolygon.append(new_polygon)
        return MultiPolygon(multipolygon)

    @staticmethod
    def _get_holes_union_coords(holes_coords):
        """Combine all holes into one object and return its coordinates."""
        union_coords = holes_coords
        if len(holes_coords) > 1:
            polygons = [Polygon(coords) for coords in holes_coords]
            union = unary_union(polygons)
            if isinstance(union, Polygon):
                union_coords = [list(union.exterior.coords)]
            elif isinstance(union, MultiPolygon):
                union_coords = [list(poly.exterior.coords) for poly in union.geoms]
            else:
                raise TypeError("Incorrect object type after polygon/holes union: %s" % type(union))
        return union_coords

    @staticmethod
    def _get_bounding_boxes(shapely_polygons):
        """Get list of bounding boxes for shapely polygons."""
        bboxes = []
        for poly in shapely_polygons:
            polygon_bbox = poly.bounds
            bbox = list(polygon_bbox)
            bboxes.append(bbox)
        return bboxes

    def _collect_all_patches(self):
        """Combine patches from all polygons into one output (_level0_patches and self._mask_level_patches)."""
        for polygon, label, counter in zip(self._shapely_polygons, self._polygon_labels, range(self._polygon_count)):
            if label not in self._patch_info:
                self._patch_info[label] = 0
            patches = self._get_polygon_patches(polygon, counter)
            self._patch_info[label] += len(patches)
            level0_patches, mask_level_patches = self._convert_patches_to_final_format(
                patches, label, self._patch_size, self._mask_patch_size, self._mask_downsample_factor
            )
            self._level0_patches.extend(level0_patches)
            self._mask_level_patches.extend(mask_level_patches)

    @staticmethod
    def _convert_patches_to_final_format(patches, label_name, patch_size, mask_patch_size, mask_downsample_factor):
        """Join all computed patch data into their final format, for both mask level and level zero."""
        patches_level0 = []
        patches_mask_level = []
        level0_size = (patch_size, patch_size)
        mask_level_size = (mask_patch_size, mask_patch_size)
        for patch_tuple in patches:
            level0_location = (
                round(patch_tuple[0] * mask_downsample_factor),
                round(patch_tuple[1] * mask_downsample_factor),
            )
            mask_level_location = (patch_tuple[0], patch_tuple[1])
            patches_level0.append((level0_location, level0_size, label_name))
            patches_mask_level.append((mask_level_location, mask_level_size, label_name))
        return (patches_level0, patches_mask_level)

    @staticmethod
    def _exclude_duplicate_patches(level0_patches, mask_level_patches, patch_info, wsi_name):
        """Find and exclude duplicate patches.

        - all values: location, size and label must be the same for two patches to be counted as duplicates
        - this function should not change patch ordering
        """
        patch_counter = Counter(level0_patches)
        patch_diff = sum(patch_counter.values()) - len(patch_counter)
        if patch_diff > 0:
            distinct_mask_level_patches = list(OrderedDict.fromkeys(mask_level_patches))
            level0_patches.clear()
            mask_level_patches.clear()
            print_out("%s: Number of duplicate patches found and excluded: %d" % (wsi_name, patch_diff))
            duplicates = {tuple_key: counter for tuple_key, counter in patch_counter.items() if counter > 1}
            # update level zero patch data
            level0_patches.extend(list(patch_counter.keys()))
            # update mask level patch data
            mask_level_patches.extend(distinct_mask_level_patches)
            # update patch counts by subtracting counter values above "1"
            for patch_data, counter in duplicates.items():
                label = patch_data[-1]
                patch_info[label] -= counter - 1

    def save_preview_image(
        self,
        image_file,
        patch_color="red",
        bbox_color="lime",
        polygon_color="blue",
        holes_color="teal",
        buffer_color="black",
        thickness=2,
        patch_markers=False,
        level_or_minsize=None,
        drawn_labels=[],
        drawn_patches_labels=[],
    ):
        """Save preview image presenting calculated patches and image/polygon boundaries.

        Draw and save patch/polygon preview image based on mask size (default) or an arbitrary WSI level.

        Parameters
        ----------
        image_file : str
            Image file name or path.

        patch_color : str, default="red"
            Color of drawn patches and patch markers.

        bbox_color : str, default="lime"
            Color of drawn polygon bounding boxes.

        polygon_color : str, default="blue"
            Color of drawn polygons.

        holes_color : str, default="teal"
            Color of drawn polygon holes.

        buffer_color : str, default="black"
            Color of drawn polygons after applying the buffer.

        thickness : int, default=2
            Thickness of the drawing line.

        patch_markers : bool, default=False
            Whether to draw patch markers or not.

        level_or_minsize : int, optional
            Determines physical dimensions of the saved image. It's either a WSI level or a minimal desired size
            in pixels.

        drawn_labels : list of str, optional
            List of labels for drawing polygons, all other polygons will be ignored.

        drawn_patches_labels : list of str, optional
            List of labels for drawing patches, all other patches will be ignored.
        """
        if level_or_minsize is not None:
            preview_level = get_level_or_level(self._wsi_slide, level_or_minsize)
        else:
            preview_level = None

        if preview_level is not None and preview_level != self._mask_downsample_level:
            image_array = get_wsi_level_array(self._wsi_slide, preview_level)
            self._preview_resample_factor = (
                self._wsi_slide.level_downsamples[preview_level]
                / self._wsi_slide.level_downsamples[self._mask_downsample_level]
            )
        else:
            image_array = get_wsi_level_array(self._wsi_slide, self._mask_downsample_level)
            self._preview_resample_factor = 1

        self._drawn_labels = drawn_labels
        self._drawn_patches_labels = drawn_patches_labels
        # Array copying is required, otherwise cv2.polylines error: SO #23830618
        image_array = image_array.copy()

        if patch_color is not None:
            self._draw_preview_patches(image_array, patch_color, thickness, patch_markers)
        if bbox_color is not None:
            self._draw_preview_bboxes(image_array, bbox_color, thickness)
        if polygon_color is not None:
            self._draw_preview_polygons(image_array, polygon_color, thickness)
        if holes_color is not None:
            self._draw_preview_holes(image_array, holes_color, thickness)
        if buffer_color is not None and self._polygon_buffer:
            self._draw_preview_buffers(image_array, buffer_color, thickness)
        preview_image = Image.fromarray(image_array)
        preview_image.save(image_file)

    def _draw_preview_patches(self, image_array, patch_color, thickness, patch_markers):
        """Draw patch polygons on image array."""
        prf = self._preview_resample_factor
        preview_patch_size = round(self._mask_patch_size / prf)
        marker_size = thickness * 2
        marker_thickness = max(thickness * 2, round(preview_patch_size * 0.05))
        scaled_patches = []
        markers = []
        for one_patch_data in self._mask_level_patches:
            patch_location = one_patch_data[0]
            patch_label = one_patch_data[2]
            scaled_patch_location = (round(patch_location[0] / prf), round(patch_location[1] / prf))
            if self._drawn_labels:
                if patch_label not in self._drawn_labels:
                    continue
            else:
                if self._drawn_patches_labels and patch_label not in self._drawn_patches_labels:
                    continue
            scaled_patch = self._convert_patch_to_point_list(scaled_patch_location, preview_patch_size)
            scaled_patches.append(scaled_patch)
            if patch_markers:
                marker = self._convert_patch_to_point_list(scaled_patch_location, marker_size)
                markers.append(marker)

        self._draw_polygons(image_array, scaled_patches, patch_color, thickness)
        if patch_markers:
            self._draw_polygons(image_array, markers, patch_color, marker_thickness)

    @staticmethod
    def _convert_patch_to_point_list(patch_location, mask_patch_size):
        """Convert patch (location + size) to list of points."""
        mps = mask_patch_size
        return [
            (patch_location[0], patch_location[1]),
            (patch_location[0], patch_location[1] + mps),
            (patch_location[0] + mps, patch_location[1] + mps),
            (patch_location[0] + mps, patch_location[1]),
        ]

    def _draw_preview_bboxes(self, image_array, bbox_color, thickness):
        """Draw bounding boxes on image array."""
        prf = self._preview_resample_factor
        scaled_bboxes = []
        for polygon_label, bbox in zip(self._polygon_labels, self._bounding_boxes):
            if self._drawn_labels and polygon_label not in self._drawn_labels:
                continue
            scaled_bbox = self._convert_bbox_to_point_list(
                [round(bbox[0] / prf), round(bbox[1] / prf), round(bbox[2] / prf), round(bbox[3] / prf)]
            )
            scaled_bboxes.append(scaled_bbox)
        self._draw_polygons(image_array, scaled_bboxes, bbox_color, thickness)

    def _draw_preview_polygons(self, image_array, polygon_color, thickness):
        """Draw annotation polygons on image array."""
        prf = self._preview_resample_factor
        scaled_polygons_points = []
        for polygon_label, polygon_points in zip(self._polygon_labels, self._polygon_drawing_points):
            if self._drawn_labels and polygon_label not in self._drawn_labels:
                continue
            scaled_polygon_points = [(round(point[0] / prf), round(point[1] / prf)) for point in polygon_points]
            scaled_polygons_points.append(scaled_polygon_points)
        self._draw_polygons(image_array, scaled_polygons_points, polygon_color, thickness)

    def _draw_preview_holes(self, image_array, holes_color, thickness):
        """Draw annotation holes on image array."""
        prf = self._preview_resample_factor
        scaled_holes_points = []
        for polygon_label, polygon_holes in zip(self._polygon_labels, self._polygon_drawing_holes):
            if self._drawn_labels and polygon_label not in self._drawn_labels:
                continue
            for single_hole in polygon_holes:
                scaled_hole_points = [(round(point[0] / prf), round(point[1] / prf)) for point in single_hole]
                scaled_holes_points.append(scaled_hole_points)
        self._draw_polygons(image_array, scaled_holes_points, holes_color, thickness)

    def _draw_preview_buffers(self, image_array, buffer_color, thickness):
        """Draw annotation polygon buffers on image array."""

        def scale_coords(coords_list, factor):
            scaled_coords = [(round(point[0] / factor), round(point[1] / factor)) for point in coords_list]
            return scaled_coords

        polygon_buffer_points = []
        prf = self._preview_resample_factor
        for counter, (polygon_label, poly) in enumerate(zip(self._polygon_labels, self._shapely_polygons)):
            if self._drawn_labels and polygon_label not in self._drawn_labels:
                continue
            if isinstance(poly, Polygon):
                if self._polygon_buffer[counter]:
                    polygon_buffer_points.append(scale_coords(list(poly.exterior.coords), prf))
                    for interior_poly in poly.interiors:
                        polygon_buffer_points.append(scale_coords(list(interior_poly.coords), prf))
            elif isinstance(poly, MultiPolygon):
                if self._polygon_buffer[counter]:
                    for sub_poly in poly.geoms:
                        polygon_buffer_points.append(scale_coords(list(sub_poly.exterior.coords), prf))
                        for interior_poly in sub_poly.interiors:
                            polygon_buffer_points.append(scale_coords(list(interior_poly.coords), prf))
        self._draw_polygons(image_array, polygon_buffer_points, buffer_color, thickness)

    @staticmethod
    def _draw_polygons(image_array, polygons_points, color, thickness):
        """Draw generic polygons on image array.

        polygons_points must be converted to integer values before passing to this method
        """
        polygon_data = [np.asarray(poly_points, dtype=np.int32) for poly_points in polygons_points]
        color_rgb = [int(val * 255) for val in colors.hex2color(colors.cnames[color])]
        cv2.polylines(image_array, polygon_data, True, color_rgb, thickness=thickness)

    @staticmethod
    def _convert_bbox_to_point_list(bbox):
        """Convert bounding box rectangle to list of points."""
        return [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]

    @staticmethod
    def _get_shapely_bbox_points(shapely_polygon):
        """Get polygon bounding box points.

        Values must be rounded as they are used in comparisons.
        """
        return tuple(roundfl(point) for point in shapely_polygon.bounds)

    def _add_valid_patch(
        self, patch_list, patch_location, shapely_polygon, shapely_polygon_index, shapely_polygon_bbox
    ):
        """Add patch to list of valid patches."""
        foreground_ratio = self._foreground_ratio[shapely_polygon_index]
        overlap_ratio = self._overlap_ratio[shapely_polygon_index]
        mask_patch_overlap_area = roundfl(self._mask_patch_size * self._mask_patch_size * overlap_ratio, 8)

        if self._is_patch_valid(
            self._mask_array,
            self._mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            self._full_overlap_check,
        ):
            patch_list.append(patch_location)

    @staticmethod
    def _is_patch_valid(
        mask_array,
        mask_patch_size,
        patch_location,
        shapely_polygon,
        shapely_polygon_bbox,
        foreground_ratio,
        mask_patch_overlap_area,
        full_overlap_check,
    ):
        """Determine if patch is valid.

        - There are edge cases where retrieved (via slicing) mask patches will go beyond mask array dimensions
        (right side, bottom, or bottom right corner) and then retrieved NumPy array for mask patch will be
        automatically truncated by NumPy (bad). Such truncated arrays must be padded with background values (0s)
        as they still may include tissue region and may become valid patches (based on user settings for foreground
        and polygon overlapping).
        - shapely_polygon and shapely_polygon_bbox are passed separately, as calculating bounding box is quite expensive
        (it's a regular property in shapely, not a cached property).
        """
        is_valid = False
        x1, y1 = patch_location
        # integer coordinates are only used for slicing
        x1_int = round(x1)
        y1_int = round(y1)
        x2 = x1 + mask_patch_size
        y2 = y1 + mask_patch_size
        x2_int = round(x2)
        y2_int = round(y2)
        mask_patch = mask_array[x1_int:x2_int, y1_int:y2_int]
        overlap_check_required = full_overlap_check or BasePatches._is_overlap_check_required(
            shapely_polygon_bbox, x1, y1, x2, y2
        )
        # Notes:
        # - in background/foreground checking mask_patch has integer dimensions
        # - in overlapping checking mask_polygon has float coordinates
        # Thus those two checks in some cases will not be 100% equivalent
        mask_patch_size_x = x2_int - x1_int
        mask_patch_size_y = y2_int - y1_int
        if mask_patch.shape[0] < mask_patch_size_x or mask_patch.shape[1] < mask_patch_size_y:
            mask_patch = BasePatches._pad_mask_patch_array(mask_patch, (mask_patch_size_x, mask_patch_size_y))
        if BasePatches._is_foreground(mask_patch, foreground_ratio):
            if overlap_check_required:
                mask_polygon = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)])
                if BasePatches._is_overlapping(shapely_polygon, mask_polygon, mask_patch_overlap_area):
                    is_valid = True
            else:
                is_valid = True
        return is_valid

    @staticmethod
    def _is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2):
        """Determine if detailed overlap check must be performed.

        - Detailed overlap check involves creating a mask level patch polygon (using shapely), which is computationally
        expensive.
        - Patch locations on the whole image are always based on the whole image bounding box, which is always
        a rectangle. Thus only patches with edges exceeding polygon bounding box should be checked using shapely.
        This is not true for polygon based patch locations, where each mask patch must be checked against overlapping
        with polygon (annotation), which may take various forms (rectangles, circles, triangles, etc.).
        """
        required = False
        x_min, y_min, x_max, y_max = shapely_polygon_bbox
        if roundfl(x1) < x_min or roundfl(y1) < y_min or roundfl(x2) > x_max or roundfl(y2) > y_max:
            required = True
        return required

    @staticmethod
    def _pad_mask_patch_array(mask_patch, mask_patch_size):
        """Pad array to given size."""
        mask_patch_shape_x, mask_patch_shape_y = mask_patch.shape
        mask_patch_size_x, mask_patch_size_y = mask_patch_size
        diff_x = diff_y = 0
        if mask_patch_shape_x < mask_patch_size_x:
            diff_x = mask_patch_size_x - mask_patch_shape_x
        if mask_patch_shape_y < mask_patch_size_y:
            diff_y = mask_patch_size_y - mask_patch_shape_y
        # default NumPy padding value is 0, zeros below indicate coordinates
        return np.pad(mask_patch, ((0, diff_x), (0, diff_y)))

    @staticmethod
    def _is_foreground(mask, threshold):
        """Determine if mask nonzero elements are above threshold."""
        return roundfl(np.count_nonzero(mask) / mask.size) >= threshold

    @staticmethod
    def _is_overlapping(shapely_polygon, mask_polygon, patch_overlap_area):
        """Determine if mask intersection elements are above threshold.

        Float comparison accuracy is lowered here to eliminate accumulated rounding errors.
        """
        return roundfl(mask_polygon.intersection(shapely_polygon).area, 8) >= patch_overlap_area

    def _create_param_info(self):
        """Create dictionary with class parameters used for patch computing."""
        excluded_params = ("self", "kwargs", "__class__")
        region_params = [
            (pname, pvalue) for pname, pvalue in self._region_params.items() if pname not in excluded_params
        ]
        mixin_params = [(pname, pvalue) for pname, pvalue in self._mixin_params.items() if pname not in excluded_params]
        abstract_params = [
            (pname, pvalue) for pname, pvalue in self._abstract_params.items() if pname not in excluded_params
        ]
        for param_list in (region_params, mixin_params, abstract_params):
            for param in param_list:
                self._param_info[param[0]] = param[1]

    @property
    def patch_count(self):
        """Return the number of calculated patches."""
        return len(self._level0_patches)

    @property
    def patch_data(self):
        """Return patch data (location, size, label) for all calculated patches at level 0."""
        return self._level0_patches

    @property
    def patch_data_mask_level(self):
        """Return patch data (location, size, label) for all calculated patches at the mask level."""
        return self._mask_level_patches

    @property
    def patch_info(self):
        """Return patch information (counts and labels) for all calculated patches."""
        return dict(sorted(self._patch_info.items()))

    @property
    def patch_labels(self):
        """Return distinct polygon labels used in patch calculations."""
        return sorted(list(set(self._polygon_labels)))

    @property
    def param_info(self):
        """Return parameter information used in patch calculations."""
        return self._param_info

    @property
    def level_or_mpp(self):
        """Return level or MPP value used in patch calculations."""
        return self._level_or_mpp

    @property
    def shapely_polygons(self):
        """Return Shapely polygon objects representing polygons in patch calculations."""
        return self._shapely_polygons

    @property
    def polygon_labels(self):
        """Return all polygon labels used in patch calculations."""
        return self._polygon_labels

    @property
    def patch_size(self):
        """Return provided patch size."""
        return self._patch_size

    @property
    def mask_shape(self):
        """Return provided mask shape."""
        return self._mask_array.shape

    @property
    def wsi_file(self):
        """Return provided WSI file name."""
        return self._wsi_file

    @property
    def wsi_slide(self):
        """Return WSI slide object created during patch calculations."""
        return self._wsi_slide

    @property
    def class_name(self):
        """Return the name of the patches object class."""
        return self.__class__.__name__
