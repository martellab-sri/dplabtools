# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for patch base class (BasePatches).

Tested classes:
    BasePatches
"""

import copy
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.validation import make_valid

from dplabtools.slides.patches.locations.base import BasePatches
from dplabtools.slides.utils import AnnotationPolygon, MaskData
from testutils import make_test_path


class PatchesTestMock(BasePatches):
    """Mock class implementing one abstract method present in BasePatches (and one method from mixins)."""

    _filter_polygons = True
    _polygons_overlap_threshold = 1

    def __init__(self, polygon_data, check_polygons=True, **kwargs):
        """Init method."""
        self._region_params = locals()
        self._mixin_params = locals()
        self._polygons = polygon_data
        self._check_polygons = check_polygons
        self._included_labels = []
        self._excluded_labels = []
        super().__init__(**kwargs)

    def _get_mask_polygons(self):
        return self._polygons

    def _get_polygon_patches(self, polygon, counter):
        return [(10, 20), (130, 140), (250, 260), (350, 370)]

    def _expand_params(self):
        pass

    @property
    def polygons_overlap_threshold(self):
        return self._polygons_overlap_threshold


class TestPatchesBaseStaticMethods(TestCase):
    """Tests for static methods in BasePatches class."""

    def setUp(self):
        self.polygons = [
            AnnotationPolygon(
                points=[(100, 100), (200, 200), (300, 300)],
                label="label1",
            ),
            AnnotationPolygon(
                points=[(200, 200), (400, 400), (600, 600), (800, 800)],
                label="label2",
                holes=[
                    [(250, 250), (350, 350), (450, 450), (550, 550)],
                    [(20, 20), (40, 40), (60, 60), (80, 80)],
                ],
            ),
            AnnotationPolygon(
                points=[(100, 100), (200, 200), (500, 500)],
                label="label2",
            ),
            AnnotationPolygon(
                points=[(100, 100), (200, 200), (400, 400)],
                label="label1",
                holes=[[(10, 10), (20, 20), (40, 40), (50, 50)]],
            ),
        ]
        self.polygon_points = [[(1, 1), (4, 1), (1, 4)], [(1, 1), (4, 1), (4, 4), (1, 5)]]

    def test__get_relative_downsample_factor(self):
        output_value = 0.5
        result_value = PatchesTestMock._get_relative_resample_factor(2, 4)
        self.assertEqual(result_value, output_value)

    def test__get_mask_patch_size(self):
        # helper function for easier output prediction
        def get_result(patch_size, mask_downsample_factor, target_resample_factor):
            relative_downsample_factor = PatchesTestMock._get_relative_resample_factor(
                mask_downsample_factor, target_resample_factor
            )
            mask_patch_size = PatchesTestMock._get_mask_patch_size(patch_size, relative_downsample_factor)
            return mask_patch_size

        # 1.
        patch_size = 256
        mask_downsample_factor = 4
        target_resample_factor = 1
        output_value = 64
        result_value = get_result(patch_size, mask_downsample_factor, target_resample_factor)
        self.assertEqual(result_value, output_value)
        # 2.
        patch_size = 256
        mask_downsample_factor = 16
        target_resample_factor = 1
        output_value = 16
        result_value = get_result(patch_size, mask_downsample_factor, target_resample_factor)
        self.assertEqual(result_value, output_value)
        # 3.
        patch_size = 64
        mask_downsample_factor = 16
        target_resample_factor = 16
        output_value = 64
        result_value = get_result(patch_size, mask_downsample_factor, target_resample_factor)
        self.assertEqual(result_value, output_value)
        # 4.
        patch_size = 128
        mask_downsample_factor = 4
        target_resample_factor = 16
        output_value = 512
        result_value = get_result(patch_size, mask_downsample_factor, target_resample_factor)
        self.assertEqual(result_value, output_value)

    def test__get_polygon_labels(self):
        output_labels = ["label1", "label2", "label2", "label1"]
        result_labels = PatchesTestMock._get_polygon_labels(self.polygons)
        self.assertEqual(result_labels, output_labels)

    def test__get_polygon_drawing_points(self):
        output_points = [
            [(100, 100), (200, 200), (300, 300)],
            [(200, 200), (400, 400), (600, 600), (800, 800)],
            [(100, 100), (200, 200), (500, 500)],
            [(100, 100), (200, 200), (400, 400)],
        ]
        result_points = PatchesTestMock._get_polygon_drawing_points(self.polygons)
        self.assertEqual(result_points, output_points)

    def test__get_polygon_drawing_holes1(self):
        output_holes = [
            [],
            [[(250, 250), (350, 350), (450, 450), (550, 550)], [(20, 20), (40, 40), (60, 60), (80, 80)]],
            [],
            [[(10, 10), (20, 20), (40, 40), (50, 50)]],
        ]
        result_holes = PatchesTestMock._get_polygon_drawing_holes(self.polygons)
        self.assertEqual(result_holes, output_holes)

    def test__get_polygon_drawing_holes2(self):
        polygons = [
            AnnotationPolygon(points=[(100, 100), (200, 200), (300, 300)], label="label1"),
            AnnotationPolygon(points=[(200, 200), (400, 400), (600, 600), (800, 800)], label="label2"),
            AnnotationPolygon(points=[(100, 100), (200, 200), (500, 500)], label="label3"),
            AnnotationPolygon(points=[(100, 100), (200, 200), (400, 400)], label="label4"),
        ]
        output_holes = [
            [],
            [],
            [],
            [],
        ]
        result_holes = PatchesTestMock._get_polygon_drawing_holes(polygons)
        self.assertEqual(result_holes, output_holes)

    def test__get_polygon_drawing_holes3(self):
        polygons = [
            AnnotationPolygon(
                points=[(100, 100), (200, 200), (300, 300)],
                label="label1",
                holes=[[(20, 20), (40, 40), (60, 60), (80, 80)]],
            ),
            AnnotationPolygon(
                points=[(200, 200), (400, 400), (600, 600), (800, 800)],
                label="label2",
                holes=[[(250, 250), (350, 350), (450, 450), (550, 550)]],
            ),
            AnnotationPolygon(
                points=[(100, 100), (200, 200), (500, 500)],
                label="label2",
                holes=[[(220, 220), (240, 240), (260, 260), (280, 280)]],
            ),
            AnnotationPolygon(
                points=[(100, 100), (200, 200), (400, 400)],
                label="label1",
                holes=[[(10, 10), (20, 20), (40, 40), (50, 50)]],
            ),
        ]
        output_holes = [
            [[(20, 20), (40, 40), (60, 60), (80, 80)]],
            [[(250, 250), (350, 350), (450, 450), (550, 550)]],
            [[(220, 220), (240, 240), (260, 260), (280, 280)]],
            [[(10, 10), (20, 20), (40, 40), (50, 50)]],
        ]
        result_holes = PatchesTestMock._get_polygon_drawing_holes(polygons)
        self.assertEqual(result_holes, output_holes)

    def test__get_shapely_polygons(self):
        buffer_list = [0.5, 0.7]
        wsi_name = "slide1.svs"
        polygons = [
            AnnotationPolygon(points=self.polygon_points[0], label="label1"),
            AnnotationPolygon(points=self.polygon_points[1], label="label2"),
        ]
        output_polygons = [
            Polygon(self.polygon_points[0]).buffer(buffer_list[0]),
            Polygon(self.polygon_points[1]).buffer(buffer_list[1]),
        ]
        result_polygons = PatchesTestMock._get_shapely_polygons(polygons, buffer_list, wsi_name)
        self.assertEqual(list(result_polygons[0].exterior.coords), list(output_polygons[0].exterior.coords))
        self.assertEqual(list(result_polygons[1].exterior.coords), list(output_polygons[1].exterior.coords))
        self.assertIsInstance(result_polygons[0], Polygon)
        self.assertIsInstance(result_polygons[1], Polygon)
        # fail on invalid buffer list
        polygon_points = [[(779, 652), (779, 653), (779, 653), (779, 652), (779, 652)]]
        polygons = [AnnotationPolygon(points=polygon_points, label="label1")]
        buffer_list = [0, 1]
        with self.assertRaises(IndexError):
            PatchesTestMock._get_shapely_polygons(polygons, buffer_list, wsi_name)
        # fail on empty (area=0) polygon
        buffer_list = [0]
        with self.assertRaises(ValueError):
            PatchesTestMock._get_shapely_polygons(polygons, buffer_list, wsi_name)

    def test__create_valid_polygon(self):
        polygon_buffer = 0
        # valid without holes
        polygon = AnnotationPolygon(
            points=[(50, 50), (50, 100), (100, 100), (100, 50)],
            label="",
        )
        result_polygon = PatchesTestMock._create_valid_polygon(polygon, polygon_buffer, "")
        self.assertIsInstance(result_polygon, Polygon)
        self.assertEqual(result_polygon.area, 2500)
        # valid with holes
        polygon = AnnotationPolygon(
            points=[(50, 50), (50, 100), (100, 100), (100, 50)],
            label="",
            holes=[[(70, 70), (70, 80), (80, 80), (80, 70)]],
        )
        result_polygon = PatchesTestMock._create_valid_polygon(polygon, polygon_buffer, "")
        self.assertIsInstance(result_polygon, Polygon)
        self.assertEqual(result_polygon.area, 2400)
        #
        invalid_polygon = AnnotationPolygon(
            points=[(50, 50), (50, 100), (70, 70), (30, 50), (20, 70)],
            label="",
        )
        invalid_polygon_with_holes = AnnotationPolygon(
            points=[(50, 50), (50, 100), (70, 70), (30, 50), (20, 70)],
            label="",
            holes=[
                [(30, 54), (30, 59), (35, 59), (35, 54)],  # area is 25
                [(55, 70), (55, 76), (61, 76), (61, 70)],  # area is 36
                [(45, 55), (45, 57), (47, 57), (47, 55)],  # area is 4
            ],
        )
        #
        # invalid without holes (will be fixed)
        result_polygon = PatchesTestMock._create_valid_polygon(invalid_polygon, polygon_buffer, "")
        self.assertIsInstance(result_polygon, MultiPolygon)
        self.assertEqual(round(result_polygon.area), 586)
        # invalid with holes (will be fixed)
        result_polygon = PatchesTestMock._create_valid_polygon(invalid_polygon_with_holes, polygon_buffer, "")
        self.assertIsInstance(result_polygon, MultiPolygon)
        self.assertEqual(round(result_polygon.area), 586 - 25 - 36 - 4)
        # invalid without holes, with positive buffer (will be fixed and resized)
        polygon_buffer = 1
        result_polygon = PatchesTestMock._create_valid_polygon(invalid_polygon, polygon_buffer, "")
        self.assertIsInstance(result_polygon, Polygon)
        self.assertTrue(round(result_polygon.area) > 586)
        # invalid with holes, with positive buffer (will be fixed and resized)
        polygon_buffer = 1
        result_polygon = PatchesTestMock._create_valid_polygon(invalid_polygon_with_holes, polygon_buffer, "")
        self.assertIsInstance(result_polygon, Polygon)
        self.assertTrue(round(result_polygon.area) > 586 - 25 - 36 - 4)
        #
        # invalid with holes, with negative buffer (will be fixed and resized)
        polygon_buffer = -1
        result_polygon = PatchesTestMock._create_valid_polygon(invalid_polygon_with_holes, polygon_buffer, "")
        self.assertIsInstance(result_polygon, MultiPolygon)
        self.assertTrue(round(result_polygon.area) < 586 - 25 - 36 - 4)

    @patch("dplabtools.slides.patches.locations.base.make_valid")
    def test__create_valid_polygon_not_made_valid(self, mock_func):
        # invalid without buffer (will not be fixed)
        invalid_polygon = AnnotationPolygon(points=[(50, 50), (50, 100), (70, 70), (30, 50), (20, 70)], label="")
        mock_func.return_value = Polygon(invalid_polygon.points)
        polygon_buffer = 0
        with self.assertRaises(ValueError):
            PatchesTestMock._create_valid_polygon(invalid_polygon, polygon_buffer, "")
        # Note: applying buffer without using make_valid would produce incomplete polygon, so it's not tested here

    @patch("dplabtools.slides.patches.locations.base.fix_polygon_type")
    def test__create_valid_polygon_type_error(self, mock_func):
        # invalid without buffer (will not be fixed)
        invalid_polygon_plus_line = AnnotationPolygon(
            points=[(50, 50), (50, 100), (70, 70), (30, 50), (20, 70), (20, 100), (20, 70)], label=""
        )
        mock_func.return_value = make_valid(Polygon(invalid_polygon_plus_line.points))
        polygon_buffer = 0
        with self.assertRaises(TypeError):
            PatchesTestMock._create_valid_polygon(invalid_polygon_plus_line, polygon_buffer, "")

    def test__create_valid_polygon_multi(self):
        # create multipolygon and check type
        multi_polygon = AnnotationPolygon(points=[(50, 50), (50, 100), (70, 70), (30, 50), (20, 70)], label="")
        polygon_buffer = 0
        result_polygon = PatchesTestMock._create_valid_polygon(multi_polygon, polygon_buffer, "")
        self.assertIsInstance(result_polygon, MultiPolygon)

    def test__get_bounding_boxes(self):
        buffer_list = [0, 0]
        output_bboxes = [[1, 1, 4, 4], [1, 1, 4, 5]]
        polygons = [
            Polygon(self.polygon_points[0]).buffer(buffer_list[0]),
            Polygon(self.polygon_points[1]).buffer(buffer_list[1]),
        ]
        result_bboxes = PatchesTestMock._get_bounding_boxes(polygons)
        self.assertEqual(result_bboxes, output_bboxes)

    def test__convert_patches_to_final_format(self):
        patches = [(1, 2), (3, 4), (5, 6)]
        label_name = "label"
        patch_size = 5
        mask_patch_size = 11
        mask_downsample_factor = 15
        output_patches = (
            [((15, 30), (5, 5), "label"), ((45, 60), (5, 5), "label"), ((75, 90), (5, 5), "label")],
            [((1, 2), (11, 11), "label"), ((3, 4), (11, 11), "label"), ((5, 6), (11, 11), "label")],
        )
        result_patches = PatchesTestMock._convert_patches_to_final_format(
            patches, label_name, patch_size, mask_patch_size, mask_downsample_factor
        )
        self.assertEqual(result_patches, output_patches)

    def test__exclude_duplicate_patches1(self):
        level0_patches_before = [
            ((100, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area3"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((300, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area2"),
            ((100, 100), (256, 256), "area2"),
            ((100, 100), (256, 256), "area3"),
            ((200, 100), (256, 256), "area3"),
            ((400, 100), (256, 256), "area3"),
            ((500, 100), (256, 256), "area3"),
            ((500, 100), (256, 256), "area3"),
            ((600, 100), (256, 256), "area3"),
        ]
        mask_level_patches_before = [
            ((25, 25), (64, 64), "area2"),
            ((50, 25), (64, 64), "area3"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((75, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area2"),
            ((50, 25), (64, 64), "area2"),
            ((50, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area3"),
            ((50, 25), (64, 64), "area3"),
            ((100, 25), (64, 64), "area3"),
            ((125, 25), (64, 64), "area3"),
            ((125, 25), (64, 64), "area3"),
            ((150, 25), (64, 64), "area3"),
        ]
        patch_info_before = {"area1": 6, "area2": 5, "area3": 7}
        wsi_name = "wsi_duplicates1.svs"
        PatchesTestMock._exclude_duplicate_patches(
            level0_patches_before, mask_level_patches_before, patch_info_before, wsi_name
        )
        level0_patches_after = [
            ((100, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area3"),
            ((100, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((300, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area2"),
            ((100, 100), (256, 256), "area3"),
            ((400, 100), (256, 256), "area3"),
            ((500, 100), (256, 256), "area3"),
            ((600, 100), (256, 256), "area3"),
        ]
        mask_level_patches_after = [
            ((25, 25), (64, 64), "area2"),
            ((50, 25), (64, 64), "area3"),
            ((25, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((75, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area3"),
            ((100, 25), (64, 64), "area3"),
            ((125, 25), (64, 64), "area3"),
            ((150, 25), (64, 64), "area3"),
        ]
        patch_info_after = {"area1": 3, "area2": 2, "area3": 5}
        self.assertEqual(level0_patches_before, level0_patches_after)
        self.assertEqual(mask_level_patches_before, mask_level_patches_after)
        self.assertEqual(patch_info_before, patch_info_after)

    def test__exclude_duplicate_patches2(self):
        level0_patches_before = [
            ((600, 100), (256, 256), "area3"),
            ((200, 100), (256, 256), "area3"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), ""),
            ((200, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((300, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), ""),
            ((100, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area2"),
            ((100, 100), (256, 256), "area2"),
            ((100, 100), (256, 256), ""),
            ((200, 100), (256, 256), "area3"),
            ((400, 100), (256, 256), "area3"),
            ((500, 100), (256, 256), "area3"),
            ((500, 100), (256, 256), "area3"),
            ((100, 100), (256, 256), ""),
        ]
        mask_level_patches_before = [
            ((150, 25), (64, 64), "area3"),
            ((50, 25), (64, 64), "area3"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), ""),
            ((50, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((75, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), ""),
            ((25, 25), (64, 64), "area2"),
            ((50, 25), (64, 64), "area2"),
            ((50, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), ""),
            ((50, 25), (64, 64), "area3"),
            ((100, 25), (64, 64), "area3"),
            ((125, 25), (64, 64), "area3"),
            ((125, 25), (64, 64), "area3"),
            ((25, 25), (64, 64), ""),
        ]
        patch_info_before = {"area1": 5, "area2": 4, "area3": 6, "": 4}
        wsi_name = "wsi_duplicates2.svs"
        PatchesTestMock._exclude_duplicate_patches(
            level0_patches_before, mask_level_patches_before, patch_info_before, wsi_name
        )
        level0_patches_after = [
            ((600, 100), (256, 256), "area3"),
            ((200, 100), (256, 256), "area3"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), ""),
            ((200, 100), (256, 256), "area1"),
            ((300, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area2"),
            ((400, 100), (256, 256), "area3"),
            ((500, 100), (256, 256), "area3"),
        ]
        mask_level_patches_after = [
            ((150, 25), (64, 64), "area3"),
            ((50, 25), (64, 64), "area3"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), ""),
            ((50, 25), (64, 64), "area1"),
            ((75, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area2"),
            ((50, 25), (64, 64), "area2"),
            ((100, 25), (64, 64), "area3"),
            ((125, 25), (64, 64), "area3"),
        ]
        patch_info_after = {"area1": 3, "area2": 2, "area3": 4, "": 1}
        self.assertEqual(level0_patches_before, level0_patches_after)
        self.assertEqual(mask_level_patches_before, mask_level_patches_after)
        self.assertEqual(patch_info_before, patch_info_after)

    def test__exclude_duplicate_patches3(self):
        level0_patches_before = [
            ((100, 100), (257, 257), "area2"),
            ((200, 100), (257, 257), "area3"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (257, 257), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((200, 100), (257, 257), "area1"),
            ((200, 100), (258, 258), "area1"),
            ((300, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area2"),
            ((200, 100), (257, 257), "area2"),
            ((100, 100), (258, 258), "area2"),
            ((100, 100), (256, 256), "area3"),
            ((200, 100), (256, 256), "area3"),
            ((400, 100), (256, 256), "area3"),
            ((500, 100), (257, 257), "area3"),
            ((500, 100), (256, 256), "area3"),
            ((600, 100), (256, 256), "area3"),
        ]
        mask_level_patches_before = [
            ((25, 25), (65, 65), "area2"),
            ((50, 25), (65, 65), "area3"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (65, 65), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((50, 25), (65, 65), "area1"),
            ((50, 25), (66, 66), "area1"),
            ((75, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area2"),
            ((50, 25), (65, 65), "area2"),
            ((50, 25), (66, 66), "area2"),
            ((25, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area3"),
            ((50, 25), (64, 64), "area3"),
            ((100, 25), (64, 64), "area3"),
            ((125, 25), (65, 65), "area3"),
            ((125, 25), (64, 64), "area3"),
            ((150, 25), (64, 64), "area3"),
        ]
        patch_info_before = {"area1": 6, "area2": 5, "area3": 7}
        wsi_name = "wsi_duplicates3.svs"
        PatchesTestMock._exclude_duplicate_patches(
            level0_patches_before, mask_level_patches_before, patch_info_before, wsi_name
        )
        level0_patches_after = [
            ((100, 100), (257, 257), "area2"),
            ((200, 100), (257, 257), "area3"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (257, 257), "area1"),
            ((200, 100), (256, 256), "area1"),
            ((200, 100), (257, 257), "area1"),
            ((200, 100), (258, 258), "area1"),
            ((300, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area2"),
            ((200, 100), (256, 256), "area2"),
            ((200, 100), (257, 257), "area2"),
            ((100, 100), (258, 258), "area2"),
            ((100, 100), (256, 256), "area3"),
            ((200, 100), (256, 256), "area3"),
            ((400, 100), (256, 256), "area3"),
            ((500, 100), (257, 257), "area3"),
            ((500, 100), (256, 256), "area3"),
            ((600, 100), (256, 256), "area3"),
        ]
        mask_level_patches_after = [
            ((25, 25), (65, 65), "area2"),
            ((50, 25), (65, 65), "area3"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (65, 65), "area1"),
            ((50, 25), (64, 64), "area1"),
            ((50, 25), (65, 65), "area1"),
            ((50, 25), (66, 66), "area1"),
            ((75, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area2"),
            ((50, 25), (65, 65), "area2"),
            ((50, 25), (66, 66), "area2"),
            ((25, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area3"),
            ((50, 25), (64, 64), "area3"),
            ((100, 25), (64, 64), "area3"),
            ((125, 25), (65, 65), "area3"),
            ((125, 25), (64, 64), "area3"),
            ((150, 25), (64, 64), "area3"),
        ]
        patch_info_after = {"area1": 6, "area2": 5, "area3": 7}
        self.assertEqual(level0_patches_before, level0_patches_after)
        self.assertEqual(mask_level_patches_before, mask_level_patches_after)
        self.assertEqual(patch_info_before, patch_info_after)

    def test__exclude_duplicate_patches4(self):
        level0_patches_before = [
            ((100, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area2"),
            ((100, 100), (256, 256), "area3"),
            ((100, 100), (256, 256), "area4"),
            ((200, 100), (256, 256), "area5"),
            ((200, 100), (256, 256), "area6"),
            ((200, 100), (256, 256), "area7"),
            ((300, 100), (256, 256), "area8"),
            ((100, 100), (256, 256), "area9"),
            ((200, 100), (256, 256), "area10"),
            ((200, 100), (256, 256), "area11"),
            ((100, 100), (256, 256), "area12"),
            ((100, 100), (256, 256), "area13"),
            ((200, 100), (256, 256), "area14"),
            ((400, 100), (256, 256), "area15"),
            ((500, 100), (256, 256), "area16"),
            ((500, 100), (256, 256), "area17"),
            ((600, 100), (256, 256), "area18"),
        ]
        mask_level_patches_before = [
            ((25, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area3"),
            ((25, 25), (64, 64), "area4"),
            ((50, 25), (64, 64), "area5"),
            ((50, 25), (64, 64), "area6"),
            ((50, 25), (64, 64), "area7"),
            ((75, 25), (64, 64), "area8"),
            ((25, 25), (64, 64), "area9"),
            ((50, 25), (64, 64), "area10"),
            ((50, 25), (64, 64), "area11"),
            ((25, 25), (64, 64), "area12"),
            ((25, 25), (64, 64), "area13"),
            ((50, 25), (64, 64), "area14"),
            ((100, 25), (64, 64), "area15"),
            ((125, 25), (64, 64), "area16"),
            ((125, 25), (64, 64), "area17"),
            ((150, 25), (64, 64), "area18"),
        ]
        patch_info_before = {
            "area1": 1,
            "area2": 1,
            "area3": 1,
            "area4": 1,
            "area5": 1,
            "area6": 1,
            "area7": 1,
            "area8": 1,
            "area9": 1,
            "area10": 1,
            "area11": 1,
            "area12": 1,
            "area13": 1,
            "area14": 1,
            "area15": 1,
            "area16": 1,
            "area17": 1,
            "area18": 1,
        }
        wsi_name = "wsi_duplicates4.svs"
        PatchesTestMock._exclude_duplicate_patches(
            level0_patches_before, mask_level_patches_before, patch_info_before, wsi_name
        )
        level0_patches_after = [
            ((100, 100), (256, 256), "area1"),
            ((200, 100), (256, 256), "area2"),
            ((100, 100), (256, 256), "area3"),
            ((100, 100), (256, 256), "area4"),
            ((200, 100), (256, 256), "area5"),
            ((200, 100), (256, 256), "area6"),
            ((200, 100), (256, 256), "area7"),
            ((300, 100), (256, 256), "area8"),
            ((100, 100), (256, 256), "area9"),
            ((200, 100), (256, 256), "area10"),
            ((200, 100), (256, 256), "area11"),
            ((100, 100), (256, 256), "area12"),
            ((100, 100), (256, 256), "area13"),
            ((200, 100), (256, 256), "area14"),
            ((400, 100), (256, 256), "area15"),
            ((500, 100), (256, 256), "area16"),
            ((500, 100), (256, 256), "area17"),
            ((600, 100), (256, 256), "area18"),
        ]
        mask_level_patches_after = [
            ((25, 25), (64, 64), "area1"),
            ((50, 25), (64, 64), "area2"),
            ((25, 25), (64, 64), "area3"),
            ((25, 25), (64, 64), "area4"),
            ((50, 25), (64, 64), "area5"),
            ((50, 25), (64, 64), "area6"),
            ((50, 25), (64, 64), "area7"),
            ((75, 25), (64, 64), "area8"),
            ((25, 25), (64, 64), "area9"),
            ((50, 25), (64, 64), "area10"),
            ((50, 25), (64, 64), "area11"),
            ((25, 25), (64, 64), "area12"),
            ((25, 25), (64, 64), "area13"),
            ((50, 25), (64, 64), "area14"),
            ((100, 25), (64, 64), "area15"),
            ((125, 25), (64, 64), "area16"),
            ((125, 25), (64, 64), "area17"),
            ((150, 25), (64, 64), "area18"),
        ]
        patch_info_after = {
            "area1": 1,
            "area2": 1,
            "area3": 1,
            "area4": 1,
            "area5": 1,
            "area6": 1,
            "area7": 1,
            "area8": 1,
            "area9": 1,
            "area10": 1,
            "area11": 1,
            "area12": 1,
            "area13": 1,
            "area14": 1,
            "area15": 1,
            "area16": 1,
            "area17": 1,
            "area18": 1,
        }
        self.assertEqual(level0_patches_before, level0_patches_after)
        self.assertEqual(mask_level_patches_before, mask_level_patches_after)
        self.assertEqual(patch_info_before, patch_info_after)

    def test__exclude_duplicate_patches5(self):
        level0_patches_before = [
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area1"),
            ((100, 100), (256, 256), "area1"),
        ]
        mask_level_patches_before = [
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area1"),
            ((25, 25), (64, 64), "area1"),
        ]
        patch_info_before = {"area1": 5}
        wsi_name = "wsi_duplicates5.svs"
        PatchesTestMock._exclude_duplicate_patches(
            level0_patches_before, mask_level_patches_before, patch_info_before, wsi_name
        )
        level0_patches_after = [
            ((100, 100), (256, 256), "area1"),
        ]
        mask_level_patches_after = [
            ((25, 25), (64, 64), "area1"),
        ]
        patch_info_after = {"area1": 1}
        self.assertEqual(level0_patches_before, level0_patches_after)
        self.assertEqual(mask_level_patches_before, mask_level_patches_after)
        self.assertEqual(patch_info_before, patch_info_after)

    def test__draw_polygons(self):
        blank_image = Image.new(mode="RGB", size=(128, 196))
        # array copying required by this recent change in Pillow:
        # https://github.com/python-pillow/Pillow/issues/6581
        result_image_array = np.copy(np.asarray(blank_image))
        polygons_points = [
            [(10, 10), (10, 30), (30, 30), (30, 10)],
            [(60, 20), (60, 70), (100, 70), (100, 20)],
            [(30, 100), (110, 100), (110, 80), (30, 80)],
        ]
        color = "red"
        thickness = 1
        output_image_array = np.copy(np.asarray(blank_image))
        # XY coordinaes below are transposed (numpy notation)
        # poly1
        output_image_array[10, 10:31] = [255, 0, 0]
        output_image_array[30, 10:31] = [255, 0, 0]
        output_image_array[10:31, 10] = [255, 0, 0]
        output_image_array[10:31, 30] = [255, 0, 0]
        # poly2
        output_image_array[20, 60:101] = [255, 0, 0]
        output_image_array[70, 60:101] = [255, 0, 0]
        output_image_array[20:71, 60] = [255, 0, 0]
        output_image_array[20:71, 100] = [255, 0, 0]
        # poly3
        output_image_array[100, 30:111] = [255, 0, 0]
        output_image_array[80, 30:111] = [255, 0, 0]
        output_image_array[80:101, 30] = [255, 0, 0]
        output_image_array[80:101, 110] = [255, 0, 0]
        PatchesTestMock._draw_polygons(result_image_array, polygons_points, color, thickness)
        np.testing.assert_equal(result_image_array, output_image_array)

    def test__convert_bbox_to_point_list(self):
        bbox = [1, 2, 3, 4]
        output_point_list = [(1, 2), (3, 2), (3, 4), (1, 4)]
        result_point_list = PatchesTestMock._convert_bbox_to_point_list(bbox)
        self.assertEqual(result_point_list, output_point_list)

    def test__convert_patch_to_point_list(self):
        patch_location = (125, 80)
        mask_patch_size = 30
        output_point_list = [(125, 80), (125, 80 + 30), (125 + 30, 80 + 30), (125 + 30, 80)]
        result_point_list = PatchesTestMock._convert_patch_to_point_list(patch_location, mask_patch_size)
        self.assertEqual(result_point_list, output_point_list)

    def test__get_shapely_bbox_points(self):
        shapely_polygon = Polygon([(1, 2), (3, 2), (3, 4), (1, 4)])
        output_bbox = (1, 2, 3, 4)
        result_bbox = PatchesTestMock._get_shapely_bbox_points(shapely_polygon)
        self.assertEqual(tuple(result_bbox), output_bbox)
        #
        shapely_polygon = Polygon(
            [
                (0.1111111111111111111, 2),
                (3, 2),
                (3, 4.66666666666666666666),
                (0.1111111111111111111, 4.66666666666666666666),
            ]
        )
        output_bbox = (0.1111111111, 2, 3, 4.6666666667)
        result_bbox = PatchesTestMock._get_shapely_bbox_points(shapely_polygon)
        self.assertEqual(tuple(result_bbox), output_bbox)

    def test__is_foreground(self):
        # mask 100% full, threshold 0.8
        mask = np.ones((30, 30), dtype=int)
        threshold = 0.8
        result = PatchesTestMock._is_foreground(mask, threshold)
        self.assertTrue(result)
        # mask 100% full, threshold 1
        threshold = 1
        result = PatchesTestMock._is_foreground(mask, threshold)
        self.assertTrue(result)
        # mask 50% full, threshold 1
        mask[:16, :16] = 0
        threshold = 1
        result = PatchesTestMock._is_foreground(mask, threshold)
        self.assertFalse(result)
        # mask 50% full, threshold 0.25
        threshold = 0.25
        result = PatchesTestMock._is_foreground(mask, threshold)
        self.assertTrue(result)

    def test__is_overlapping(self):
        # two polygons with 25% overlap, each polygon has area equal 10
        shapely_polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
        mask_polygon = Polygon([(5, 5), (5, 15), (15, 15), (15, 5), (5, 5)])
        # exact overlapping (25% threshold)
        patch_overlap_area = 10 * 10 * 0.25
        result = PatchesTestMock._is_overlapping(shapely_polygon, mask_polygon, patch_overlap_area)
        self.assertTrue(result)
        # (24% overlap threshold)
        patch_overlap_area = 10 * 10 * 0.24
        result = PatchesTestMock._is_overlapping(shapely_polygon, mask_polygon, patch_overlap_area)
        self.assertTrue(result)
        # (26% overlap threshold)
        patch_overlap_area = 10 * 10 * 0.26
        result = PatchesTestMock._is_overlapping(shapely_polygon, mask_polygon, patch_overlap_area)
        self.assertFalse(result)

    def test_pad_mask_patch_array(self):
        # x dimension mismatch, square patch
        mask_patch_size = (30, 30)
        mask_patch = np.ones((10, 30), dtype=int)
        output_array = np.ones((30, 30), dtype=int)
        output_array[10:, ...] = 0
        result_array = PatchesTestMock._pad_mask_patch_array(mask_patch, mask_patch_size)
        np.testing.assert_equal(result_array, output_array)
        # y dimension mismatch, square patch
        mask_patch_size = (30, 30)
        mask_patch = np.ones((30, 15), dtype=int)
        output_array = np.ones((30, 30), dtype=int)
        output_array[..., 15:] = 0
        result_array = PatchesTestMock._pad_mask_patch_array(mask_patch, mask_patch_size)
        np.testing.assert_equal(result_array, output_array)
        # x and y mismatch, square patch
        mask_patch_size = (30, 30)
        mask_patch = np.ones((20, 10), dtype=int)
        output_array = np.ones((30, 30), dtype=int)
        output_array[20:, ...] = 0
        output_array[..., 10:] = 0
        result_array = PatchesTestMock._pad_mask_patch_array(mask_patch, mask_patch_size)
        np.testing.assert_equal(result_array, output_array)
        # both dimensions mismatch, non square patch
        mask_patch_size = (40, 30)
        mask_patch = np.ones((10, 20), dtype=int)
        output_array = np.ones((40, 30), dtype=int)
        output_array[10:, ...] = 0
        output_array[..., 20:] = 0
        result_array = PatchesTestMock._pad_mask_patch_array(mask_patch, mask_patch_size)
        np.testing.assert_equal(result_array, output_array)
        # both dimensions mismatch, non square patch
        mask_patch_size = (30, 40)
        mask_patch = np.ones((20, 10), dtype=int)
        output_array = np.ones((30, 40), dtype=int)
        output_array[20:, ...] = 0
        output_array[..., 10:] = 0
        result_array = PatchesTestMock._pad_mask_patch_array(mask_patch, mask_patch_size)
        np.testing.assert_equal(result_array, output_array)
        # both dimensions mismatch, non square patch
        mask_patch_size = (40, 30)
        mask_patch = np.ones((20, 10), dtype=int)
        output_array = np.ones((40, 30), dtype=int)
        output_array[20:, ...] = 0
        output_array[..., 10:] = 0
        result_array = PatchesTestMock._pad_mask_patch_array(mask_patch, mask_patch_size)
        np.testing.assert_equal(result_array, output_array)

    def test__is_patch_valid(self):
        shapely_polygon = Polygon([(10, 10), (10, 60), (60, 60), (60, 10), (10, 10)])
        shapely_polygon_bbox = shapely_polygon.bounds
        mask_patch_size = 15
        foreground_ratio = 1
        patch_location = (20, 40)
        full_overlap_check = True
        # empty mask case
        mask_array = np.zeros((100, 100), dtype=int)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 1
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        # other easy cases
        mask_array = np.ones((100, 100), dtype=int)
        foreground_ratio = 1
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 1
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        foreground_ratio = 0.5
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        foreground_ratio = 0.5
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 1
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        foreground_ratio = 1
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 1.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        #
        patch_location = (60, 80)
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        # 3 tests for border cases of overlapping area (created polygons overlap at 25%)
        patch_location = (35, 35)
        mask_patch_size = 50
        foreground_ratio = 0.1
        #
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.25
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.24
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.26
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        # 3 tests for border cases of foreground threshold (half of mask is empty)
        mask_array[..., 50:] = 0
        shapely_polygon = Polygon([(10, 10), (10, 70), (70, 70), (70, 10), (10, 10)])
        mask_patch_size = 20
        patch_location = (40, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.1
        foreground_ratio = 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        foreground_ratio = 0.49
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        foreground_ratio = 0.51
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        # bonus test (where foreground passes, shrink polygon and use higher overlap criteria to make it fail)
        shapely_polygon = Polygon([(10, 10), (10, 70), (50, 70), (50, 10), (10, 10)])
        foreground_ratio = 0.5
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.51
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        # finally, double border case (both overlap and foreground criteria barely pass):
        shapely_polygon = Polygon([(10, 10), (10, 70), (50, 70), (50, 10), (10, 10)])
        foreground_ratio = 0.50
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        # tests using full_overlap_check=False
        full_overlap_check = False
        mask_array = np.ones((100, 100), dtype=int)
        shapely_polygon = Polygon([(10, 10), (10, 70), (50, 70), (50, 10), (10, 10)])
        shapely_polygon_bbox = shapely_polygon.bounds
        foreground_ratio = 1
        mask_patch_size = 20
        #
        patch_location = (20, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (40, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (40, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.49
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (40, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.51
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        #
        patch_location = (20, 0)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (20, 0)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.49
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (20, 0)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.51
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        #
        patch_location = (0, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (0, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.49
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (0, 40)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.51
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)
        #
        patch_location = (20, 60)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.5
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (20, 60)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.49
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertTrue(result)
        #
        patch_location = (20, 60)
        mask_patch_overlap_area = mask_patch_size * mask_patch_size * 0.51
        result = PatchesTestMock._is_patch_valid(
            mask_array,
            mask_patch_size,
            patch_location,
            shapely_polygon,
            shapely_polygon_bbox,
            foreground_ratio,
            mask_patch_overlap_area,
            full_overlap_check,
        )
        self.assertFalse(result)

    def test__is_overlap_check_required(self):
        shapely_polygon_bbox = (0, 0, 100, 100)
        x1, y1, x2, y2 = (10, 10, 20, 20)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertFalse(result)
        #
        x1, y1, x2, y2 = (0, 0, 100, 100)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertFalse(result)
        #
        x1, y1, x2, y2 = (-10, 10, 20, 20)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertTrue(result)
        #
        x1, y1, x2, y2 = (10, -10, 20, 20)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertTrue(result)
        #
        x1, y1, x2, y2 = (10, 10, 120, 20)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertTrue(result)
        #
        x1, y1, x2, y2 = (10, 10, 20, 120)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertTrue(result)
        #
        x1, y1, x2, y2 = (-10, -10, 120, 120)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertTrue(result)
        #
        x1, y1, x2, y2 = (-10, -10, -20, -20)
        result = PatchesTestMock._is_overlap_check_required(shapely_polygon_bbox, x1, y1, x2, y2)
        self.assertTrue(result)


class TestPatchesBaseStaticMethodsPolygonChecks(TestCase):
    """Tests for static methods related to polygon checks in BasePatches class."""

    def setUp(self):
        # Set of polygons: all internal polygons must be valid (is_valid must return True)
        # Overlapping in this context means only external boundaries, overlapping via holes does not count
        #
        # Set #1: nested and overlapping polygons, but no multipolygons
        # contained polygons:
        poly1 = Polygon([(50, 50), (50, 1000), (2000, 1000), (2000, 50)])
        poly2a = Polygon([(100, 100), (100, 900), (900, 900), (900, 100)])
        poly2b = Polygon([(1000, 100), (1000, 900), (1900, 900), (1900, 100)])
        poly3a = Polygon([(200, 200), (200, 400), (500, 400), (500, 200)])
        poly3b = Polygon([(550, 200), (550, 400), (850, 400), (850, 200)])
        poly3c = Polygon([(1100, 200), (1100, 400), (1400, 400), (1400, 200)])
        poly3d = Polygon([(1500, 200), (1500, 400), (1800, 400), (1800, 200)])
        poly4a = Polygon([(250, 250), (250, 350), (300, 350), (300, 250)])
        poly4b = Polygon([(400, 250), (400, 350), (450, 350), (450, 250)])
        poly4c = Polygon([(600, 250), (600, 350), (650, 350), (650, 250)])
        poly4d = Polygon([(700, 250), (700, 350), (750, 350), (750, 250)])
        poly4e = Polygon([(1150, 250), (1150, 350), (1200, 350), (1200, 250)])
        poly4f = Polygon([(1250, 250), (1250, 350), (1300, 350), (1300, 250)])
        poly4g = Polygon([(1550, 250), (1550, 350), (1600, 350), (1600, 250)])
        poly4h = Polygon([(1650, 250), (1650, 350), (1700, 350), (1700, 250)])
        poly0 = Polygon([(20, 20), (20, 1020), (2020, 1020), (2020, 20)])
        # overlapping polygons:
        poly5 = Polygon([(3000, 50), (3000, 200), (3500, 200), (3500, 50)])
        poly6 = Polygon([(3200, 50), (3200, 200), (3700, 200), (3700, 50)])
        poly7 = Polygon([(3400, 50), (3400, 200), (3900, 200), (3900, 50)])
        poly8 = Polygon([(3600, 50), (3600, 200), (4100, 200), (4100, 50)])
        # neither contained nor overlapping
        poly9 = Polygon([(2500, 600), (2500, 700), (2900, 700), (2900, 600)])

        self.shapely_polygons1 = [
            poly1,  # 0
            poly2a,  # 1
            poly2b,  # 2
            poly3a,  # 3
            poly3b,  # 4
            poly3c,  # 5
            poly3d,  # 6
            poly4a,  # 7
            poly4b,  # 8
            poly4c,  # 9
            poly4d,  # 10
            poly4e,  # 11
            poly4f,  # 12
            poly4g,  # 13
            poly4h,  # 14
            poly0,  # 15
            poly5,  # 16
            poly6,  # 17
            poly7,  # 18
            poly8,  # 19
            poly9,  # 20
        ]

        self.contained_polygons1 = {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            1: [3, 4, 7, 8, 9, 10],
            2: [5, 6, 11, 12, 13, 14],
            3: [7, 8],
            4: [9, 10],
            5: [11, 12],
            6: [13, 14],
            15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        }

        self.overlapping_polygons1 = {16: [17, 18], 17: [18, 19], 18: [19]}

        #
        # Set #2: nested polygons and multipolygons, but no overlapping
        poly1 = Polygon([(50, 50), (50, 1000), (1000, 1000), (1000, 50)])
        poly2a = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        poly2b = Polygon([(400, 400), (400, 600), (600, 600), (600, 400)])
        poly2c_part1 = Polygon([(100, 700), (100, 900), (300, 900), (300, 700)])
        poly2c_part2 = Polygon([(400, 700), (400, 900), (600, 900), (600, 700)])
        poly2c_part3 = Polygon([(700, 700), (700, 900), (900, 900), (900, 700)])
        poly2c_part4 = Polygon([(700, 400), (700, 600), (900, 600), (900, 400)])
        poly2c = MultiPolygon([poly2c_part1, poly2c_part2, poly2c_part3, poly2c_part4])
        poly3e1 = Polygon([(150, 750), (150, 850), (250, 850), (250, 750)])
        poly3e1_1 = Polygon([(180, 780), (180, 830), (230, 830), (230, 780)])
        poly3e2 = Polygon([(750, 750), (750, 850), (850, 850), (850, 750)])
        poly3f_part1 = Polygon([(410, 710), (410, 880), (520, 880), (520, 710)])
        poly3f_part2 = Polygon([(530, 710), (530, 880), (550, 880), (550, 710)])
        poly3f_part3 = Polygon([(560, 710), (560, 880), (580, 880), (580, 710)])
        poly3f = MultiPolygon([poly3f_part1, poly3f_part2, poly3f_part3])
        poly3a = Polygon([(150, 150), (150, 200), (200, 200), (200, 150)])
        poly3b = Polygon([(210, 150), (210, 200), (260, 200), (260, 150)])
        poly3c = Polygon([(450, 450), (450, 500), (500, 500), (500, 450)])
        poly3d = Polygon([(510, 450), (510, 500), (560, 500), (560, 450)])
        poly4a = Polygon([(160, 160), (160, 180), (180, 180), (180, 160)])
        poly4b = Polygon([(220, 160), (220, 180), (240, 180), (240, 160)])
        poly4c = Polygon([(430, 730), (430, 760), (460, 760), (460, 730)])
        poly2d = Polygon([(500, 200), (500, 300), (900, 300), (900, 200)])

        self.shapely_polygons2 = [
            poly1,  # 0
            poly2a,  # 1
            poly2b,  # 2
            poly2c,  # 3
            poly3e1,  # 4
            poly3e1_1,  # 5
            poly3e2,  # 6
            poly3f,  # 7
            poly3a,  # 8
            poly3b,  # 9
            poly3c,  # 10
            poly3d,  # 11
            poly4a,  # 12
            poly4b,  # 13
            poly4c,  # 14
            poly2d,  # 15
        ]

        self.contained_polygons2 = {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [8, 9, 12, 13],
            2: [10, 11],
            3: [4, 5, 6, 7, 14],
            4: [5],
            7: [14],
            8: [12],
            9: [13],
        }

        self.overlapping_polygons2 = {}

        # Set #3: nested and overlapping polygons with holes
        # contained polygons:
        poly1 = Polygon(
            [(50, 50), (50, 1000), (2000, 1000), (2000, 50)], holes=[[(170, 950), (170, 980), (500, 980), (500, 950)]]
        )
        poly2a = Polygon(
            [(100, 100), (100, 900), (900, 900), (900, 100)],
            holes=[
                [(170, 700), (170, 800), (500, 800), (500, 700)],
                [(170, 500), (170, 600), (500, 600), (500, 500)],
                [(170, 370), (170, 450), (530, 450), (530, 370)],
            ],
        )
        poly2b = Polygon(
            [(1000, 100), (1000, 900), (1900, 900), (1900, 100)],
            holes=[
                [(1050, 700), (1050, 800), (1500, 800), (1500, 700)],
                [(1050, 600), (1050, 650), (1500, 650), (1500, 600)],
                [(1050, 500), (1050, 550), (1500, 550), (1500, 500)],
            ],
        )
        poly3a = Polygon([(200, 200), (200, 400), (500, 400), (500, 200)])
        poly3b = Polygon([(550, 200), (550, 400), (850, 400), (850, 200)])
        poly3c = Polygon([(1100, 200), (1100, 400), (1400, 400), (1400, 200)])
        poly3d = Polygon([(1500, 200), (1500, 400), (1800, 400), (1800, 200)])
        poly4a = Polygon([(250, 250), (250, 350), (300, 350), (300, 250)])
        poly4b = Polygon([(400, 250), (400, 350), (450, 350), (450, 250)])
        poly4c = Polygon([(600, 250), (600, 350), (650, 350), (650, 250)])
        poly4d = Polygon([(700, 250), (700, 350), (750, 350), (750, 250)])
        poly4e = Polygon([(1150, 250), (1150, 350), (1200, 350), (1200, 250)])
        poly4f = Polygon([(1250, 250), (1250, 350), (1300, 350), (1300, 250)])
        poly4g = Polygon([(1550, 250), (1550, 350), (1600, 350), (1600, 250)])
        poly4h = Polygon([(1650, 250), (1650, 350), (1700, 350), (1700, 250)])
        poly0 = Polygon([(20, 20), (20, 1020), (2020, 1020), (2020, 20)])
        # overlapping polygons:
        poly5 = Polygon([(3000, 50), (3000, 200), (3500, 200), (3500, 50)])
        poly6 = Polygon([(3200, 50), (3200, 200), (3700, 200), (3700, 50)])
        poly7 = Polygon([(3400, 50), (3400, 200), (3900, 200), (3900, 50)])
        poly8 = Polygon(
            [(3600, 50), (3600, 200), (4100, 200), (4100, 50)],
            holes=[[(3650, 300), (3650, 330), (3700, 330), (3700, 300)]],
        )
        # neither contained nor overlapping
        poly9 = Polygon(
            [(2500, 600), (2500, 700), (2900, 700), (2900, 600)],
            holes=[[(2600, 650), (2600, 680), (2650, 680), (2650, 650)]],
        )

        self.shapely_polygons3 = [
            poly1,  # 0
            poly2a,  # 1
            poly2b,  # 2
            poly3a,  # 3
            poly3b,  # 4
            poly3c,  # 5
            poly3d,  # 6
            poly4a,  # 7
            poly4b,  # 8
            poly4c,  # 9
            poly4d,  # 10
            poly4e,  # 11
            poly4f,  # 12
            poly4g,  # 13
            poly4h,  # 14
            poly0,  # 15
            poly5,  # 16
            poly6,  # 17
            poly7,  # 18
            poly8,  # 19
            poly9,  # 20
        ]

        self.contained_polygons3 = {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            1: [3, 4, 7, 8, 9, 10],
            2: [5, 6, 11, 12, 13, 14],
            3: [7, 8],
            4: [9, 10],
            5: [11, 12],
            6: [13, 14],
            15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        }

        self.overlapping_polygons3 = {16: [17, 18], 17: [18, 19], 18: [19]}

        # Set #4: nested polygons and multipolygons with overlapping and holes
        poly1 = Polygon([(50, 50), (50, 1000), (1000, 1000), (1000, 50)])
        poly2a = Polygon(
            [(100, 100), (100, 300), (300, 300), (300, 100)], holes=[[(150, 230), (150, 250), (200, 250), (200, 230)]]
        )
        poly2b = Polygon(
            [(400, 400), (400, 600), (600, 600), (600, 400)], holes=[[(470, 470), (470, 480), (480, 480), (480, 470)]]
        )
        poly2c_part1 = Polygon([(100, 700), (100, 900), (300, 900), (300, 700)])
        poly2c_part2 = Polygon([(400, 700), (400, 900), (600, 900), (600, 700)])
        poly2c_part3 = Polygon(
            [(700, 700), (700, 900), (900, 900), (900, 700)], holes=[[(780, 780), (780, 800), (800, 800), (800, 780)]]
        )
        poly2c_part4 = Polygon(
            [(700, 400), (700, 600), (900, 600), (900, 400)], holes=[[(750, 500), (750, 580), (800, 580), (800, 500)]]
        )
        poly2c = MultiPolygon([poly2c_part1, poly2c_part2, poly2c_part3, poly2c_part4])
        poly3e1 = Polygon([(150, 750), (150, 850), (250, 850), (250, 750)])
        poly3e1_1 = Polygon(
            [(180, 780), (180, 830), (230, 830), (230, 780)], holes=[[(200, 800), (200, 810), (210, 810), (210, 800)]]
        )
        poly3e2 = Polygon([(750, 750), (750, 850), (850, 850), (850, 750)])
        poly3f_part1 = Polygon(
            [(410, 710), (410, 880), (520, 880), (520, 710)], holes=[[(450, 800), (450, 810), (470, 810), (470, 800)]]
        )
        poly3f_part2 = Polygon([(530, 710), (530, 880), (550, 880), (550, 710)])
        poly3f_part3 = Polygon([(560, 710), (560, 880), (580, 880), (580, 710)])
        poly3f = MultiPolygon([poly3f_part1, poly3f_part2, poly3f_part3])
        poly3a = Polygon([(150, 150), (150, 200), (200, 200), (200, 150)])
        poly3b = Polygon([(210, 150), (210, 200), (260, 200), (260, 150)])
        poly3c = Polygon([(450, 450), (450, 500), (500, 500), (500, 450)])
        poly3d = Polygon([(510, 450), (510, 500), (560, 500), (560, 450)])
        poly4a = Polygon([(160, 160), (160, 180), (180, 180), (180, 160)])
        poly4b = Polygon([(220, 160), (220, 180), (240, 180), (240, 160)])
        poly4c = Polygon([(430, 730), (430, 760), (460, 760), (460, 730)])
        poly2d = Polygon(
            [(500, 200), (500, 300), (900, 300), (900, 200)], holes=[[(600, 250), (600, 280), (650, 280), (650, 250)]]
        )
        poly5_part1 = Polygon(
            [(850, 550), (850, 630), (930, 630), (930, 550)], holes=[[(860, 560), (860, 580), (880, 580), (880, 560)]]
        )
        poly5_part2 = Polygon([(920, 400), (920, 450), (940, 450), (940, 400)])
        poly5 = MultiPolygon([poly5_part1, poly5_part2])

        self.shapely_polygons4 = [
            poly1,  # 0
            poly2a,  # 1
            poly2b,  # 2
            poly2c,  # 3
            poly3e1,  # 4
            poly3e1_1,  # 5
            poly3e2,  # 6
            poly3f,  # 7
            poly3a,  # 8
            poly3b,  # 9
            poly3c,  # 10
            poly3d,  # 11
            poly4a,  # 12
            poly4b,  # 13
            poly4c,  # 14
            poly2d,  # 15
            poly5,  # 16
        ]

        self.contained_polygons4 = {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            1: [8, 9, 12, 13],
            2: [10, 11],
            3: [4, 5, 6, 7, 14],
            4: [5],
            7: [14],
            8: [12],
            9: [13],
        }

        self.overlapping_polygons4 = {3: [16]}

    def test__run_polygons_check(self):
        mask_patch_size = 10
        polygons_overlap_threshold = 10
        # independent polygons with holes
        poly1 = Polygon(
            shell=[(1200, 50), (1200, 200), (1700, 200), (1700, 50)],
            holes=[[(1300, 100), (1300, 150), (1400, 150), (1400, 100)]],
        )
        poly2 = Polygon(
            shell=[(3200, 50), (3200, 200), (3700, 200), (3700, 50)],
            holes=[[(3300, 100), (3300, 150), (3400, 150), (3400, 100)]],
        )
        shapely_polygons = [poly1, poly2]
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide1")
        self.assertTrue(poly1.equals(shapely_polygons[0]))
        self.assertTrue(poly2.equals(shapely_polygons[1]))
        # single multipolygon with holes
        poly1 = Polygon([(2000, 50), (2000, 200), (2500, 200), (2500, 50)])
        poly2 = Polygon(
            shell=[(3200, 50), (3200, 200), (3700, 200), (3700, 50)],
            holes=[[(3300, 100), (3300, 150), (3400, 150), (3400, 100)]],
        )
        multipolygon = MultiPolygon([poly1, poly2])
        shapely_polygons = [multipolygon]
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide2")
        self.assertTrue(multipolygon.equals(shapely_polygons[0]))
        # overlapping polygons without (will generate 5 log entries):
        poly1 = Polygon([(3000, 50), (3000, 200), (3500, 200), (3500, 50)])
        poly2 = Polygon([(3200, 50), (3200, 200), (3700, 200), (3700, 50)])
        poly3 = Polygon([(3400, 50), (3400, 200), (3900, 200), (3900, 50)])
        poly4 = Polygon([(3600, 50), (3600, 200), (4100, 200), (4100, 50)])
        shapely_polygons = [poly1, poly2, poly3, poly4]
        polygons_overlap_threshold = 150
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide3")
        # embedded polygons without holes (one contains another)
        poly1 = Polygon([(50, 50), (50, 1000), (1000, 1000), (1000, 50)])
        poly2 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        shapely_polygons = [poly1, poly2]
        polygons_overlap_threshold = 10
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide4")
        self.assertFalse(poly1.equals(shapely_polygons[0]))
        self.assertTrue(poly2.equals(shapely_polygons[1]))
        #
        # embedded polygons, outer polygon with holes
        poly1 = Polygon(
            [(50, 50), (50, 1000), (1000, 1000), (1000, 50)], holes=[[(500, 500), (500, 600), (600, 600), (600, 500)]]
        )
        poly2 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        shapely_polygons = [poly1, poly2]
        polygons_overlap_threshold = 10
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide5")
        self.assertFalse(poly1.equals(shapely_polygons[0]))
        self.assertTrue(poly2.equals(shapely_polygons[1]))
        # embedded polygons, both with holes
        poly1 = Polygon(
            [(50, 50), (50, 1000), (1000, 1000), (1000, 50)], holes=[[(500, 500), (500, 600), (600, 600), (600, 500)]]
        )
        poly2 = Polygon(
            [(100, 100), (100, 300), (300, 300), (300, 100)], holes=[[(200, 200), (200, 250), (250, 250), (250, 200)]]
        )
        shapely_polygons = [poly1, poly2]
        polygons_overlap_threshold = 10
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide6")
        self.assertFalse(poly1.equals(shapely_polygons[0]))
        self.assertTrue(poly2.equals(shapely_polygons[1]))
        # embedded multipolygons with holes
        poly1 = Polygon([(2000, 50), (2000, 500), (2500, 500), (2500, 50)])
        poly2 = Polygon(
            shell=[(3200, 50), (3200, 200), (3700, 200), (3700, 50)],
            holes=[[(3300, 100), (3300, 150), (3400, 150), (3400, 100)]],
        )
        poly3 = Polygon([(2100, 100), (2100, 150), (2150, 150), (2150, 100)])
        poly4 = Polygon(
            shell=[(2100, 200), (2100, 400), (2400, 400), (2400, 200)],
            holes=[[(2150, 250), (2150, 300), (2300, 300), (2300, 250)]],
        )
        multipolygon1 = MultiPolygon([poly1, poly2])
        multipolygon2 = MultiPolygon([poly3, poly4])
        shapely_polygons = [multipolygon1, multipolygon2]
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide7")
        self.assertFalse(multipolygon1.equals(shapely_polygons[0]))
        self.assertTrue(multipolygon2.equals(shapely_polygons[1]))
        # embedded polygons with more holes
        poly1 = Polygon(
            [(50, 50), (50, 1000), (1000, 1000), (1000, 50)], holes=[[(500, 500), (500, 600), (600, 600), (600, 500)]]
        )
        poly2 = Polygon(
            [(100, 100), (100, 300), (300, 300), (300, 100)],
            holes=[
                [(200, 200), (200, 250), (250, 250), (250, 200)],
                [(15, 20), (15, 25), (25, 25), (25, 20)],
            ],
        )
        shapely_polygons = [poly1, poly2]
        polygons_overlap_threshold = 10
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide8")
        self.assertFalse(poly1.equals(shapely_polygons[0]))
        self.assertTrue(poly2.equals(shapely_polygons[1]))
        # embedded multipolygons with more holes
        poly1 = Polygon([(2000, 50), (2000, 500), (2500, 500), (2500, 50)])
        poly2 = Polygon(
            shell=[(3200, 50), (3200, 200), (3700, 200), (3700, 50)],
            holes=[
                [(3300, 100), (3300, 150), (3400, 150), (3400, 100)],
                [(3300, 160), (3300, 170), (3400, 170), (3400, 160)],
            ],
        )
        poly3 = Polygon([(2100, 100), (2100, 150), (2150, 150), (2150, 100)])
        poly4 = Polygon(
            shell=[(2100, 200), (2100, 400), (2400, 400), (2400, 200)],
            holes=[
                [(2150, 250), (2150, 300), (2300, 300), (2300, 250)],
                [(2180, 210), (2180, 220), (2320, 220), (2320, 210)],
            ],
        )
        multipolygon1 = MultiPolygon([poly1, poly2])
        multipolygon2 = MultiPolygon([poly3, poly4])
        shapely_polygons = [multipolygon1, multipolygon2]
        PatchesTestMock._run_polygons_check(shapely_polygons, mask_patch_size, polygons_overlap_threshold, "slide9")
        self.assertFalse(multipolygon1.equals(shapely_polygons[0]))
        self.assertTrue(multipolygon2.equals(shapely_polygons[1]))

    def test__get_contained_and_overlapping_polygons(self):
        # set #1
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(
            self.shapely_polygons1
        )
        self.assertEqual(result_contained, self.contained_polygons1)
        self.assertEqual(result_overlapping, self.overlapping_polygons1)
        # set #2
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(
            self.shapely_polygons2
        )
        self.assertEqual(result_contained, self.contained_polygons2)
        self.assertEqual(result_overlapping, self.overlapping_polygons2)
        # set #3
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(
            self.shapely_polygons3
        )
        self.assertEqual(result_contained, self.contained_polygons3)
        self.assertEqual(result_overlapping, self.overlapping_polygons3)
        # set #4
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(
            self.shapely_polygons4
        )
        self.assertEqual(result_contained, self.contained_polygons4)
        self.assertEqual(result_overlapping, self.overlapping_polygons4)

        # two independent polygons
        poly1 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        poly2 = Polygon([(400, 400), (400, 600), (600, 600), (600, 400)])
        shapely_polygons = [poly1, poly2]
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(shapely_polygons)
        self.assertEqual(result_contained, {})
        self.assertEqual(result_overlapping, {})
        # two polygons, one with hole, contained
        poly1 = Polygon(
            [(100, 100), (100, 500), (500, 500), (500, 100)], holes=[[(250, 250), (250, 300), (300, 300), (300, 250)]]
        )
        poly2 = Polygon([(200, 200), (200, 400), (400, 400), (400, 200)])
        shapely_polygons = [poly1, poly2]
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(shapely_polygons)
        self.assertEqual(result_contained, {0: [1]})
        self.assertEqual(result_overlapping, {})
        # two polygons, one with hole, overlapping
        poly1 = Polygon(
            [(100, 100), (100, 500), (500, 500), (500, 100)], holes=[[(250, 250), (250, 300), (300, 300), (300, 250)]]
        )
        poly2 = Polygon([(200, 200), (200, 400), (800, 400), (800, 200)])
        shapely_polygons = [poly1, poly2]
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(shapely_polygons)
        self.assertEqual(result_contained, {})
        self.assertEqual(result_overlapping, {0: [1]})
        # multipolygon with hole and polygon, contained
        poly1a = Polygon(
            [(100, 100), (100, 500), (500, 500), (500, 100)], holes=[[(250, 250), (250, 300), (300, 300), (300, 250)]]
        )
        poly1b = Polygon([(10, 10), (10, 50), (50, 50), (50, 10)])
        multipolygon = MultiPolygon([poly1a, poly1b])
        poly2 = Polygon([(200, 200), (200, 400), (400, 400), (400, 200)])
        shapely_polygons = [multipolygon, poly2]
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(shapely_polygons)
        self.assertEqual(result_contained, {0: [1]})
        self.assertEqual(result_overlapping, {})
        # multipolygon with hole and polygon, overlapping
        poly1a = Polygon(
            [(100, 100), (100, 500), (500, 500), (500, 100)], holes=[[(250, 250), (250, 300), (300, 300), (300, 250)]]
        )
        poly1b = Polygon([(10, 10), (10, 50), (50, 50), (50, 10)])
        multipolygon = MultiPolygon([poly1a, poly1b])
        poly2 = Polygon([(200, 200), (200, 400), (800, 400), (00, 200)])
        shapely_polygons = [multipolygon, poly2]
        result_contained, result_overlapping = PatchesTestMock._get_contained_and_overlapping_polygons(shapely_polygons)
        self.assertEqual(result_contained, {})
        self.assertEqual(result_overlapping, {0: [1]})

    def test__replace_polygons(self):
        # set #1
        polygons_before = copy.deepcopy(self.shapely_polygons1)
        PatchesTestMock._replace_polygons(self.shapely_polygons1, self.contained_polygons1, "slide1")
        self.assertFalse(polygons_before[0].equals(self.shapely_polygons1[0]))
        self.assertFalse(polygons_before[1].equals(self.shapely_polygons1[1]))
        self.assertFalse(polygons_before[2].equals(self.shapely_polygons1[2]))
        self.assertFalse(polygons_before[3].equals(self.shapely_polygons1[3]))
        self.assertFalse(polygons_before[4].equals(self.shapely_polygons1[4]))
        self.assertFalse(polygons_before[5].equals(self.shapely_polygons1[5]))
        self.assertFalse(polygons_before[6].equals(self.shapely_polygons1[6]))
        self.assertTrue(polygons_before[7].equals(self.shapely_polygons1[7]))
        self.assertTrue(polygons_before[8].equals(self.shapely_polygons1[8]))
        self.assertTrue(polygons_before[9].equals(self.shapely_polygons1[9]))
        self.assertTrue(polygons_before[10].equals(self.shapely_polygons1[10]))
        self.assertTrue(polygons_before[11].equals(self.shapely_polygons1[11]))
        self.assertTrue(polygons_before[12].equals(self.shapely_polygons1[12]))
        self.assertTrue(polygons_before[13].equals(self.shapely_polygons1[13]))
        self.assertTrue(polygons_before[14].equals(self.shapely_polygons1[14]))
        self.assertFalse(polygons_before[15].equals(self.shapely_polygons1[15]))
        self.assertTrue(polygons_before[16].equals(self.shapely_polygons1[16]))
        self.assertTrue(polygons_before[17].equals(self.shapely_polygons1[17]))
        self.assertTrue(polygons_before[18].equals(self.shapely_polygons1[18]))
        self.assertTrue(polygons_before[19].equals(self.shapely_polygons1[19]))
        self.assertTrue(polygons_before[20].equals(self.shapely_polygons1[20]))
        # set #2
        polygons_before = copy.deepcopy(self.shapely_polygons2)
        PatchesTestMock._replace_polygons(self.shapely_polygons2, self.contained_polygons2, "slide2")
        self.assertFalse(polygons_before[0].equals(self.shapely_polygons2[0]))
        self.assertFalse(polygons_before[1].equals(self.shapely_polygons2[1]))
        self.assertFalse(polygons_before[2].equals(self.shapely_polygons2[2]))
        self.assertFalse(polygons_before[3].equals(self.shapely_polygons2[3]))
        self.assertFalse(polygons_before[4].equals(self.shapely_polygons2[4]))
        self.assertTrue(polygons_before[5].equals(self.shapely_polygons2[5]))
        self.assertTrue(polygons_before[6].equals(self.shapely_polygons2[6]))
        self.assertFalse(polygons_before[7].equals(self.shapely_polygons2[7]))
        self.assertFalse(polygons_before[8].equals(self.shapely_polygons2[8]))
        self.assertFalse(polygons_before[9].equals(self.shapely_polygons2[9]))
        self.assertTrue(polygons_before[10].equals(self.shapely_polygons2[10]))
        self.assertTrue(polygons_before[11].equals(self.shapely_polygons2[11]))
        self.assertTrue(polygons_before[12].equals(self.shapely_polygons2[12]))
        self.assertTrue(polygons_before[13].equals(self.shapely_polygons2[13]))
        self.assertTrue(polygons_before[14].equals(self.shapely_polygons2[14]))
        self.assertTrue(polygons_before[15].equals(self.shapely_polygons2[15]))
        # set #3
        polygons_before = copy.deepcopy(self.shapely_polygons3)
        PatchesTestMock._replace_polygons(self.shapely_polygons3, self.contained_polygons3, "slide3")
        self.assertFalse(polygons_before[0].equals(self.shapely_polygons3[0]))
        self.assertFalse(polygons_before[1].equals(self.shapely_polygons3[1]))
        self.assertFalse(polygons_before[2].equals(self.shapely_polygons3[2]))
        self.assertFalse(polygons_before[3].equals(self.shapely_polygons3[3]))
        self.assertFalse(polygons_before[4].equals(self.shapely_polygons3[4]))
        self.assertFalse(polygons_before[5].equals(self.shapely_polygons3[5]))
        self.assertFalse(polygons_before[6].equals(self.shapely_polygons3[6]))
        self.assertTrue(polygons_before[7].equals(self.shapely_polygons3[7]))
        self.assertTrue(polygons_before[8].equals(self.shapely_polygons3[8]))
        self.assertTrue(polygons_before[9].equals(self.shapely_polygons3[9]))
        self.assertTrue(polygons_before[10].equals(self.shapely_polygons3[10]))
        self.assertTrue(polygons_before[11].equals(self.shapely_polygons3[11]))
        self.assertTrue(polygons_before[12].equals(self.shapely_polygons3[12]))
        self.assertTrue(polygons_before[13].equals(self.shapely_polygons3[13]))
        self.assertTrue(polygons_before[14].equals(self.shapely_polygons3[14]))
        self.assertFalse(polygons_before[15].equals(self.shapely_polygons3[15]))
        self.assertTrue(polygons_before[16].equals(self.shapely_polygons3[16]))
        self.assertTrue(polygons_before[17].equals(self.shapely_polygons3[17]))
        self.assertTrue(polygons_before[18].equals(self.shapely_polygons3[18]))
        self.assertTrue(polygons_before[19].equals(self.shapely_polygons3[19]))
        self.assertTrue(polygons_before[20].equals(self.shapely_polygons3[20]))
        # set #4
        polygons_before = copy.deepcopy(self.shapely_polygons4)
        PatchesTestMock._replace_polygons(self.shapely_polygons4, self.contained_polygons4, "slide4")
        self.assertFalse(polygons_before[0].equals(self.shapely_polygons4[0]))
        self.assertFalse(polygons_before[1].equals(self.shapely_polygons4[1]))
        self.assertFalse(polygons_before[2].equals(self.shapely_polygons4[2]))
        self.assertFalse(polygons_before[3].equals(self.shapely_polygons4[3]))
        self.assertFalse(polygons_before[4].equals(self.shapely_polygons4[4]))
        self.assertTrue(polygons_before[5].equals(self.shapely_polygons4[5]))
        self.assertTrue(polygons_before[6].equals(self.shapely_polygons4[6]))
        self.assertFalse(polygons_before[7].equals(self.shapely_polygons4[7]))
        self.assertFalse(polygons_before[8].equals(self.shapely_polygons4[8]))
        self.assertFalse(polygons_before[9].equals(self.shapely_polygons4[9]))
        self.assertTrue(polygons_before[10].equals(self.shapely_polygons4[10]))
        self.assertTrue(polygons_before[11].equals(self.shapely_polygons4[11]))
        self.assertTrue(polygons_before[12].equals(self.shapely_polygons4[12]))
        self.assertTrue(polygons_before[13].equals(self.shapely_polygons4[13]))
        self.assertTrue(polygons_before[14].equals(self.shapely_polygons4[14]))
        self.assertTrue(polygons_before[15].equals(self.shapely_polygons4[15]))
        self.assertTrue(polygons_before[16].equals(self.shapely_polygons4[16]))
        # poly1 is invalid (overlapping holes)
        poly1 = Polygon(
            [(100, 100), (100, 900), (900, 900), (900, 100)],
            holes=[[(170, 700), (170, 800), (500, 800), (500, 700)], [(270, 700), (270, 800), (500, 800), (500, 700)]],
        )
        poly2 = Polygon([(170, 200), (170, 300), (500, 300), (500, 200)])
        shapely_polygons = [poly1, poly2]
        contained_polygons = {0: [1]}
        polygons_before = copy.deepcopy(shapely_polygons)
        PatchesTestMock._replace_polygons(shapely_polygons, contained_polygons, "slide5")
        # comparison between poly1 and shapely_polygons[0] is not possible as poly1 is invalid
        self.assertTrue(shapely_polygons[0].is_valid)
        self.assertTrue(polygons_before[1].equals(shapely_polygons[1]))
        # no replacement, polygons are not related
        poly1 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        poly2 = Polygon([(400, 400), (400, 600), (600, 600), (600, 400)])
        shapely_polygons = [poly1, poly2]
        contained_polygons = {}
        polygons_before = copy.deepcopy(shapely_polygons)
        PatchesTestMock._replace_polygons(shapely_polygons, contained_polygons, "slide6")
        self.assertTrue(polygons_before[0].equals(shapely_polygons[0]))
        self.assertTrue(polygons_before[1].equals(shapely_polygons[1]))
        # two identical polygons (holes will be added and then removing by make_valid)
        poly1 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        poly2 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        shapely_polygons = [poly1, poly2]
        contained_polygons = {0: [1], 1: [0]}
        polygons_before = copy.deepcopy(shapely_polygons)
        PatchesTestMock._replace_polygons(shapely_polygons, contained_polygons, "slide7")
        self.assertTrue(polygons_before[0].equals(shapely_polygons[0]))
        self.assertTrue(polygons_before[1].equals(shapely_polygons[1]))

    @patch("dplabtools.slides.patches.locations.base.BasePatches._get_replacement_polygon")
    def test__replace_polygons_type_error1(self, mock_func):
        poly1 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        poly2 = Polygon([(400, 400), (400, 600), (600, 600), (600, 400)])
        shapely_polygons = [poly1, poly2]
        contained_polygons = {0: [1]}
        invalid_polygon = Polygon([(50, 50), (50, 100), (70, 70), (30, 50), (20, 70)])
        # this will return MultiPolygon
        mock_func.return_value = make_valid(invalid_polygon)
        with self.assertRaises(TypeError):
            PatchesTestMock._replace_polygons(shapely_polygons, contained_polygons, "slide1")

    @patch("dplabtools.slides.patches.locations.base.BasePatches._get_replacement_multipolygon")
    def test__replace_polygons_type_error2(self, mock_func):
        poly1 = Polygon([(100, 100), (100, 300), (300, 300), (300, 100)])
        poly2 = Polygon([(400, 400), (400, 600), (600, 600), (600, 400)])
        poly3 = Polygon([(700, 700), (700, 900), (900, 900), (900, 700)])
        shapely_polygons = [MultiPolygon([poly1, poly2]), poly3]
        contained_polygons = {0: [1]}
        polygon = Polygon([(50, 50), (50, 100), (100, 100), (100, 50)])
        # this will return Polygon
        mock_func.return_value = polygon
        with self.assertRaises(TypeError):
            PatchesTestMock._replace_polygons(shapely_polygons, contained_polygons, "slide2")

    def test__get_replacement_polygon(self):
        # polygon_index (parent) must refer to Polygon, not MultiPolygon
        #
        # using set #1
        polygon_index = 0
        output_polygon = Polygon(
            shell=[(50, 50), (50, 1000), (2000, 1000), (2000, 50)],
            holes=[
                [(100, 100), (100, 900), (900, 900), (900, 100)],
                [(1000, 100), (1000, 900), (1900, 900), (1900, 100)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons1, polygon_index, self.contained_polygons1
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 2
        output_polygon = Polygon(
            shell=[(1000, 100), (1000, 900), (1900, 900), (1900, 100)],
            holes=[
                [(1100, 200), (1100, 400), (1400, 400), (1400, 200)],
                [(1500, 200), (1500, 400), (1800, 400), (1800, 200)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons1, polygon_index, self.contained_polygons1
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 6
        output_polygon = Polygon(
            shell=[(1500, 200), (1500, 400), (1800, 400), (1800, 200)],
            holes=[
                [(1550, 250), (1550, 350), (1600, 350), (1600, 250)],
                [(1650, 250), (1650, 350), (1700, 350), (1700, 250)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons1, polygon_index, self.contained_polygons1
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 15
        output_polygon = Polygon(
            shell=[(20, 20), (20, 1020), (2020, 1020), (2020, 20)],
            holes=[[(50, 50), (50, 1000), (2000, 1000), (2000, 50)]],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons1, polygon_index, self.contained_polygons1
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        # using set #2
        polygon_index = 0
        output_polygon = Polygon(
            shell=[(50, 50), (50, 1000), (1000, 1000), (1000, 50)],
            holes=[
                [(100, 100), (100, 300), (300, 300), (300, 100)],
                [(400, 400), (400, 600), (600, 600), (600, 400)],
                [(100, 700), (100, 900), (300, 900), (300, 700)],
                [(400, 700), (400, 900), (600, 900), (600, 700)],
                [(700, 700), (700, 900), (900, 900), (900, 700)],
                [(700, 400), (700, 600), (900, 600), (900, 400)],
                [(500, 200), (500, 300), (900, 300), (900, 200)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons2, polygon_index, self.contained_polygons2
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 1
        output_polygon = Polygon(
            shell=[(100, 100), (100, 300), (300, 300), (300, 100)],
            holes=[
                [(150, 150), (150, 200), (200, 200), (200, 150)],
                [(210, 150), (210, 200), (260, 200), (260, 150)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons2, polygon_index, self.contained_polygons2
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 2
        output_polygon = Polygon(
            shell=[(400, 400), (400, 600), (600, 600), (600, 400)],
            holes=[
                [(450, 450), (450, 500), (500, 500), (500, 450)],
                [(510, 450), (510, 500), (560, 500), (560, 450)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons2, polygon_index, self.contained_polygons2
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        # using set #3
        polygon_index = 0
        output_polygon = Polygon(
            shell=[(50, 50), (50, 1000), (2000, 1000), (2000, 50)],
            holes=[
                [(100, 100), (100, 900), (900, 900), (900, 100)],
                [(1000, 100), (1000, 900), (1900, 900), (1900, 100)],
                [(170, 950), (170, 980), (500, 980), (500, 950)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons3, polygon_index, self.contained_polygons3
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 1
        output_polygon = Polygon(
            shell=[(100, 100), (100, 900), (900, 900), (900, 100)],
            holes=[
                [
                    (500, 200),
                    (200, 200),
                    (200, 370),
                    (170, 370),
                    (170, 450),
                    (530, 450),
                    (530, 370),
                    (500, 370),
                    (500, 200),
                ],
                [(850, 400), (850, 200), (550, 200), (550, 400), (850, 400)],
                [(500, 500), (170, 500), (170, 600), (500, 600), (500, 500)],
                [(500, 700), (170, 700), (170, 800), (500, 800), (500, 700)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons3, polygon_index, self.contained_polygons3
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 2
        output_polygon = Polygon(
            shell=[(1000, 100), (1000, 900), (1900, 900), (1900, 100)],
            holes=[
                [(1050, 700), (1050, 800), (1500, 800), (1500, 700)],
                [(1050, 600), (1050, 650), (1500, 650), (1500, 600)],
                [(1050, 500), (1050, 550), (1500, 550), (1500, 500)],
                [(1100, 200), (1100, 400), (1400, 400), (1400, 200)],
                [(1500, 200), (1500, 400), (1800, 400), (1800, 200)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons3, polygon_index, self.contained_polygons3
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 3
        output_polygon = Polygon(
            shell=[(200, 200), (200, 400), (500, 400), (500, 200)],
            holes=[[(250, 250), (250, 350), (300, 350), (300, 250)], [(400, 250), (400, 350), (450, 350), (450, 250)]],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons3, polygon_index, self.contained_polygons3
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 6
        output_polygon = Polygon(
            shell=[(1500, 200), (1500, 400), (1800, 400), (1800, 200)],
            holes=[
                [(1550, 250), (1550, 350), (1600, 350), (1600, 250)],
                [(1650, 250), (1650, 350), (1700, 350), (1700, 250)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons3, polygon_index, self.contained_polygons3
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        # using set #4
        polygon_index = 0
        output_polygon = Polygon(
            shell=[(50, 50), (50, 1000), (1000, 1000), (1000, 50)],
            holes=[
                [(300, 100), (100, 100), (100, 300), (300, 300), (300, 100)],
                [(900, 200), (500, 200), (500, 300), (900, 300), (900, 200)],
                [(940, 400), (920, 400), (920, 450), (940, 450), (940, 400)],
                [(600, 400), (400, 400), (400, 600), (600, 600), (600, 400)],
                [
                    (850, 630),
                    (930, 630),
                    (930, 550),
                    (900, 550),
                    (900, 400),
                    (700, 400),
                    (700, 600),
                    (850, 600),
                    (850, 630),
                ],
                [(300, 700), (100, 700), (100, 900), (300, 900), (300, 700)],
                [(400, 700), (400, 900), (600, 900), (600, 700), (400, 700)],
                [(700, 700), (700, 900), (900, 900), (900, 700), (700, 700)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons4, polygon_index, self.contained_polygons4
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 1
        output_polygon = Polygon(
            shell=[(100, 100), (100, 300), (300, 300), (300, 100)],
            holes=[
                [(150, 150), (150, 200), (200, 200), (200, 150)],
                [(210, 150), (210, 200), (260, 200), (260, 150)],
                [(150, 230), (150, 250), (200, 250), (200, 230)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons4, polygon_index, self.contained_polygons4
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 2
        output_polygon = Polygon(
            shell=[(400, 400), (400, 600), (600, 600), (600, 400)],
            holes=[
                [(450, 450), (450, 500), (500, 500), (500, 450)],
                [(510, 450), (510, 500), (560, 500), (560, 450)],
            ],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons4, polygon_index, self.contained_polygons4
        )
        self.assertTrue(result_polygon.equals(output_polygon))
        #
        polygon_index = 4
        output_polygon = Polygon(
            shell=[(150, 750), (150, 850), (250, 850), (250, 750)],
            holes=[[(180, 780), (180, 830), (230, 830), (230, 780)]],
        )
        result_polygon = PatchesTestMock._get_replacement_polygon(
            self.shapely_polygons4, polygon_index, self.contained_polygons4
        )
        self.assertTrue(result_polygon.equals(output_polygon))

    def test__get_replacement_multipolygon(self):
        # polygon_index (parent) must refer to MultiPolygon, not Polygon
        #
        # using set #2
        polygon_index = 3
        output_polygon_part1 = Polygon(
            shell=[(100, 700), (100, 900), (300, 900), (300, 700)],
            holes=[
                [(150, 750), (150, 850), (250, 850), (250, 750)],
            ],
        )
        output_polygon_part2 = Polygon(
            shell=[(400, 700), (400, 900), (600, 900), (600, 700)],
            holes=[
                [(410, 710), (410, 880), (520, 880), (520, 710)],
                [(530, 710), (530, 880), (550, 880), (550, 710)],
                [(560, 710), (560, 880), (580, 880), (580, 710)],
            ],
        )
        output_polygon_part3 = Polygon(
            shell=[(700, 700), (700, 900), (900, 900), (900, 700)],
            holes=[
                [(750, 750), (750, 850), (850, 850), (850, 750)],
            ],
        )
        output_polygon_part4 = Polygon(
            shell=[(700, 400), (700, 600), (900, 600), (900, 400)],
            holes=[],
        )
        output_multipolygon = MultiPolygon(
            [output_polygon_part1, output_polygon_part2, output_polygon_part3, output_polygon_part4]
        )
        result_multipolygon = PatchesTestMock._get_replacement_multipolygon(
            self.shapely_polygons2, polygon_index, self.contained_polygons2
        )
        self.assertTrue(result_multipolygon.equals(output_multipolygon))
        #
        polygon_index = 7
        output_polygon_part1 = Polygon(
            shell=[(410, 710), (410, 880), (520, 880), (520, 710)],
            holes=[
                [(430, 730), (430, 760), (460, 760), (460, 730)],
            ],
        )
        output_polygon_part2 = Polygon(
            shell=[(530, 710), (530, 880), (550, 880), (550, 710)],
            holes=[],
        )
        output_polygon_part3 = Polygon(
            shell=[(560, 710), (560, 880), (580, 880), (580, 710)],
            holes=[],
        )
        output_multipolygon = MultiPolygon([output_polygon_part1, output_polygon_part2, output_polygon_part3])
        result_multipolygon = PatchesTestMock._get_replacement_multipolygon(
            self.shapely_polygons2, polygon_index, self.contained_polygons2
        )
        self.assertTrue(result_multipolygon.equals(output_multipolygon))
        # using set #4
        polygon_index = 3
        output_polygon_part1 = Polygon(
            shell=[(100, 700), (100, 900), (300, 900), (300, 700)],
            holes=[
                [(150, 750), (150, 850), (250, 850), (250, 750)],
            ],
        )
        output_polygon_part2 = Polygon(
            shell=[(400, 700), (400, 900), (600, 900), (600, 700)],
            holes=[
                [(410, 710), (410, 880), (520, 880), (520, 710)],
                [(530, 710), (530, 880), (550, 880), (550, 710)],
                [(560, 710), (560, 880), (580, 880), (580, 710)],
            ],
        )
        output_polygon_part3 = Polygon(
            shell=[(700, 700), (700, 900), (900, 900), (900, 700)],
            holes=[
                [(750, 750), (750, 850), (850, 850), (850, 750)],
            ],
        )
        output_polygon_part4 = Polygon(
            shell=[(700, 400), (700, 600), (900, 600), (900, 400)],
            holes=[[(750, 500), (750, 580), (800, 580), (800, 500)]],
        )
        output_multipolygon = MultiPolygon(
            [output_polygon_part1, output_polygon_part2, output_polygon_part3, output_polygon_part4]
        )
        result_multipolygon = PatchesTestMock._get_replacement_multipolygon(
            self.shapely_polygons4, polygon_index, self.contained_polygons4
        )
        self.assertTrue(result_multipolygon.equals(output_multipolygon))
        #
        polygon_index = 7
        output_polygon_part1 = Polygon(
            shell=[(410, 710), (410, 880), (520, 880), (520, 710)],
            holes=[[(430, 730), (430, 760), (460, 760), (460, 730)], [(450, 800), (450, 810), (470, 810), (470, 800)]],
        )
        output_polygon_part2 = Polygon(
            shell=[(530, 710), (530, 880), (550, 880), (550, 710)],
            holes=[],
        )
        output_polygon_part3 = Polygon(
            shell=[(560, 710), (560, 880), (580, 880), (580, 710)],
            holes=[],
        )
        output_multipolygon = MultiPolygon([output_polygon_part1, output_polygon_part2, output_polygon_part3])
        result_multipolygon = PatchesTestMock._get_replacement_multipolygon(
            self.shapely_polygons4, polygon_index, self.contained_polygons4
        )
        self.assertTrue(result_multipolygon.equals(output_multipolygon))

    def test__get_holes_union_coords(self):
        # one hole
        holes_coords = [[(1, 1), (1, 2), (2, 2), (2, 1)]]
        result_coords = PatchesTestMock._get_holes_union_coords(holes_coords)
        self.assertEqual(result_coords, holes_coords)
        # two identical holes
        holes_coords = [[(1, 1), (1, 2), (2, 2), (2, 1)], [(1, 1), (1, 2), (2, 2), (2, 1)]]
        output_coords = [(1, 1), (1, 2), (2, 2), (2, 1)]
        result_coords = PatchesTestMock._get_holes_union_coords(holes_coords)
        self.assertEqual(len(result_coords), 1)
        self.assertTrue(Polygon(output_coords).equals(Polygon(result_coords[0])))
        # three idependent holes
        holes_coords = [
            [(1, 1), (1, 2), (2, 2), (2, 1)],
            [(3, 3), (3, 4), (4, 4), (4, 3)],
            [(5, 5), (5, 6), (6, 6), (6, 5)],
        ]
        output_coords = [
            [(1, 1), (1, 2), (2, 2), (2, 1)],
            [(3, 3), (3, 4), (4, 4), (4, 3)],
            [(5, 5), (5, 6), (6, 6), (6, 5)],
        ]
        result_coords = PatchesTestMock._get_holes_union_coords(holes_coords)
        self.assertEqual(len(result_coords), 3)
        self.assertTrue(Polygon(output_coords[0]).equals(Polygon(result_coords[0])))
        self.assertTrue(Polygon(output_coords[1]).equals(Polygon(result_coords[1])))
        self.assertTrue(Polygon(output_coords[2]).equals(Polygon(result_coords[2])))
        # two overlapping holes
        holes_coords = [[(0, 0), (0, 2), (2, 2), (2, 0)], [(1, 1), (1, 3), (3, 3), (3, 1)]]
        output_coords = [[(0, 0), (0, 2), (1, 2), (1, 3), (3, 3), (3, 1), (2, 1), (2, 0)]]
        result_coords = PatchesTestMock._get_holes_union_coords(holes_coords)
        self.assertEqual(len(result_coords), 1)
        self.assertTrue(Polygon(output_coords[0]).equals(Polygon(result_coords[0])))

    @patch("dplabtools.slides.patches.locations.base.unary_union")
    def test__get_holes_union_coords_error(self, mock_func):
        holes_coords = [[(1, 1), (1, 2), (2, 2), (2, 1)], [(3, 3), (3, 4), (4, 4), (4, 3)]]
        mock_func.return_value = LineString([(0, 0), (5, 5)])
        with self.assertRaises(TypeError):
            PatchesTestMock._get_holes_union_coords(holes_coords)


class TestPatchesBaseProperties(TestCase):
    """Tests for properties in BasePatches class."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")

    def test_properties_and_save_images(self):
        # Since this is mocked data, foreground and overlap thresholds are ignored
        result_test_image_tif0 = make_test_path("saved_data/preview/test_preview_level0.tif")
        result_test_image_tif1 = make_test_path("saved_data/preview/test_preview_level1.tif")
        result_test_image_tif2 = make_test_path("saved_data/preview/test_preview_level2.tif")
        polygons = [
            AnnotationPolygon(points=[(100, 100), (150, 300), (300, 200), (200, 100)], label="label1"),
            AnnotationPolygon(points=[(400, 400), (450, 600), (600, 600), (600, 500)], label="label2"),
        ]
        mask_array = np.ones((640, 768), dtype=int)
        output_patch_data_level0 = [
            ((40, 80), (128, 128), "label1"),
            ((520, 560), (128, 128), "label1"),
            ((1000, 1040), (128, 128), "label1"),
            ((1400, 1480), (128, 128), "label1"),
            ((40, 80), (128, 128), "label2"),
            ((520, 560), (128, 128), "label2"),
            ((1000, 1040), (128, 128), "label2"),
            ((1400, 1480), (128, 128), "label2"),
        ]
        output_polygons = [
            list(Polygon([(100, 100), (150, 300), (300, 200), (200, 100), (100, 100)]).exterior.coords),
            list(Polygon([(400, 400), (450, 600), (600, 600), (600, 500), (400, 400)]).exterior.coords),
        ]
        # level 0 patches tests
        #
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif0)
        # patch_count
        self.assertEqual(patches.patch_count, 2 * 4)
        # patch_data at level 0
        self.assertEqual(patches.patch_data, output_patch_data_level0)
        # patch_data at mask level
        output_patch_data_mask_level = [
            ((10, 20), (32, 32), "label1"),
            ((130, 140), (32, 32), "label1"),
            ((250, 260), (32, 32), "label1"),
            ((350, 370), (32, 32), "label1"),
            ((10, 20), (32, 32), "label2"),
            ((130, 140), (32, 32), "label2"),
            ((250, 260), (32, 32), "label2"),
            ((350, 370), (32, 32), "label2"),
        ]
        self.assertEqual(patches.patch_data_mask_level, output_patch_data_mask_level)
        # patch_info
        self.assertEqual(patches.patch_info, {"label1": 4, "label2": 4})
        # patch_labels
        self.assertEqual(patches.patch_labels, ["label1", "label2"])
        # param_info
        output_param_info = {
            "wsi_file": self.wsi_file_tif,
            "mask_data": str(MaskData(mask_data=mask_array)),
            "patch_size": 128,
            "level_or_mpp": 0,
            "foreground_ratio": 0.8,
            "overlap_ratio": 0.8,
            "polygon_buffer": 0,
            "polygon_data": polygons,
            "check_polygons": True,
        }
        self.assertEqual(patches.param_info, output_param_info)
        # level_or_mpp
        self.assertEqual(patches.level_or_mpp, 0)
        # polygons
        result_polygons = [list(poly.exterior.coords) for poly in patches.shapely_polygons]
        self.assertEqual(result_polygons, output_polygons)
        # polygon labels:
        self.assertEqual(patches.polygon_labels, ["label1", "label2"])
        # patch size
        self.assertEqual(patches.patch_size, 128)
        # mask shape
        self.assertEqual(patches.mask_shape[0], 640)
        self.assertEqual(patches.mask_shape[1], 768)
        # wsi file
        self.assertEqual(patches.wsi_file, self.wsi_file_tif)
        # wsi slide
        self.assertEqual(patches.wsi_slide.mpp_data, (0.25, 0.25))
        # class name
        self.assertEqual(patches.class_name, patches.__class__.__name__)
        #
        # level 1 patches tests
        #
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif1)
        # patch_count
        self.assertEqual(patches.patch_count, 2 * 4)
        # patch_data at level 0
        self.assertEqual(patches.patch_data, output_patch_data_level0)
        # patch_data at mask level
        output_patch_data_mask_level = [
            ((10, 20), (128, 128), "label1"),
            ((130, 140), (128, 128), "label1"),
            ((250, 260), (128, 128), "label1"),
            ((350, 370), (128, 128), "label1"),
            ((10, 20), (128, 128), "label2"),
            ((130, 140), (128, 128), "label2"),
            ((250, 260), (128, 128), "label2"),
            ((350, 370), (128, 128), "label2"),
        ]
        self.assertEqual(patches.patch_data_mask_level, output_patch_data_mask_level)
        # patch_info
        self.assertEqual(patches.patch_info, {"label1": 4, "label2": 4})
        # patch_labels
        self.assertEqual(patches.patch_labels, ["label1", "label2"])
        # param_info
        output_param_info = {
            "wsi_file": self.wsi_file_tif,
            "mask_data": str(MaskData(mask_data=mask_array)),
            "patch_size": 128,
            "level_or_mpp": 1,
            "foreground_ratio": 0.8,
            "overlap_ratio": 0.8,
            "polygon_buffer": 0,
            "polygon_data": polygons,
            "check_polygons": True,
        }
        self.assertEqual(patches.param_info, output_param_info)
        # level_or_mpp
        self.assertEqual(patches.level_or_mpp, 1)
        # polygons
        result_polygons = [list(poly.exterior.coords) for poly in patches.shapely_polygons]
        self.assertEqual(result_polygons, output_polygons)
        # polygon labels:
        self.assertEqual(patches.polygon_labels, ["label1", "label2"])
        #
        # level 2 patches tests
        #
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=2,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif2)
        # patch_count
        self.assertEqual(patches.patch_count, 2 * 4)
        # patch_data at level 0
        self.assertEqual(patches.patch_data, output_patch_data_level0)
        # patch_data at mask level
        output_patch_data_mask_level = [
            ((10, 20), (512, 512), "label1"),
            ((130, 140), (512, 512), "label1"),
            ((250, 260), (512, 512), "label1"),
            ((350, 370), (512, 512), "label1"),
            ((10, 20), (512, 512), "label2"),
            ((130, 140), (512, 512), "label2"),
            ((250, 260), (512, 512), "label2"),
            ((350, 370), (512, 512), "label2"),
        ]
        self.assertEqual(patches.patch_data_mask_level, output_patch_data_mask_level)
        # patch_info
        self.assertEqual(patches.patch_info, {"label1": 4, "label2": 4})
        # patch_labels
        self.assertEqual(patches.patch_labels, ["label1", "label2"])
        # param_info
        output_param_info = {
            "wsi_file": self.wsi_file_tif,
            "mask_data": str(MaskData(mask_data=mask_array)),
            "patch_size": 128,
            "level_or_mpp": 2,
            "foreground_ratio": 0.8,
            "overlap_ratio": 0.8,
            "polygon_buffer": 0,
            "polygon_data": polygons,
            "check_polygons": True,
        }
        self.assertEqual(patches.param_info, output_param_info)
        # level_or_mpp
        self.assertEqual(patches.level_or_mpp, 2)
        # polygons
        result_polygons = [list(poly.exterior.coords) for poly in patches.shapely_polygons]
        self.assertEqual(result_polygons, output_polygons)
        # polygon labels:
        self.assertEqual(patches.polygon_labels, ["label1", "label2"])


class TestDuplicatePatchesThreeLabels(TestCase):
    """Tests for label based properties in BasePatches class.

    Removing duplicate patches is also tested here.
    """

    def test_three_labels(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        # Polygon points below are not used in computing output, it's all about the number of polygons,
        # not their shapes/points
        polygons = [
            AnnotationPolygon(points=[(100, 100), (100, 110), (110, 110), (110, 100)], label="label3"),
            AnnotationPolygon(points=[(200, 200), (200, 210), (210, 210), (210, 200)], label="label1"),
            AnnotationPolygon(points=[(300, 300), (300, 310), (310, 310), (310, 300)], label="label2"),
            AnnotationPolygon(points=[(400, 400), (400, 410), (410, 410), (410, 400)], label="label1"),
            AnnotationPolygon(points=[(500, 500), (500, 510), (510, 510), (510, 500)], label="label2"),
            AnnotationPolygon(points=[(600, 600), (600, 610), (610, 610), (610, 600)], label="label3"),
            AnnotationPolygon(points=[(700, 700), (700, 710), (710, 710), (710, 700)], label="label1"),
        ]
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_labels, ["label1", "label2", "label3"])
        self.assertEqual(patches.polygon_labels, ["label3", "label1", "label2", "label1", "label2", "label3", "label1"])
        # Patches are duplicated (fixed polygons in PatchesTestMock), so log message about excluding duplicates
        # will be generated here. Before excluding the values are: "label1": 4 * 3, "label2": 4 * 2, "label3": 4 * 2
        self.assertEqual(patches.patch_info, {"label1": 4, "label2": 4, "label3": 4})


class TestPatchesPreviewImage(TestCase):
    """Tests for saving preview image in BasePatches class."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")

    def test_save_preview_image(self):
        # This is just a drawing accuracy test, patch location computing is not included here
        # images have been saved in the previous test
        result_test_image_tif0 = make_test_path("saved_data/preview/test_preview_level0.tif")
        result_test_image_tif1 = make_test_path("saved_data/preview/test_preview_level1.tif")
        result_test_image_tif2 = make_test_path("saved_data/preview/test_preview_level2.tif")
        #
        output_image0 = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_level0.tif"))
        output_image_array0 = np.asarray(output_image0)
        output_image0.close()
        result_image0 = Image.open(result_test_image_tif0)
        result_image_array0 = np.asarray(result_image0)
        result_image0.close()
        np.testing.assert_equal(result_image_array0, output_image_array0)
        #
        output_image1 = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_level1.tif"))
        output_image_array1 = np.asarray(output_image1)
        output_image1.close()
        result_image1 = Image.open(result_test_image_tif1)
        result_image_array1 = np.asarray(result_image1)
        result_image1.close()
        np.testing.assert_equal(result_image_array1, output_image_array1)
        #
        output_image2 = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_level2.tif"))
        output_image_array2 = np.asarray(output_image2)
        output_image2.close()
        result_image2 = Image.open(result_test_image_tif2)
        result_image_array2 = np.asarray(result_image2)
        result_image2.close()
        np.testing.assert_equal(result_image_array2, output_image_array2)

    def test_save_preview_image_with_polygon_buffer1(self):
        # test polygon
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_level1_buffer.tif")
        mask_array = np.ones((640, 768), dtype=int)
        polygons = [AnnotationPolygon(points=[(100, 100), (100, 200), (200, 200), (200, 100)], label="label1")]
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=1,
            polygon_buffer=-20,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif)
        #
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_level1_buffer.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_preview_image_with_polygon_buffer2(self):
        # test multipolygon with two holes
        mask_array = np.ones((640, 768), dtype=int)
        polygons = [
            AnnotationPolygon(
                points=[(260, 90), (140, 320), (370, 370), (60, 60), (80, 220)],
                label="",
                holes=[
                    [(205, 275), (205, 320), (255, 320), (255, 275)],
                    [(90, 140), (90, 170), (120, 170), (120, 140)],
                ],
            )
        ]
        # 1. positive buffer - multipolygon transitions to polygon with two holes
        result_test_image_tif1 = make_test_path("saved_data/preview/test_preview_level1_buffer_multi1.tif")
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=1,
            polygon_buffer=5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif1)
        output_image = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_level1_buffer_multi1.tif")
        )
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif1)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # 2. negative buffer - multipolygon remain multipolygon with two holes, but only for small buffer values
        result_test_image_tif2 = make_test_path("saved_data/preview/test_preview_level1_buffer_multi2.tif")
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=1,
            polygon_buffer=-5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif2)
        output_image = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_level1_buffer_multi2.tif")
        )
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif2)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_preview_image_with_patch_markers1(self):
        # Marker thickness based on patch size
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_level1_markers.tif")
        mask_array = np.ones((640, 768), dtype=int)
        polygons = [AnnotationPolygon(points=[(100, 100), (100, 200), (200, 200), (200, 100)], label="label1")]
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, patch_markers=True)
        #
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_level1_markers.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_preview_image_with_patch_markers2(self):
        # Marker thickness based on provided thickness
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_level0_markers.tif")
        mask_array = np.ones((160, 192), dtype=int)
        polygons = [AnnotationPolygon(points=[(25, 25), (25, 50), (50, 50), (50, 25)], label="label1")]
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, patch_markers=True)
        #
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_level0_markers.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_preview_image_with_holes(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_level1_holes.tif")
        mask_array = np.ones((640, 768), dtype=int)
        polygons = [
            AnnotationPolygon(
                points=[(100, 100), (100, 200), (200, 200), (200, 100)],
                label="label1",
                holes=[
                    [(120, 120), (120, 140), (140, 140), (140, 120)],
                    [(160, 160), (160, 180), (180, 180), (180, 160)],
                ],
            )
        ]
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            check_polygons=False,
        )
        patches.save_preview_image(result_test_image_tif, patch_markers=True)
        #
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_level1_holes.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_images_without_all_colors(self):
        result_test_image_patches_tif = make_test_path("saved_data/preview/test_preview_only_patches.tif")
        result_test_image_bboxes_tif = make_test_path("saved_data/preview/test_preview_only_bboxes.tif")
        result_test_image_polygons_tif = make_test_path("saved_data/preview/test_preview_only_polygons.tif")
        result_test_image_buffers_tif = make_test_path("saved_data/preview/test_preview_only_buffers.tif")
        mask_array = np.ones((640, 768), dtype=int)
        polygons = [
            AnnotationPolygon(points=[(100, 100), (150, 300), (300, 200), (200, 100)], label="label1"),
            AnnotationPolygon(points=[(400, 400), (450, 600), (600, 600), (600, 500)], label="label2"),
        ]
        patches = PatchesTestMock(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=[-10, -15],
        )
        patches.save_preview_image(
            result_test_image_patches_tif, bbox_color=None, polygon_color=None, buffer_color=None
        )
        patches.save_preview_image(
            result_test_image_bboxes_tif, patch_color=None, polygon_color=None, buffer_color=None
        )
        patches.save_preview_image(result_test_image_polygons_tif, patch_color=None, bbox_color=None, buffer_color=None)
        patches.save_preview_image(result_test_image_buffers_tif, patch_color=None, bbox_color=None, polygon_color=None)
        #
        output_image_patches = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_only_patches.tif")
        )
        output_image_patches_array = np.asarray(output_image_patches)
        output_image_patches.close()
        result_image_patches = Image.open(result_test_image_patches_tif)
        result_image_patches_array = np.asarray(result_image_patches)
        result_image_patches.close()
        np.testing.assert_equal(result_image_patches_array, output_image_patches_array)
        #
        output_image_bboxes = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_only_bboxes.tif"))
        output_image_bboxes_array = np.asarray(output_image_bboxes)
        output_image_bboxes.close()
        result_image_bboxes = Image.open(result_test_image_bboxes_tif)
        result_image_bboxes_array = np.asarray(result_image_bboxes)
        result_image_bboxes.close()
        np.testing.assert_equal(result_image_bboxes_array, output_image_bboxes_array)
        #
        output_image_polygons = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_only_polygons.tif")
        )
        output_image_polygons_array = np.asarray(output_image_polygons)
        output_image_polygons.close()
        result_image_polygons = Image.open(result_test_image_polygons_tif)
        result_image_polygons_array = np.asarray(result_image_polygons)
        result_image_polygons.close()
        np.testing.assert_equal(result_image_polygons_array, output_image_polygons_array)
        #
        output_image_buffers = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_only_buffers.tif")
        )
        output_image_buffers_array = np.asarray(output_image_buffers)
        output_image_buffers.close()
        result_image_buffers = Image.open(result_test_image_buffers_tif)
        result_image_buffers_array = np.asarray(result_image_buffers)
        result_image_buffers.close()
        np.testing.assert_equal(result_image_buffers_array, output_image_buffers_array)


class TestPatchesPreviewImagePreviewLevel(TestCase):
    """Tests for saving preview image with different preview levels."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(
            points=[(20, 20), (20, 250), (250, 250), (250, 20)],
            label="area1",
            holes=[[(150, 150), (150, 230), (230, 230), (230, 150)], [(40, 170), (40, 220), (80, 220), (80, 170)]],
        )
        poly2 = AnnotationPolygon(
            points=[(300, 350), (300, 700), (620, 700)],
            label="area2",
            holes=[[(330, 500), (330, 550), (390, 550), (390, 500)]],
        )
        poly3 = AnnotationPolygon(points=[(300, 40), (320, 190), (450, 240), (540, 40)], label="area3", holes=[])
        self.polygons = [poly1, poly2, poly3]

    def test_save_preview_image_without_level_defined(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_preview_level_not_defined.tif")
        patches = PatchesTestMock(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=8,
        )
        patches.save_preview_image(result_test_image_tif, patch_markers=True)
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_preview_level_not_defined.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_save_preview_image_with_level_is_mask(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_preview_level_is_mask.tif")
        patches = PatchesTestMock(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=8,
        )
        patches.save_preview_image(result_test_image_tif, patch_markers=True, level_or_minsize=1)
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_preview_level_is_mask.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_save_preview_image_with_level_zero(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_preview_level0.tif")
        patches = PatchesTestMock(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=8,
        )
        patches.save_preview_image(result_test_image_tif, patch_markers=True, level_or_minsize=0)
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_preview_level0.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_save_preview_image_with_level_two(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_preview_level2.tif")
        patches = PatchesTestMock(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=8,
        )
        patches.save_preview_image(result_test_image_tif, patch_markers=True, thickness=1, level_or_minsize=2)
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_preview_level2.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_save_preview_image_with_level_from_size(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_preview_level_from_size.tif")
        patches = PatchesTestMock(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=8,
        )
        patches.save_preview_image(result_test_image_tif, level_or_minsize=500)
        #
        result_image_preview = Image.open(result_test_image_tif)
        width, height = result_image_preview.width, result_image_preview.height
        result_image_preview.close()
        self.assertEqual((width, height), (640, 768))


class TestPatchesPreviewImageDrawnLabelsVariousElements(TestCase):
    """Tests for saving preview image with different labels."""

    class PatchesTestMockDrawnLabels(PatchesTestMock):
        """Mock class implementing different patches per polygon."""

        def _get_polygon_patches(self, polygon, counter):
            patches = []
            if counter == 0:
                patches = [(30, 30), (80, 80)]
            elif counter == 1:
                patches = [(360, 600), (500, 650)]
            if counter == 2:
                patches = [(350, 150), (450, 100)]
            return patches

    def setUp(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(
            points=[(20, 20), (20, 250), (250, 250), (250, 20)],
            label="area1",
            holes=[[(150, 150), (150, 230), (230, 230), (230, 150)], [(40, 170), (40, 220), (80, 220), (80, 170)]],
        )
        poly2 = AnnotationPolygon(
            points=[(300, 350), (300, 700), (620, 700)],
            label="area2",
            holes=[[(330, 500), (330, 550), (390, 550), (390, 500)]],
        )
        poly3 = AnnotationPolygon(points=[(300, 40), (320, 190), (450, 240), (540, 40)], label="area3", holes=[])
        polygons = [poly1, poly2, poly3]
        self.patches = self.PatchesTestMockDrawnLabels(
            polygon_data=polygons,
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=8,
        )

    def test_preview_all_drawn_default(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_all_default.tif")
        self.patches.save_preview_image(result_test_image_tif, patch_markers=True)
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_all_default.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_all_drawn_listed(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_all_listed.tif")
        self.patches.save_preview_image(
            result_test_image_tif, patch_markers=True, drawn_labels=["area1", "area2", "area3"]
        )
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_all_listed.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_area1(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_area1.tif")
        self.patches.save_preview_image(result_test_image_tif, patch_markers=True, drawn_labels=["area1"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_area1.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_area2(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_area2.tif")
        self.patches.save_preview_image(result_test_image_tif, patch_markers=True, drawn_labels=["area2"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_area2.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_area3(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_area3.tif")
        self.patches.save_preview_image(result_test_image_tif, patch_markers=True, drawn_labels=["area3"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_area3.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_area1_and_area3(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_area1_area3.tif")
        self.patches.save_preview_image(result_test_image_tif, patch_markers=True, drawn_labels=["area1", "area3"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_area1_area3.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_area_not_found(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_area_not_found.tif")
        self.patches.save_preview_image(result_test_image_tif, patch_markers=True, drawn_labels=["area66"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_area_not_found.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_drawn_patches_default(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_patches_default.tif")
        self.patches.save_preview_image(result_test_image_tif, patch_markers=True)
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_patches_default.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_drawn_patches_nofilters(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_patches_nofilters.tif")
        self.patches.save_preview_image(
            result_test_image_tif, patch_markers=True, drawn_labels=[], drawn_patches_labels=[]
        )
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_patches_nofilters.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_drawn_patches_area1(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_patches_area1.tif")
        self.patches.save_preview_image(
            result_test_image_tif, patch_markers=True, drawn_labels=[], drawn_patches_labels=["area1"]
        )
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_patches_area1.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_drawn_patches_area1_area3(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_patches_area1_area3.tif")
        self.patches.save_preview_image(
            result_test_image_tif, patch_markers=True, drawn_labels=[], drawn_patches_labels=["area1", "area3"]
        )
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_patches_area1_area3.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_drawn_patches_override(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_patches_override.tif")
        self.patches.save_preview_image(
            result_test_image_tif, patch_markers=True, drawn_labels=["area1"], drawn_patches_labels=["area2", "area3"]
        )
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_patches_override.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_drawn_patches_not_found(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_patches_not_found.tif")
        self.patches.save_preview_image(
            result_test_image_tif, patch_markers=True, drawn_labels=[], drawn_patches_labels=["area66"]
        )
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_patches_not_found.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)


class TestPatchesPreviewImageDrawnLabelsMultipleLabels(TestCase):
    """Tests for saving preview image with different labels."""

    class PatchesTestMockDrawnLabels(PatchesTestMock):
        """Mock class implementing empty patch list."""

        def _get_polygon_patches(self, polygon, counter):
            return []

    def setUp(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 100), (100, 100), (100, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(220, 20), (220, 100), (300, 100), (300, 20)], label="area1")
        poly3 = AnnotationPolygon(points=[(420, 20), (420, 100), (500, 100), (500, 20)], label="area1")
        poly4 = AnnotationPolygon(points=[(20, 220), (20, 300), (100, 300), (100, 220)], label="")
        poly5 = AnnotationPolygon(points=[(220, 220), (220, 300), (300, 300), (300, 220)], label="")
        poly6 = AnnotationPolygon(points=[(420, 220), (420, 300), (500, 300), (500, 220)], label="")
        poly7 = AnnotationPolygon(points=[(20, 420), (20, 500), (100, 500), (100, 420)], label="area2")
        poly8 = AnnotationPolygon(points=[(220, 420), (220, 500), (300, 500), (300, 420)], label="area2")
        poly9 = AnnotationPolygon(points=[(420, 420), (420, 500), (500, 500), (500, 420)], label="area2")
        poly10 = AnnotationPolygon(points=[(20, 620), (20, 700), (100, 700), (100, 620)], label="area3")
        poly11 = AnnotationPolygon(points=[(220, 620), (220, 700), (300, 700), (300, 620)], label="area3")
        poly12 = AnnotationPolygon(points=[(420, 620), (420, 700), (500, 700), (500, 620)], label="area3")
        polygons = [poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9, poly10, poly11, poly12]
        self.patches = self.PatchesTestMockDrawnLabels(
            polygon_data=polygons,
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            polygon_buffer=0,
        )

    def test_preview_multi_all(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_multi_all.tif")
        self.patches.save_preview_image(result_test_image_tif)
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_multi_all.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_multi_area1(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_multi_area1.tif")
        self.patches.save_preview_image(result_test_image_tif, drawn_labels=["area1"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_multi_area1.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_multi_area2(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_multi_area2.tif")
        self.patches.save_preview_image(result_test_image_tif, drawn_labels=["area2"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_multi_area2.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_multi_area3(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_multi_area3.tif")
        self.patches.save_preview_image(result_test_image_tif, drawn_labels=["area3"])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_multi_area3.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)

    def test_preview_multi_area_empty(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_drawn_labels_multi_empty.tif")
        self.patches.save_preview_image(result_test_image_tif, drawn_labels=[""])
        #
        output_image_preview = Image.open(
            make_test_path("ref_data/slides/patches/preview/ref_preview_drawn_labels_multi_empty.tif")
        )
        output_image_preview_array = np.asarray(output_image_preview)
        output_image_preview.close()
        result_image_preview = Image.open(result_test_image_tif)
        result_image_preview_array = np.asarray(result_image_preview)
        result_image_preview.close()
        np.testing.assert_equal(result_image_preview_array, output_image_preview_array)
