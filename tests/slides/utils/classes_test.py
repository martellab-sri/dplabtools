# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for classes in slides.utils.classes."""

from unittest import TestCase

import numpy as np
from PIL import Image

from dplabtools.slides.utils import AnnotationPolygon, MaskData, PolygonData
from testutils import make_test_path


class TestAnnotationPolygon(TestCase):
    """Tests for AnnotationPolygon class."""

    def setUp(self):
        self.polygon = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )

    def test_eq1(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertEqual(polygon1, polygon2)

    def test_eq2(self):
        polygon1 = AnnotationPolygon(
            points=[[10, 10], (10, 100), [100, 100], (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), [10, 100], (100, 100), [100, 10]],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertEqual(polygon1, polygon2)

    def test_eq3(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), [30, 50], (50, 50), [50, 30]]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[[30, 30], (30, 50), [50, 50], (50, 30)]],
        )
        self.assertEqual(polygon1, polygon2)

    def test_eq4(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[
                [(30, 30), [30, 50], (50, 50), [50, 30]],
                [(130, 130), [130, 150], (150, 150), [150, 130]],
            ],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[
                [[30, 30], (30, 50), [50, 50], (50, 30)],
                [[130, 130], (130, 150), [150, 150], (150, 130)],
            ],
        )
        self.assertEqual(polygon1, polygon2)

    def test_eq5(self):
        polygon1 = AnnotationPolygon(
            points=[(11, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq6(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label1",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label2",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq7(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(31, 30), (30, 50), (50, 50), (50, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq8(self):
        polygon1 = AnnotationPolygon(
            points=[[10, 10], (10, 100), [100, 100], (100, 10)],
            label="label",
            holes=[[(31, 30), (30, 50), (50, 50), (50, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), [10, 100], (100, 100), [100, 10]],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq9(self):
        polygon1 = AnnotationPolygon(
            points=[(11, 10), (10, 100), (100, 100), (100, 10)],
            label="label1",
            holes=[[(30, 30), [30, 50], (50, 50), [50, 30]]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label1",
            holes=[[[30, 30], (30, 50), [50, 50], (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq10(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), [10, 100], [100, 100], (100, 10)],
            label="label1",
            holes=[[(30, 30), [30, 50], (50, 50), [50, 30]]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), [10, 100], [100, 100], (100, 10)],
            label="label2",
            holes=[[[30, 30], (30, 50), [50, 50], (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq11(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 11)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq12(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (51, 30)]],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[[(30, 30), (30, 50), (50, 50), (50, 30)]],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq13(self):
        polygon1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[
                [(30, 30), [30, 50], (50, 50), [50, 30]],
                [(130, 130), [130, 150], (150, 150), [150, 130]],
            ],
        )
        polygon2 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label",
            holes=[
                [[30, 30], (30, 50), [50, 50], (50, 30)],
                [[130, 130], (130, 150), [150, 150], (150, 131)],
            ],
        )
        self.assertNotEqual(polygon1, polygon2)

    def test_eq14(self):
        self.assertNotEqual(self.polygon, list)

    def test_eq15(self):
        self.assertEqual(self.polygon, self.polygon)

    def test_eq16(self):
        polygon1 = AnnotationPolygon(points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label")
        polygon2 = AnnotationPolygon(points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label")
        self.assertEqual(polygon1, polygon2)

    def test_str(self):
        output_text = (
            "[[(10, 10), (10, 100), (100, 100), (100, 10)], 'label', [[(30, 30), (30, 50), (50, 50), (50, 30)]]]"
        )
        result_text = str(self.polygon)
        self.assertEqual(result_text, output_text)

    def test_repr(self):
        output_text = (
            "[[(10, 10), (10, 100), (100, 100), (100, 10)], 'label', [[(30, 30), (30, 50), (50, 50), (50, 30)]]]"
        )
        result_text = repr(self.polygon)
        self.assertEqual(result_text, output_text)

    def test_property_data_dict(self):
        output_dict = {
            "points": [(10, 10), (10, 100), (100, 100), (100, 10)],
            "label": "label",
            "holes": [[(30, 30), (30, 50), (50, 50), (50, 30)]],
        }
        result_dict = self.polygon.data_dict
        self.assertEqual(result_dict, output_dict)

    def test_property_points(self):
        output_points = [(10, 10), (10, 100), (100, 100), (100, 10)]
        result_points = self.polygon.points
        self.assertEqual(result_points, output_points)

    def test_property_label(self):
        output_label = "label"
        result_label = self.polygon.label
        self.assertEqual(result_label, output_label)

    def test_property_holes(self):
        output_holes = [[(30, 30), (30, 50), (50, 50), (50, 30)]]
        result_holes = self.polygon.holes
        self.assertEqual(result_holes, output_holes)


class TestMaskData(TestCase):
    """Tests for MaskData class."""

    def test_read_uncompressed_array2d_from_file(self):
        array_file = make_test_path("ref_data/slides/utils/array.npy")
        mask_data = MaskData(mask_data=array_file)
        self.assertEqual(mask_data.mask_array.shape, (160, 192))
        self.assertEqual(str(mask_data), array_file)
        self.assertEqual(repr(mask_data), array_file)

    def test_read_compressed_array2d_from_file_default_key(self):
        array_file = make_test_path("ref_data/slides/utils/array1.npz")
        mask_data = MaskData(mask_data=array_file)
        self.assertEqual(mask_data.mask_array.shape, (160, 192))
        self.assertEqual(str(mask_data), array_file)
        self.assertEqual(repr(mask_data), array_file)

    def test_read_compressed_array2d_from_file_custom_key(self):
        # back up the original key value as class MaskData has no reset option
        key_value = MaskData.npz_data_key
        MaskData.npz_data_key = "custom"
        array_file = make_test_path("ref_data/slides/utils/array2.npz")
        mask_data = MaskData(mask_data=array_file)
        MaskData.npz_data_key = key_value
        self.assertEqual(mask_data.mask_array.shape, (160, 192))
        self.assertEqual(str(mask_data), array_file)
        self.assertEqual(repr(mask_data), array_file)

    def test_read_image2d_from_file(self):
        image_file = make_test_path("ref_data/slides/utils/mask2d.png")
        mask_data = MaskData(mask_data=image_file)
        self.assertEqual(mask_data.mask_array.shape, (10, 30))
        self.assertEqual(str(mask_data), image_file)
        self.assertEqual(repr(mask_data), image_file)

    def test_read_image3d_from_file(self):
        image_file = make_test_path("ref_data/slides/utils/mask3d.png")
        with self.assertRaises(ValueError):
            MaskData(mask_data=image_file)

    def test_read_array2d_from_memory(self):
        array_data = np.ones((33, 44), dtype=bool)
        mask_data = MaskData(mask_data=array_data)
        self.assertEqual(mask_data.mask_array.shape, (33, 44))

    def test_read_array3d_from_memory(self):
        array_data = np.ones((33, 44, 4), dtype=bool)
        with self.assertRaises(ValueError):
            MaskData(mask_data=array_data)

    def test_read_image2d_from_memory(self):
        image_data = Image.new(size=(20, 40), mode="L")
        mask_data = MaskData(mask_data=image_data)
        self.assertEqual(mask_data.mask_array.shape, (20, 40))

    def test_read_image3d_from_memory(self):
        image_data = Image.new(size=(20, 40), mode="RGB")
        with self.assertRaises(ValueError):
            MaskData(mask_data=image_data)

    def test_read_wrong_type_from_file(self):
        text_file = make_test_path("ref_data/slides/utils/test.txt")
        with self.assertRaises(TypeError):
            MaskData(mask_data=text_file)

    def test_read_wrong_type_from_memory(self):
        with self.assertRaises(TypeError):
            MaskData(mask_data=5)

    def test_str1(self):
        array_data = np.ones((5, 14), dtype=bool)
        mask_data = MaskData(mask_data=array_data)
        output_text = "NumPy array [shape=(5, 14), size=70, type=bool]"
        result_text = str(mask_data)
        self.assertEqual(result_text, output_text)

    def test_str2(self):
        image_data = Image.new(size=(13, 7), mode="L")
        mask_data = MaskData(mask_data=image_data)
        output_text = "NumPy array [shape=(13, 7), size=91, type=bool]"
        result_text = str(mask_data)
        self.assertEqual(result_text, output_text)

    def test_repr1(self):
        array_data = np.ones((5, 10), dtype=bool)
        mask_data = MaskData(mask_data=array_data)
        output_text = "NumPy array [shape=(5, 10), size=50, type=bool]"
        result_text = repr(mask_data)
        self.assertEqual(result_text, output_text)

    def test_repr2(self):
        image_data = Image.new(size=(9, 11), mode="L")
        mask_data = MaskData(mask_data=image_data)
        output_text = "NumPy array [shape=(9, 11), size=99, type=bool]"
        result_text = repr(mask_data)
        self.assertEqual(result_text, output_text)

    def test_properties(self):
        array_data = np.ones((15, 30), dtype=bool)
        mask_data = MaskData(mask_data=array_data)
        mask_array = mask_data.mask_array
        self.assertEqual(mask_array.shape, array_data.shape)
        self.assertEqual(mask_array.size, array_data.size)
        self.assertEqual(mask_array.dtype, array_data.dtype)


class TestPolygonData(TestCase):
    """Tests for PolygonData class."""

    def test_annotation_polygons_list_multi(self):
        poly1 = AnnotationPolygon(points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label1")
        poly2 = AnnotationPolygon(points=[(20, 20), (20, 200), (200, 200), (200, 20)], label="label2")
        polygon_list = [poly1, poly2]
        output_polygons = [
            AnnotationPolygon(points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label1"),
            AnnotationPolygon(points=[(20, 20), (20, 200), (200, 200), (200, 20)], label="label2"),
        ]
        polygon_data = PolygonData(polygon_data=polygon_list)
        result_polygons = polygon_data.polygons
        self.assertEqual(result_polygons, output_polygons)

    def test_annotation_polygons_list_single(self):
        poly1 = AnnotationPolygon(points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label1")
        polygon_list = [poly1]
        output_polygons = [AnnotationPolygon(points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label1")]
        polygon_data = PolygonData(polygon_data=polygon_list)
        result_polygons = polygon_data.polygons
        self.assertEqual(result_polygons, output_polygons)

    def test_annotation_polygons_nonlist_single(self):
        poly1 = AnnotationPolygon(points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label1")
        with self.assertRaises(TypeError):
            PolygonData(polygon_data=poly1)

    def test_json_doc_list_multi(self):
        json_doc = """[
                        {"points": [[1, 2], [3, 4], [5, 6]], "label": "label1", "holes": []},
                        {"points": [[7, 8], [9, 10], [11, 12]], "label": "label2", "holes": []}
                 ]"""
        output_polygons = [
            AnnotationPolygon(points=[(1, 2), (3, 4), (5, 6)], label="label1"),
            AnnotationPolygon(points=[(7, 8), (9, 10), (11, 12)], label="label2"),
        ]
        polygon_data = PolygonData(polygon_data=json_doc)
        result_polygons = polygon_data.polygons
        self.assertEqual(result_polygons, output_polygons)

    def test_json_doc_list_single(self):
        json_doc = '[{"points": [[1, 2], [3, 4], [5, 6]], "label": "label1", "holes": []}]'
        output_polygons = [AnnotationPolygon(points=[(1, 2), (3, 4), (5, 6)], label="label1")]
        polygon_data = PolygonData(polygon_data=json_doc)
        result_polygons = polygon_data.polygons
        self.assertEqual(result_polygons, output_polygons)

    def test_json_doc_list_single_invalid(self):
        json_doc = '[{"some_points": [[1, 2], [3, 4], [5, 6]], "label": "label1", "holes": []}]'
        with self.assertRaises(ValueError):
            PolygonData(polygon_data=json_doc)

    def test_json_file(self):
        json_file = make_test_path("ref_data/slides/utils/test3.json")
        output_polygons = [
            AnnotationPolygon(points=[(10, 20), (30, 40), (50, 60)], label="ABC1"),
            AnnotationPolygon(points=[(70, 80), (90, 100), (110, 120)], label="ABC2"),
        ]
        polygon_data = PolygonData(polygon_data=json_file)
        result_polygons = polygon_data.polygons
        self.assertEqual(result_polygons, output_polygons)

    def test_nonlist_wrong_type(self):
        poly1 = 33
        with self.assertRaises(TypeError):
            PolygonData(polygon_data=poly1)

    def test_list_wrong_type(self):
        poly1 = [33]
        with self.assertRaises(ValueError):
            PolygonData(polygon_data=poly1)

    def test_str1(self):
        json_doc = '[{"points": [[1, 2], [3, 4], [5, 6]], "label": "label1", "holes": []}]'
        polygon_data = PolygonData(polygon_data=json_doc)
        output_text = '[{"points": [[1, 2], [3, 4], [5, 6]], "label": "label1", "holes": []}]'
        result_text = str(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_str2(self):
        json_file = make_test_path("ref_data/slides/utils/test3.json")
        polygon_data = PolygonData(polygon_data=json_file)
        output_text = make_test_path("ref_data/slides/utils/test3.json")
        result_text = str(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_str3(self):
        poly1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)], label="label1", holes=[[(20, 30), (30, 40), (40, 50)]]
        )
        polygon_list = [poly1]
        polygon_data = PolygonData(polygon_data=polygon_list)
        output_text = "[[[(10, 10), (10, 100), (100, 100), (100, 10)], 'label1', [[(20, 30), (30, 40), (40, 50)]]]]"
        result_text = str(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_str4(self):
        poly1 = AnnotationPolygon(
            points=[(10, 10), (10, 100), (100, 100), (100, 10)],
            label="label1",
            holes=[[(20, 30), (30, 40), (40, 50)]],
        )
        poly2 = AnnotationPolygon(
            points=[(20, 20), (20, 200), (200, 200), (200, 20)],
            label="label2",
        )
        polygon_list = [poly1, poly2]
        polygon_data = PolygonData(polygon_data=polygon_list)
        output_text = "[[[(10, 10), (10, 100), (100, 100), (100, 10)], 'label1', [[(20, 30), (30, 40), (40, 50)]]], [[(20, 20), (20, 200), (200, 200), (200, 20)], 'label2', []]]"
        result_text = str(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_repr1(self):
        json_doc = (
            '[{"points": [[10, 20], [30, 40], [50, 60]], "label": "label1", "holes": [[[20, 30], [30, 40], [40, 50]]]}]'
        )
        polygon_data = PolygonData(polygon_data=json_doc)
        output_text = (
            '[{"points": [[10, 20], [30, 40], [50, 60]], "label": "label1", "holes": [[[20, 30], [30, 40], [40, 50]]]}]'
        )
        result_text = repr(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_repr2(self):
        json_file = make_test_path("ref_data/slides/utils/test3.json")
        polygon_data = PolygonData(polygon_data=json_file)
        output_text = make_test_path("ref_data/slides/utils/test3.json")
        result_text = repr(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_repr3(self):
        poly1 = AnnotationPolygon(
            points=[(1, 1), (1, 10), (10, 10), (10, 1)], label="label1", holes=[[(2, 3), (3, 4), (4, 5)]]
        )
        polygon_list = [poly1]
        polygon_data = PolygonData(polygon_data=polygon_list)
        output_text = "[[[(1, 1), (1, 10), (10, 10), (10, 1)], 'label1', [[(2, 3), (3, 4), (4, 5)]]]]"
        result_text = repr(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_repr4(self):
        poly1 = AnnotationPolygon(
            points=[(80, 90), (10, 100), (100, 100), (100, 10)],
            label="label1",
            holes=[[(20, 30), (30, 40), (40, 50)]],
        )
        poly2 = AnnotationPolygon(
            points=[(50, 60), (20, 200), (200, 200), (200, 20)],
            label="label2",
        )
        polygon_list = [poly1, poly2]
        polygon_data = PolygonData(polygon_data=polygon_list)
        output_text = "[[[(80, 90), (10, 100), (100, 100), (100, 10)], 'label1', [[(20, 30), (30, 40), (40, 50)]]], [[(50, 60), (20, 200), (200, 200), (200, 20)], 'label2', []]]"
        result_text = repr(polygon_data)
        self.assertEqual(result_text, output_text)

    def test_properties(self):
        poly1 = AnnotationPolygon(points=[(1, 1), (1, 10), (10, 10), (10, 1)], label="label1")
        polygon_list = [poly1]
        output_polygons = [AnnotationPolygon(points=[(1, 1), (1, 10), (10, 10), (10, 1)], label="label1")]
        polygon_data = PolygonData(polygon_data=polygon_list)
        result_polygons = polygon_data.polygons
        self.assertEqual(result_polygons, output_polygons)
