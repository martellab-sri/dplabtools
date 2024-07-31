# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions included in slides.utils.data."""

import json
from unittest import TestCase

import numpy as np
from PIL import Image

from dplabtools.slides.utils import data
from testutils import make_test_path


class TestUtilsData(TestCase):
    """Tests for functions included in slides.utils.data."""

    def test_get_np_array(self):
        ref_type = type(np.ones((5, 5)))
        # test array as input
        input_array = np.ones((5, 5))
        result_array = data.get_np_array(input_array)
        self.assertEqual(type(result_array), ref_type)
        # test string (uncompressed file) as input
        input_array = make_test_path("ref_data/slides/utils/array.npy")
        result_array = data.get_np_array(input_array)
        self.assertEqual(type(result_array), ref_type)
        # test string (compressed file, default key) as input
        input_array = make_test_path("ref_data/slides/utils/array1.npz")
        result_array = data.get_np_array(input_array)
        self.assertEqual(type(result_array), ref_type)
        # test string (compressed file, custom key) as input
        input_array = make_test_path("ref_data/slides/utils/array2.npz")
        result_array = data.get_np_array(input_array, data_key="custom")
        self.assertEqual(type(result_array), ref_type)
        # test other type (exception expected)
        with self.assertRaises(TypeError):
            input_array = 123
            result_array = data.get_np_array(input_array)

    def test_get_pil_image(self):
        ref_types = ("Image", "PngImageFile")
        # test image as input
        input_image = Image.new(mode="RGB", size=(20, 20))
        result_image = data.get_pil_image(input_image)
        self.assertTrue(result_image.__class__.__name__ in ref_types)
        # test string (file) as input
        input_image = make_test_path("ref_data/slides/utils/ref_png.png")
        result_image = data.get_pil_image(input_image)
        self.assertTrue(result_image.__class__.__name__ in ref_types)
        with self.assertRaises(TypeError):
            input_image = 123
            result_image = data.get_pil_image(input_image)

    def test_get_json_obj(self):
        # test string as valid JSON object (one polygon)
        input_str = '[{"points": [[1, 2], [3, 4], [5, 6]], "label": "abc", "holes": []}]'
        output_list = [{"points": [[1, 2], [3, 4], [5, 6]], "label": "abc", "holes": []}]
        result_list = data.get_json_obj(json_data=input_str)
        self.assertEqual(result_list, output_list)
        # test string as valid JSON object (two polygons)
        input_str = """[
                        {"points": [[1, 2], [3, 4], [5, 6]], "label": "abc1", "holes": []},
                        {"points": [[7, 8], [9, 10], [11, 12]], "label": "abc2", "holes": []}
                    ]"""
        output_list = [
            {"points": [[1, 2], [3, 4], [5, 6]], "label": "abc1", "holes": []},
            {"points": [[7, 8], [9, 10], [11, 12]], "label": "abc2", "holes": []},
        ]
        result_list = data.get_json_obj(json_data=input_str)
        self.assertEqual(result_list, output_list)
        # test string as valid JSON list object
        input_str = "[]"
        output_list = []
        result_list = data.get_json_obj(json_data=input_str)
        self.assertEqual(result_list, output_list)
        # test string as valid JSON object but not list wrapped - must fail
        input_str = '{"points": [[1, 2], [3, 4], [5, 6]], "label": "abc", "holes": []}'
        with self.assertRaises(ValueError):
            data.get_json_obj(json_data=input_str)
        # test string as empty JSON object - must fail
        input_str = ""
        with self.assertRaises(ValueError):
            data.get_json_obj(json_data=input_str)
        # test string as non-JSON object - valid JSON file loaded using "json.load"
        input_file = make_test_path("ref_data/slides/utils/test1.json")
        output_list = [
            {"points": [[1, 2], [3, 4], [5, 6]], "label": "abc1", "holes": []},
            {"points": [[7, 8], [9, 10], [11, 12]], "label": "abc2", "holes": []},
        ]
        result_list = data.get_json_obj(json_data=input_file)
        self.assertEqual(result_list, output_list)
        # test string as non-JSON object - valid JSON file loaded using "open"
        input_file = make_test_path("ref_data/slides/utils/test1.json")
        output_list = [
            {"points": [[1, 2], [3, 4], [5, 6]], "label": "abc1", "holes": []},
            {"points": [[7, 8], [9, 10], [11, 12]], "label": "abc2", "holes": []},
        ]
        with open(input_file, "r") as f:
            all_lines = f.read().rstrip("\n")
        result_list = data.get_json_obj(json_data=all_lines)
        self.assertEqual(result_list, output_list)
        # test string as non-JSON object - invalid JSON file
        input_file = make_test_path("ref_data/slides/utils/test2.json")
        with self.assertRaises(json.decoder.JSONDecodeError):
            data.get_json_obj(json_data=input_file)
        # test string as non-JSON object, non-existing file
        input_str = "string"
        with self.assertRaises(FileNotFoundError):
            data.get_json_obj(json_data=input_str)
