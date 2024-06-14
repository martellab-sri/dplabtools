# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions inside patchutils.

Tested functions:
    expand_scalar_param
    expand_list_param
"""

from unittest import TestCase

from dplabtools.slides.patches.locations.utils import expand_scalar_param, expand_list_param


class TestPatchUtils(TestCase):
    """Test cases for functions inside patchutils."""

    def test_expand_scalar_param(self):
        input_param = 4
        output_param = [4, 4, 4, 4, 4, 4]
        result_param = expand_scalar_param(input_param, "", 6)
        self.assertEqual(result_param, output_param)
        #
        input_param = 3.5
        output_param = [3.5, 3.5]
        result_param = expand_scalar_param(input_param, "", 2)
        self.assertEqual(result_param, output_param)
        #
        input_param = 0
        output_param = [0, 0, 0]
        result_param = expand_scalar_param(input_param, "", 3)
        self.assertEqual(result_param, output_param)
        #
        input_param = [1, 2, 3]
        output_param = [1, 2, 3]
        result_param = expand_scalar_param(input_param, "", 3)
        self.assertEqual(result_param, output_param)
        #
        input_param = [1, 2, 3]
        with self.assertRaises(ValueError):
            result_param = expand_scalar_param(input_param, "", 5)
        #
        input_param = []
        with self.assertRaises(ValueError):
            result_param = expand_scalar_param(input_param, "", 5)
        #
        input_param = "abc"
        with self.assertRaises(ValueError):
            result_param = expand_scalar_param(input_param, "", 5)

    def test_expand_list_param(self):
        input_param = [1, 2, 3, 4]
        output_param = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        result_param = expand_list_param(input_param, "", 3)
        self.assertEqual(result_param, output_param)
        #
        input_param = []
        output_param = [[], [], []]
        result_param = expand_list_param(input_param, "", 3)
        self.assertEqual(result_param, output_param)
        #
        input_param = [[1, 2], [3, 4], [5, 6]]
        output_param = [[1, 2], [3, 4], [5, 6]]
        result_param = expand_list_param(input_param, "", 3)
        self.assertEqual(result_param, output_param)
        #
        input_param = [[1, 2], 3, 4, 5]
        output_param = [[1, 2], 3, 4, 5]
        result_param = expand_list_param(input_param, "", 4)
        self.assertEqual(result_param, output_param)
        #
        input_param = [1, 2, [3, 4, 5]]
        output_param = [1, 2, [3, 4, 5]]
        result_param = expand_list_param(input_param, "", 3)
        self.assertEqual(result_param, output_param)
        #
        input_param = [[1, 2], [3, 4]]
        with self.assertRaises(ValueError):
            result_param = expand_list_param(input_param, "", 3)
        #
        input_param = "abc"
        with self.assertRaises(ValueError):
            result_param = expand_list_param(input_param, "", 3)
        #
        input_param = 1
        with self.assertRaises(ValueError):
            result_param = expand_list_param(input_param, "", 3)
