# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions included in slides.utils.mask."""

from unittest import TestCase

import numpy as np

from dplabtools.slides.utils import mask


class TestUtilsMask(TestCase):
    """Tests for functions included in slides.utils.mask."""

    def test_get_mask_bounding_box(self):
        # all ones
        input_array = np.ones((100, 200), dtype=int)
        result_tuple = mask.get_mask_bounding_box(input_array)
        output_tuple = (0, 0, 99, 199)
        self.assertEqual(result_tuple, output_tuple)
        # all zeros
        input_array = np.zeros((100, 200), dtype=int)
        result_tuple = mask.get_mask_bounding_box(input_array)
        output_tuple = (0, 0, 0, 0)
        self.assertEqual(result_tuple, output_tuple)
        # partially filled mask
        input_array = np.zeros((100, 200), dtype=int)
        input_array[50:70, 110:150] = 1
        result_tuple = mask.get_mask_bounding_box(input_array)
        output_tuple = (50, 110, 69, 149)
        self.assertEqual(result_tuple, output_tuple)
        # partially filled rows
        input_array = np.zeros((100, 200), dtype=int)
        input_array[:70, ...] = 1
        result_tuple = mask.get_mask_bounding_box(input_array)
        output_tuple = (0, 0, 69, 199)
        self.assertEqual(result_tuple, output_tuple)
        # partially filled rows
        input_array = np.zeros((100, 200), dtype=int)
        input_array[70:, ...] = 1
        result_tuple = mask.get_mask_bounding_box(input_array)
        output_tuple = (70, 0, 99, 199)
        self.assertEqual(result_tuple, output_tuple)
        # partially filled columns
        input_array = np.zeros((100, 200), dtype=int)
        input_array[..., :40] = 1
        result_tuple = mask.get_mask_bounding_box(input_array)
        output_tuple = (0, 0, 99, 39)
        self.assertEqual(result_tuple, output_tuple)
        # partially filled columns
        input_array = np.zeros((100, 200), dtype=int)
        input_array[..., 40:] = 1
        result_tuple = mask.get_mask_bounding_box(input_array)
        output_tuple = (0, 40, 99, 199)
        self.assertEqual(result_tuple, output_tuple)

    def test_get_mask_pixel_step(self):
        patch_size = 256
        patch_stride = 1
        downsample_factor = 4
        output_value = 64
        result_value = mask.get_mask_pixel_step(patch_size, patch_stride, downsample_factor)
        self.assertEqual(result_value, output_value)
        #
        patch_size = 128
        patch_stride = 0.5
        downsample_factor = 2
        output_value = 32
        result_value = mask.get_mask_pixel_step(patch_size, patch_stride, downsample_factor)
        self.assertEqual(result_value, output_value)
        #
        patch_size = 512
        patch_stride = 0.25
        downsample_factor = 16
        output_value = 8
        result_value = mask.get_mask_pixel_step(patch_size, patch_stride, downsample_factor)
        self.assertEqual(result_value, output_value)
        #
        patch_size = 64
        patch_stride = 0.25
        downsample_factor = 16
        output_value = 1
        result_value = mask.get_mask_pixel_step(patch_size, patch_stride, downsample_factor)
        self.assertEqual(result_value, output_value)
        #
        patch_size = 64
        patch_stride = 0.025
        downsample_factor = 16
        output_value = 0.1
        result_value = mask.get_mask_pixel_step(patch_size, patch_stride, downsample_factor)
        self.assertEqual(result_value, output_value)
        #
        patch_size = 64
        patch_stride = 0.0025
        downsample_factor = 16
        with self.assertRaises(ValueError):
            mask.get_mask_pixel_step(patch_size, patch_stride, downsample_factor)
        #
        patch_size = 512
        patch_stride = 5 + 1
        downsample_factor = 4
        with self.assertRaises(ValueError):
            mask.get_mask_pixel_step(patch_size, patch_stride, downsample_factor)
