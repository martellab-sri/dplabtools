# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions included in slides.utils.image."""

from unittest import TestCase

import numpy as np
from PIL import Image

from dplabtools.slides.utils import image
from testutils import make_test_path


class TestUtilsImageTif(TestCase):
    """Tests for tif functions included in slides.utils.image."""

    def set_pretest_data(self):
        self.input_array = np.empty((250, 180, 4), dtype=np.uint8)
        self.input_array[0:250, 0:90] = [0, 91, 187, 255]
        self.input_array[0:250, 90:180] = [255, 213, 0, 255]
        self.resolution_data = (250.0, 250.0, "CENTIMETER")

    def test_save_tif_image_with_resolution(self):
        """Run two tests: with compression and without.

        NOTES:
        Adding compression may make results non-deterministic (between library versions or different OSes),
        so bit-by-bit comparison cannot be done. For instance this (one pixel comparison):

              np.testing.assert_equal(result_image_array[0,-1], output_image_array[0,-1])

        may yield:
              Mismatched elements: 2 / 3 (66.7%)
              Max absolute difference: 255
              Max relative difference: 255.
               x: array([  0,  91, 187], dtype=uint8)
               y: array([  1,  90, 187], dtype=uint8)

        That's why only some smaller areas of images will be compared when compression is used.
        """
        # no compression
        result_test_image_tif = make_test_path("saved_data/utils/test_tif1.tif")
        self.set_pretest_data()
        image.save_tif_image_with_resolution(
            np.transpose(self.input_array), result_test_image_tif, self.resolution_data, jpeg_compression=False
        )
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif1.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # with compression (RGB only)
        result_test_image_tif = make_test_path("saved_data/utils/test_tif2.tif")
        self.set_pretest_data()
        self.input_array = self.input_array[:, :, 0:3]
        image.save_tif_image_with_resolution(
            np.transpose(self.input_array), result_test_image_tif, self.resolution_data, jpeg_compression=True
        )
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif2.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array[10:20, 10:20], output_image_array[10:20, 10:20])
        np.testing.assert_equal(result_image_array[210:220, 110:120], output_image_array[210:220, 110:120])

    def test_pad_image_zero_background(self):
        # zero background present at bottom right
        result_test_image_tif = make_test_path("saved_data/utils/test_tif3_bottom.tif")
        input_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif3a.tif"))
        result_image = image.pad_image_zero_background(input_image, background_color=[0, 255, 0])
        result_image.save(result_test_image_tif)
        input_image.close()
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif3b.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # zero background not present (input image should not change)
        input_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif4.tif"))
        result_image = image.pad_image_zero_background(input_image, background_color=[0, 255, 0])
        input_image_array = np.asarray(input_image)
        result_image_array = np.asarray(result_image)
        input_image.close()
        np.testing.assert_equal(result_image_array, input_image_array)
        # zero background present at top left
        result_test_image_tif = make_test_path("saved_data/utils/test_tif3_top.tif")
        input_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif3c.tif"))
        result_image = image.pad_image_zero_background(input_image, background_color=[0, 255, 0])
        result_image.save(result_test_image_tif)
        input_image.close()
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif3d.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)


class TestUtilsImageColors(TestCase):
    """Tests for color related functions included in slides.utils.image."""

    def test_get_transparency(self):
        input_value = 0.6
        output_value = 153
        result_value = image.get_transparency(input_value)
        self.assertEqual(result_value, output_value)
        #
        input_value = 0
        output_value = 0
        result_value = image.get_transparency(input_value)
        self.assertEqual(result_value, output_value)
        #
        input_value = 1
        output_value = 255
        result_value = image.get_transparency(input_value)
        self.assertEqual(result_value, output_value)

    def test_get_color_hex(self):
        input_value = "red"
        output_value = "#FF0000"
        result_value = image.get_color_hex(input_value)
        self.assertEqual(result_value, output_value)
        #
        input_value = "#12aa22"
        output_value = "#12aa22"
        result_value = image.get_color_hex(input_value)
        self.assertEqual(result_value, output_value)
        #
        input_value = "#12BB44"
        output_value = "#12BB44"
        result_value = image.get_color_hex(input_value)
        self.assertEqual(result_value, output_value)

    def test_get_color_rgb(self):
        input_value = "#FF0000"
        output_value = (255, 0, 0)
        result_value = image.get_color_rgb(input_value)
        self.assertEqual(result_value, output_value)
        #
        input_value = "#cc00ff"
        output_value = (204, 0, 255)
        result_value = image.get_color_rgb(input_value)
        self.assertEqual(result_value, output_value)
