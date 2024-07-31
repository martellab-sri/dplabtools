# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions included in slides.utils.wsi."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from dplabtools.slides import GenericSlide
from dplabtools.slides.utils import wsi
from testutils import make_test_path


class TestUtilsWsi(TestCase):
    """Tests for functions included in slides.utils.wsi."""

    wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
    wsi_slide = GenericSlide(wsi_file=wsi_file)

    def test_get_wsi_name(self):
        input_string = "/tmp/dir/aaa.txt"
        output_string = "aaa.txt"
        result_string = wsi.get_wsi_name(input_string)
        self.assertEqual(result_string, output_string)

    def test_get_wsi_id(self):
        input_string = "/tmp/dir/aaa.txt"
        output_string = "aaa"
        result_string = wsi.get_wsi_id(input_string)
        self.assertEqual(result_string, output_string)

    def test_get_wsi_level_image(self):
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_scan_level1.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = wsi.get_wsi_level_image(self.wsi_slide, level=1)
        result_image_array = np.asarray(result_image)
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_get_wsi_level_array(self):
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_scan_level1.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_array = wsi.get_wsi_level_array(self.wsi_slide, level=1)
        np.testing.assert_equal(result_array, output_image_array)

    def test_get_wsi_level_zero_array(self):
        output_image = Image.open(make_test_path("wsi/board-flat.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_array = wsi.get_wsi_level_zero_array(self.wsi_slide.slide_file)
        np.testing.assert_equal(result_array, output_image_array)

    def test_find_wsi_level(self):
        # level 1
        input_tuple = (640, 768)
        output_level = 1
        result_level = wsi.find_wsi_level(self.wsi_slide, input_tuple)
        self.assertEqual(result_level, output_level)
        # level 2
        input_tuple = (160, 192)
        output_level = 2
        result_level = wsi.find_wsi_level(self.wsi_slide, input_tuple)
        self.assertEqual(result_level, output_level)
        # not found
        with self.assertRaises(ValueError):
            input_tuple = (333, 333)
            result_level = wsi.find_wsi_level(self.wsi_slide, input_tuple)

    def test_get_wsi_downsample_factor(self):
        # level 1
        input_tuple = (640, 768)
        output_factor = 4.0
        result_factor = wsi.get_wsi_downsample_factor(self.wsi_slide, input_tuple)
        self.assertEqual(result_factor, output_factor)
        # level 2
        input_tuple = (160, 192)
        output_factor = 16.0
        result_factor = wsi.get_wsi_downsample_factor(self.wsi_slide, input_tuple)
        self.assertEqual(result_factor, output_factor)
        # not found
        with self.assertRaises(ValueError):
            input_tuple = (333, 333)
            result_factor = wsi.find_wsi_level(self.wsi_slide, input_tuple)
        # check types
        input_tuple = (160, 192)
        output_factor_type = float
        result_factor_type = type(wsi.get_wsi_downsample_factor(self.wsi_slide, input_tuple))
        self.assertEqual(output_factor_type, result_factor_type)

    def test_compute_wsi_resolution_data1(self):
        wsi_file = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide = GenericSlide(wsi_file=wsi_file)
        downsample_factor = wsi_slide.level_downsamples[1]
        output_tuple = (9920.2783, 9920.2783, "CENTIMETER")
        result_tuple = wsi.compute_wsi_resolution_data(wsi_slide, downsample_factor)
        result_tuple = (round(result_tuple[0], 4), round(result_tuple[1], 4), result_tuple[2])
        self.assertEqual(result_tuple, output_tuple)

    @patch("dplabtools.slides.libs.genericslide.GenericSlide._get_mpp_data")
    def test_compute_wsi_resolution_data2(self, mock_func):
        mock_func.return_value = (0.001, 0.001)
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide = GenericSlide(wsi_file=wsi_file)
        downsample_factor = 16
        output_tuple = (625000.0, 625000.0, "CENTIMETER")
        result_tuple = wsi.compute_wsi_resolution_data(wsi_slide, downsample_factor)
        self.assertEqual(result_tuple, output_tuple)

    def test_find_nearest_wsi_level(self):
        # level 1
        input_minsize = 100
        output_level_found = 2
        result_level_found = wsi.find_nearest_wsi_level(self.wsi_slide, input_minsize, allow_level_zero=False)
        self.assertEqual(result_level_found, output_level_found)
        # level 2
        input_minsize = 500
        output_level_found = 1
        result_level_found = wsi.find_nearest_wsi_level(self.wsi_slide, input_minsize, allow_level_zero=False)
        self.assertEqual(result_level_found, output_level_found)
        # not found
        input_minsize = 1000
        with self.assertRaises(ValueError):
            result_level_found = wsi.find_nearest_wsi_level(self.wsi_slide, input_minsize, allow_level_zero=False)
        # level 0
        input_minsize = 1000
        output_level_found = 0
        result_level_found = wsi.find_nearest_wsi_level(self.wsi_slide, input_minsize, allow_level_zero=True)
        self.assertEqual(result_level_found, output_level_found)

    def test_get_resampled_wsi_images_one(self):
        result_test_image_tif = make_test_path("saved_data/utils/test_tif5.tif")
        resized_image_list = wsi.get_resampled_wsi_images(self.wsi_file, [(160 * 3, 192 * 3)])
        resized_image_list[0].save(result_test_image_tif)
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif5.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_get_resampled_wsi_images_multi(self):
        resized_image_list = wsi.get_resampled_wsi_images(
            self.wsi_file, [(160 * 3, 192 * 3), (160 * 2, 192 * 2), (160 * 4, 192 * 4)]
        )
        for num, img in enumerate(resized_image_list, start=6):
            result_test_image_tif = make_test_path("saved_data/utils/test_tif%d.tif" % num)
            img.save(result_test_image_tif)
            output_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif%d.tif" % num))
            output_image_array = np.asarray(output_image)
            output_image.close()
            result_image = Image.open(result_test_image_tif)
            result_image_array = np.asarray(result_image)
            result_image.close()
            np.testing.assert_equal(result_image_array, output_image_array)

    def test_get_resampled_wsi_images_changed_filter(self):
        # reference image was processed using LANCZOS
        result_test_image_tif = make_test_path("saved_data/utils/test_tif5_filter.tif")
        resized_image_list = wsi.get_resampled_wsi_images(self.wsi_file, [(160 * 3, 192 * 3)], resize_filter="NEAREST")
        resized_image_list[0].save(result_test_image_tif)
        output_image = Image.open(make_test_path("ref_data/slides/utils/ref_tif5.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # images must be different as different filters were used
        self.assertFalse(np.array_equal(result_image_array, output_image_array))

    def test_get_resampled_tiles_one(self):
        wsi_tile = Image.new(mode="RGB", size=(250, 250))
        size_list = [(50, 70)]
        result_tiles = wsi.get_resampled_tiles(wsi_tile, size_list)
        self.assertEqual(result_tiles[0].width, 50)
        self.assertEqual(result_tiles[0].height, 70)

    def test_get_resampled_tiles_multi(self):
        wsi_tile = Image.new(mode="RGB", size=(250, 250))
        size_list = [(50, 70), (40, 50), (20, 30)]
        result_tiles = wsi.get_resampled_tiles(wsi_tile, size_list)
        self.assertEqual(result_tiles[0].width, 50)
        self.assertEqual(result_tiles[0].height, 70)
        self.assertEqual(result_tiles[1].width, 40)
        self.assertEqual(result_tiles[1].height, 50)
        self.assertEqual(result_tiles[2].width, 20)
        self.assertEqual(result_tiles[2].height, 30)

    def test_get_resampled_tiles_changed_filter(self):
        # use previous test image as a tile (must be real image for different filters test)
        wsi_tile = Image.open(make_test_path("ref_data/slides/utils/ref_tif5.tif"))
        size_list = [(50, 70)]
        result_tiles1 = wsi.get_resampled_tiles(wsi_tile, size_list)
        result_tiles2 = wsi.get_resampled_tiles(wsi_tile, size_list, resize_filter="NEAREST")
        wsi_tile.close()
        self.assertEqual(result_tiles1[0].width, 50)
        self.assertEqual(result_tiles1[0].height, 70)
        self.assertEqual(result_tiles2[0].width, 50)
        self.assertEqual(result_tiles2[0].height, 70)
        tiles1_array = np.asarray(result_tiles1[0])
        tiles2_array = np.asarray(result_tiles2[0])
        # images must be different as different filters were used
        self.assertFalse(np.array_equal(tiles1_array, tiles2_array))

    def test_get_level_and_mpp(self):
        input_level_or_mpp = 3.14
        result_level, result_mpp = wsi.get_level_and_mpp(input_level_or_mpp)
        self.assertIsNone(result_level)
        self.assertEqual(result_mpp, 3.14)
        #
        input_level_or_mpp = 2
        result_level, result_mpp = wsi.get_level_and_mpp(input_level_or_mpp)
        self.assertEqual(result_level, 2)
        self.assertIsNone(result_mpp)
        #
        input_level_or_mpp = 0
        result_level, result_mpp = wsi.get_level_and_mpp(input_level_or_mpp)
        self.assertEqual(result_level, 0)
        self.assertIsNone(result_mpp)
        #
        input_level_or_mpp = 0.0
        with self.assertRaises(ValueError):
            wsi.get_level_and_mpp(input_level_or_mpp)
        #
        input_level_or_mpp = "abc"
        with self.assertRaises(ValueError):
            wsi.get_level_and_mpp(input_level_or_mpp)

    def test_get_basepatch_mpp(self):
        level_mpp_values = {1: "aaa", 2: "bbb", 3: "ccc"}
        #
        basepatch_level_or_mpp = 3
        output_mpp = "ccc"
        result_mpp = wsi.get_basepatch_mpp(basepatch_level_or_mpp, level_mpp_values)
        self.assertEqual(result_mpp, output_mpp)
        #
        basepatch_level_or_mpp = 1.5
        output_mpp = 1.5
        result_mpp = wsi.get_basepatch_mpp(basepatch_level_or_mpp, level_mpp_values)
        self.assertEqual(result_mpp, output_mpp)

    def test_get_level_or_level(self):
        level_or_minsize = 0
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 0)
        #
        level_or_minsize = 2
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 2)
        #
        level_or_minsize = 50
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 50)
        #
        level_or_minsize = 100
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 2)
        #
        level_or_minsize = 500
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 1)
        #
        level_or_minsize = 1000
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 0)

    @patch("dplabtools.slides.utils.wsi.LEVEL_MINSIZE_SPLIT", 45)
    def test_get_level_or_level_split(self):
        level_or_minsize = 40
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 40)
        #
        level_or_minsize = 50
        result_level = wsi.get_level_or_level(self.wsi_slide, level_or_minsize)
        self.assertEqual(result_level, 2)

    def test_get_target_resample_factor(self):
        patch_level = 2
        patch_mpp = None
        level_downsamples = [1, 4, 16]
        level_mpp_values = []
        result_value = wsi.get_target_resample_factor(patch_level, patch_mpp, level_downsamples, level_mpp_values)
        output_value = 16
        self.assertEqual(result_value, output_value)
        #
        patch_level = 0
        patch_mpp = None
        level_downsamples = [1, 4, 16]
        level_mpp_values = []
        result_value = wsi.get_target_resample_factor(patch_level, patch_mpp, level_downsamples, level_mpp_values)
        output_value = 1
        self.assertEqual(result_value, output_value)
        #
        patch_level = None
        patch_mpp = 6
        level_downsamples = []
        level_mpp_values = [3, 5, 7]
        result_value = wsi.get_target_resample_factor(patch_level, patch_mpp, level_downsamples, level_mpp_values)
        output_value = 6 / 3
        self.assertEqual(result_value, output_value)
