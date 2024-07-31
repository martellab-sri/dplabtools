# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for GenericSlide.

Tested classes:
    GenericSlide
"""

import os
import warnings
from unittest import TestCase, skipIf
from unittest.mock import patch
from collections import OrderedDict

import numpy as np
import PIL
from PIL import Image
from skimage import io

from dplabtools.slides import GenericSlide
from testconfig import fast_tests_only
from testutils import make_test_path
from testconsts import testdata_dir


class TestGenericSlideBasicProperties(TestCase):
    """Tests for basic slide properties."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)

    def test_property_magnification(self):
        output_magnification = 40
        result_magnification = self.wsi_slide_svs.magnification
        self.assertEqual(result_magnification, output_magnification)
        self.assertEqual(type(result_magnification), type(output_magnification))

    def test_property_level_dimensions(self):
        output_level_dimensions = ((27818, 31316), (6954, 7829), (1738, 1957))
        result_level_dimensions = self.wsi_slide_svs.level_dimensions
        self.assertEqual(result_level_dimensions, output_level_dimensions)

    def test_property_level_downsamples(self):
        output_level_downsamples = (1.0, 4.000144, 16.003899)
        result_level_downsamples = self.wsi_slide_svs.level_downsamples
        result_level_downsamples = tuple(round(ld, 6) for ld in result_level_downsamples)
        self.assertEqual(result_level_downsamples, output_level_downsamples)

    def test_property_mpp_data(self):
        output_mpp_data = (0.252, 0.252)
        result_mpp_data = self.wsi_slide_svs.mpp_data
        self.assertEqual(result_mpp_data, output_mpp_data)

    def test_property_level_mpp_values(self):
        output_mpp_values = (0.252, 1.00804, 4.03298)
        result_mpp_values = self.wsi_slide_svs.level_mpp_values
        result_mpp_values = tuple(round(mpp, 5) for mpp in result_mpp_values)
        self.assertEqual(result_mpp_values, output_mpp_values)

    def test_property_level_count(self):
        output_count = 3
        result_count = self.wsi_slide_svs.level_count
        self.assertEqual(result_count, output_count)

    def test_property_thumbnail_image(self):
        result_image = self.wsi_slide_svs.thumbnail_image
        self.assertTrue(result_image.width > 0)
        self.assertTrue(result_image.height > 0)

    def test_property_lib_name(self):
        result_lib_name = self.wsi_slide_svs.lib_name
        self.assertTrue(len(result_lib_name) > 0)

    def test_property_all_properties(self):
        result_all_properties = self.wsi_slide_svs.all_properties
        self.assertTrue(len(result_all_properties) > 10)
        self.assertTrue(len(result_all_properties.keys()) > 10)
        self.assertTrue(len(result_all_properties.values()) > 10)

    def test_property_level_dimensions_extra(self):
        output_level_dimensions = OrderedDict()
        result_level_dimensions = self.wsi_slide_svs.level_dimensions_extra
        self.assertEqual(result_level_dimensions, output_level_dimensions)

    def test_property_level_resamples_extra(self):
        output_level_resamples = OrderedDict()
        result_level_resamples = self.wsi_slide_svs.level_resamples_extra
        self.assertEqual(result_level_resamples, output_level_resamples)

    def test_property_mpp_values_extra(self):
        output_mpp_values = ()
        result_mpp_values = self.wsi_slide_svs.level_mpp_values_extra
        self.assertEqual(result_mpp_values, output_mpp_values)

    def test_property_level_count_extra(self):
        output_count = 0
        result_count = self.wsi_slide_svs.level_count_extra
        self.assertEqual(result_count, output_count)


class TestGenericSlideFileAndObjectProperties(TestCase):
    """Tests for basic file related properties."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)

    def test_property_slide_file(self):
        output_file = os.path.join(testdata_dir, "wsi", "TUPAC-TE-234.svs")
        result_file = self.wsi_slide_svs.slide_file
        self.assertEqual(result_file, output_file)

    def test_property_slide_name(self):
        output_name = "TUPAC-TE-234.svs"
        result_name = self.wsi_slide_svs.slide_name
        self.assertEqual(result_name, output_name)

    def test_property_slide_id(self):
        output_id = "TUPAC-TE-234"
        result_id = self.wsi_slide_svs.slide_id
        self.assertEqual(result_id, output_id)

    def test_property_slide_object(self):
        result_object = self.wsi_slide_svs.slide_object
        self.assertIsNotNone(result_object)


class TestGenericSlideClassAttributesProperties(TestCase):
    """Tests for properties implemented as class attributes."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)
        self.wsi_slide_svs.set_external_mpp(1.3)

    def tearDown(self):
        self.wsi_slide_svs.set_external_mpp(None)

    def test_class_attributes_properties(self):
        self.assertIsNotNone(self.wsi_slide_svs.external_mpp)
        self.assertIsNotNone(self.wsi_slide_svs.mpp_round_decimal_places)
        self.assertIsNotNone(self.wsi_slide_svs.range_min_mpp)
        self.assertIsNotNone(self.wsi_slide_svs.range_max_magnification)
        self.assertIsNotNone(self.wsi_slide_svs.resampling_filter)
        self.assertIsNotNone(self.wsi_slide_svs.mpp_level_margin)
        self.assertIsNotNone(self.wsi_slide_svs.padding_margin_pixels)
        self.assertIsNotNone(self.wsi_slide_svs.level_zero_resampling)


class TestGenericSlideExternalMPPProperty(TestCase):
    """Tests for externally provided MPP data."""

    def test_mpp_missing_and_ignored(self):
        wsi_file_svs1 = make_test_path("wsi/TCGA-OL-A5RX-01Z-00-DX1.15A0D4F4-2744-4D44-8883-27FF83D9C824.svs")
        GenericSlide.set_external_mpp(0.345)
        wsi_slide_svs1 = GenericSlide(wsi_file=wsi_file_svs1)
        output_mpp_data1 = (0.345, 0.345)
        result_mpp_data1 = wsi_slide_svs1.mpp_data
        self.assertEqual(result_mpp_data1, output_mpp_data1)
        self.assertEqual(wsi_slide_svs1.external_mpp, 0.345)
        #  MPP set previously should be ignored as embedded MPP is present
        wsi_file_svs2 = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs2 = GenericSlide(wsi_file=wsi_file_svs2)
        output_mpp_data2 = (0.252, 0.252)
        result_mpp_data2 = wsi_slide_svs2.mpp_data
        self.assertEqual(result_mpp_data2, output_mpp_data2)
        # Resetting  MPP value at the end is required so subsequent tests would not fail
        GenericSlide.set_external_mpp(None)
        #
        self.assertEqual(wsi_slide_svs1.external_mpp, None)
        self.assertEqual(wsi_slide_svs2.external_mpp, None)

    def test_mpp_embedded_present(self):
        # in all 3 cases external MPP should be ignored
        wsi_file_svs1 = make_test_path("wsi/TUPAC-TE-234.svs")
        GenericSlide.set_external_mpp(0.999)
        wsi_slide_svs1 = GenericSlide(wsi_file=wsi_file_svs1)
        output_mpp_data1 = (0.252, 0.252)
        result_mpp_data1 = wsi_slide_svs1.mpp_data
        self.assertEqual(result_mpp_data1, output_mpp_data1)
        #
        wsi_file_svs2 = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_svs2 = GenericSlide(wsi_file=wsi_file_svs2)
        output_mpp_data2 = (0.25, 0.25)
        result_mpp_data2 = wsi_slide_svs2.mpp_data
        self.assertEqual(result_mpp_data2, output_mpp_data2)
        #
        wsi_file_svs3 = make_test_path("wsi/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.tif")
        wsi_slide_svs3 = GenericSlide(wsi_file=wsi_file_svs3)
        output_mpp_data3 = (0.5, 0.5)
        result_mpp_data3 = wsi_slide_svs3.mpp_data
        self.assertEqual(result_mpp_data3, output_mpp_data3)
        #
        GenericSlide.set_external_mpp(None)

    def test_mpp_all_missing(self):
        wsi_file_svs1 = make_test_path("wsi/TCGA-OL-A5RX-01Z-00-DX1.15A0D4F4-2744-4D44-8883-27FF83D9C824.svs")
        GenericSlide.set_external_mpp(1.111)
        wsi_slide_svs1 = GenericSlide(wsi_file=wsi_file_svs1)
        output_mpp_data1 = (1.111, 1.111)
        result_mpp_data1 = wsi_slide_svs1.mpp_data
        self.assertEqual(result_mpp_data1, output_mpp_data1)
        self.assertEqual(wsi_slide_svs1.external_mpp, 1.111)
        #
        wsi_file_svs2 = make_test_path("wsi/TCGA-OL-A5RX-01Z-00-DX1.15A0D4F4-2744-4D44-8883-27FF83D9C824.svs")
        GenericSlide.set_external_mpp(1.222)
        wsi_slide_svs2 = GenericSlide(wsi_file=wsi_file_svs2)
        output_mpp_data2 = (1.222, 1.222)
        result_mpp_data2 = wsi_slide_svs2.mpp_data
        self.assertEqual(result_mpp_data2, output_mpp_data2)
        self.assertEqual(wsi_slide_svs2.external_mpp, 1.222)
        #
        wsi_file_svs3 = make_test_path("wsi/TCGA-OL-A5RX-01Z-00-DX1.15A0D4F4-2744-4D44-8883-27FF83D9C824.svs")
        GenericSlide.set_external_mpp(1.333)
        wsi_slide_svs3 = GenericSlide(wsi_file=wsi_file_svs3)
        output_mpp_data3 = (1.333, 1.333)
        result_mpp_data3 = wsi_slide_svs3.mpp_data
        self.assertEqual(result_mpp_data3, output_mpp_data3)
        self.assertEqual(wsi_slide_svs3.external_mpp, 1.333)
        #
        GenericSlide.set_external_mpp(None)


class TestGenericSlideMPPDataProperty(TestCase):
    """Tests for reading MPP data property using different scenarios."""

    def tearDown(self):
        GenericSlide.set_mpp_round_decimal_places(5)

    def test_mpp_data_custom_function(self):
        wsi_file_tif = make_test_path("wsi/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        result_mpp_data = wsi_slide_tif.mpp_data
        output_mpp_data = (0.5, 0.5)
        self.assertEqual(result_mpp_data, output_mpp_data)

    def test_mpp_data_not_available(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        result_mpp_data = wsi_slide_tif.mpp_data
        output_mpp_data = (None, None)
        self.assertEqual(result_mpp_data, output_mpp_data)

    @patch("dplabtools.slides.libs.genericslide.GenericSlide._get_mpp_data")
    def test_mpp_data_tif_rounded1(self, mock_func):
        GenericSlide.set_mpp_round_decimal_places(5)
        mock_func.return_value = (0.22632099999999999, 0.22631600000000002)
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        result_mpp_data = wsi_slide_tif.mpp_data
        output_mpp_data = (0.22632, 0.22632)
        self.assertEqual(result_mpp_data, output_mpp_data)

    @patch("dplabtools.slides.libs.genericslide.GenericSlide._get_mpp_data")
    def test_mpp_data_tif_rounded2(self, mock_func):
        GenericSlide.set_mpp_round_decimal_places(2)
        mock_func.return_value = (0.22632099999999999, 0.22631600000000002)
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        result_mpp_data = wsi_slide_tif.mpp_data
        output_mpp_data = (0.23, 0.23)
        self.assertEqual(result_mpp_data, output_mpp_data)

    @patch("dplabtools.slides.libs.genericslide.GenericSlide._get_mpp_data")
    def test_mpp_data_not_supported(self, mock_func):
        GenericSlide.set_mpp_round_decimal_places(5)
        mock_func.return_value = (0.11111111111111, 0.22222222222222)
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=wsi_file_tif)


class TestGenericSlideLevelMPPValuesProperty(TestCase):
    """Tests for reading level MPP values."""

    def test_mpp_not_available(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        output_mpp_values = (None, None, None)
        result_mpp_values = wsi_slide_tif.level_mpp_values
        self.assertEqual(result_mpp_values, output_mpp_values)


class TestGenericSlidelLabelImageProperty(TestCase):
    """Tests for reading embedded image label."""

    def test_image_label_present(self):
        wsi_file_svs = make_test_path("wsi/JP2K-33003-1.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs)
        result_image = wsi_slide_svs.label_image
        result_image_size = (result_image.width, result_image.height)
        output_image_size = (415, 422)
        self.assertEqual(result_image_size, output_image_size)

    def test_image_label_not_present(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        result_image = wsi_slide_tif.label_image
        self.assertIsNone(result_image)


class TestGenericSlidelMagnification(TestCase):
    """Tests for reading embedded magnification value."""

    def test_magnification_present(self):
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs)
        result_magnification = wsi_slide_svs.magnification
        self.assertEqual(result_magnification, 40)

    def test_magnification_not_present(self):
        # this will also produce a log message
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        result_magnification = wsi_slide_tif.magnification
        self.assertIsNone(result_magnification)


class TestGenericSlideGetRegionArray(TestCase):
    """Tests for reading region as numpy array."""

    def test_get_region_array1(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        level = 0
        location = (0, 0)
        size = (400, 500)
        region_image = wsi_slide_tif.get_region(location, level, size)
        region_image_array = np.asarray(region_image)
        region_array = wsi_slide_tif.get_region_array(location, level, size)
        self.assertEqual(region_array.shape, tuple(reversed(size)) + (3,))
        np.testing.assert_equal(region_image_array, region_array)

    def test_get_region_array2(self):
        wsi_file_tif = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        level = 1
        location = (20000, 20000)
        size = (1000, 1200)
        region_image = wsi_slide_tif.get_region(location, level, size)
        region_image_array = np.asarray(region_image)
        region_array = wsi_slide_tif.get_region_array(location, level, size)
        self.assertEqual(region_array.shape, tuple(reversed(size)) + (3,))
        np.testing.assert_equal(region_image_array, region_array)

    def test_get_region_array3(self):
        wsi_file_tif = make_test_path("wsi/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        level = 0
        location = (5000, 5000)
        size = (400, 300)
        region_image = wsi_slide_tif.get_region(location, level, size)
        region_image_array = np.asarray(region_image)
        region_array = wsi_slide_tif.get_region_array(location, level, size)
        self.assertEqual(region_array.shape, tuple(reversed(size)) + (3,))
        np.testing.assert_equal(region_image_array, region_array)

    def test_get_region_array4(self):
        wsi_file_tif = make_test_path("wsi/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        level = 1
        location = (5000, 6000)
        size = (300, 300)
        region_image = wsi_slide_tif.get_region(location, level, size)
        region_image_array = np.asarray(region_image)
        region_array = wsi_slide_tif.get_region_array(location, level, size)
        self.assertEqual(region_array.shape, tuple(reversed(size)) + (3,))
        np.testing.assert_equal(region_image_array, region_array)


class TestGenericSlideGetRegionLocationCheckSpanWholeImage(TestCase):
    """Tests for location checks while reading patches using level values."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif)

    def test_get_region_out_of_range(self):
        level_or_mpp = 0
        location = (-1, -1)
        size = (2560 + 2, 3072 + 2)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        #
        location = (-1, 1)
        size = (2560 + 2, 3072)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        #
        location = (1, -1)
        size = (2560, 3072 + 2)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        #
        location = (-282, 2278)
        size = (819, 819)
        region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        self.assertEqual(region.size, size)
        #
        location = (2300, -282)
        size = (819, 819)
        region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        self.assertEqual(region.size, size)
        #
        location = (-282, 2278)
        size = (819, 5819)
        region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        self.assertEqual(region.size, size)
        #
        location = (2300, -282)
        size = (5819, 819)
        region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        self.assertEqual(region.size, size)
        #
        location = (-100, 2000)
        size = (20, 2000)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        #
        location = (2000, -100)
        size = (2000, 20)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)


class TestGenericSlideGetRegionWithoutMPP(TestCase):
    """Tests for reading patches using level values."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)
        self.wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif)

    def test_get_region_out_of_range(self):
        level_or_mpp = 0
        size = (100, 100)
        #
        location = (10000, 10000)
        region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        self.assertEqual(region.size, size)
        #
        location = (100000, 100000)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (1000, 100000)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (100000, 1000)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (-100, -100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (-100, 100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (100, -100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (-100, -99)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (-99, -100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        level_or_mpp = 1
        size = (25, 25)
        #
        location = (-100, -100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (-100, 100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (100, -100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (-100, -99)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        #
        location = (-99, -100)
        with self.assertRaises(ValueError):
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)

    def test_get_region_in_range(self):
        # tiffslide will throw "out of bounds" warning here, so it's silenced temporarily
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="location")
            level_or_mpp = 0
            size = (100, 100)
            location = (-99, -99)
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
            self.assertEqual(region.size, size)
            #
            level_or_mpp = 1
            size = (25, 25)
            location = (-99, -99)
            region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
            self.assertEqual(region.size, size)

    def test_get_region0_no_padding(self):
        result_region_tif = make_test_path("saved_data/libs/test_region0.tif")
        location = (256 * 3, 256 * 5)
        level_or_mpp = 0
        size = (512, 512)
        region = self.wsi_slide_tif.get_region(location, level_or_mpp, size, skip_padding=True)
        region.save(result_region_tif)
        output_image = Image.open(make_test_path("ref_data/slides/libs/regions/region0.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_region_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # check if output is in RGB format
        np.testing.assert_equal(result_image_array.shape[2], 3)

    def test_get_region0_with_padding(self):
        result_region_tif = make_test_path("saved_data/libs/test_region0.tif")
        location = (256 * 3, 256 * 5)
        level_or_mpp = 0
        size = (512, 512)
        region = self.wsi_slide_tif.get_region(location, level_or_mpp, size)
        region.save(result_region_tif)
        output_image = Image.open(make_test_path("ref_data/slides/libs/regions/region0.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_region_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_get_region1_no_padding(self):
        result_region_tif = make_test_path("saved_data/libs/test_region1.tif")
        location = (256 * 4, 256 * 6)
        level_or_mpp = 1
        size = (256, 256)
        region = self.wsi_slide_tif.get_region(location, level_or_mpp, size, skip_padding=True)
        region.save(result_region_tif)
        output_image = Image.open(make_test_path("ref_data/slides/libs/regions/region1.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_region_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_get_region_with_padding(self):
        result_region_tif = make_test_path("saved_data/libs/test_region2.tif")
        location = (26660, 24500)
        level_or_mpp = 1
        size = (512, 512)
        white_padded_box = (290, 0, 512, 512)
        region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        region.save(result_region_tif)
        output_image = Image.open(make_test_path("ref_data/slides/libs/regions/region2.tif")).crop(white_padded_box)
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_region_tif).crop(white_padded_box)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # only white padded area is compared, after image cropping
        np.testing.assert_equal(result_image_array, output_image_array)
        # all pixel components must be 255
        self.assertEqual(np.min(result_image_array), 255)

    def test_get_region_different_levels_at_once_no_padding(self):
        result_region_tif1 = make_test_path("saved_data/libs/test_multi_region1.tif")
        result_region_tif2 = make_test_path("saved_data/libs/test_multi_region2.tif")
        result_region_tif3 = make_test_path("saved_data/libs/test_multi_region3.tif")
        location1 = (256 * 9, 256 * 7)
        location2 = (256, 0)
        location3 = (256 * 2, 256 * 3)
        level_or_mpp1 = 0
        level_or_mpp2 = 1
        level_or_mpp3 = 2
        size1 = (256, 256)
        size2 = (512, 512)
        size3 = (128, 128)
        region1 = self.wsi_slide_tif.get_region(location1, level_or_mpp1, size1, skip_padding=True)
        region2 = self.wsi_slide_tif.get_region(location2, level_or_mpp2, size2, skip_padding=True)
        region3 = self.wsi_slide_tif.get_region(location3, level_or_mpp3, size3, skip_padding=True)
        region1.save(result_region_tif1)
        region2.save(result_region_tif2)
        region3.save(result_region_tif3)
        # image1
        output_image1 = Image.open(make_test_path("ref_data/slides/libs/multi/multi_region1.tif"))
        output_image_array1 = np.asarray(output_image1)
        output_image1.close()
        result_image1 = Image.open(result_region_tif1)
        result_image_array1 = np.asarray(result_image1)
        result_image1.close()
        np.testing.assert_equal(result_image_array1, output_image_array1)
        # image2
        output_image2 = Image.open(make_test_path("ref_data/slides/libs/multi/multi_region2.tif"))
        output_image_array2 = np.asarray(output_image2)
        output_image2.close()
        result_image2 = Image.open(result_region_tif2)
        result_image_array2 = np.asarray(result_image2)
        result_image2.close()
        np.testing.assert_equal(result_image_array2, output_image_array2)
        # image3
        output_image3 = Image.open(make_test_path("ref_data/slides/libs/multi/multi_region3.tif"))
        output_image_array3 = np.asarray(output_image3)
        output_image3.close()
        result_image3 = Image.open(result_region_tif3)
        result_image_array3 = np.asarray(result_image3)
        result_image3.close()
        np.testing.assert_equal(result_image_array3, output_image_array3)

    def test_get_region_different_levels_at_once_with_padding(self):
        result_region_tif1 = make_test_path("saved_data/libs/test_multi_region1.tif")
        result_region_tif2 = make_test_path("saved_data/libs/test_multi_region2.tif")
        result_region_tif3 = make_test_path("saved_data/libs/test_multi_region3.tif")
        location1 = (256 * 9, 256 * 7)
        location2 = (256, 0)
        location3 = (256 * 2, 256 * 3)
        level_or_mpp1 = 0
        level_or_mpp2 = 1
        level_or_mpp3 = 2
        size1 = (256, 256)
        size2 = (512, 512)
        size3 = (128, 128)
        region1 = self.wsi_slide_tif.get_region(location1, level_or_mpp1, size1)
        region2 = self.wsi_slide_tif.get_region(location2, level_or_mpp2, size2)
        region3 = self.wsi_slide_tif.get_region(location3, level_or_mpp3, size3)
        region1.save(result_region_tif1)
        region2.save(result_region_tif2)
        region3.save(result_region_tif3)
        # image1
        output_image1 = Image.open(make_test_path("ref_data/slides/libs/multi/multi_region1.tif"))
        output_image_array1 = np.asarray(output_image1)
        output_image1.close()
        result_image1 = Image.open(result_region_tif1)
        result_image_array1 = np.asarray(result_image1)
        result_image1.close()
        np.testing.assert_equal(result_image_array1, output_image_array1)
        # image2
        output_image2 = Image.open(make_test_path("ref_data/slides/libs/multi/multi_region2.tif"))
        output_image_array2 = np.asarray(output_image2)
        output_image2.close()
        result_image2 = Image.open(result_region_tif2)
        result_image_array2 = np.asarray(result_image2)
        result_image2.close()
        np.testing.assert_equal(result_image_array2, output_image_array2)
        # image3
        output_image3 = Image.open(make_test_path("ref_data/slides/libs/multi/multi_region3.tif"))
        output_image_array3 = np.asarray(output_image3)
        output_image3.close()
        result_image3 = Image.open(result_region_tif3)
        result_image_array3 = np.asarray(result_image3)
        result_image3.close()
        np.testing.assert_equal(result_image_array3, output_image_array3)


class TestGenericSlideGetRegionWithMPP(TestCase):
    """Tests for reading patches using MPP values.

    Notes
    -----
    Those tests must use an actual WSI file.
    """

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)

    def test_get_region_and_cache_different_levels_at_once_no_padding_mpp(self):
        # this will also test if caching (_mpp_wsi_level_cache) works, all tests done in memory
        # so we have to read two patches per level, to test adding and retrieving from cache
        output_cache_dict = {0.252: 0, 1.008036: 1, 4.032983: 2}
        location0 = (10000, 20000)
        location1 = (11000, 21000)
        location2 = (12000, 22000)
        level_or_mpp0 = 0.252 * 1
        level_or_mpp1 = 0.252 * 4.000144
        level_or_mpp2 = 0.252 * 16.003899
        size0 = (1024, 512)
        size1 = (512, 256)
        size2 = (256, 128)
        # read patches using mpp
        region_via_mpp0 = self.wsi_slide_svs.get_region(location0, level_or_mpp0, size0, skip_padding=True)
        region_via_mpp1 = self.wsi_slide_svs.get_region(location1, level_or_mpp1, size1, skip_padding=True)
        region_via_mpp2 = self.wsi_slide_svs.get_region(location2, level_or_mpp2, size2, skip_padding=True)
        # compare cache
        cached_dict_rounded = {round(k, 6): v for k, v in self.wsi_slide_svs._prop_mpp_wsi_level_cache.items()}
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        # read patches using level
        region_via_level0 = self.wsi_slide_svs.get_region(location0, 0, size0, skip_padding=True)
        region_via_level1 = self.wsi_slide_svs.get_region(location1, 1, size1, skip_padding=True)
        region_via_level2 = self.wsi_slide_svs.get_region(location2, 2, size2, skip_padding=True)
        # compare patches
        np.testing.assert_equal(np.asarray(region_via_mpp0), np.asarray(region_via_level0))
        np.testing.assert_equal(np.asarray(region_via_mpp1), np.asarray(region_via_level1))
        np.testing.assert_equal(np.asarray(region_via_mpp2), np.asarray(region_via_level2))
        # read more patches using mpp
        region_via_mpp0 = self.wsi_slide_svs.get_region(location0, level_or_mpp0, size0, skip_padding=True)
        region_via_mpp1 = self.wsi_slide_svs.get_region(location1, level_or_mpp1, size1, skip_padding=True)
        region_via_mpp2 = self.wsi_slide_svs.get_region(location2, level_or_mpp2, size2, skip_padding=True)
        # compare cache again
        cached_dict_rounded = {round(k, 6): v for k, v in self.wsi_slide_svs._prop_mpp_wsi_level_cache.items()}
        self.assertEqual(cached_dict_rounded, output_cache_dict)

    def test_get_region_different_levels_at_once_with_padding_mpp(self):
        # this will also test if caching (_resample_cache) works, all tests done in memory
        # so we have to read two patches per level, to test adding and retrieving from cache
        output_cache_dict_step1 = {
            "0.252": {"L": 0, "B": 1.0, "Z": 1.0},
            "1.008036288": {"L": 0, "B": 4.000144, "Z": 4.000144},
            "4.032982548": {"L": 0, "B": 16.003899, "Z": 16.003899},
        }
        output_cache_dict_step2 = {
            "0.252": {"L": 0, "B": 1.0, "Z": 1.0},
            "1.008036288": {"L": 0, "B": 4.000144, "Z": 4.000144},
            "4.032982548": {"L": 0, "B": 16.003899, "Z": 16.003899},
            "0": {"L": 0, "B": 1.0, "Z": 1.0},
            "1": {"L": 1, "B": 1.0, "Z": 4.000144},
            "2": {"L": 2, "B": 1.0, "Z": 16.003899},
        }
        location0 = (10000, 31300)
        location1 = (27700, 21000)
        location2 = (27700, 31300)
        level_or_mpp0 = 0.252  # 0.252 * 1
        level_or_mpp1 = 1.008036288  # 0.252 * 4.000144
        level_or_mpp2 = 4.032982548  # 0.252 * 16.003899
        size0 = (1024, 512)
        size1 = (512, 256)
        size2 = (256, 128)
        # read patches using mpp
        region_via_mpp0 = self.wsi_slide_svs.get_region(location0, level_or_mpp0, size0)
        region_via_mpp1 = self.wsi_slide_svs.get_region(location1, level_or_mpp1, size1)
        region_via_mpp2 = self.wsi_slide_svs.get_region(location2, level_or_mpp2, size2)
        # compare cache
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()}
            for k1, v1 in self.wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict_step1)
        # read patches using level
        region_via_level0 = self.wsi_slide_svs.get_region(location0, 0, size0)
        region_via_level1 = self.wsi_slide_svs.get_region(location1, 1, size1)
        region_via_level2 = self.wsi_slide_svs.get_region(location2, 2, size2)
        # compare cache again
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()}
            for k1, v1 in self.wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict_step2)
        # compare patches
        np.testing.assert_equal(np.asarray(region_via_mpp0), np.asarray(region_via_level0))
        np.testing.assert_equal(np.asarray(region_via_mpp1), np.asarray(region_via_level1))
        np.testing.assert_equal(np.asarray(region_via_mpp2), np.asarray(region_via_level2))
        # read more patches using mpp
        region_via_mpp0 = self.wsi_slide_svs.get_region(location0, level_or_mpp0, size0)
        region_via_mpp1 = self.wsi_slide_svs.get_region(location1, level_or_mpp1, size1)
        region_via_mpp2 = self.wsi_slide_svs.get_region(location2, level_or_mpp2, size2)
        # compare cache again
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()}
            for k1, v1 in self.wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict_step2)

    def test_get_region_different_levels_at_once_mixed_padding_mpp(self):
        location0 = (10000, 31300)
        location1 = (27700, 21000)
        location2 = (27700, 31300)
        level_or_mpp0 = 0.252 * 1
        level_or_mpp1 = 0.252 * 4.000144
        level_or_mpp2 = 0.252 * 16.003899
        size0 = (1024, 512)
        size1 = (512, 256)
        size2 = (256, 128)
        # read patches using mpp
        region_via_mpp0 = self.wsi_slide_svs.get_region(location0, level_or_mpp0, size0, skip_padding=True)
        region_via_mpp1 = self.wsi_slide_svs.get_region(location1, level_or_mpp1, size1)
        region_via_mpp2 = self.wsi_slide_svs.get_region(location2, level_or_mpp2, size2, skip_padding=True)
        # read patches using level
        region_via_level0 = self.wsi_slide_svs.get_region(location0, 0, size0)
        region_via_level1 = self.wsi_slide_svs.get_region(location1, 1, size1, skip_padding=True)
        region_via_level2 = self.wsi_slide_svs.get_region(location2, 2, size2)
        # compare patches
        self.assertEqual(np.any(np.not_equal(np.asarray(region_via_mpp0), np.asarray(region_via_level0))), True)
        self.assertEqual(np.any(np.not_equal(np.asarray(region_via_mpp1), np.asarray(region_via_level1))), True)
        self.assertEqual(np.any(np.not_equal(np.asarray(region_via_mpp2), np.asarray(region_via_level2))), True)

    def test_get_region_without_resampling_mpp(self):
        location = (100, 100)
        level_or_mpp = 0.252 * 2
        size = (200, 200)
        with self.assertRaises(ValueError):
            self.wsi_slide_svs.get_region(location, level_or_mpp, size)


class TestGenericSlideGetRegionBranchingNoDownsampling(TestCase):
    """Test all get_region branching when using level."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif)

    # case1
    def test_level_no_resampling_skip_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case1.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case2
    def test_level_no_resampling_with_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case2.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case3
    def test_level_no_resampling_skip_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case3.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case4
    def test_level_no_resampling_with_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case4.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_mpp_no_resampling_skip_padding_mpp_resampling_not_enabled(self):
        size = (256, 256)
        location = (0, 0)
        with self.assertRaises(ValueError):
            self.wsi_slide_tif.get_region(location, 0.2, size, skip_padding=True)


class TestGenericSlideGetRegionBranchingWSIResampling(TestCase):
    """Test all get_region branching when using WSI resampling."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.75])

    # case5
    def test_level_wsi_resampling_skip_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case5.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case6
    def test_level_wsi_resampling_with_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case6.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case7
    def test_level_wsi_resampling_skip_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case7.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case8
    def test_level_wsi_resampling_with_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case8.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case9
    def test_mpp_wsi_resampling_skip_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case9.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case10
    def test_mpp_wsi_resampling_with_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case10.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case11
    def test_mpp_wsi_resampling_skip_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case11.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case12
    def test_mpp_wsi_resampling_with_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case12.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_mpp_wsi_resampling_skip_padding_mpp_no_match(self):
        size = (256, 256)
        location = (0, 0)
        with self.assertRaises(ValueError):
            self.wsi_slide_tif.get_region(location, 0.9, size, skip_padding=True)


class TestGenericSlideGetRegionBranchingTileResampling(TestCase):
    """Test all get_region branching when using TILE resampling."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif, resampling_mode="tile")

    # case13
    def test_level_tile_resampling_skip_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case13.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case14
    def test_level_tile_resampling_with_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case14.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case15
    def test_level_tile_resampling_skip_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case15.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case16
    def test_level_tile_resampling_with_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case16.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case17
    def test_mpp_tile_resampling_skip_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case17.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case18
    def test_mpp_tile_resampling_with_padding(self):
        size = (256, 256)
        location = (0, 0)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case18.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case19
    def test_mpp_tile_resampling_skip_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=True)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case19.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    # case20
    def test_mpp_tile_resampling_with_padding_negative_location(self):
        size = (256, 256)
        location = (-10, -10)
        region = self.wsi_slide_tif.get_region(location, 0.75, size, skip_padding=False)
        output_image = Image.open(make_test_path("ref_data/slides/libs/getregion/case20.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image_array = np.asarray(region)
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_mpp_tile_resampling_skip_padding_mpp_out_range(self):
        size = (256, 256)
        location = (0, 0)
        with self.assertRaises(ValueError):
            self.wsi_slide_tif.get_region(location, 0.0001, size, skip_padding=True)


class TestGenericSlideGetRegionBranchingResamplingCrossCheck(TestCase):
    """Cross check between WSI and TILE resampling methods - where possible."""

    def test_resampling_modes_crosscheck(self):
        #
        image1 = Image.open(make_test_path("ref_data/slides/libs/getregion/case5.tif"))
        image1_array = np.asarray(image1)
        image1.close()
        image2 = Image.open(make_test_path("ref_data/slides/libs/getregion/case13.tif"))
        image2_array = np.asarray(image2)
        image2.close()
        np.testing.assert_equal(image1_array, image2_array)
        #
        image1 = Image.open(make_test_path("ref_data/slides/libs/getregion/case6.tif"))
        image1_array = np.asarray(image1)
        image1.close()
        image2 = Image.open(make_test_path("ref_data/slides/libs/getregion/case14.tif"))
        image2_array = np.asarray(image2)
        image2.close()
        np.testing.assert_equal(image1_array, image2_array)
        #
        image1 = Image.open(make_test_path("ref_data/slides/libs/getregion/case8.tif"))
        image1_array = np.asarray(image1)
        image1.close()
        image2 = Image.open(make_test_path("ref_data/slides/libs/getregion/case16.tif"))
        image2_array = np.asarray(image2)
        image2.close()
        np.testing.assert_equal(image1_array, image2_array)
        #
        image1 = Image.open(make_test_path("ref_data/slides/libs/getregion/case9.tif"))
        image1_array = np.asarray(image1)
        image1.close()
        image2 = Image.open(make_test_path("ref_data/slides/libs/getregion/case17.tif"))
        image2_array = np.asarray(image2)
        image2.close()
        self.assertFalse(np.array_equal(image1_array, image2_array))
        #
        image1 = Image.open(make_test_path("ref_data/slides/libs/getregion/case10.tif"))
        image1_array = np.asarray(image1)
        image1.close()
        image2 = Image.open(make_test_path("ref_data/slides/libs/getregion/case18.tif"))
        image2_array = np.asarray(image2)
        image2.close()
        self.assertFalse(np.array_equal(image1_array, image2_array))
        #
        image1 = Image.open(make_test_path("ref_data/slides/libs/getregion/case11.tif"))
        image1_array = np.asarray(image1)
        image1.close()
        image2 = Image.open(make_test_path("ref_data/slides/libs/getregion/case19.tif"))
        image2_array = np.asarray(image2)
        image2.close()
        self.assertFalse(np.array_equal(image1_array, image2_array))
        #
        image1 = Image.open(make_test_path("ref_data/slides/libs/getregion/case12.tif"))
        image1_array = np.asarray(image1)
        image1.close()
        image2 = Image.open(make_test_path("ref_data/slides/libs/getregion/case20.tif"))
        image2_array = np.asarray(image2)
        image2.close()
        self.assertFalse(np.array_equal(image1_array, image2_array))


class TestGenericSlideBackgroundPadding(TestCase):
    """Tests for background padding when reading patches.

    Notes
    -----
    - test tif image dimensions are: 2560 x 3072
    - tests below must be run on level 0 to avoid pixel rounding
    - failing can only be determined by presence/absence of log messages
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif)

    def tearDown(self):
        GenericSlide.set_padding_margin_pixels(10)

    def test_padding_x_default(self):
        # no padding action: location_x does not exceed margin
        size = (20, 25)
        location_x = 2560 - (10 + size[0])
        location = (location_x, 100)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # padding triggered: location_x exceeds margin by one pixel
        location_x = 2560 - (10 + size[0]) + 1
        location = (location_x, 100)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)

    def test_padding_y_default(self):
        # no padding action: location_y does not exceed margin
        size = (35, 30)
        location_y = 3072 - (10 + size[1])
        location = (100, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # padding triggered: margin exceeded by one pixel
        location_y = 3072 - (10 + size[1]) + 1
        location = (100, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)

    def test_padding_xy_defult(self):
        # no padding action: no dimension exceeds margin
        size = (40, 50)
        location_x = 2560 - (10 + size[0])
        location_y = 3072 - (10 + size[1])
        location = (location_x, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # padding triggered: margin exceeded by one pixel in both dimensions
        location_x = 2560 - (10 + size[0]) + 1
        location_y = 3072 - (10 + size[1]) + 1
        location = (location_x, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)

    def test_padding_x_custom(self):
        GenericSlide.set_padding_margin_pixels(5)
        # no padding action: location_x does not exceed margin
        size = (20, 25)
        location_x = 2560 - (5 + size[0])
        location = (location_x, 100)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # padding triggered: location_x exceeds margin by one pixel
        location_x = 2560 - (5 + size[0]) + 1
        location = (location_x, 100)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)

    def test_padding_y_custom(self):
        GenericSlide.set_padding_margin_pixels(5)
        # no padding action: location_y does not exceed margin
        size = (35, 30)
        location_y = 3072 - (5 + size[1])
        location = (100, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # padding triggered: margin exceeded by one pixel
        location_y = 3072 - (5 + size[1]) + 1
        location = (100, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)

    def test_padding_xy_custom(self):
        GenericSlide.set_padding_margin_pixels(5)
        # no padding action: no dimension exceeds margin
        size = (40, 50)
        location_x = 2560 - (5 + size[0])
        location_y = 3072 - (5 + size[1])
        location = (location_x, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # padding triggered: margin exceeded by one pixel in both dimensions
        location_x = 2560 - (5 + size[0]) + 1
        location_y = 3072 - (5 + size[1]) + 1
        location = (location_x, location_y)
        region = self.wsi_slide_tif.get_region(location, 0, size)
        self.assertEqual(region.size, size)


class TestGenericSlideMPPMargin(TestCase):
    """Tests for MPP margin - conversion from MPP to WSI level."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)

    def tearDown(self):
        GenericSlide.set_mpp_level_margin(0.003)

    def test_mpp_margin_above_default(self):
        location = (10000, 20000)
        level_or_mpp = 0.252 + 0.004
        size = (256, 256)
        with self.assertRaises(ValueError):
            self.wsi_slide_svs.get_region(location, level_or_mpp, size)

    def test_mpp_margin_equal_default(self):
        location = (10000, 20000)
        level_or_mpp = 0.252 + 0.003
        size = (256, 256)
        with self.assertRaises(ValueError):
            self.wsi_slide_svs.get_region(location, level_or_mpp, size)

    def test_mpp_margin_below_default(self):
        location = (10000, 20000)
        level_or_mpp = 0.252 + 0.002
        size = (256, 256)
        region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        self.assertEqual(region.size, size)

    def test_mpp_margin_above_custom(self):
        GenericSlide.set_mpp_level_margin(0.1)
        location = (10000, 20000)
        level_or_mpp = 0.252 + 0.15
        size = (256, 256)
        with self.assertRaises(ValueError):
            self.wsi_slide_svs.get_region(location, level_or_mpp, size)

    def test_mpp_margin_equal_custom(self):
        GenericSlide.set_mpp_level_margin(0.1)
        location = (10000, 20000)
        level_or_mpp = 0.252 + 0.1
        size = (256, 256)
        with self.assertRaises(ValueError):
            self.wsi_slide_svs.get_region(location, level_or_mpp, size)

    def test_mpp_margin_below_custom(self):
        GenericSlide.set_mpp_level_margin(0.1)
        location = (10000, 20000)
        level_or_mpp = 0.252 + 0.05
        size = (256, 256)
        region = self.wsi_slide_svs.get_region(location, level_or_mpp, size)
        self.assertEqual(region.size, size)

    def test_mpp_match_no_extra_level_present(self):
        # test for MPP match (part I)
        location = (10000, 20000)
        level_or_mpp = 0.5
        size = (256, 256)
        with self.assertRaises(ValueError):
            self.wsi_slide_svs.get_region(location, level_or_mpp, size)


class TestGenericSlideResamplingOneLevel(TestCase):
    """Tests for WSI resampling with one extra level."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)

    def tearDown(self):
        GenericSlide.set_range_min_mpp(0.001)

    def test_resampling_not_possible_no_mpp_data(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression.tif")
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=wsi_file_tif, resampling_mode="wsi", extra_mpps=[3.5])

    def test_resampling_not_possible_wrong_type(self):
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[6])

    def test_resampling_mpp_range_above(self):
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[100000000.1])

    def test_resampling_mpp_range_below1(self):
        GenericSlide.set_range_min_mpp(0.0000001)
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.000000001])

    def test_resampling_mpp_range_below2(self):
        GenericSlide.set_range_min_mpp(0.5)
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.4])

    @skipIf(fast_tests_only, "Skipping resampling test, running fast tests only")
    def test_resampling_image_and_cache(self):
        output_size = (935, 1052)
        output_cache = {7.5: -1}
        wsi_slide_svs_mpp = GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[7.5])
        # this will also test level_images_extra property
        result_image = wsi_slide_svs_mpp.level_images_extra[7.5]
        result_size = (result_image.width, result_image.height)
        self.assertEqual(result_size, output_size)
        cached_dict_rounded = {round(k, 6): v for k, v in wsi_slide_svs_mpp._prop_mpp_wsi_level_cache.items()}
        self.assertEqual(cached_dict_rounded, output_cache)
        # read some patches in various ways and check cache again
        location = (1000, 2000)
        size = (25, 25)
        # level 0
        region = wsi_slide_svs_mpp.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # level 0 via mpp
        region = wsi_slide_svs_mpp.get_region(location, 0.252, size)
        self.assertEqual(region.size, size)
        output_cache.update({0.252: 0})
        # extra level mpp
        region = wsi_slide_svs_mpp.get_region(location, 7.5, size)
        self.assertEqual(region.size, size)
        output_cache.update({7.5: -1})
        # not present mpp (part II of MPP match)
        with self.assertRaises(ValueError):
            output_cache.update({0.5: -1})
            wsi_slide_svs_mpp.get_region(location, 0.5, size)
        # compare cache again at the end
        cached_dict_rounded = {round(k, 6): round(v, 6) for k, v in wsi_slide_svs_mpp._prop_mpp_wsi_level_cache.items()}
        self.assertEqual(cached_dict_rounded, output_cache)

    def test_resampling_skipped_and_cache(self):
        # this will also produce a screen message (resampling skipped...)
        wsi_slide_svs_mpp = GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.252])
        output_cache = {0.252: 0}
        self.assertEqual(wsi_slide_svs_mpp._prop_mpp_wsi_level_cache, output_cache)


class TestGenericSlideResamplingMultiLevel(TestCase):
    """Tests for WSI resampling with multiple extra levels."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_slide_svs = GenericSlide(wsi_file=self.wsi_file_svs)

    def test_resampling_not_possible_no_mpp_data(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression.tif")
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.5, 0.1, 0.9])

    def test_resampling_not_possible_wrong_types(self):
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.5, 0.1, 2])

    def test_resampling_not_possible_not_unique(self):
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.5, 0.1, 0.5])

    def test_resampling_mpp_range_above(self):
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.5, 100000000.1, 3.1])

    def test_resampling_mpp_range_below(self):
        with self.assertRaises(ValueError):
            GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[2, 0.000000001, 1.3])

    def test_resampling_all_skipped_and_cache(self):
        wsi_slide_svs_mpp = GenericSlide(
            wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.252, 0.252 * 4, 0.252 * 16]
        )
        output_cache = {0.252: 0, 1.008: 1, 4.032: 2}
        self.assertEqual(wsi_slide_svs_mpp._prop_mpp_wsi_level_cache, output_cache)
        # also test level_images_extra property
        self.assertEqual(wsi_slide_svs_mpp.level_images_extra.keys(), OrderedDict().keys())

    @skipIf(fast_tests_only, "Skipping resampling test, running fast tests only")
    def test_resampling_some_skipped_and_cache(self):
        wsi_slide_svs_mpp = GenericSlide(
            wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.252, 0.252 * 4, 6.5, 0.252 * 16, 8.55]
        )
        output_cache = {0.252: 0, 1.008: 1, 6.5: -1, 4.032: 2, 8.55: -1}
        self.assertEqual(wsi_slide_svs_mpp._prop_mpp_wsi_level_cache, output_cache)
        # also test level_images_extra property
        self.assertEqual(wsi_slide_svs_mpp.level_images_extra.keys(), OrderedDict({6.5: -1, 8.55: -1}).keys())

    @skipIf(fast_tests_only, "Skipping resampling test, running fast tests only")
    def test_resampling_image_and_cache_and_properties(self):
        mpp_list = [8.8, 0.252, 5.5, 7.9, 9.5]
        # one mpp value should get skipped in processing
        output_size_list = [(797, 897), (1275, 1435), (887, 999), (738, 831)]
        output_cache = {8.8: -1, 0.252: 0, 5.5: -1, 7.9: -1, 9.5: -1}
        wsi_slide_svs_mpp = GenericSlide(wsi_file=self.wsi_file_svs, resampling_mode="wsi", extra_mpps=mpp_list)
        # this will also test level_images_extra property
        result_images = [img for mpp, img in wsi_slide_svs_mpp.level_images_extra.items()]
        result_size_list = [(result_image.width, result_image.height) for result_image in result_images]
        self.assertEqual(result_size_list, output_size_list)
        # compare cache
        cached_dict_rounded = {round(k, 6): v for k, v in wsi_slide_svs_mpp._prop_mpp_wsi_level_cache.items()}
        self.assertEqual(cached_dict_rounded, output_cache)
        # read some patches in various ways and check cache again
        location = (3000, 4000)
        size = (35, 35)
        # level 0
        region = wsi_slide_svs_mpp.get_region(location, 0, size)
        self.assertEqual(region.size, size)
        # level 1
        region = wsi_slide_svs_mpp.get_region(location, 1, size)
        self.assertEqual(region.size, size)
        # level 2
        region = wsi_slide_svs_mpp.get_region(location, 2, size)
        self.assertEqual(region.size, size)
        # level 0 via mpp
        region = wsi_slide_svs_mpp.get_region(location, 0.252 * 1, size)
        self.assertEqual(region.size, size)
        output_cache.update({0.252: 0})
        # level 1 via mpp
        region = wsi_slide_svs_mpp.get_region(location, 0.252 * 4, size)
        self.assertEqual(region.size, size)
        output_cache.update({0.252 * 4: 1})
        # level 2 via mpp
        region = wsi_slide_svs_mpp.get_region(location, 0.252 * 16, size)
        self.assertEqual(region.size, size)
        output_cache.update({0.252 * 16: 2})
        # extra level mpp
        region = wsi_slide_svs_mpp.get_region(location, 9.5, size)
        self.assertEqual(region.size, size)
        output_cache.update({9.5: -1})
        # extra level mpp
        region = wsi_slide_svs_mpp.get_region(location, 7.9, size)
        self.assertEqual(region.size, size)
        output_cache.update({9.5: -1})
        # extra level mpp
        region = wsi_slide_svs_mpp.get_region(location, 8.8, size)
        self.assertEqual(region.size, size)
        output_cache.update({8.8: -1})
        # extra level mpp
        region = wsi_slide_svs_mpp.get_region(location, 5.5, size)
        self.assertEqual(region.size, size)
        output_cache.update({5.5: -1})
        # not present mpp
        with self.assertRaises(ValueError):
            output_cache.update({4.4: -1})
            wsi_slide_svs_mpp.get_region(location, 4.4, size)
        # compare cache again
        cached_dict_rounded = {round(k, 6): round(v, 6) for k, v in wsi_slide_svs_mpp._prop_mpp_wsi_level_cache.items()}
        self.assertEqual(cached_dict_rounded, output_cache)
        # extra properties in resampling
        output_level_dimensions = OrderedDict({8.8: (797, 897), 5.5: (1275, 1435), 7.9: (887, 999), 9.5: (738, 831)})
        result_level_dimensions = wsi_slide_svs_mpp.level_dimensions_extra
        self.assertEqual(result_level_dimensions, output_level_dimensions)
        #
        output_level_resamples = OrderedDict({8.8: 34.92063, 5.5: 21.82540, 7.9: 31.34921, 9.5: 37.69841})
        result_level_resamples = wsi_slide_svs_mpp.level_resamples_extra
        result_level_resamples = OrderedDict({k: round(v, 5) for k, v in result_level_resamples.items()})
        self.assertEqual(result_level_resamples, output_level_resamples)
        #
        output_mpp_values = (8.8, 5.5, 7.9, 9.5)
        result_mpp_values = wsi_slide_svs_mpp.level_mpp_values_extra
        self.assertEqual(result_mpp_values, output_mpp_values)
        #
        output_count = 4
        result_count = wsi_slide_svs_mpp.level_count_extra
        self.assertEqual(result_count, output_count)


class TestGenericSlideGetRegionMPPMultiLevel(TestCase):
    """More advanced tests for reading regions from multiple levels."""

    def tearDown(self):
        GenericSlide.set_level_zero_resampling(True)

    @skipIf(fast_tests_only, "Skipping resampling test, running fast tests only")
    def test_get_region_different_levels_at_once_mpp_level_zero_resampling_wsi_mode(self):
        # this will also test if caching (_resample_cache) works, all tests done in memory
        # so we have to read two patches per level, to test adding and retrieving from cache
        GenericSlide.set_level_zero_resampling(True)
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs, resampling_mode="wsi", extra_mpps=[5.5, 6.6, 7.7, 8.8])
        #
        # test 1: testing cache only
        output_cache_dict = {
            "5.5": {"L": 0, "B": 21.825397, "Z": 21.825397},
            "6.6": {"L": 0, "B": 26.190476, "Z": 26.190476},
            "7.7": {"L": 0, "B": 30.555556, "Z": 30.555556},
            "8.8": {"L": 0, "B": 34.920635, "Z": 34.920635},
        }
        location = (15000, 17300)
        size = (250, 250)
        region_via_mpp1 = wsi_slide_svs.get_region(location, 5.5, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 6.6, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 7.7, size)
        region_via_mpp4 = wsi_slide_svs.get_region(location, 8.8, size)
        self.assertEqual(region_via_mpp1.size, (250, 250))
        self.assertEqual(region_via_mpp2.size, (250, 250))
        self.assertEqual(region_via_mpp3.size, (250, 250))
        self.assertEqual(region_via_mpp4.size, (250, 250))
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        #
        # test 2: test cache and patches (read WSI levels using MPP and level)
        output_cache_dict = {
            "0.252": {"L": 0, "B": 1.0, "Z": 1.0},
            "1.008036288": {"L": 0, "B": 4.000144, "Z": 4.000144},
            "4.032982548": {"L": 0, "B": 16.003899, "Z": 16.003899},
            "0": {"L": 0, "B": 1.0, "Z": 1.0},
            "1": {"L": 1, "B": 1.0, "Z": 4.000144},
            "2": {"L": 2, "B": 1.0, "Z": 16.003899},
            "5.5": {"L": 0, "B": 21.825397, "Z": 21.825397},
            "6.6": {"L": 0, "B": 26.190476, "Z": 26.190476},
            "7.7": {"L": 0, "B": 30.555556, "Z": 30.555556},
            "8.8": {"L": 0, "B": 34.920635, "Z": 34.920635},
        }
        level_or_mpp0 = 0.252  # 0.252 * 1
        level_or_mpp1 = 1.008036288  # 0.252 * 4.000144
        level_or_mpp2 = 4.032982548  # 0.252 * 16.003899
        region_via_mpp0 = wsi_slide_svs.get_region(location, level_or_mpp0, size)
        region_via_mpp1 = wsi_slide_svs.get_region(location, level_or_mpp1, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, level_or_mpp2, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        # compare patches
        np.testing.assert_equal(np.asarray(region_via_mpp0), np.asarray(region_via_level0))
        np.testing.assert_equal(np.asarray(region_via_mpp1), np.asarray(region_via_level1))
        np.testing.assert_equal(np.asarray(region_via_mpp2), np.asarray(region_via_level2))
        # read more patches using mpp/level
        region_via_mpp0 = wsi_slide_svs.get_region(location, 5.5, size)
        region_via_mpp1 = wsi_slide_svs.get_region(location, 6.6, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 7.7, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 8.8, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache again
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        # test extra images created in resampling
        self.assertEqual(len(wsi_slide_svs.level_images_extra), 4)

    @skipIf(fast_tests_only, "Skipping resampling test, running fast tests only")
    def test_get_region_different_levels_at_once_mpp_variable_resampling_wsi_mode(self):
        # MPP 6.6 is not read via get_region but should still be present in the created cache.
        GenericSlide.set_level_zero_resampling(False)
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(
            wsi_file=wsi_file_svs, resampling_mode="wsi", extra_mpps=[0.7, 1.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
        )
        #
        # test 1: testing cache only
        output_cache_dict = {
            "0.7": {"L": 0, "B": 2.777778, "Z": 2.777778},
            "1.0": {"L": 0, "B": 3.968254, "Z": 3.968254},
            "1.1": {"L": 1, "B": 1.091231, "Z": 4.365079},
            "2.2": {"L": 1, "B": 2.182461, "Z": 8.730159},
            "3.3": {"L": 1, "B": 3.273692, "Z": 13.095238},
            "4.4": {"L": 2, "B": 1.091004, "Z": 17.460317},
            "5.5": {"L": 2, "B": 1.363755, "Z": 21.825397},
            "6.6": {"L": 2, "B": 1.636506, "Z": 26.190476},
        }
        location = (15000, 17300)
        size = (250, 250)
        region_via_mpp1 = wsi_slide_svs.get_region(location, 0.7, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 1.0, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 2.2, size)
        region_via_mpp4 = wsi_slide_svs.get_region(location, 4.4, size)
        region_via_mpp5 = wsi_slide_svs.get_region(location, 5.5, size)
        self.assertEqual(region_via_mpp1.size, (250, 250))
        self.assertEqual(region_via_mpp2.size, (250, 250))
        self.assertEqual(region_via_mpp3.size, (250, 250))
        self.assertEqual(region_via_mpp4.size, (250, 250))
        self.assertEqual(region_via_mpp5.size, (250, 250))
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        #
        # test 2: test cache and patches (read WSI levels using MPP and level)
        output_cache_dict = {
            "0.252": {"L": 0, "B": 1.0, "Z": 1.0},
            "1.008036288": {"L": 1, "B": 1.0, "Z": 4.000144},
            "4.032982548": {"L": 2, "B": 1.0, "Z": 16.003899},
            "0": {"L": 0, "B": 1.0, "Z": 1.0},
            "1": {"L": 1, "B": 1.0, "Z": 4.000144},
            "2": {"L": 2, "B": 1.0, "Z": 16.003899},
            "0.7": {"L": 0, "B": 2.777778, "Z": 2.777778},
            "1.0": {"L": 0, "B": 3.968254, "Z": 3.968254},
            "1.1": {"L": 1, "B": 1.091231, "Z": 4.365079},
            "2.2": {"L": 1, "B": 2.182461, "Z": 8.730159},
            "3.3": {"L": 1, "B": 3.273692, "Z": 13.095238},
            "4.4": {"L": 2, "B": 1.091004, "Z": 17.460317},
            "5.5": {"L": 2, "B": 1.363755, "Z": 21.825397},
            "6.6": {"L": 2, "B": 1.636506, "Z": 26.190476},
        }
        level_or_mpp0 = 0.252  # 0.252 * 1
        level_or_mpp1 = 1.008036288  # 0.252 * 4.000144
        level_or_mpp2 = 4.032982548  # 0.252 * 16.003899
        region_via_mpp0 = wsi_slide_svs.get_region(location, level_or_mpp0, size)
        region_via_mpp1 = wsi_slide_svs.get_region(location, level_or_mpp1, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, level_or_mpp2, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        # compare patches
        np.testing.assert_equal(np.asarray(region_via_mpp0), np.asarray(region_via_level0))
        np.testing.assert_equal(np.asarray(region_via_mpp1), np.asarray(region_via_level1))
        np.testing.assert_equal(np.asarray(region_via_mpp2), np.asarray(region_via_level2))
        # read more patches using mpp/level
        region_via_mpp0 = wsi_slide_svs.get_region(location, 1.1, size)
        region_via_mpp1 = wsi_slide_svs.get_region(location, 2.2, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 3.3, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 5.5, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache again
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        # test extra images created in resampling
        self.assertEqual(len(wsi_slide_svs.level_images_extra), 8)

    def test_get_region_different_levels_at_once_mpp_level_zero_resampling_tile_mode(self):
        # this will also test if caching (_resample_cache) works, all tests done in memory
        # so we have to read two patches per level, to test adding and retrieving from cache
        GenericSlide.set_level_zero_resampling(True)
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs, resampling_mode="tile")
        #
        # test 1: testing cache only
        output_cache_dict = {
            "0.3": {"L": 0, "B": 1.190476, "Z": 1.190476},
            "0.35": {"L": 0, "B": 1.388889, "Z": 1.388889},
            "0.4": {"L": 0, "B": 1.587302, "Z": 1.587302},
        }
        location = (15000, 17300)
        size = (50, 60)  # patches must be small for speed
        region_via_mpp1 = wsi_slide_svs.get_region(location, 0.30, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 0.35, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 0.40, size)
        self.assertEqual(region_via_mpp1.size, (50, 60))
        self.assertEqual(region_via_mpp2.size, (50, 60))
        self.assertEqual(region_via_mpp3.size, (50, 60))
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        #
        # test 2: test cache and patches (read WSI levels using MPP and level)
        output_cache_dict = {
            "0.252": {"L": 0, "B": 1.0, "Z": 1.0},
            "1.008036288": {"L": 0, "B": 4.000144, "Z": 4.000144},
            "4.032982548": {"L": 0, "B": 16.003899, "Z": 16.003899},
            "0": {"L": 0, "B": 1.0, "Z": 1.0},
            "1": {"L": 1, "B": 1.0, "Z": 4.000144},
            "2": {"L": 2, "B": 1.0, "Z": 16.003899},
            "0.3": {"L": 0, "B": 1.190476, "Z": 1.190476},
            "0.35": {"L": 0, "B": 1.388889, "Z": 1.388889},
            "0.4": {"L": 0, "B": 1.587302, "Z": 1.587302},
        }
        level_or_mpp0 = 0.252  # 0.252 * 1
        level_or_mpp1 = 1.008036288  # 0.252 * 4.000144
        level_or_mpp2 = 4.032982548  # 0.252 * 16.003899
        region_via_mpp0 = wsi_slide_svs.get_region(location, level_or_mpp0, size)
        region_via_mpp1 = wsi_slide_svs.get_region(location, level_or_mpp1, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, level_or_mpp2, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        # compare patches
        np.testing.assert_equal(np.asarray(region_via_mpp0), np.asarray(region_via_level0))
        np.testing.assert_equal(np.asarray(region_via_mpp1), np.asarray(region_via_level1))
        np.testing.assert_equal(np.asarray(region_via_mpp2), np.asarray(region_via_level2))
        # read more patches using mpp/level
        region_via_mpp1 = wsi_slide_svs.get_region(location, 0.30, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 0.35, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 0.40, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache again
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)

    def test_get_region_different_levels_at_once_mpp_variable_resampling_tile_mode(self):
        GenericSlide.set_level_zero_resampling(False)
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs, resampling_mode="tile")
        #
        # test 1: testing cache only
        output_cache_dict = {
            "0.7": {"L": 0, "B": 2.777778, "Z": 2.777778},
            "1.0": {"L": 0, "B": 3.968254, "Z": 3.968254},
            "2.2": {"L": 1, "B": 2.182461, "Z": 8.730159},
            "4.4": {"L": 2, "B": 1.091004, "Z": 17.460317},
            "5.5": {"L": 2, "B": 1.363755, "Z": 21.825397},
        }
        location = (15000, 17300)
        size = (40, 60)
        region_via_mpp1 = wsi_slide_svs.get_region(location, 0.7, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 1.0, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 2.2, size)
        region_via_mpp4 = wsi_slide_svs.get_region(location, 4.4, size)
        region_via_mpp5 = wsi_slide_svs.get_region(location, 5.5, size)
        self.assertEqual(region_via_mpp1.size, (40, 60))
        self.assertEqual(region_via_mpp2.size, (40, 60))
        self.assertEqual(region_via_mpp3.size, (40, 60))
        self.assertEqual(region_via_mpp4.size, (40, 60))
        self.assertEqual(region_via_mpp5.size, (40, 60))
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        #
        # test 2: test cache and patches (read WSI levels using MPP and level)
        output_cache_dict = {
            "0.252": {"L": 0, "B": 1.0, "Z": 1.0},
            "1.008036288": {"L": 1, "B": 1.0, "Z": 4.000144},
            "4.032982548": {"L": 2, "B": 1.0, "Z": 16.003899},
            "0": {"L": 0, "B": 1.0, "Z": 1.0},
            "1": {"L": 1, "B": 1.0, "Z": 4.000144},
            "2": {"L": 2, "B": 1.0, "Z": 16.003899},
            "0.7": {"L": 0, "B": 2.777778, "Z": 2.777778},
            "1.0": {"L": 0, "B": 3.968254, "Z": 3.968254},
            "2.2": {"L": 1, "B": 2.182461, "Z": 8.730159},
            "4.4": {"L": 2, "B": 1.091004, "Z": 17.460317},
            "5.5": {"L": 2, "B": 1.363755, "Z": 21.825397},
        }
        level_or_mpp0 = 0.252  # 0.252 * 1
        level_or_mpp1 = 1.008036288  # 0.252 * 4.000144
        level_or_mpp2 = 4.032982548  # 0.252 * 16.003899
        region_via_mpp0 = wsi_slide_svs.get_region(location, level_or_mpp0, size)
        region_via_mpp1 = wsi_slide_svs.get_region(location, level_or_mpp1, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, level_or_mpp2, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)
        # compare patches
        np.testing.assert_equal(np.asarray(region_via_mpp0), np.asarray(region_via_level0))
        np.testing.assert_equal(np.asarray(region_via_mpp1), np.asarray(region_via_level1))
        np.testing.assert_equal(np.asarray(region_via_mpp2), np.asarray(region_via_level2))
        # test 3: read more patches using mpp/level
        output_cache_dict = {
            "0.252": {"L": 0, "B": 1.0, "Z": 1.0},
            "1.008036288": {"L": 1, "B": 1.0, "Z": 4.000144},
            "4.032982548": {"L": 2, "B": 1.0, "Z": 16.003899},
            "0": {"L": 0, "B": 1.0, "Z": 1.0},
            "1": {"L": 1, "B": 1.0, "Z": 4.000144},
            "2": {"L": 2, "B": 1.0, "Z": 16.003899},
            "0.7": {"L": 0, "B": 2.777778, "Z": 2.777778},
            "1.0": {"L": 0, "B": 3.968254, "Z": 3.968254},
            "1.1": {"L": 1, "B": 1.091231, "Z": 4.365079},
            "2.2": {"L": 1, "B": 2.182461, "Z": 8.730159},
            "3.3": {"L": 1, "B": 3.273692, "Z": 13.095238},
            "4.4": {"L": 2, "B": 1.091004, "Z": 17.460317},
            "5.5": {"L": 2, "B": 1.363755, "Z": 21.825397},
        }
        region_via_mpp0 = wsi_slide_svs.get_region(location, 1.1, size)
        region_via_mpp1 = wsi_slide_svs.get_region(location, 2.2, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 3.3, size)
        region_via_mpp2 = wsi_slide_svs.get_region(location, 4.4, size)
        region_via_mpp3 = wsi_slide_svs.get_region(location, 5.5, size)
        region_via_level0 = wsi_slide_svs.get_region(location, 0, size)
        region_via_level1 = wsi_slide_svs.get_region(location, 1, size)
        region_via_level2 = wsi_slide_svs.get_region(location, 2, size)
        # compare cache again
        cached_dict_rounded = {
            k1: {k2: round(v2, 6) for k2, v2 in v1.items()} for k1, v1 in wsi_slide_svs._prop_resample_cache.items()
        }
        self.assertEqual(cached_dict_rounded, output_cache_dict)


class TestGenericSlideResamplingMultiLevelBoard(TestCase):
    """Tests for fake WSI resampling with multiple extra levels.

    Some of the output patches will be padded.
    """

    class DummyGenericSlide(GenericSlide):
        """Dummy class with MPP data embedded."""

        def _lib_get_mpp_data(self):
            return (0.2, 0.2)

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression.tif")

    def test_board_resampling_get_region(self):
        wsi_slide_tif = self.DummyGenericSlide(
            wsi_file=self.wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.1, 0.4, 1.6, 6.4]
        )
        # level 0
        region1 = wsi_slide_tif.get_region((256, 256), 0, (256, 512))
        output_image1 = Image.open(make_test_path("ref_data/slides/patches/mpp/mpp_patch1.tif"))
        output_image1_array = np.asarray(output_image1)
        np.testing.assert_equal(np.asarray(region1), np.asarray(output_image1_array))
        # level 1
        region2 = wsi_slide_tif.get_region((256, 256), 1, (256, 512))
        output_image2 = Image.open(make_test_path("ref_data/slides/patches/mpp/mpp_patch2.tif"))
        output_image2_array = np.asarray(output_image2)
        np.testing.assert_equal(np.asarray(region2), np.asarray(output_image2_array))
        # level 2
        region3 = wsi_slide_tif.get_region((256, 256), 2, (256, 512))
        output_image3 = Image.open(make_test_path("ref_data/slides/patches/mpp/mpp_patch3.tif"))
        output_image3_array = np.asarray(output_image3)
        np.testing.assert_equal(np.asarray(region3), np.asarray(output_image3_array))
        # mpp 0.4
        region4 = wsi_slide_tif.get_region((256, 256), 0.4, (256, 512))
        output_image4 = Image.open(make_test_path("ref_data/slides/patches/mpp/mpp_patch4.tif"))
        output_image4_array = np.asarray(output_image4)
        np.testing.assert_equal(np.asarray(region4), np.asarray(output_image4_array))
        # mpp 1.6
        region5 = wsi_slide_tif.get_region((256, 256), 1.6, (256, 512))
        output_image5 = Image.open(make_test_path("ref_data/slides/patches/mpp/mpp_patch5.tif"))
        output_image5_array = np.asarray(output_image5)
        np.testing.assert_equal(np.asarray(region5), np.asarray(output_image5_array))
        # mpp 6.4
        region6 = wsi_slide_tif.get_region((256, 256), 6.4, (256, 512))
        output_image6 = Image.open(make_test_path("ref_data/slides/patches/mpp/mpp_patch6.tif"))
        output_image6_array = np.asarray(output_image6)
        np.testing.assert_equal(np.asarray(region6), np.asarray(output_image6_array))
        # mpp 0.1 (upsampling)
        region7 = wsi_slide_tif.get_region((256, 256), 0.1, (256, 512))
        output_image7 = Image.open(make_test_path("ref_data/slides/patches/mpp/mpp_patch7.tif"))
        output_image7_array = np.asarray(output_image7)
        np.testing.assert_equal(np.asarray(region7), np.asarray(output_image7_array))

    def test_board_resampling_image_properties(self):
        wsi_slide_tif = self.DummyGenericSlide(
            wsi_file=self.wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.4, 1.6, 6.4]
        )
        # level_dimensions_extra
        output_level_dimensions = OrderedDict({0.4: (1280, 1536), 1.6: (320, 384), 6.4: (80, 96)})
        result_level_dimensions = wsi_slide_tif.level_dimensions_extra
        self.assertEqual(result_level_dimensions, output_level_dimensions)
        # level_resamples_extra
        output_level_resamples = OrderedDict({0.4: 2, 1.6: 8, 6.4: 32})
        result_level_resamples = wsi_slide_tif.level_resamples_extra
        self.assertEqual(result_level_resamples, output_level_resamples)
        # level_mpp_values_extra
        output_mpp_values = (0.4, 1.6, 6.4)
        result_mpp_values = wsi_slide_tif.level_mpp_values_extra
        self.assertEqual(result_mpp_values, output_mpp_values)
        # level_count_extra
        output_count = 3
        result_count = wsi_slide_tif.level_count_extra
        self.assertEqual(result_count, output_count)

    def test_board_resampling_image_properties_mpp_overlap1(self):
        wsi_slide_tif = self.DummyGenericSlide(
            wsi_file=self.wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.2, 1.6, 6.4]
        )
        # level_dimensions_extra
        output_level_dimensions = OrderedDict({1.6: (320, 384), 6.4: (80, 96)})
        result_level_dimensions = wsi_slide_tif.level_dimensions_extra
        self.assertEqual(result_level_dimensions, output_level_dimensions)
        # level_resamples_extra
        output_level_resamples = OrderedDict({1.6: 8, 6.4: 32})
        result_level_resamples = wsi_slide_tif.level_resamples_extra
        self.assertEqual(result_level_resamples, output_level_resamples)
        # level_mpp_values_extra
        output_mpp_values = (1.6, 6.4)
        result_mpp_values = wsi_slide_tif.level_mpp_values_extra
        self.assertEqual(result_mpp_values, output_mpp_values)
        # level_count_extra
        output_count = 2
        result_count = wsi_slide_tif.level_count_extra
        self.assertEqual(result_count, output_count)

    def test_board_resampling_image_properties_mpp_overlap2(self):
        wsi_slide_tif = self.DummyGenericSlide(
            wsi_file=self.wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.2, 0.8, 6.4]
        )
        # level_dimensions_extra
        output_level_dimensions = OrderedDict({6.4: (80, 96)})
        result_level_dimensions = wsi_slide_tif.level_dimensions_extra
        self.assertEqual(result_level_dimensions, output_level_dimensions)
        # level_resamples_extra
        output_level_resamples = OrderedDict({6.4: 32})
        result_level_resamples = wsi_slide_tif.level_resamples_extra
        self.assertEqual(result_level_resamples, output_level_resamples)
        # level_mpp_values_extra
        output_mpp_values = (6.4,)
        result_mpp_values = wsi_slide_tif.level_mpp_values_extra
        self.assertEqual(result_mpp_values, output_mpp_values)
        # level_count_extra
        output_count = 1
        result_count = wsi_slide_tif.level_count_extra
        self.assertEqual(result_count, output_count)


class TestGenericSlideResamplingMultiLevelMultiColorBoard(TestCase):
    """Tests for resampling with different bases represented by different colors."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-colors.tif")

    def tearDown(self):
        GenericSlide.set_level_zero_resampling(True)

    def test_resampling_level_zero_wsi_mode(self):
        GenericSlide.set_level_zero_resampling(True)
        wsi_slide_tif = GenericSlide(
            wsi_file=self.wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.3, 0.6, 1.5, 2.5, 4.5, 5.5]
        )
        region1_image = wsi_slide_tif.get_region((256, 256), 0.3, (512, 512))
        region2_image = wsi_slide_tif.get_region((256, 256), 0.6, (512, 512))
        region3_image = wsi_slide_tif.get_region((256, 256), 1.5, (256, 256))
        region4_image = wsi_slide_tif.get_region((256, 256), 2.5, (128, 128))
        region5_image = wsi_slide_tif.get_region((256, 256), 4.5, (64, 64))
        region6_image = wsi_slide_tif.get_region((256, 256), 5.5, (64, 64))
        region1_image_array = np.asarray(region1_image)
        region2_image_array = np.asarray(region2_image)
        region3_image_array = np.asarray(region3_image)
        region4_image_array = np.asarray(region4_image)
        region5_image_array = np.asarray(region5_image)
        region6_image_array = np.asarray(region6_image)
        self.assertEqual(tuple(region1_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region2_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region3_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region4_image_array[10, 10]), (255, 0, 0))
        self.assertEqual(tuple(region5_image_array[10, 10]), (255, 0, 0))
        self.assertEqual(tuple(region6_image_array[7, 7]), (255, 0, 0))

    def test_resampling_level_zero_tile_mode(self):
        GenericSlide.set_level_zero_resampling(True)
        wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif, resampling_mode="tile")
        region1_image = wsi_slide_tif.get_region((256, 256), 0.3, (512, 512))
        region2_image = wsi_slide_tif.get_region((256, 256), 0.6, (512, 512))
        region3_image = wsi_slide_tif.get_region((256, 256), 1.5, (256, 256))
        region4_image = wsi_slide_tif.get_region((256, 256), 2.5, (128, 128))
        region5_image = wsi_slide_tif.get_region((256, 256), 4.5, (64, 64))
        region6_image = wsi_slide_tif.get_region((256, 256), 5.5, (64, 64))
        region1_image_array = np.asarray(region1_image)
        region2_image_array = np.asarray(region2_image)
        region3_image_array = np.asarray(region3_image)
        region4_image_array = np.asarray(region4_image)
        region5_image_array = np.asarray(region5_image)
        region6_image_array = np.asarray(region6_image)
        self.assertEqual(tuple(region1_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region2_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region3_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region4_image_array[10, 10]), (255, 0, 0))
        self.assertEqual(tuple(region5_image_array[10, 10]), (255, 0, 0))
        self.assertEqual(tuple(region6_image_array[7, 7]), (255, 0, 0))

    def test_resampling_variable_level_wsi_mode(self):
        GenericSlide.set_level_zero_resampling(False)
        wsi_slide_tif = GenericSlide(
            wsi_file=self.wsi_file_tif, resampling_mode="wsi", extra_mpps=[0.3, 0.6, 1.5, 2.5, 4.5, 5.5]
        )
        region1_image = wsi_slide_tif.get_region((256, 256), 0.3, (512, 512))
        region2_image = wsi_slide_tif.get_region((256, 256), 0.6, (512, 512))
        region3_image = wsi_slide_tif.get_region((256, 256), 1.5, (256, 256))
        region4_image = wsi_slide_tif.get_region((256, 256), 2.5, (128, 128))
        region5_image = wsi_slide_tif.get_region((256, 256), 4.5, (64, 64))
        region6_image = wsi_slide_tif.get_region((256, 256), 5.5, (64, 64))
        region1_image_array = np.asarray(region1_image)
        region2_image_array = np.asarray(region2_image)
        region3_image_array = np.asarray(region3_image)
        region4_image_array = np.asarray(region4_image)
        region5_image_array = np.asarray(region5_image)
        region6_image_array = np.asarray(region6_image)
        self.assertEqual(tuple(region1_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region2_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region3_image_array[5, 5]), (0, 255, 0))
        self.assertEqual(tuple(region4_image_array[10, 10]), (0, 255, 0))
        self.assertEqual(tuple(region5_image_array[10, 10]), (0, 0, 255))
        self.assertEqual(tuple(region6_image_array[7, 7]), (0, 0, 255))

    def test_resampling_variable_level_tile_mode(self):
        GenericSlide.set_level_zero_resampling(False)
        wsi_slide_tif = GenericSlide(wsi_file=self.wsi_file_tif, resampling_mode="tile")
        region1_image = wsi_slide_tif.get_region((256, 256), 0.3, (512, 512))
        region2_image = wsi_slide_tif.get_region((256, 256), 0.6, (512, 512))
        region3_image = wsi_slide_tif.get_region((256, 256), 1.5, (256, 256))
        region4_image = wsi_slide_tif.get_region((256, 256), 2.5, (128, 128))
        region5_image = wsi_slide_tif.get_region((256, 256), 4.5, (64, 64))
        region6_image = wsi_slide_tif.get_region((256, 256), 5.5, (64, 64))
        region1_image_array = np.asarray(region1_image)
        region2_image_array = np.asarray(region2_image)
        region3_image_array = np.asarray(region3_image)
        region4_image_array = np.asarray(region4_image)
        region5_image_array = np.asarray(region5_image)
        region6_image_array = np.asarray(region6_image)
        self.assertEqual(tuple(region1_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region2_image_array[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(region3_image_array[5, 5]), (0, 255, 0))
        self.assertEqual(tuple(region4_image_array[10, 10]), (0, 255, 0))
        self.assertEqual(tuple(region5_image_array[10, 10]), (0, 0, 255))
        self.assertEqual(tuple(region6_image_array[7, 7]), (0, 0, 255))


class TestGenericSlideStaticMethods(TestCase):
    """Tests for static methods present in GenericSlide."""

    def test_check_resampling_args(self):
        wsi_name = "1.svs"
        #
        resampling_mode = 1
        extra_mpps = []
        with self.assertRaises(ValueError):
            GenericSlide._check_resampling_args(resampling_mode, extra_mpps, wsi_name)
        #
        resampling_mode = "wsi"
        extra_mpps = []
        with self.assertRaises(ValueError):
            GenericSlide._check_resampling_args(resampling_mode, extra_mpps, wsi_name)
        #
        resampling_mode = "tile"
        extra_mpps = [1, 2, 3]
        with self.assertRaises(ValueError):
            GenericSlide._check_resampling_args(resampling_mode, extra_mpps, wsi_name)
        #
        resampling_mode = None
        extra_mpps = [1, 2, 3]
        with self.assertRaises(ValueError):
            GenericSlide._check_resampling_args(resampling_mode, extra_mpps, wsi_name)

    def test__check_mpp_data(self):
        wsi_name = "slide1.svs"
        mpp_data = (0.2, 0.2)
        GenericSlide._check_mpp_data(mpp_data, wsi_name)
        mpp_data = (None, None)
        with self.assertRaises(ValueError):
            GenericSlide._check_mpp_data(mpp_data, wsi_name)
        mpp_data = (None, 0.1)
        with self.assertRaises(ValueError):
            GenericSlide._check_mpp_data(mpp_data, wsi_name)

    def test__check_extra_mpp_types(self):
        wsi_name = "slide3.svs"
        mpp_list = [1.1, 2.2, 3.3]
        GenericSlide._check_extra_mpp_types(mpp_list, wsi_name)
        mpp_list = [1.1, 2, 3.3]
        with self.assertRaises(ValueError):
            GenericSlide._check_extra_mpp_types(mpp_list, wsi_name)
        mpp_list = [None, 2.2, 3.3]
        with self.assertRaises(ValueError):
            GenericSlide._check_extra_mpp_types(mpp_list, wsi_name)

    def test__check_extra_mpp_unique(self):
        wsi_name = "slide4.svs"
        mpp_list = [1.1, 2.2, 3.3, 4.4]
        GenericSlide._check_extra_mpp_unique(mpp_list, wsi_name)
        mpp_list = [1.1, 2.2, 1.1, 4.4]
        with self.assertRaises(ValueError):
            GenericSlide._check_extra_mpp_unique(mpp_list, wsi_name)

    def test__get_max_mpp_for_range(self):
        wsi_name = "slide1.svs"
        level_mpp_values = (0.1, 1, 10)
        magnification = 0.5
        range_max_magnification = 14
        result_max_mpp = GenericSlide._get_max_mpp_for_range(
            level_mpp_values, magnification, range_max_magnification, wsi_name
        )
        self.assertEqual(result_max_mpp, 0.05)
        magnification = None
        range_max_magnification = 15
        result_max_mpp = GenericSlide._get_max_mpp_for_range(
            level_mpp_values, magnification, range_max_magnification, wsi_name
        )
        self.assertEqual(result_max_mpp, 1.5)

    def test__check_mpp_range(self):
        wsi_name = "wsi.svs"
        min_mpp = 0.1
        max_mpp = 2
        input_mpp_value = 1
        GenericSlide._check_mpp_range([input_mpp_value], min_mpp, max_mpp, wsi_name)
        input_mpp_value = 2
        GenericSlide._check_mpp_range([input_mpp_value], min_mpp, max_mpp, wsi_name)
        input_mpp_value = 3
        with self.assertRaises(ValueError):
            GenericSlide._check_mpp_range([input_mpp_value], min_mpp, max_mpp, wsi_name)
        input_mpp_value = 0.01
        with self.assertRaises(ValueError):
            GenericSlide._check_mpp_range([input_mpp_value], min_mpp, max_mpp, wsi_name)

    def test__find_mpp_wsi_level(self):
        level_mpp_values = (0.1, 1, 10)
        mpp_level_margin = 0.05
        #
        input_mpp_value = 0.2
        result_level = GenericSlide._find_mpp_wsi_level(level_mpp_values, input_mpp_value, mpp_level_margin)
        self.assertEqual(result_level, -1)
        #
        input_mpp_value = 0.1
        result_level = GenericSlide._find_mpp_wsi_level(level_mpp_values, input_mpp_value, mpp_level_margin)
        self.assertEqual(result_level, 0)
        #
        input_mpp_value = 10
        result_level = GenericSlide._find_mpp_wsi_level(level_mpp_values, input_mpp_value, mpp_level_margin)
        self.assertEqual(result_level, 2)
        #
        input_mpp_value = 1.05
        result_level = GenericSlide._find_mpp_wsi_level(level_mpp_values, input_mpp_value, mpp_level_margin)
        self.assertEqual(result_level, -1)
        #
        input_mpp_value = 1.049
        result_level = GenericSlide._find_mpp_wsi_level(level_mpp_values, input_mpp_value, mpp_level_margin)
        self.assertEqual(result_level, 1)
        #
        level_mpp_values = (None, None, None)
        result_level = GenericSlide._find_mpp_wsi_level(level_mpp_values, input_mpp_value, mpp_level_margin)
        self.assertEqual(result_level, -1)

    def test__get_resampling_batches(self):
        mpps = ["a", "b", "c", "d", "e", "f", "g", "h"]
        # other inner cache keys ("Z" and "B") are purposely not created here
        resample_cache = {
            "a": {"L": 1},
            "b": {"L": 0},
            "c": {"L": 1},
            "d": {"L": 3},
            "e": {"L": 0},
            "f": {"L": 2},
            "g": {"L": 2},
            "h": {"L": 1},
        }
        output_batches = {0: ["b", "e"], 1: ["a", "c", "h"], 2: ["f", "g"], 3: ["d"]}
        result_batches = GenericSlide._get_resampling_batches(mpps, resample_cache)
        self.assertEqual(result_batches, output_batches)

    def test__get_wsi_level_array(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide = GenericSlide(wsi_file=wsi_file)
        #
        level_array = GenericSlide._get_wsi_level_array(wsi_slide, level=0)
        self.assertEqual(level_array.shape, (3072, 2560, 3))
        #
        level_array = GenericSlide._get_wsi_level_array(wsi_slide, level=2)
        self.assertEqual(level_array.shape, (192, 160, 3))

    def test__get_batch_resample_size_list(self):
        level_size = (1000, 1500)
        mpps = [20, 50, 75, 100, 120]
        # other inner cache keys ("Z" and "L") are purposely not created here
        resample_cache = {"20": {"B": 1}, "50": {"B": 2}, "75": {"B": 3}, "100": {"B": 4}, "120": {"B": 5.5}}
        output_list = [(1000, 1500), (500, 750), (333, 500), (250, 375), (182, 273)]
        result_list = GenericSlide._get_batch_resample_size_list(level_size, mpps, resample_cache)
        self.assertEqual(result_list, output_list)

    def test__get_resampled_wsi_level_images(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        level_array = io.imread(wsi_file)
        resample_size_list = [(300, 600), (200, 400), (100, 200)]
        resampling_filter = "LANCZOS"
        images = GenericSlide._get_resampled_wsi_level_images(level_array, resample_size_list, resampling_filter)
        self.assertIsInstance(images[0], PIL.Image.Image)
        self.assertIsInstance(images[1], PIL.Image.Image)
        self.assertIsInstance(images[2], PIL.Image.Image)
        self.assertEqual(images[0].width, 300)
        self.assertEqual(images[0].height, 600)
        self.assertEqual(images[1].width, 200)
        self.assertEqual(images[1].height, 400)
        self.assertEqual(images[2].width, 100)
        self.assertEqual(images[2].height, 200)

    def test__get_resampled_wsi_level_images_unknown_filter(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        level_array = io.imread(wsi_file)
        resample_size_list = [(300, 600), (200, 400), (100, 200)]
        resampling_filter = "ABC123"
        with self.assertRaises(KeyError):
            GenericSlide._get_resampled_wsi_level_images(level_array, resample_size_list, resampling_filter)

    def test__get_base_resample_level(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide = GenericSlide(wsi_file=wsi_file)
        #
        mpp = 1
        level = GenericSlide._get_base_resample_level(wsi_slide, mpp, level_zero_resampling=True)
        self.assertEqual(level, 0)
        #
        mpp = 1
        level = GenericSlide._get_base_resample_level(wsi_slide, mpp, level_zero_resampling=False)
        self.assertNotEqual(level, 0)

    def test__get_resampled_tile(self):
        base_level_region = Image.new(mode="RGB", size=(200, 200))
        size = (110, 120)
        resampling_filter = "LANCZOS"
        result_tile = GenericSlide._get_resampled_tile(base_level_region, size, resampling_filter)
        self.assertEqual(result_tile.width, 110)
        self.assertEqual(result_tile.height, 120)

    def test__get_resampled_tile_unknown_filter(self):
        base_level_region = Image.new(mode="RGB", size=(200, 200))
        size = (110, 120)
        resampling_filter = "ABC123"
        with self.assertRaises(KeyError):
            GenericSlide._get_resampled_tile(base_level_region, size, resampling_filter)


class TestGenericSlideWSIResamplingIdenticalFilters(TestCase):
    """Compare two identical WSI resampling methods.

    Methods in this class must be run in the order they are written.
    """

    image_filter1 = None

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.location = (250, 250)
        self.level_or_mpp = 0.7
        self.size = (300, 400)

    def tearDown(self):
        GenericSlide.set_resampling_filter("LANCZOS")

    def test_wsi_resampling_filter_default(self):
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="wsi", extra_mpps=[self.level_or_mpp])
        self.__class__.image_filter1 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)

    def test_wsi_resampling_filter_other(self):
        GenericSlide.set_resampling_filter("LANCZOS")
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="wsi", extra_mpps=[self.level_or_mpp])
        image_filter2 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)
        # test if arrays are identical
        image_filter1_array = np.asarray(self.image_filter1)
        image_filter2_array = np.asarray(image_filter2)
        np.testing.assert_equal(image_filter1_array, image_filter2_array)


class TestGenericSlideWSIResamplingDifferentFilters(TestCase):
    """Compare two different WSI resampling methods.

    Methods in this class must be run in the order they are written.
    """

    image_filter1 = None

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.location = (250, 250)
        self.level_or_mpp = 0.7
        self.size = (300, 400)

    def tearDown(self):
        GenericSlide.set_resampling_filter("LANCZOS")

    def test_wsi_resampling_filter_default(self):
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="wsi", extra_mpps=[self.level_or_mpp])
        self.__class__.image_filter1 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)

    def test_wsi_resampling_filter_other(self):
        GenericSlide.set_resampling_filter("NEAREST")
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="wsi", extra_mpps=[self.level_or_mpp])
        image_filter2 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)
        # test if arrays are different
        image_filter1_array = np.asarray(self.image_filter1)
        image_filter2_array = np.asarray(image_filter2)
        self.assertFalse(np.array_equal(image_filter1_array, image_filter2_array))


class TestGenericSlideTileResamplingIdenticalFilters(TestCase):
    """Compare two identical TILE resampling methods.

    Methods in this class must be run in the order they are written.
    """

    tile_filter1 = None

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.location = (250, 250)
        self.level_or_mpp = 0.9
        self.size = (400, 300)

    def tearDown(self):
        GenericSlide.set_resampling_filter("LANCZOS")

    def test_tile_resampling_filter_default(self):
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="tile")
        self.__class__.tile_filter1 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)

    def test_tile_resampling_filter_other_same(self):
        GenericSlide.set_resampling_filter("LANCZOS")
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="tile")
        tile_filter2 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)
        # test if arrays are identical
        tile_filter1_array = np.asarray(self.tile_filter1)
        tile_filter2_array = np.asarray(tile_filter2)
        np.testing.assert_equal(tile_filter1_array, tile_filter2_array)


class TestGenericSlideTileResamplingDifferentFilters(TestCase):
    """Compare two different TILE resampling methods.

    Methods in this class must be run in the order they are written.
    """

    tile_filter1 = None

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.location = (250, 250)
        self.level_or_mpp = 0.9
        self.size = (400, 300)

    def tearDown(self):
        GenericSlide.set_resampling_filter("LANCZOS")

    def test_tile_resampling_filter_default(self):
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="tile")
        self.__class__.tile_filter1 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)

    def test_tile_resampling_filter_other_different(self):
        GenericSlide.set_resampling_filter("NEAREST")
        wsi_slide = GenericSlide(wsi_file=self.wsi_file, resampling_mode="tile")
        tile_filter2 = wsi_slide.get_region(self.location, self.level_or_mpp, self.size)
        # test if arrays are different
        tile_filter1_array = np.asarray(self.tile_filter1)
        tile_filter2_array = np.asarray(tile_filter2)
        self.assertFalse(np.array_equal(tile_filter1_array, tile_filter2_array))


class TestGenericSlidePublicMethods(TestCase):
    """Tests for other public methods present in GenericSlide."""

    def tearDown(self):
        GenericSlide.set_range_max_magnification(40)

    def test_check_mpp_range_out_of_range(self):
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs)
        with self.assertRaises(ValueError):
            wsi_slide_svs.check_mpp_range(0.000000001)
        with self.assertRaises(ValueError):
            wsi_slide_svs.check_mpp_range(100000000)
        wsi_slide_svs.check_mpp_range(1)

    def test_check_mpp_range_with_resampling(self):
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs, resampling_mode="tile")
        wsi_slide_svs.check_mpp_range(0.3)

    def test_mpp_out_of_range_magnification_missing(self):
        # test if range checking is possible when mpp data is available, but magnification is not available
        GenericSlide.set_range_max_magnification(23)
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        # log message should be generated here ("Setting max magnification to 23 in MPP range checking")
        wsi_slide_tif.check_mpp_range(0.3)

    def test_check_mpp_data(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        mpp_data = (1, 2, 3)
        wsi_slide_tif.check_mpp_data(mpp_data)
        #
        mpp_data = (None, None, None)
        with self.assertRaises(ValueError):
            wsi_slide_tif.check_mpp_data(mpp_data)

    def test_get_property(self):
        """Returned value may be upper or lower case depending on WSI library."""
        wsi_file_tif = make_test_path("wsi/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        output_property = "centimeter"
        result_property = wsi_slide_tif.get_property("tiff.ResolutionUnit")
        self.assertEqual(result_property.upper(), output_property.upper())
        #
        with self.assertRaises(ValueError):
            wsi_slide_tif.get_property("pink unicorn")
        # read more properties from other images
        wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs)
        result_property = wsi_slide_svs.get_property("tiff.ImageDescription")
        self.assertTrue(len(result_property) > 50)
        #
        wsi_file_svs = make_test_path("wsi/JP2K-33003-1.svs")
        wsi_slide_svs = GenericSlide(wsi_file=wsi_file_svs)
        output_property = "6797"
        result_property = wsi_slide_svs.get_property("aperio.ImageID")
        self.assertEqual(result_property, output_property)
        #
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        output_property = "None"
        result_property = wsi_slide_tif.get_property("tiff.ResolutionUnit")
        self.assertEqual(result_property.upper(), output_property.upper())
