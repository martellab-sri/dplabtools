# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for flat image files using Pillow."""

import os
import glob
from unittest import TestCase

import numpy as np
from PIL import Image

from dplabtools.slides import GenericSlide
from dplabtools.slides.patches import PolygonRegionGridPatches, DiskPatchExtractor, MultiResDiskPatchExtractor
from dplabtools.slides.utils import AnnotationPolygon

from testutils import make_test_path


class TestGenericSlidePillowBasicProperties(TestCase):
    """Tests for basic slide properties using Pillow."""

    def setUp(self):
        self.wsi_file_flat = make_test_path("wsi/wsi-flat.jpg")
        self.wsi_slide_flat = GenericSlide(wsi_file=self.wsi_file_flat)

    def test_property_magnification(self):
        result_magnification = self.wsi_slide_flat.magnification
        self.assertIsNone(result_magnification)

    def test_property_level_dimensions(self):
        output_level_dimensions = ((2500, 2000),)
        result_level_dimensions = self.wsi_slide_flat.level_dimensions
        self.assertEqual(result_level_dimensions, output_level_dimensions)

    def test_property_level_downsamples(self):
        output_level_downsamples = (1.0,)
        result_level_downsamples = self.wsi_slide_flat.level_downsamples
        result_level_downsamples = tuple(round(ld, 6) for ld in result_level_downsamples)
        self.assertEqual(result_level_downsamples, output_level_downsamples)

    def test_property_mpp_data(self):
        output_mpp_data = (None, None)
        result_mpp_data = self.wsi_slide_flat.mpp_data
        self.assertEqual(result_mpp_data, output_mpp_data)

    def test_property_level_mpp_values(self):
        output_mpp_values = (None,)
        result_mpp_values = self.wsi_slide_flat.level_mpp_values
        self.assertEqual(result_mpp_values, output_mpp_values)

    def test_property_level_count(self):
        output_count = 1
        result_count = self.wsi_slide_flat.level_count
        self.assertEqual(result_count, output_count)

    def test_property_thumbnail_image(self):
        result_image = self.wsi_slide_flat.thumbnail_image
        self.assertTrue(result_image.width > 0)
        self.assertTrue(result_image.height > 0)

    def test_property_label_image(self):
        result_image = self.wsi_slide_flat.label_image
        self.assertIsNone(result_image)

    def test_property_lib_name(self):
        result_lib_name = self.wsi_slide_flat.lib_name
        self.assertTrue(len(result_lib_name) > 0)


class TestGenericSlidePillowChangeMPP(TestCase):
    """Tests when changing MPP between slides."""

    def test_change_mpp(self):
        GenericSlide.set_external_mpp(0.666)
        wsi_file_flat1 = make_test_path("wsi/wsi-flat.jpg")
        wsi_slide_flat1 = GenericSlide(wsi_file=wsi_file_flat1)
        output_mpp_data1 = (0.666, 0.666)
        result_mpp_data1 = wsi_slide_flat1.mpp_data
        self.assertEqual(result_mpp_data1, output_mpp_data1)
        GenericSlide.set_external_mpp(0.888)
        wsi_file_flat2 = make_test_path("wsi/wsi-flat.jpg")
        wsi_slide_flat2 = GenericSlide(wsi_file=wsi_file_flat2)
        output_mpp_data2 = (0.888, 0.888)
        result_mpp_data2 = wsi_slide_flat2.mpp_data
        self.assertEqual(result_mpp_data2, output_mpp_data2)


class TestPolygonPillowGetRegion(TestCase):
    """Tests for get region function using Pillow."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-flat.tif")
        GenericSlide.set_external_mpp(None)

    def test_get_region_level(self):
        wsi_slide_flat = GenericSlide(wsi_file=self.wsi_file_tif)
        result_region_tif = make_test_path("saved_data/libs/test_region3.tif")
        location = (1024, 768)
        size = (256, 256)
        level = 0
        region = wsi_slide_flat.get_region(location, level, size)
        region.save(result_region_tif)
        output_image = Image.open(make_test_path("ref_data/slides/libs/regions/region3.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_region_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_get_region_mpp1(self):
        # set MPP and read as level 0 MPP
        GenericSlide.set_external_mpp(0.128)
        wsi_slide_flat = GenericSlide(wsi_file=self.wsi_file_tif)
        result_region_tif = make_test_path("saved_data/libs/test_region4.tif")
        location = (1024, 2048)
        size = (256, 256)
        mpp = 0.128
        region = wsi_slide_flat.get_region(location, mpp, size)
        region.save(result_region_tif)
        output_image = Image.open(make_test_path("ref_data/slides/libs/regions/region4.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_region_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # also test set_external_mpp function
        mpp_data = wsi_slide_flat.mpp_data
        self.assertEqual(mpp_data, (0.128, 0.128))

    def test_get_region_mpp2(self):
        # read MPP without setting MPP
        wsi_slide_flat = GenericSlide(wsi_file=self.wsi_file_tif)
        location = (1024, 2048)
        size = (256, 256)
        mpp = 0.256
        with self.assertRaises(ValueError):
            wsi_slide_flat.get_region(location, mpp, size)

    def test_get_region_mpp3(self):
        # set MPP but read different MPP
        GenericSlide.set_external_mpp(0.115)
        wsi_slide_flat = GenericSlide(wsi_file=self.wsi_file_tif)
        location = (1024, 2048)
        size = (256, 256)
        mpp = 0.256
        with self.assertRaises(ValueError):
            wsi_slide_flat.get_region(location, mpp, size)

    def test_get_region_array(self):
        wsi_slide_flat = GenericSlide(wsi_file=self.wsi_file_tif)
        location = (1024, 768)
        size = (256, 256)
        level = 0
        with self.assertRaises(NotImplementedError):
            wsi_slide_flat.get_region_array(location, level, size)


class TestPolygonPillowRegionGridPatches(TestCase):
    """Tests for basic patch extraction using Pillow."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-flat.tif")
        self.mask_array = np.ones((2560, 3072), dtype=int)
        poly1 = AnnotationPolygon(points=[(256, 256), (256, 1020), (1020, 1020), (1020, 256)], label="area1")
        poly2 = AnnotationPolygon(points=[(1280, 1536), (1280, 2800), (2500, 2800)], label="area2")
        self.polygons = [poly1, poly2]
        GenericSlide.set_external_mpp(None)

    def test_count_patches(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 19)

    def test_extract_patches(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        output_dir = make_test_path("saved_data/patches29")
        DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            filename_separator="__",
            create_subdirs=True,
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 19)

    def test_extract_multires_patches(self):
        GenericSlide.set_external_mpp(0.5)
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        output_dir = make_test_path("saved_data/patches30")
        MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=3,
            levels_or_mpps=[0, 0.5, 0.75],
            resampling_mode="wsi",
            image_type="tif",
            create_subdirs=True,
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 19 * 3)
        # changing MPP value in the same context (on the fly)
        GenericSlide.set_external_mpp(0.25)
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        output_dir = make_test_path("saved_data/patches31")
        MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=3,
            levels_or_mpps=[0, 0.5, 0.75, 0.88],
            resampling_mode="wsi",
            image_type="tif",
            create_subdirs=True,
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 19 * 4)
