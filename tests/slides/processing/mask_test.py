# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for mask generation.

Tested classes:
    WSIPolygonMask
    WSITissueMask
"""

import os
from unittest import TestCase

import numpy as np
from PIL import Image

from dplabtools.slides.processing import WSIPolygonMask, WSITissueMask
from dplabtools.slides.utils import AnnotationPolygon
from testutils import make_test_path


class TestWSIPolygonMaskStaticMethodCreate(TestCase):
    """Tests for create static method in WSIPolygonMask."""

    def test__create_mask(self):
        # single rectangle - flexible fit
        mask_size = (500, 700)
        poly1 = AnnotationPolygon(points=[(100, 300), (100, 400), (200, 400), (200, 300)], label="")
        polygons = [poly1]
        result_mask = WSIPolygonMask._create_mask(mask_size, polygons)
        output_mask = np.full(mask_size, False, dtype=bool)
        output_mask[100 : 200 + 1, 300 : 400 + 1] = True
        np.testing.assert_equal(result_mask, output_mask)
        # single rectangle - one direction fit
        mask_size = (310, 410)
        poly1 = AnnotationPolygon(points=[(100, 300), (100, 400), (200, 400), (200, 300)], label="")
        polygons = [poly1]
        result_mask = WSIPolygonMask._create_mask(mask_size, polygons)
        output_mask = np.full(mask_size, False, dtype=bool)
        output_mask[100 : 200 + 1, 300 : 400 + 1] = True
        np.testing.assert_equal(result_mask, output_mask)
        # rectangle set
        mask_size = (500, 700)
        poly1 = AnnotationPolygon(points=[(100, 100), (100, 200), (200, 200), (200, 100)], label="")
        poly2 = AnnotationPolygon(points=[(300, 300), (300, 400), (400, 400), (400, 300)], label="")
        poly3 = AnnotationPolygon(points=[(400, 400), (400, 450), (450, 450), (450, 400)], label="")
        polygons = [poly1, poly2, poly3]
        result_mask = WSIPolygonMask._create_mask(mask_size, polygons)
        output_mask = np.full(mask_size, False, dtype=bool)
        output_mask[100 : 200 + 1, 100 : 200 + 1] = True
        output_mask[300 : 400 + 1, 300 : 400 + 1] = True
        output_mask[400 : 450 + 1, 400 : 450 + 1] = True
        np.testing.assert_equal(result_mask, output_mask)
        # triangle
        mask_size = (20, 20)
        poly1 = AnnotationPolygon(points=[(0, 0), (0, 10), (10, 0)], label="")
        polygons = [poly1]
        result_mask = WSIPolygonMask._create_mask(mask_size, polygons)
        output_mask = np.full(mask_size, False, dtype=bool)
        output_mask[0, 0 : 10 + 1] = True
        output_mask[1, 0 : 9 + 1] = True
        output_mask[2, 0 : 8 + 1] = True
        output_mask[3, 0 : 7 + 1] = True
        output_mask[4, 0 : 6 + 1] = True
        output_mask[5, 0 : 5 + 1] = True
        output_mask[6, 0 : 4 + 1] = True
        output_mask[7, 0 : 3 + 1] = True
        output_mask[8, 0 : 2 + 1] = True
        output_mask[9, 0 : 1 + 1] = True
        output_mask[10, 0 : 0 + 1] = True
        np.testing.assert_equal(result_mask, output_mask)
        # full coverage
        mask_size = (50, 50)
        poly1 = AnnotationPolygon(points=[(0, 0), (0, 50), (50, 50), (50, 0)], label="")
        polygons = [poly1]
        result_mask = WSIPolygonMask._create_mask(mask_size, polygons)
        output_mask = np.full(mask_size, False, dtype=bool)
        output_mask[0:50, 0:50] = True
        np.testing.assert_equal(result_mask, output_mask)
        # oversized rectangle (x)
        mask_size = (20, 200)
        poly1 = AnnotationPolygon(points=[(0, 0), (0, 50), (50, 50), (50, 0)], label="")
        polygons = [poly1]
        with self.assertRaises(ValueError):
            WSIPolygonMask._create_mask(mask_size, polygons)
        # oversized rectangle (y)
        mask_size = (200, 20)
        poly1 = AnnotationPolygon(points=[(0, 0), (0, 50), (50, 50), (50, 0)], label="")
        polygons = [poly1]
        with self.assertRaises(ValueError):
            WSIPolygonMask._create_mask(mask_size, polygons)
        # oversized rectangle (xy)
        mask_size = (20, 20)
        poly1 = AnnotationPolygon(points=[(0, 0), (0, 50), (50, 50), (50, 0)], label="")
        polygons = [poly1]
        with self.assertRaises(ValueError):
            WSIPolygonMask._create_mask(mask_size, polygons)


class TestWSIPolygonMaskSavingFiles(TestCase):
    """Tests for saving files in polygon mask class."""

    def test_polygon_mask_saving_files(self):
        poly1 = AnnotationPolygon(points=[(100, 100), (100, 200), (500, 200), (500, 100)], label="")
        polygons = [poly1]
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        save_dir = make_test_path("saved_data/masks")
        array_file_name = "polygon_mask.npz"
        png_file_name = "polygon_mask.png"
        array_file_path = os.path.join(save_dir, array_file_name)
        png_file_path = os.path.join(save_dir, png_file_name)
        mask = WSIPolygonMask(wsi_file=wsi_file_tif, level_or_minsize=1, polygons=polygons)
        mask.save_array(array_file_path)
        mask.save_png(png_file_path)
        # read saved files
        mask_array = np.load(array_file_path)["data"]
        self.assertIsInstance(mask_array, np.ndarray)
        self.assertEqual(mask_array.shape, (640, 768))
        mask_png = Image.open(png_file_path)
        self.assertIsInstance(mask_png, Image.Image)
        self.assertEqual((mask_png.width, mask_png.height), (640, 768))
        mask_png.close()


class TestWSIPolygonMaskSavingOverlay(TestCase):
    """Tests for saving overlays in polygon mask class."""

    def test_polygon_mask_saving_overlay(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        save_dir = make_test_path("saved_data/masks")
        poly1 = AnnotationPolygon(points=[(128, 64), (128, 256), (512, 256), (512, 64)], label="")
        polygons = [poly1]
        mask = WSIPolygonMask(wsi_file=wsi_file_tif, level_or_minsize=1, polygons=polygons)
        png_file_name = "test_overlay9.png"
        png_file_path = os.path.join(save_dir, png_file_name)
        mask.save_overlay_png(png_file_path)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay9.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)


class TestWSITissueMaskStaticMethodCreate(TestCase):
    """Tests for create static method in WSITissueMask."""

    def test__create_mask(self):
        image_array = np.full((100, 100, 3), 255, dtype=np.uint8)
        image_array[20:40, 60:90] = [178, 92, 142]
        output_mask = np.full((100, 100), False, dtype=bool)
        output_mask[20:40, 60:90] = True
        #
        mode = "lab"
        threshold = 0.1
        result_mask = WSITissueMask._create_mask(image_array, mode, threshold)
        np.testing.assert_equal(result_mask, output_mask)
        #
        mode = "hsv"
        threshold = 0.1
        result_mask = WSITissueMask._create_mask(image_array, mode, threshold)
        np.testing.assert_equal(result_mask, output_mask)
        #
        mode = "otsu"
        threshold = None
        result_mask = WSITissueMask._create_mask(image_array, mode, threshold)
        np.testing.assert_equal(result_mask, output_mask)
        #
        mode = "kiwi"
        threshold = None
        with self.assertRaises(ValueError):
            WSITissueMask._create_mask(image_array, mode, threshold)


class TestWSITissueMaskStaticMethodTransform(TestCase):
    """Tests for transform static method in WSITissueMask."""

    def setUp(self):
        self.mask_array = np.full((100, 110), False, dtype=bool)
        self.output_mask = np.full((100, 110), False, dtype=bool)

    def test__transform_mask_binary_fill_holes1(self):
        level = 0
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0,
            remove_all_holes=True,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_binary_fill_holes2(self):
        level = 2
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0,
            remove_all_holes=True,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_pass1(self):
        # max_allowed_area is 2
        level = 3
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.1,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_pass2(self):
        # max_allowed_area is 3
        level = 2
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.01,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_pass3(self):
        # max_allowed_area is 5
        level = 1
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.001,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_pass4(self):
        # max_allowed_area is 7
        level = 0
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.0001,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_fail1(self):
        # max_allowed_area is 1
        level = 3
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.01,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25, 65] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_fail2(self):
        # max_allowed_area is 1
        level = 2
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.001,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25, 65] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_fail3(self):
        # max_allowed_area is 1
        level = 1
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.0001,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25, 65] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_one_pixel_fail4(self):
        # max_allowed_area is 1
        level = 0
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25, 65] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.00001,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25, 65] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_pass1(self):
        # max_allowed_area is 6554
        level = 0
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.1,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_pass2(self):
        # max_allowed_area is 410
        level = 1
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.1,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_pass3(self):
        # max_allowed_area is 103
        level = 2
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.4,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_pass4(self):
        # max_allowed_area is 105
        level = 3
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=6.5,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_fail1(self):
        # max_allowed_area is 66
        level = 0
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.001,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25:35, 65:75] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_fail2(self):
        # max_allowed_area is 41
        level = 1
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.01,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25:35, 65:75] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_fail3(self):
        # max_allowed_area is 26
        level = 2
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.1,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25:35, 65:75] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_fail4(self):
        # max_allowed_area is 17
        level = 3
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:35, 65:75] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=1,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25:35, 65:75] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_barely_pass(self):
        # max_allowed_area is 26, hole area is 24
        level = 2
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:31, 65:69] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.1,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_holes_many_pixels_barely_fail(self):
        # max_allowed_area is 26, hole area is 28
        level = 2
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:32, 65:69] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0.1,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:40, 60:90] = True
        self.output_mask[25:32, 65:69] = False
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_one_pixel_pass1(self):
        # min_allowed_size is 2
        level = 3
        self.mask_array[25, 65] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.1,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_one_pixel_pass2(self):
        # min_allowed_size is 26
        level = 2
        self.mask_array[25, 65] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.1,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_one_pixel_pass3(self):
        # min_allowed_size is 410
        level = 1
        self.mask_array[25, 65] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.1,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_one_pixel_fail1(self):
        # min_allowed_size is 1
        level = 3
        self.mask_array[25, 65] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.01,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[25, 65] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_one_pixel_fail2(self):
        # min_allowed_size is 1
        level = 2
        self.mask_array[25, 65] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.001,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[25, 65] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_one_pixel_fail3(self):
        # min_allowed_size is 1
        level = 1
        self.mask_array[25, 65] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.0001,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[25, 65] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_many_pixels_barely_pass(self):
        # min_allowed_size is 26, object area is 24
        level = 2
        self.mask_array[20:26, 60:64] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.1,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__transform_mask_remove_small_objects_many_pixels_barely_fail(self):
        # min_allowed_size is 26, object area is 28
        level = 2
        self.mask_array[20:27, 60:64] = True
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0.1,
            remove_all_holes=False,
            close_fill_kernel_size=0,
        )
        self.output_mask[20:27, 60:64] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__close_fill_kernel_size1(self):
        # pass, hole removed
        level = 0
        self.mask_array[20:40, 60:90] = True
        self.mask_array[25:30, 65:70] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=10,
        )
        # shifted by one, for some reason
        self.output_mask[20 + 1 : 40 + 1, 60 + 1 : 90 + 1] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__close_fill_kernel_size2(self):
        # pass, hole removed + changed shape and moved by one pixel
        level = 0
        self.mask_array[10:90, 10:90] = True
        self.mask_array[15:30, 25:40] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=20,
        )
        self.output_mask[10 + 1 : 90 + 10, 10 + 1 : 90 + 1] = True
        np.testing.assert_equal(result_mask, self.output_mask)

    def test__close_fill_kernel_size3(self):
        # fail, hole not removed
        level = 0
        self.mask_array[10:90, 10:90] = True
        self.mask_array[15:30, 25:40] = False
        result_mask = WSITissueMask._transform_mask(
            self.mask_array,
            level,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0,
            remove_all_holes=False,
            close_fill_kernel_size=15,
        )
        self.output_mask[10:90, 10:90] = True
        self.output_mask[15:30, 25:40] = False
        np.testing.assert_equal(result_mask, self.output_mask)


class TestWSITissueMaskProperties(TestCase):
    """Tests for properties in mask classes."""

    def test_mask_properties(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        # this will also test level_or_minsize
        mask = WSITissueMask(wsi_file=wsi_file_tif, level_or_minsize=100)
        self.assertEqual(mask.level, 2)
        self.assertIsInstance(mask.array, np.ndarray)
        self.assertEqual(mask.array.shape, (160, 192))


class TestWSITissueMaskSavingFiles(TestCase):
    """Tests for saving files in tissue mask class."""

    def test_tissue_mask_saving_files(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        save_dir = make_test_path("saved_data/masks")
        array_file_name = "tissue_mask.npz"
        png_file_name = "tissue_mask.png"
        array_file_path = os.path.join(save_dir, array_file_name)
        png_file_path = os.path.join(save_dir, png_file_name)
        mask = WSITissueMask(wsi_file=wsi_file_tif, level_or_minsize=2)
        mask.save_array(array_file_path)
        mask.save_png(png_file_path)
        # read saved files
        mask_array = np.load(array_file_path)["data"]
        self.assertIsInstance(mask_array, np.ndarray)
        self.assertEqual(mask_array.shape, (160, 192))
        mask_png = Image.open(png_file_path)
        self.assertIsInstance(mask_png, Image.Image)
        self.assertEqual((mask_png.width, mask_png.height), (160, 192))
        mask_png.close()


class TestWSITissueMaskSavingOverlay(TestCase):
    """Tests for saving overlays in tissue mask class."""

    wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
    save_dir = make_test_path("saved_data/masks")
    mask = WSITissueMask(wsi_file=wsi_file_tif, level_or_minsize=2)

    def test_tissue_mask_saving_overlay1(self):
        # default params
        png_file_name = "test_overlay1.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay1.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_tissue_mask_saving_overlay2(self):
        # no mask color, others default
        png_file_name = "test_overlay2.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path, mask_color=None)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay2.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_tissue_mask_saving_overlay3(self):
        # different mask color + different transparency
        png_file_name = "test_overlay3.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path, mask_color="black", mask_alpha=0.9)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay3.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_tissue_mask_saving_overlay4(self):
        # different mask color + different transparency + no outline
        png_file_name = "test_overlay4.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path, mask_color="black", mask_alpha=0.9, outline_color=None)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay4.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_tissue_mask_saving_overlay5(self):
        # no outline color, others default
        png_file_name = "test_overlay5.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path, outline_color=None)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay5.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_tissue_mask_saving_overlay6(self):
        # different outline color, no mask color
        png_file_name = "test_overlay6.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path, mask_color=None, outline_color="pink")
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay6.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_tissue_mask_saving_overlay7(self):
        # different thickness, no mask color
        png_file_name = "test_overlay7.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path, mask_color=None, outline_thickness=5)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay7.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_tissue_mask_saving_overlay8(self):
        # no mask color, no outline color
        png_file_name = "test_overlay8.png"
        png_file_path = os.path.join(self.save_dir, png_file_name)
        self.mask.save_overlay_png(png_file_path, mask_color=None, outline_color=None)
        # read saved files
        result_image = Image.open(png_file_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/masks/ref_overlay8.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
