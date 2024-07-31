# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for heatmap generation.

Tested classes:
    WSIHeatmap
"""

import os
from unittest import TestCase

import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

from dplabtools.slides import GenericSlide
from dplabtools.slides.processing import WSIHeatmap
from testutils import make_test_path


class TestWSIHeatmap(TestCase):
    """Generic tests."""

    def test__process_heatmap_data(self):
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        self.assertIsInstance(heatmap._probs_array, np.ndarray)
        self.assertIsInstance(heatmap._probs_array_transposed, np.ndarray)
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=None, vmax=None)
        self.assertIsInstance(heatmap._probs_array, np.ndarray)
        self.assertIsInstance(heatmap._probs_array_transposed, np.ndarray)
        #
        input_array = np.ones((100, 200))
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=None, vmax=None)
        #
        input_array = np.ones((100, 200))
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=1, vmax=1)
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=2, vmax=2)


class TestWSIHeatmapStaticMethods(TestCase):
    """Tests for static methods in WSIHeatmap class."""

    def test__get_pixel_data(self):
        # no background color, no transparency, all tissue
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        result_array = WSIHeatmap._get_pixel_data(input_array, None, "jet", 1, 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # no background color, no transparency, non tissue present
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        input_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        output_array[30:70, 40:80] = [0, 0, 0, 0]
        result_array = WSIHeatmap._get_pixel_data(input_array, None, "jet", 1, 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # background is green (#008000), no transparency, non tissue present
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        input_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        output_array[30:70, 40:80] = [0, 128, 0, 255]
        result_array = WSIHeatmap._get_pixel_data(input_array, "green", "jet", 1, 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # background is green (#008000), transparency is 50%, non tissue present
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        input_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        output_array[30:70, 40:80] = [0, 128, 0, 128]
        output_array[:, :, 3] = 128
        result_array = WSIHeatmap._get_pixel_data(input_array, "green", "jet", 0.5, 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # no background color, transparency 50%, non tissue present
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        input_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        output_array[30:70, 40:80] = [0, 0, 0, 0]
        output_array[0:30, 0:200, 3] = 128
        output_array[70:100, 0:200, 3] = 128
        output_array[0:100, 0:40, 3] = 128
        output_array[0:100, 80:200, 3] = 128
        result_array = WSIHeatmap._get_pixel_data(input_array, None, "jet", 0.5, 0, 1)
        np.testing.assert_equal(result_array, output_array)

    def test__get_colormapped_array(self):
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        # jet
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        result_array = WSIHeatmap._get_colormapped_array(input_array, "jet", 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # rainbow
        output_array = np.full((100, 200, 4), [255, 0, 0, 255])
        output_array[0:50, 0:50] = [127, 0, 255, 255]
        result_array = WSIHeatmap._get_colormapped_array(input_array, "rainbow", 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # different vmin/vmax
        input_array = np.full((100, 200), 0.75)
        input_array[0:50, 0:50] = 0.25
        result_array = WSIHeatmap._get_colormapped_array(input_array, "jet", 0.25, 0.75)
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        np.testing.assert_equal(result_array, output_array)
        # non matching vmin/vmax ranges
        input_array = np.full((100, 200), 0.75)
        input_array[0:50, 0:50] = 0.25
        result_array = WSIHeatmap._get_colormapped_array(input_array, "jet", 0, 1)
        output_array = np.full((100, 200, 4), [255, 148, 0, 255])
        output_array[0:50, 0:50] = [0, 128, 255, 255]
        np.testing.assert_equal(result_array, output_array)

    def test__apply_background(self):
        input_cmapped_array = np.full((100, 200, 4), [127, 0, 0, 255])
        input_cmapped_array[0:50, 0:50] = [0, 0, 127, 255]
        input_probs_array = np.full((100, 200), 0.3)
        input_probs_array[30:70, 40:80] = np.nan
        output_array = np.copy(input_cmapped_array)
        output_array[30:70, 40:80] = [255, 148, 0, 55]
        WSIHeatmap._apply_background(input_cmapped_array, input_probs_array, "#ff9400", 55)
        np.testing.assert_equal(input_cmapped_array, output_array)

    def test__apply_transparency(self):
        input_cmapped_array = np.full((100, 200, 4), [127, 0, 0, 255])
        input_cmapped_array[0:50, 0:50] = [0, 0, 127, 255]
        input_probs_array = np.full((100, 200), 0.3)
        # bottom right corner is NaN, tranparency should not be applied
        input_probs_array[70:100, 150:200] = np.nan
        output_array = np.copy(input_cmapped_array)
        output_array[0:70, 0:150, 3] = 77
        output_array[0:70, 150:200, 3] = 77
        output_array[70:100, 0:150, 3] = 77
        WSIHeatmap._apply_transparency(input_cmapped_array, input_probs_array, 77)
        np.testing.assert_equal(input_cmapped_array, output_array)

    def test__remove_transparency(self):
        input_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array = np.full((100, 200, 3), [127, 0, 0])
        result_array = WSIHeatmap._remove_transparency(input_array)
        np.testing.assert_equal(result_array, output_array)


class TestWSIHeatmapSavingFiles(TestCase):
    """Tests for saving files in WSIHeatmap class."""

    def test_save_png(self):
        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_file = "test_heatmap1.png"
        result_image_file = os.path.join(save_dir, result_image_file)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_png(result_image_file)
        # read saved image
        result_image = Image.open(result_image_file)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_png_draw(self):
        def draw_function(image_draw, *args):
            rect = args[0]
            rect_polygon = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
            image_draw.polygon(rect_polygon, outline="lime", width=2)

        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_file = "test_heatmap2.png"
        result_image_file = os.path.join(save_dir, result_image_file)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_png(result_image_file, draw_fn=draw_function, draw_args=((50, 70, 70, 90),))
        # read saved image
        result_image = Image.open(result_image_file)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap2.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_colorbar_png(self):
        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_file = "test_heatmap3.png"
        result_image_file = os.path.join(save_dir, result_image_file)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_colorbar_png(result_image_file)
        # read saved image
        result_image = Image.open(result_image_file)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap3.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_colorbar_png_interpolation(self):
        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_file1 = "test_heatmap_interpolation1.png"
        result_image_file1 = os.path.join(save_dir, result_image_file1)
        result_image_file2 = "test_heatmap_interpolation2.png"
        result_image_file2 = os.path.join(save_dir, result_image_file2)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        # save
        heatmap.save_colorbar_png(result_image_file1)
        heatmap.save_colorbar_png(result_image_file2, interpolation="bicubic")
        # compare
        heatmap_image1 = Image.open(result_image_file1)
        heatmap_image2 = Image.open(result_image_file2)
        heatmap_array1 = np.asarray(heatmap_image1)
        heatmap_array2 = np.asarray(heatmap_image2)
        heatmap_image1.close()
        heatmap_image2.close()
        self.assertFalse(np.array_equal(heatmap_array1, heatmap_array2))

    def test_save_colorbar_png_dpi(self):
        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_file1 = "test_heatmap_dpi1.png"
        result_image_file1 = os.path.join(save_dir, result_image_file1)
        result_image_file2 = "test_heatmap_dpi2.png"
        result_image_file2 = os.path.join(save_dir, result_image_file2)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        # save
        heatmap.save_colorbar_png(result_image_file1, dpi=100)
        heatmap.save_colorbar_png(result_image_file2, dpi=200)
        # compare
        heatmap_image1 = Image.open(result_image_file1)
        heatmap_image2 = Image.open(result_image_file2)
        heatmap_size1 = (heatmap_image1.width, heatmap_image1.height)
        heatmap_size2 = (heatmap_image2.width, heatmap_image2.height)
        heatmap_image1.close()
        heatmap_image2.close()
        self.assertGreater(heatmap_size2[0], heatmap_size1[0])
        self.assertGreater(heatmap_size2[1], heatmap_size1[1])

    def test_save_colorbar_png_draw(self):
        def draw_function(current_axes, *args):
            region = args[0]
            x0, y0, x1, y1 = region
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.5, edgecolor="lime", facecolor="none")
            current_axes.add_patch(rect)

        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_file = "test_heatmap4.png"
        result_image_file = os.path.join(save_dir, result_image_file)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_colorbar_png(result_image_file, draw_fn=draw_function, draw_args=((50, 70, 70, 90),))
        # read saved image
        result_image = Image.open(result_image_file)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap4.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_tif(self):
        # test WSI file must include MPP data
        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        input_array[100:120, 140:160] = np.nan
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        save_dir = make_test_path("saved_data/heatmaps1")
        #
        # background is undefined -> no compression used
        result_image_name = "test_heatmap1.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_tif(result_image_path, wsi_file_tif)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 16, wsi_slide.mpp_data[1] * 16))
        self.assertTrue(os.path.getsize(result_image_path) > 100000)
        #
        # force no compression when background defined + no transparency
        result_image_name = "test_heatmap2.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color="yellow", vmin=0, vmax=1)
        heatmap.save_tif(result_image_path, wsi_file_tif, allow_compression=False)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap2.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 16, wsi_slide.mpp_data[1] * 16))
        self.assertTrue(os.path.getsize(result_image_path) > 100000)
        #
        # background is undefined + transparency -> no compression used
        result_image_name = "test_heatmap3.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, alpha=0.5, vmin=0, vmax=1)
        heatmap.save_tif(result_image_path, wsi_file_tif)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap3.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 16, wsi_slide.mpp_data[1] * 16))
        self.assertTrue(os.path.getsize(result_image_path) > 100000)
        #
        # background is defined + transparency -> no compression used
        result_image_name = "test_heatmap4.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color="orange", alpha=0.5, vmin=0, vmax=1)
        heatmap.save_tif(result_image_path, wsi_file_tif)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap4.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 16, wsi_slide.mpp_data[1] * 16))
        self.assertTrue(os.path.getsize(result_image_path) > 100000)
        #
        # allow compression, results are undeterministic between different envs
        result_image_name = "test_heatmap5.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color="lime", vmin=0, vmax=1)
        heatmap.save_tif(result_image_path, wsi_file_tif)
        # check file size
        self.assertTrue(os.path.exists(result_image_path))
        self.assertTrue(os.path.getsize(result_image_path) < 20000)

    def test_save_tif_downsample_factor(self):
        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        input_array[100:120, 140:160] = np.nan
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        save_dir = make_test_path("saved_data/heatmaps1")
        #
        # save df=1
        result_image_name = "test_heatmap6.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_tif(result_image_path, wsi_file_tif, downsample_factor=1)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap6.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, wsi_slide.mpp_data)
        #
        # save df=4
        result_image_name = "test_heatmap7.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_tif(result_image_path, wsi_file_tif, downsample_factor=4)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap7.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 4, wsi_slide.mpp_data[1] * 4))

    def test_save_tif_draw(self):
        def draw_function(pixel_data, *args):
            rect = args[0]
            pixel_data[rect[0] : rect[1], rect[2] : rect[3]] = [0, 127, 0, 255]

        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        input_array[100:120, 140:160] = np.nan
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_name = "test_heatmap8.tif"
        result_image_path = os.path.join(save_dir, result_image_name)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, alpha=0.5, vmin=0, vmax=1)
        heatmap.save_tif(
            result_image_path,
            wsi_file_tif,
            draw_fn=draw_function,
            draw_args=((140, 160, 40, 60),),
        )
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap8.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 16, wsi_slide.mpp_data[1] * 16))

    def test_save_overlay_png(self):
        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        wsi_file_tif = make_test_path("wsi/board-multi-layer-compression-clean.tif")
        save_dir = make_test_path("saved_data/heatmaps1")
        #
        # no transparency + cutoff
        result_image_file = "test_overlay1.png"
        result_image_path = os.path.join(save_dir, result_image_file)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_overlay_png(result_image_path, wsi_file_tif, cutoff=0.5)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_overlay1.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        #
        # transparency 0.5 + cutoff
        result_image_file = "test_overlay2.png"
        result_image_path = os.path.join(save_dir, result_image_file)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, alpha=0.5, vmin=0, vmax=1)
        heatmap.save_overlay_png(result_image_path, wsi_file_tif, cutoff=0.5)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_overlay2.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        #
        # transparency 0.9 + no cut-off
        result_image_file = "test_overlay3.png"
        result_image_path = os.path.join(save_dir, result_image_file)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, alpha=0.9, vmin=0, vmax=1)
        heatmap.save_overlay_png(result_image_path, wsi_file_tif, cutoff=None)
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_overlay3.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_overlay_png_draw(self):
        def draw_function(image_draw, *args):
            rect = args[0]
            rect_polygon = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
            image_draw.polygon(rect_polygon, outline="lime", width=2)

        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        wsi_file_tif = make_test_path("wsi/board-multi-layer-compression-clean.tif")
        save_dir = make_test_path("saved_data/heatmaps1")
        result_image_file = "test_overlay4.png"
        result_image_path = os.path.join(save_dir, result_image_file)
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None, vmin=0, vmax=1)
        heatmap.save_overlay_png(
            result_image_path, wsi_file_tif, cutoff=0.5, draw_fn=draw_function, draw_args=((50, 70, 70, 90),)
        )
        # read saved image
        result_image = Image.open(result_image_path)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_overlay4.png")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)

    def test_save_all_files(self):
        """Run this test to ensure that saving any files does not modify the original array with source values."""
        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0.5
        input_array[140:, 172:] = np.nan
        # backup original array first
        input_array_backup = np.copy(input_array)
        # save all possible files
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        save_dir = make_test_path("saved_data/heatmaps2")
        # default settings
        heatmap = WSIHeatmap(heatmap_data=input_array)
        png_file = "file1.png"
        png_file_path = os.path.join(save_dir, png_file)
        heatmap.save_png(png_file_path)
        colorbar_png_file = "file2.png"
        colorbar_png_file_path = os.path.join(save_dir, colorbar_png_file)
        heatmap.save_colorbar_png(colorbar_png_file_path)
        overlay_png_file = "file3.png"
        overlay_png_file_path = os.path.join(save_dir, overlay_png_file)
        heatmap.save_overlay_png(overlay_png_file_path, wsi_file_tif)
        tif_file = "file4.tif"
        tif_file_path = os.path.join(save_dir, tif_file)
        heatmap.save_tif(tif_file_path, wsi_file_tif)
        # without background
        heatmap = WSIHeatmap(heatmap_data=input_array, background_color=None)
        png_file = "file5.png"
        png_file_path = os.path.join(save_dir, png_file)
        heatmap.save_png(png_file_path)
        colorbar_png_file = "file6.png"
        colorbar_png_file_path = os.path.join(save_dir, colorbar_png_file)
        heatmap.save_colorbar_png(colorbar_png_file_path)
        overlay_png_file = "file7.png"
        overlay_png_file_path = os.path.join(save_dir, overlay_png_file)
        heatmap.save_overlay_png(overlay_png_file_path, wsi_file_tif)
        tif_file = "file8.tif"
        tif_file_path = os.path.join(save_dir, tif_file)
        heatmap.save_tif(tif_file_path, wsi_file_tif)
        # with transparency
        heatmap = WSIHeatmap(heatmap_data=input_array, alpha=0.5)
        png_file = "file9.png"
        png_file_path = os.path.join(save_dir, png_file)
        heatmap.save_png(png_file_path)
        colorbar_png_file = "file10.png"
        colorbar_png_file_path = os.path.join(save_dir, colorbar_png_file)
        heatmap.save_colorbar_png(colorbar_png_file_path)
        overlay_png_file = "file11.png"
        overlay_png_file_path = os.path.join(save_dir, overlay_png_file)
        heatmap.save_overlay_png(overlay_png_file_path, wsi_file_tif)
        tif_file = "file12.tif"
        tif_file_path = os.path.join(save_dir, tif_file)
        heatmap.save_tif(tif_file_path, wsi_file_tif)
        # with data range
        heatmap = WSIHeatmap(heatmap_data=input_array, vmin=0.3, vmax=0.9)
        png_file = "file13.png"
        png_file_path = os.path.join(save_dir, png_file)
        heatmap.save_png(png_file_path)
        colorbar_png_file = "file14.png"
        colorbar_png_file_path = os.path.join(save_dir, colorbar_png_file)
        heatmap.save_colorbar_png(colorbar_png_file_path)
        overlay_png_file = "file15.png"
        overlay_png_file_path = os.path.join(save_dir, overlay_png_file)
        heatmap.save_overlay_png(overlay_png_file_path, wsi_file_tif)
        tif_file = "file16.tif"
        tif_file_path = os.path.join(save_dir, tif_file)
        heatmap.save_tif(tif_file_path, wsi_file_tif)
        # separate test for cutoff values
        heatmap = WSIHeatmap(heatmap_data=input_array)
        overlay_png_file = "file17.png"
        overlay_png_file_path = os.path.join(save_dir, overlay_png_file)
        heatmap.save_overlay_png(overlay_png_file_path, wsi_file_tif, cutoff=0.7)
        overlay_png_file = "file18.png"
        overlay_png_file_path = os.path.join(save_dir, overlay_png_file)
        heatmap.save_overlay_png(overlay_png_file_path, wsi_file_tif, cutoff=0.2)
        # compare arrays
        np.testing.assert_equal(input_array, input_array_backup)


class TestWSIHeatmapProperties(TestCase):
    """Tests for properties in WSIHeatmap class."""

    def test__pixel_data_property(self):
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        heatmap = WSIHeatmap(heatmap_data=np.transpose(input_array), background_color=None, vmin=0, vmax=1)
        np.testing.assert_equal(heatmap._pixel_data, output_array)
