# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024-2026 Sunnybrook Research Institute - All Rights Reserved.
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


def assert_one_image(result_image_path, output_image_path):
    """Evaluate two images after conversion to arrays."""
    result_image = Image.open(result_image_path)
    result_image_array = np.asarray(result_image)
    result_image.close()
    output_image = Image.open(output_image_path)
    output_image_array = np.asarray(output_image)
    output_image.close()
    np.testing.assert_equal(result_image_array, output_image_array)


class TestWSIHeatmapInit(TestCase):
    """Tests related to object creation ."""

    def test_init(self):
        # wrong heatmap data dimensions
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        with self.assertRaises(ValueError):
            WSIHeatmap(heatmap_data=input_array, wsi_file=wsi_file_tif)

    def test__process_heatmap_data(self):
        """Test processing stages and parameters present in _process_heatmap_data."""
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        self.assertIsInstance(heatmap._heatmap_data_transposed, np.ndarray)
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            background_color=None,
            vmin=None,
            vmax=None,
        )
        self.assertIsInstance(heatmap._heatmap_data_transposed, np.ndarray)
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(
                heatmap_data=input_array,
                wsi_file=None,
                background_color=None,
                vmin=0.6,
                vmax=1,
            )
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(
                heatmap_data=input_array,
                wsi_file=None,
                background_color=None,
                vmin=0.5,
                vmax=0.7,
            )
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(
                heatmap_data=input_array,
                wsi_file=None,
                background_color=None,
                vmin=0.6,
                vmax=0.7,
            )
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            background_color=None,
            vmin=0.5,
            vmax=1,
        )
        self.assertIsInstance(heatmap._heatmap_data_transposed, np.ndarray)
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            background_color=None,
            vmin=0,
            vmax=2,
        )
        self.assertIsInstance(heatmap._heatmap_data_transposed, np.ndarray)
        #
        input_array = np.ones((100, 200))
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(
                heatmap_data=input_array,
                wsi_file=None,
                background_color=None,
                vmin=None,
                vmax=None,
            )
        #
        input_array = np.ones((100, 200))
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(
                heatmap_data=input_array,
                wsi_file=None,
                background_color=None,
                vmin=1,
                vmax=1,
            )
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        with self.assertRaises(ValueError):
            heatmap = WSIHeatmap(
                heatmap_data=input_array,
                wsi_file=None,
                background_color=None,
                vmin=2,
                vmax=2,
            )
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            background_color=None,
            vmin=None,
            vmax=None,
            cutoff=0.51,
        )
        self.assertEqual(np.nanmin(heatmap._heatmap_data_transposed), 1.0)
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            background_color=None,
            vmin=None,
            vmax=None,
            cutoff=0.5,
        )
        self.assertEqual(np.nanmin(heatmap._heatmap_data_transposed), 0.5)
        #
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0.5
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            background_color=None,
            vmin=None,
            vmax=None,
            cutoff=0.49,
        )
        self.assertEqual(np.nanmin(heatmap._heatmap_data_transposed), 0.5)


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
        # no background color, no transparency, non tissue is present
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        input_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        output_array[30:70, 40:80] = [0, 0, 0, 0]
        result_array = WSIHeatmap._get_pixel_data(input_array, None, "jet", 1, 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # background is green (#008000), no transparency, non tissue is present
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        input_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        output_array[30:70, 40:80] = [0, 128, 0, 255]
        result_array = WSIHeatmap._get_pixel_data(input_array, "green", "jet", 1, 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # background is green (#008000), transparency is 50%, non tissue is present
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        input_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array = np.full((100, 200, 4), [64, 64, 0, 255])
        output_array[0:50, 0:50] = [0, 64, 64, 255]
        output_array[30:70, 40:80] = [0, 128, 0, 255]
        result_array = WSIHeatmap._get_pixel_data(input_array, "green", "jet", 0.5, 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # no background color, transparency 50%, non tissue is present
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

    def test__get_colormapped_data(self):
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        # jet
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        result_array = WSIHeatmap._get_colormapped_data(input_array, "jet", 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # rainbow
        output_array = np.full((100, 200, 4), [255, 0, 0, 255])
        output_array[0:50, 0:50] = [127, 0, 255, 255]
        result_array = WSIHeatmap._get_colormapped_data(input_array, "rainbow", 0, 1)
        np.testing.assert_equal(result_array, output_array)
        # different vmin/vmax
        input_array = np.full((100, 200), 0.75)
        input_array[0:50, 0:50] = 0.25
        result_array = WSIHeatmap._get_colormapped_data(input_array, "jet", 0.25, 0.75)
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        np.testing.assert_equal(result_array, output_array)
        # non matching vmin/vmax ranges
        input_array = np.full((100, 200), 0.75)
        input_array[0:50, 0:50] = 0.25
        result_array = WSIHeatmap._get_colormapped_data(input_array, "jet", 0, 1)
        output_array = np.full((100, 200, 4), [255, 148, 0, 255])
        output_array[0:50, 0:50] = [0, 128, 255, 255]
        np.testing.assert_equal(result_array, output_array)
        # handling nan values
        input_array = np.full((100, 200), 0.75)
        input_array[0:50, 0:50] = 0.25
        input_array[0:20, 0:20] = np.nan
        result_array = WSIHeatmap._get_colormapped_data(input_array, "jet", 0, 1)
        output_array = np.full((100, 200, 4), [255, 148, 0, 255])
        output_array[0:50, 0:50] = [0, 128, 255, 255]
        output_array[0:20, 0:20] = [0, 0, 0, 0]
        np.testing.assert_equal(result_array, output_array)

    def test__apply_transparency(self):
        input_cmapped_array = np.full((100, 200, 4), [127, 0, 0, 255])
        input_cmapped_array[0:50, 0:50] = [0, 0, 127, 255]
        input_probs_array = np.full((100, 200), 0.3)
        # bottom right corner is NaN, transparency should not be applied
        input_probs_array[70:100, 150:200] = np.nan
        output_array = np.copy(input_cmapped_array)
        output_array[0:70, 0:150, 3] = 77
        output_array[0:70, 150:200, 3] = 77
        output_array[70:100, 0:150, 3] = 77
        WSIHeatmap._apply_transparency(input_cmapped_array, input_probs_array, 77)
        np.testing.assert_equal(input_cmapped_array, output_array)

    def test__apply_background(self):
        # alpha=1 (transparency=255)
        input_cmapped_array = np.full((100, 200, 4), [127, 0, 0, 255])
        input_cmapped_array[0:50, 0:50] = [0, 0, 127, 255]
        input_probs_array = np.full((100, 200), 0.3)
        input_probs_array[30:70, 40:80] = np.nan
        output_array = np.copy(input_cmapped_array)
        output_array[30:70, 40:80] = [255, 148, 0, 255]
        WSIHeatmap._apply_background(input_cmapped_array, input_probs_array, "#ff9400", False)
        np.testing.assert_equal(input_cmapped_array, output_array)
        # alpha=0.5 (transparency=128)
        input_cmapped_array = np.full((100, 200, 4), [127, 0, 0, 128])
        input_cmapped_array[0:50, 0:50] = [0, 0, 127, 128]
        input_probs_array = np.full((100, 200), 0.3)
        input_probs_array[30:70, 40:80] = np.nan
        output_array = np.full((100, 200, 4), [191, 74, 102, 255])
        output_array[0:50, 0:50] = [127, 74, 165, 255]
        output_array[30:70, 40:80] = [255, 148, 204, 255]
        WSIHeatmap._apply_background(input_cmapped_array, input_probs_array, "#ff94cc", True)
        np.testing.assert_equal(input_cmapped_array, output_array)

    def test__remove_transparency(self):
        input_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array = np.full((100, 200, 3), [127, 0, 0])
        result_array = WSIHeatmap._remove_transparency(input_array)
        np.testing.assert_equal(result_array, output_array)


class TestWSIHeatmapSavingFiles(TestCase):
    """Tests for saving files in WSIHeatmap class - all file types.

    Tests differ by parameters passed to WSIHeatmap.
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        self.save_dir = make_test_path("saved_data/heatmaps")

    def test_save_file1a(self):
        result_png_file = "test_heatmap1a.png"
        result_colorbar_png_file = "test_heatmap1a_colorbar.png"
        result_tif_file = "test_heatmap1a.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[0:30, 160:192] = 0.5
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color=None,
            alpha=1,
            vmin=0,
            vmax=1,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file1b(self):
        # must produce same result as file1a
        result_png_file = "test_heatmap1b.png"
        result_colorbar_png_file = "test_heatmap1b_colorbar.png"
        result_tif_file = "test_heatmap1b.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[0:30, 160:192] = 0.5
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color=None,
            alpha=1,
            vmin=None,
            vmax=None,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap1.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file2(self):
        result_png_file = "test_heatmap2.png"
        result_colorbar_png_file = "test_heatmap2_colorbar.png"
        result_tif_file = "test_heatmap2.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[0:30, 160:192] = 0.5
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color=None,
            alpha=1,
            vmin=0,
            vmax=1,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap2.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap2_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap2.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file3(self):
        result_png_file = "test_heatmap3.png"
        result_colorbar_png_file = "test_heatmap3_colorbar.png"
        result_tif_file = "test_heatmap3.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[0:30, 160:192] = 0.5
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color="yellow",
            alpha=1,
            vmin=0,
            vmax=1,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap3.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap3_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap3.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file4(self):
        result_png_file = "test_heatmap4.png"
        result_colorbar_png_file = "test_heatmap4_colorbar.png"
        result_tif_file = "test_heatmap4.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[0:30, 160:192] = 0.5
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            background_color="yellow",
            alpha=1,
            vmin=0,
            vmax=1,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap4.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap4_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap4.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file5(self):
        result_png_file = "test_heatmap5.png"
        result_colorbar_png_file = "test_heatmap5_colorbar.png"
        result_tif_file = "test_heatmap5.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[0:30, 160:192] = 0.5
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            background_color=None,
            alpha=1,
            vmin=0,
            vmax=1,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap5.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap5_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap5.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file6(self):
        result_png_file = "test_heatmap6.png"
        result_colorbar_png_file = "test_heatmap6_colorbar.png"
        result_tif_file = "test_heatmap6.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[0:30, 160:192] = 0.5
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            background_color=None,
            alpha=0.5,
            vmin=0,
            vmax=1,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap6.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap6_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap6.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file7(self):
        result_png_file = "test_heatmap7.png"
        result_colorbar_png_file = "test_heatmap7_colorbar.png"
        result_tif_file = "test_heatmap7.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0.4
        input_array[0:30, 160:192] = 0.7
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            background_color=None,
            alpha=1,
            vmin=0.2,
            vmax=1.5,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap7.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap7_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap7.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file8(self):
        result_png_file = "test_heatmap8.png"
        result_colorbar_png_file = "test_heatmap8_colorbar.png"
        result_tif_file = "test_heatmap8.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0.4
        input_array[0:30, 160:192] = 0.7
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            background_color=None,
            alpha=1,
            vmin=0.2,
            vmax=1.5,
            cutoff=0.5,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap8.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap8_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap8.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file9(self):
        result_png_file = "test_heatmap9.png"
        result_colorbar_png_file = "test_heatmap9_colorbar.png"
        result_tif_file = "test_heatmap9.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0.4
        input_array[0:30, 160:192] = 0.7
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color=None,
            alpha=0.4,
            vmin=None,
            vmax=None,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap9.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap9_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap9.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file10(self):
        # wsi is None and color background, no alpha
        result_png_file = "test_heatmap10.png"
        result_colorbar_png_file = "test_heatmap10_colorbar.png"
        result_tif_file = "test_heatmap10.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0.4
        input_array[0:30, 160:192] = 0.7
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color="pink",
            alpha=1,
            vmin=None,
            vmax=None,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap10.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap10_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap10.tif")
        assert_one_image(result_tif_path, output_image_path)

    def test_save_file11(self):
        # wsi is None and color background, alpha
        result_png_file = "test_heatmap11.png"
        result_colorbar_png_file = "test_heatmap11_colorbar.png"
        result_tif_file = "test_heatmap11.tif"
        result_png_path = os.path.join(self.save_dir, result_png_file)
        result_colorbar_png_path = os.path.join(self.save_dir, result_colorbar_png_file)
        result_tif_path = os.path.join(self.save_dir, result_tif_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0.4
        input_array[0:30, 160:192] = 0.7
        input_array[120:160, 132:192] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color="pink",
            alpha=0.5,
            vmin=None,
            vmax=None,
            cutoff=None,
        )
        heatmap.save_image(result_png_path)
        heatmap.save_colorbar_image(result_colorbar_png_path)
        heatmap.save_tif(result_tif_path)
        # eval PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap11.png")
        assert_one_image(result_png_path, output_image_path)
        # eval colorbar PNG image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap11_colorbar.png")
        assert_one_image(result_colorbar_png_path, output_image_path)
        # eval TIF image
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap11.tif")
        assert_one_image(result_tif_path, output_image_path)


class TestWSIHeatmapDrawing(TestCase):
    """Tests for drawing on heatmap image canvas."""

    def setUp(self):
        self.save_dir = make_test_path("saved_data/heatmaps")
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")

    def test_save_png_draw(self):
        def draw_function(image_draw, *args):
            rect = args[0]
            rect_polygon = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
            image_draw.polygon(rect_polygon, outline="lime", width=2)

        result_image_file = "test_heatmap_draw.png"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            alpha=0.3,
        )
        heatmap.save_image(result_image_path, draw_fn=draw_function, draw_args=((50, 70, 70, 90),))
        # eval
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap_draw.png")
        assert_one_image(result_image_path, output_image_path)

    def test_save_png_draw_overlay(self):
        def draw_function(image_draw, *args):
            rect = args[0]
            rect_polygon = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
            image_draw.polygon(rect_polygon, outline="lime", width=2)

        result_image_file = "test_heatmap_draw_overlay.png"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[50:80, 70:100] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            alpha=1,
        )
        heatmap.save_image(result_image_path, draw_fn=draw_function, draw_args=((40, 60, 60, 80),))
        # eval
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap_draw_overlay.png")
        assert_one_image(result_image_path, output_image_path)

    def test_save_colorbar_png_draw(self):
        def draw_function(current_axes, *args):
            region = args[0]
            x0, y0, x1, y1 = region
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.5, edgecolor="lime", facecolor="none")
            current_axes.add_patch(rect)

        result_image_file = "test_heatmap_colorbar_draw.png"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=None,
            overlay=False,
            background_color=None,
            alpha=0.4,
            vmin=None,
            vmax=None,
            cutoff=None,
        )
        heatmap.save_colorbar_image(result_image_path, draw_fn=draw_function, draw_args=((50, 70, 70, 90),))
        # eval
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap_colorbar_draw.png")
        assert_one_image(result_image_path, output_image_path)

    def test_save_colorbar_png_draw_overlay(self):
        def draw_function(current_axes, *args):
            region = args[0]
            x0, y0, x1, y1 = region
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.5, edgecolor="lime", facecolor="none")
            current_axes.add_patch(rect)

        result_image_file = "test_heatmap_colorbar_draw_overlay.png"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        input_array[50:80, 70:100] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            alpha=1,
        )
        heatmap.save_colorbar_image(result_image_path, draw_fn=draw_function, draw_args=((40, 60, 60, 80),))
        # eval
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap_colorbar_draw_overlay.png")
        assert_one_image(result_image_path, output_image_path)

    def test_save_tif_draw(self):
        def draw_function(pixel_data, *args):
            rect = args[0]
            pixel_data[rect[0] : rect[1], rect[2] : rect[3]] = [0, 127, 0, 255]

        result_image_file = "test_heatmap_draw.tif"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        input_array[100:120, 140:160] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color=None,
            alpha=0.5,
            vmin=None,
            vmax=None,
            cutoff=None,
        )
        heatmap.save_tif(result_image_path, draw_fn=draw_function, draw_args=((140, 160, 40, 60),))
        # eval
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap_draw.tif")
        assert_one_image(result_image_path, output_image_path)

    def test_save_tif_draw_overlay(self):
        def draw_function(pixel_data, *args):
            rect = args[0]
            pixel_data[rect[0] : rect[1], rect[2] : rect[3]] = [0, 127, 0]

        result_image_file = "test_heatmap_draw_overlay.tif"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        input_array[50:70, 150:170] = np.nan
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            alpha=1,
        )
        heatmap.save_tif(result_image_path, draw_fn=draw_function, draw_args=((140, 160, 40, 60),))
        # eval
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap_draw_overlay.tif")
        assert_one_image(result_image_path, output_image_path)


class TestWSIHeatmapSaveImage(TestCase):
    """Tests specific to saving images (not colorbar), not shared with other saving methods."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        self.save_dir = make_test_path("saved_data/heatmaps")

    def test_save_image_jpeg_with_overlay(self):
        """Test saving as JPEG and also passing image writer specific parameter (quality)."""
        result_image_file = "test_heatmap1.jpeg"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=True,
            vmin=0,
            vmax=1,
        )
        heatmap.save_image(result_image_path, quality=1)
        # check if file was created
        self.assertTrue(os.path.exists(result_image_path))

    def test_save_image_jpeg_no_overlay(self):
        """Test saving as JPEG and also passing image writer specific parameter (quality)."""
        result_image_file = "test_heatmap1.jpeg"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            overlay=False,
            background_color="orange",
            vmin=0,
            vmax=1,
        )
        heatmap.save_image(result_image_path, quality=1)
        # check if file was created
        self.assertTrue(os.path.exists(result_image_path))


class TestWSIHeatmapSaveColorbarImage(TestCase):
    """Tests specific to saving colorbar images, not shared with other saving methods."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        self.save_dir = make_test_path("saved_data/heatmaps")

    def test_save_colorbar_image_interpolation(self):
        result_image_file1 = "test_heatmap_interpolation1.png"
        result_image_path1 = os.path.join(self.save_dir, result_image_file1)
        result_image_file2 = "test_heatmap_interpolation2.png"
        result_image_path2 = os.path.join(self.save_dir, result_image_file2)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        # save
        heatmap.save_colorbar_image(result_image_path1)
        heatmap.save_colorbar_image(result_image_path2, interpolation="bicubic")
        # compare
        heatmap_image1 = Image.open(result_image_path1)
        heatmap_image2 = Image.open(result_image_path2)
        heatmap_array1 = np.asarray(heatmap_image1)
        heatmap_array2 = np.asarray(heatmap_image2)
        heatmap_image1.close()
        heatmap_image2.close()
        self.assertFalse(np.array_equal(heatmap_array1, heatmap_array2))

    def test_save_colorbar_image_dpi(self):
        result_image_file1 = "test_heatmap_dpi1.png"
        result_image_path1 = os.path.join(self.save_dir, result_image_file1)
        result_image_file2 = "test_heatmap_dpi2.png"
        result_image_path2 = os.path.join(self.save_dir, result_image_file2)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        # save
        heatmap.save_colorbar_image(result_image_path1, dpi=100)
        heatmap.save_colorbar_image(result_image_path2, dpi=200)
        # compare
        heatmap_image1 = Image.open(result_image_path1)
        heatmap_image2 = Image.open(result_image_path2)
        heatmap_size1 = (heatmap_image1.width, heatmap_image1.height)
        heatmap_size2 = (heatmap_image2.width, heatmap_image2.height)
        heatmap_image1.close()
        heatmap_image2.close()
        self.assertGreater(heatmap_size2[0], heatmap_size1[0])
        self.assertGreater(heatmap_size2[1], heatmap_size1[1])

    def test_save_colorbar_image_jpeg(self):
        """Test saving as JPEG and also passing image writer specific parameter (pad_inches)."""
        result_image_file = "test_heatmap_colorbar.jpeg"
        result_image_path = os.path.join(self.save_dir, result_image_file)
        input_array = np.ones((160, 192))
        input_array[0:50, 0:50] = 0
        heatmap = WSIHeatmap(
            heatmap_data=input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        heatmap.save_colorbar_image(result_image_path, pad_inches=2.5)
        # check if file was created
        self.assertTrue(os.path.exists(result_image_path))


class TestWSIHeatmapSavingTif(TestCase):
    """Tests specific to saving TIF, not shared with other saving methods."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        self.save_dir = make_test_path("saved_data/heatmaps")
        input_array = np.ones((160, 192))
        input_array[0:64, 0:96] = 0
        input_array[100:120, 140:160] = np.nan
        self.input_array = input_array

    def test_save_mpp(self):
        result_image_name = "test_heatmap_mpp.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path)
        output_image_path = make_test_path("ref_data/slides/processing/heatmaps/ref_heatmap_mpp.tif")
        # compare
        assert_one_image(result_image_path, output_image_path)
        wsi_slide = GenericSlide(wsi_file=self.wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_path)
        output_slide = GenericSlide(wsi_file=output_image_path)
        self.assertEqual(result_slide.mpp_data, output_slide.mpp_data)
        self.assertEqual(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 16, wsi_slide.mpp_data[1] * 16))

    def test_save_no_compression1(self):
        # background is undefined -> no compression used
        result_image_name = "test_heatmap_nocompression1.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path)
        # check image file size
        self.assertTrue(os.path.getsize(result_image_path) > 100000)

    def test_save_no_compression2(self):
        # force no compression when background is defined and transparency not present
        result_image_name = "test_heatmap_nocompression2.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color="yellow",
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path, allow_compression=False)
        # check image file size
        self.assertTrue(os.path.getsize(result_image_path) > 100000)

    def test_save_no_compression3(self):
        # background is undefined + transparency -> no compression used
        result_image_name = "test_heatmap_nocompression3.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path)
        # check image file size
        self.assertTrue(os.path.getsize(result_image_path) > 100000)

    def test_save_no_compression4(self):
        # background is defined + transparency -> no compression used
        result_image_name = "test_heatmap_nocompression4.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color="orange",
            alpha=0.5,
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path)
        # check image file size
        self.assertTrue(os.path.getsize(result_image_path) > 100000)

    def test_save_with_compression(self):
        # use compression, results are nondeterministic between different envs
        result_image_name = "test_heatmap_compression.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color="lime",
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path)
        # check image file size
        self.assertTrue(os.path.getsize(result_image_path) < 20000)

    def test_save_downsample_factor1(self):
        # save df=1
        result_image_name = "test_heatmap_df1.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path, downsample_factor=1)
        # compare
        result_slide = GenericSlide(wsi_file=result_image_path)
        self.assertEqual(result_slide.mpp_data, (0.25, 0.25))

    def test_save_downsample_factor2(self):
        # save df=4
        result_image_name = "test_heatmap_df2.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path, downsample_factor=4)
        # compare
        result_slide = GenericSlide(wsi_file=result_image_path)
        self.assertEqual(result_slide.mpp_data, (0.25 * 4, 0.25 * 4))

    def test_save_downsample_factor3(self):
        # save df=16
        result_image_name = "test_heatmap_df3.tif"
        result_image_path = os.path.join(self.save_dir, result_image_name)
        heatmap = WSIHeatmap(
            heatmap_data=self.input_array,
            wsi_file=self.wsi_file_tif,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        heatmap.save_tif(result_image_path, downsample_factor=16)
        # compare
        result_slide = GenericSlide(wsi_file=result_image_path)
        self.assertEqual(result_slide.mpp_data, (0.25 * 16, 0.25 * 16))


class TestWSIHeatmapProperties(TestCase):
    """Tests for properties in WSIHeatmap class."""

    def test__pixel_data_property(self):
        input_array = np.ones((100, 200))
        input_array[0:50, 0:50] = 0
        output_array = np.full((100, 200, 4), [127, 0, 0, 255])
        output_array[0:50, 0:50] = [0, 0, 127, 255]
        heatmap = WSIHeatmap(
            heatmap_data=np.transpose(input_array),
            wsi_file=None,
            background_color=None,
            vmin=0,
            vmax=1,
        )
        np.testing.assert_equal(heatmap._pixel_data, output_array)
