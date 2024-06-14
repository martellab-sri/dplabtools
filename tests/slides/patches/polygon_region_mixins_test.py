# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for polygon based patch classes.

Tested classes:
    PolygonRegionRandomPatches
    PolygonRegionPoissonDiskPatches
    PolygonRegionGridPatches
    PolygonRegionCustomPatches
"""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from dplabtools.slides.patches import (
    PolygonRegionRandomPatches,
    PolygonRegionPoissonDiskPatches,
    PolygonRegionGridPatches,
    PolygonRegionCustomPatches,
)
from dplabtools.slides.utils import AnnotationPolygon
from testutils import make_test_path


def round_test(number, decimal_places=10):
    # purposely hard-coded decimal places, so they would get overwritten in the mock
    return round(number, 10)


class TestPolygonRegionRandomPatches(TestCase):
    """Tests for PolygonRegionRandomPatches.

    In those tests number of patches should be low to avoid generating duplicates.
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        self.polygons = [poly1, poly2]

    def test_number_of_patches(self):
        patches = PolygonRegionRandomPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            num_patches=10,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 20)
        self.assertEqual(patches.patch_info["area1"], 10)
        self.assertEqual(patches.patch_info["area2"], 10)
        #
        patches = PolygonRegionRandomPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            num_patches=[5, 7],
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 12)
        self.assertEqual(patches.patch_info["area1"], 5)
        self.assertEqual(patches.patch_info["area2"], 7)

    def test_number_of_patches_expand_params(self):
        patches = PolygonRegionRandomPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            num_patches=[10, 7],
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 17)
        #
        patches = PolygonRegionRandomPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            num_patches=[8, 3],
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 11)

    def test_save_preview_image(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_polygon_random_level0.tif")
        patches = PolygonRegionRandomPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            num_patches=10,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # Find total count of red pixels when patches do not overlap
        max_red_pixels = ((67 * 67) - (61 * 61) - 4) * 20
        # and when patches perfectly overlap
        min_red_pixels = ((67 * 67) - (61 * 61) - 4) * 2
        red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(red_pixels_count <= max_red_pixels)
        self.assertTrue(red_pixels_count >= min_red_pixels)


class TestPolygonRegionPoissonDiskPatches(TestCase):
    """Tests for PolygonRegionPoissonDiskPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        self.polygons = [poly1, poly2]

    def test_save_preview_image(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_polygon_poisson_level0.tif")
        patches = PolygonRegionPoissonDiskPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            poisson_spacing=40,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # Find total count of red pixels, at least 30 non overlapping patches should be created
        min_red_pixels = ((19 * 19) - (13 * 13) - 4) * 30
        red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(red_pixels_count >= min_red_pixels)

    def test_poisson_spacing_expand_params(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_polygon_poisson_expand.tif")
        patches = PolygonRegionPoissonDiskPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            poisson_spacing=[40, 20],
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # Find total count of red pixels, at least 15 + 40 non overlapping patches should be created
        min_red_pixels = ((19 * 19) - (13 * 13) - 4) * 55
        red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(red_pixels_count >= min_red_pixels)

    def test_poisson_spacing_expand_params_compare_counts(self):
        patches = PolygonRegionPoissonDiskPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            poisson_spacing=[40, 20],
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertLess(patches.patch_info["area1"], patches.patch_info["area2"])
        #
        patches = PolygonRegionPoissonDiskPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=0,
            poisson_spacing=[20, 40],
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertLess(patches.patch_info["area2"], patches.patch_info["area1"])


class TestPolygonRegionGridPatches(TestCase):
    """Tests for PolygonRegionGridPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array_level1 = np.ones((640, 768), dtype=int)
        self.mask_array_level2 = np.ones((160, 192), dtype=int)
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        self.polygons = [poly1, poly2]

    def test_save_preview_image(self):
        # 1. level 0 patches, mask at level 1
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_polygon_grid_level0.tif")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        # compare images
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_polygon_grid_level0.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # compare counts
        self.assertEqual(patches.patch_count, 26)
        # compare patch info:
        self.assertEqual(patches.patch_info, {"area1": 16, "area2": 10})
        # 2. level 1 patches, mask at level 1
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_polygon_grid_level1.tif")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        # compare images
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_polygon_grid_level1.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # compare counts
        self.assertEqual(patches.patch_count, 1)
        # compare patch info:
        self.assertEqual(patches.patch_info, {"area1": 1, "area2": 0})

    def test_compare_patch_counts(self):
        # 1. Mask full (level 1), stride 0.5
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 49 + 36)
        self.assertEqual(patches.patch_info["area1"], 49)
        self.assertEqual(patches.patch_info["area2"], 36)
        # 2. Mask full (level 1), stride 0.5, overlap_ratio=1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            overlap_ratio=1,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 36 + 28)
        self.assertEqual(patches.patch_info["area1"], 36)
        self.assertEqual(patches.patch_info["area2"], 28)
        # 3. Mask full (level 2), stride 0.5, overlap_ratio=1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level2,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            overlap_ratio=1,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 36 + 28)
        self.assertEqual(patches.patch_info["area1"], 36)
        self.assertEqual(patches.patch_info["area2"], 28)
        # 4. Mask full (level 1), stride 0.5, patch_size=128
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=128,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 224 + 171)
        self.assertEqual(patches.patch_info["area1"], 224)
        self.assertEqual(patches.patch_info["area2"], 171)
        # 4. Mask full (level 1), stride 0.5, reading at level 1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=128,
            level_or_mpp=1,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 6)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 6)
        # 5. Mask full (level 1), stride 0.5, reading at level 1, overlap_ratio=1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=128,
            level_or_mpp=1,
            patch_stride=0.5,
            overlap_ratio=1,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 4 + 3)
        self.assertEqual(patches.patch_info["area1"], 4)
        self.assertEqual(patches.patch_info["area2"], 3)
        # 6. Mask full (level 1), stride 1, reading at level 1, overlap_ratio=1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=325,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 6)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 6)
        # 7. Mask full (level 1), stride 1, reading at level 1, overlap_ratio=1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=325,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.5,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 10)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 10)
        # 8. Mask full (level 1), stride 1, reading at level 1, overlap_ratio=0.6
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=325,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.6,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 6)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 6)
        # 9. Mask full (level 1), stride 1, reading at level 1, overlap_ratio=0.4
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=325,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.4,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 10)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 10)
        # buffer tests
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.1,
            foreground_ratio=0.8,
            polygon_buffer=-30,
        )
        self.assertEqual(patches.patch_count, 9 + 10)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 10)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.8,
            polygon_buffer=-30,
        )
        self.assertEqual(patches.patch_count, 4 + 3)
        self.assertEqual(patches.patch_info["area1"], 4)
        self.assertEqual(patches.patch_info["area2"], 3)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
            polygon_buffer=-30,
        )
        self.assertEqual(patches.patch_count, 8 + 3)
        self.assertEqual(patches.patch_info["area1"], 8)
        self.assertEqual(patches.patch_info["area2"], 3)
        # zero patch count tests
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=512,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
            polygon_buffer=-80,
        )
        self.assertEqual(patches.patch_count, 0)
        self.assertEqual(patches.patch_info["area1"], 0)
        self.assertEqual(patches.patch_info["area2"], 0)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
            polygon_buffer=[-100, -20],
        )
        self.assertEqual(patches.patch_count, 6)
        self.assertEqual(patches.patch_info["area1"], 0)
        self.assertEqual(patches.patch_info["area2"], 6)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
            polygon_buffer=[-10, -70],
        )
        self.assertEqual(patches.patch_count, 9)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 0)
        # non full mask tests
        local_mask_array = np.zeros((640, 768), dtype=int)
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=local_mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 0)
        self.assertEqual(patches.patch_info["area1"], 0)
        self.assertEqual(patches.patch_info["area2"], 0)
        #
        local_mask_array = np.zeros((640, 768), dtype=int)
        local_mask_array[:450, ...] = 1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=local_mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 7)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 7)
        #
        local_mask_array = np.zeros((640, 768), dtype=int)
        local_mask_array[:300, ...] = 1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=local_mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 0)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 0)
        #
        local_mask_array = np.zeros((640, 768), dtype=int)
        local_mask_array[300:500, ...] = 1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=local_mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 0 + 9)
        self.assertEqual(patches.patch_info["area1"], 0)
        self.assertEqual(patches.patch_info["area2"], 9)
        #
        local_mask_array = np.zeros((640, 768), dtype=int)
        local_mask_array[..., 50:600] = 1
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=local_mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 6 + 3)
        self.assertEqual(patches.patch_info["area1"], 6)
        self.assertEqual(patches.patch_info["area2"], 3)

    def test_polygon_buffer_expand_params(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
            polygon_buffer=[-30, -10],
        )
        self.assertEqual(patches.patch_count, 8 + 6)
        self.assertEqual(patches.patch_info["area1"], 8)
        self.assertEqual(patches.patch_info["area2"], 6)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
            polygon_buffer=[-10, -30],
        )
        self.assertEqual(patches.patch_count, 9 + 3)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 3)

    def test_patch_stride_expand_params(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5],
            overlap_ratio=0.8,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 16 + 36)
        self.assertEqual(patches.patch_info["area1"], 16)
        self.assertEqual(patches.patch_info["area2"], 36)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5],
            overlap_ratio=1,
            foreground_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 9 + 28)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 28)

    def test_polygon_overlap_ratio_expand_params(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=[0.1, 0.1],
            foreground_ratio=0.5,
        )
        self.assertEqual(patches.patch_count, 16 + 15)
        self.assertEqual(patches.patch_info["area1"], 16)
        self.assertEqual(patches.patch_info["area2"], 15)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=[0.9, 0.9],
            foreground_ratio=0.5,
        )
        self.assertEqual(patches.patch_count, 15 + 10)
        self.assertEqual(patches.patch_info["area1"], 15)
        self.assertEqual(patches.patch_info["area2"], 10)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=[1, 0.01],
            foreground_ratio=0.5,
        )
        self.assertEqual(patches.patch_count, 9 + 20)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 20)

    def test_polygon_foreground_ratio_expand_params(self):
        mask_file = make_test_path("mask/board-clean-mask.npy")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.8,
            foreground_ratio=[0.8, 0.8],
        )
        self.assertEqual(patches.patch_count, 8 + 0)
        self.assertEqual(patches.patch_info["area1"], 8)
        self.assertEqual(patches.patch_info["area2"], 0)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.8,
            foreground_ratio=[0.8, 0.1],
        )
        self.assertEqual(patches.patch_count, 8 + 10)
        self.assertEqual(patches.patch_info["area1"], 8)
        self.assertEqual(patches.patch_info["area2"], 10)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.8,
            foreground_ratio=[0.1, 0.5],
        )
        self.assertEqual(patches.patch_count, 16 + 6)
        self.assertEqual(patches.patch_info["area1"], 16)
        self.assertEqual(patches.patch_info["area2"], 6)

    def test_compare_patch_data(self):
        # 1.
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=512,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [
            ((20, 20), (512, 512), "area1"),
            ((20 + 512, 20), (512, 512), "area1"),
            ((20, 20 + 512), (512, 512), "area1"),
            ((1200, 1500 + 512), (512, 512), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [
            ((5, 5), (128, 128), "area1"),
            ((5 + 128, 5), (128, 128), "area1"),
            ((5, 5 + 128), (128, 128), "area1"),
            ((300, 375 + 128), (128, 128), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # 2.
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=512,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.8,
            foreground_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [
            ((20, 20), (512, 512), "area1"),
            ((20 + 512, 20), (512, 512), "area1"),
            ((20, 20 + 512), (512, 512), "area1"),
            ((20 + 512, 20 + 512), (512, 512), "area1"),
            ((1200, 1500 + 512), (512, 512), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [
            ((5, 5), (128, 128), "area1"),
            ((5 + 128, 5), (128, 128), "area1"),
            ((5, 5 + 128), (128, 128), "area1"),
            ((5 + 128, 5 + 128), (128, 128), "area1"),
            ((300, 375 + 128), (128, 128), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # 3.
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=512,
            level_or_mpp=0,
            patch_stride=[1, 0.5],
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [
            ((20, 20), (512, 512), "area1"),
            ((20 + 512, 20), (512, 512), "area1"),
            ((20, 20 + 512), (512, 512), "area1"),
            ((1200, 1500 + 512), (512, 512), "area2"),
            ((1200, 1500 + 512 + 256), (512, 512), "area2"),
            ((1200 + 256, 1500 + 512 + 256), (512, 512), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [
            ((5, 5), (128, 128), "area1"),
            ((5 + 128, 5), (128, 128), "area1"),
            ((5, 5 + 128), (128, 128), "area1"),
            ((300, 375 + 128), (128, 128), "area2"),
            ((300, 375 + 128 + 64), (128, 128), "area2"),
            ((300 + 64, 375 + 128 + 64), (128, 128), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # 4.
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=128,
            level_or_mpp=1,
            patch_stride=[1, 0.5],
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [
            ((20, 20), (128, 128), "area1"),
            ((20 + 512, 20), (128, 128), "area1"),
            ((20, 20 + 512), (128, 128), "area1"),
            ((1200, 1500 + 512), (128, 128), "area2"),
            ((1200, 1500 + 512 + 256), (128, 128), "area2"),
            ((1200 + 256, 1500 + 512 + 256), (128, 128), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [
            ((5, 5), (128, 128), "area1"),
            ((5 + 128, 5), (128, 128), "area1"),
            ((5, 5 + 128), (128, 128), "area1"),
            ((300, 375 + 128), (128, 128), "area2"),
            ((300, 375 + 128 + 64), (128, 128), "area2"),
            ((300 + 64, 375 + 128 + 64), (128, 128), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # 5.
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=32,
            level_or_mpp=2,
            patch_stride=[1, 0.5],
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [
            ((20, 20), (32, 32), "area1"),
            ((20 + 512, 20), (32, 32), "area1"),
            ((20, 20 + 512), (32, 32), "area1"),
            ((1200, 1500 + 512), (32, 32), "area2"),
            ((1200, 1500 + 512 + 256), (32, 32), "area2"),
            ((1200 + 256, 1500 + 512 + 256), (32, 32), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [
            ((5, 5), (128, 128), "area1"),
            ((5 + 128, 5), (128, 128), "area1"),
            ((5, 5 + 128), (128, 128), "area1"),
            ((300, 375 + 128), (128, 128), "area2"),
            ((300, 375 + 128 + 64), (128, 128), "area2"),
            ((300 + 64, 375 + 128 + 64), (128, 128), "area2"),
        ]
        self.assertEqual(result_data, output_data)

    def test_compare_patch_data_info_count(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array_level1,
            patch_size=512,
            level_or_mpp=0,
            patch_stride=1,
            overlap_ratio=0.95,
            foreground_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [
            ((20, 20), (512, 512), "area1"),
            ((20 + 512, 20), (512, 512), "area1"),
            ((20, 20 + 512), (512, 512), "area1"),
            ((1200, 1500 + 512), (512, 512), "area2"),
        ]
        self.assertEqual(result_data, output_data)
        self.assertTrue("area1" in patches.patch_info)
        self.assertTrue("area2" in patches.patch_info)
        self.assertEqual(len(patches.patch_info), 2)
        self.assertEqual(patches.patch_count, 4)


class TestPolygonRegionWithHoles(TestCase):
    """Tests for holes in polygons."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(
            points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)],
            label="area1",
            holes=[[(353, 353), (353, 403), (403, 403), (403, 353)]],
        )
        poly2 = AnnotationPolygon(
            points=[(1200, 1500), (1200, 2800), (2500, 2800)],
            label="area2",
            holes=[[(1410, 2010), (1410, 2070), (1470, 2070), (1470, 2010)]],
        )
        self.polygons = [poly1, poly2]

    def test_holes1(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
        )
        # patches.save_preview_image("HOLES-test.png", patch_markers=True)
        self.assertEqual(patches.patch_count, 8 + 7)
        self.assertEqual(patches.patch_info["area1"], 8)
        self.assertEqual(patches.patch_info["area2"], 7)


class TestPolygonRegionFiveLabels(TestCase):
    """Tests for PolygonRegionGridPatches with 5 polygons with distinct labels."""

    def test_five_polygons(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        poly3 = AnnotationPolygon(
            points=[(1200, 400), (1350, 800), (1800, 1100), (2100, 800), (1400, 200)], label="area3"
        )
        poly4 = AnnotationPolygon(points=[(40, 1100), (40, 2200), (720, 2200), (720, 1100)], label="area4")
        poly5 = AnnotationPolygon(points=[(512, 2304), (0, 3072), (1024, 3072)], label="area5")
        polygons = [poly1, poly2, poly3, poly4, poly5]
        mask_array = np.ones((640, 768), dtype=int)
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=[1, 0.5, 0.5, 0.75, 1],
            overlap_ratio=1,
            foreground_ratio=0.95,
        )
        self.assertEqual(patches.patch_count, 9 + 28 + 3 + 15 + 2)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 28)
        self.assertEqual(patches.patch_info["area3"], 3)
        self.assertEqual(patches.patch_info["area4"], 15)
        self.assertEqual(patches.patch_info["area5"], 2)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=[1, 0.5, 0.5, 0.75, 1],
            overlap_ratio=0.95,
            foreground_ratio=0.95,
        )
        self.assertEqual(patches.patch_count, 9 + 28 + 8 + 15 + 2)
        self.assertEqual(patches.patch_info["area1"], 9)
        self.assertEqual(patches.patch_info["area2"], 28)
        self.assertEqual(patches.patch_info["area3"], 8)
        self.assertEqual(patches.patch_info["area4"], 15)
        self.assertEqual(patches.patch_info["area5"], 2)
        #
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=[1, 0.5, 0.5, 0.75, 1],
            overlap_ratio=0.5,
            foreground_ratio=0.95,
        )
        self.assertEqual(patches.patch_count, 16 + 53 + 23 + 18 + 6)
        self.assertEqual(patches.patch_info["area1"], 16)
        self.assertEqual(patches.patch_info["area2"], 53)
        self.assertEqual(patches.patch_info["area3"], 23)
        self.assertEqual(patches.patch_info["area4"], 18)
        self.assertEqual(patches.patch_info["area5"], 6)
        self.assertEqual(patches.patch_info, {"area1": 16, "area2": 53, "area3": 23, "area4": 18, "area5": 6})


class TestPolygonRegionRepeatedLabels(TestCase):
    """Tests for PolygonRegionGridPatches with 5 polygons with repeated labels."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)

    def test_repeated_labels(self):
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        poly3 = AnnotationPolygon(
            points=[(1200, 400), (1350, 800), (1800, 1100), (2100, 800), (1400, 200)], label="area3"
        )
        poly4 = AnnotationPolygon(points=[(40, 1100), (40, 2200), (720, 2200), (720, 1100)], label="area1")
        poly5 = AnnotationPolygon(points=[(512, 2304), (0, 3072), (1024, 3072)], label="area2")
        polygons = [poly1, poly2, poly3, poly4, poly5]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=[1, 0.5, 0.5, 0.75, 1],
            overlap_ratio=1,
            foreground_ratio=0.95,
        )
        self.assertEqual(patches.patch_count, 9 + 28 + 3 + 15 + 2)
        self.assertEqual(patches.patch_info["area1"], 9 + 15)
        self.assertEqual(patches.patch_info["area2"], 28 + 2)
        self.assertEqual(patches.patch_info["area3"], 3)
        self.assertEqual(patches.patch_info, {"area1": 9 + 15, "area2": 28 + 2, "area3": 3})

    def test_repeated_empty_labels(self):
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        poly3 = AnnotationPolygon(
            points=[(1200, 400), (1350, 800), (1800, 1100), (2100, 800), (1400, 200)], label="area3"
        )
        poly4 = AnnotationPolygon(points=[(40, 1100), (40, 2200), (720, 2200), (720, 1100)], label="")
        poly5 = AnnotationPolygon(points=[(512, 2304), (0, 3072), (1024, 3072)], label="")
        polygons = [poly1, poly2, poly3, poly4, poly5]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=[1, 0.5, 0.5, 0.75, 1],
            overlap_ratio=1,
            foreground_ratio=0.95,
        )
        self.assertEqual(patches.patch_count, 9 + 28 + 3 + 15 + 2)
        self.assertEqual(patches.patch_info[""], 9 + 15 + 2)
        self.assertEqual(patches.patch_info["area2"], 28)
        self.assertEqual(patches.patch_info["area3"], 3)
        self.assertEqual(patches.patch_info, {"": 9 + 15 + 2, "area2": 28, "area3": 3})


class TestPolygonRegionFullOverlapCheck(TestCase):
    """Tests for "_full_overlap_check".

    When using polygon region classes for patch locations, no patch should exceed polygon borders when
    overlap_ratio is 1. This condition is imposed by "_full_overlap_check = True".
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        self.patch_size = 200
        self.level_or_mpp = 1
        self.foreground_ratio = 0.01
        self.overlap_ratio = 1
        poly1 = AnnotationPolygon(points=[(120, 120), (120, 2000), (1920, 2000), (1920, 120)], label="area1")
        self.polygons = [poly1]

    def get_result_tuple(self, patch_data):
        location_x = []
        location_y = []
        # mask is at level 1, so divide by 4
        max_x = round(1920 / 4) - self.patch_size
        max_y = round(2000 / 4) - self.patch_size
        for _data in patch_data:
            x = round(_data[0][0] / 4)
            y = round(_data[0][1] / 4)
            check_x = bool(x >= 0 and x <= max_x)
            check_y = bool(y >= 0 and y <= max_y)
            location_x.append(check_x)
            location_y.append(check_y)
        return (location_x, location_y)

    def test_random_full_overlap_check(self):
        patches = PolygonRegionRandomPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            num_patches=50,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        result_x, result_y = self.get_result_tuple(patches.patch_data)
        self.assertTrue(all(result_x))
        self.assertTrue(all(result_y))

    def test_poisson_full_overlap_check(self):
        patches = PolygonRegionPoissonDiskPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            poisson_spacing=100,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        result_x, result_y = self.get_result_tuple(patches.patch_data)
        self.assertTrue(all(result_x))
        self.assertTrue(all(result_y))

    def test_grid_full_overlap_check(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            patch_stride=1,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        result_x, result_y = self.get_result_tuple(patches.patch_data)
        self.assertTrue(all(result_x))
        self.assertTrue(all(result_y))


class TestPolygonRegionAnnotationPolygonsCheck(TestCase):
    """Extra tests for polygons checks that must be run externally."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)

    def tearDown(self):
        PolygonRegionGridPatches.set_polygons_overlap_threshold(1)

    def test__run_polygons_check_overlap_threshold(self):
        PolygonRegionGridPatches.set_polygons_overlap_threshold(0.5)
        # overlapping polygons (will generate 5 print_log entries: 3 will be recorded, 2 will be skipped):
        poly1 = AnnotationPolygon(points=[(1000, 50), (1000, 200), (1500, 200), (1500, 50)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 70), (1200, 220), (1700, 220), (1700, 70)], label="area2")
        poly3 = AnnotationPolygon(points=[(1400, 90), (1400, 240), (1900, 240), (1900, 90)], label="area3")
        poly4 = AnnotationPolygon(points=[(1600, 110), (1600, 260), (2100, 260), (2100, 110)], label="area4")
        polygons = [poly1, poly2, poly3, poly4]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 0)

    def test__run_polygons_check_enabled(self):
        poly1 = AnnotationPolygon(points=[(100, 100), (100, 2000), (1500, 2000), (1500, 100)], label="area1")
        poly2 = AnnotationPolygon(points=[(200, 700), (200, 1200), (700, 1200), (700, 700)], label="area2")
        poly3 = AnnotationPolygon(points=[(400, 1300), (400, 1800), (900, 1800), (900, 1300)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            check_polygons=True,
        )
        self.assertEqual(patches.patch_count, 21)

    def test__run_polygons_check_disabled(self):
        poly1 = AnnotationPolygon(points=[(100, 100), (100, 2000), (1500, 2000), (1500, 100)], label="area1")
        poly2 = AnnotationPolygon(points=[(200, 700), (200, 1200), (700, 1200), (700, 700)], label="area2")
        poly3 = AnnotationPolygon(points=[(400, 1300), (400, 1800), (900, 1800), (900, 1300)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            check_polygons=False,
        )
        self.assertEqual(patches.patch_count, 37)


class TestPolygonRegionIncludeExclude(TestCase):
    """Test for include/exclude arguments for polygon based classes."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        poly3 = AnnotationPolygon(
            points=[(1200, 400), (1350, 800), (1800, 1100), (2100, 800), (1400, 200)], label="area3"
        )
        poly4 = AnnotationPolygon(points=[(40, 1100), (40, 2200), (720, 2200), (720, 1100)], label="area4")
        poly5 = AnnotationPolygon(points=[(512, 2304), (0, 3072), (1024, 3072)], label="area5")
        self.polygons = [poly1, poly2, poly3, poly4, poly5]

    def test_include1(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=["area1"],
        )
        self.assertTrue("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertFalse("area5" in patches.patch_info)

    def test_include2(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=["area1", "area3"],
        )
        self.assertTrue("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertTrue("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertFalse("area5" in patches.patch_info)

    def test_include3(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=["area5"],
        )
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertTrue("area5" in patches.patch_info)

    def test_include4(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=["area1", "area2", "area3", "area4", "area5"],
        )
        self.assertTrue("area1" in patches.patch_info)
        self.assertTrue("area2" in patches.patch_info)
        self.assertTrue("area3" in patches.patch_info)
        self.assertTrue("area4" in patches.patch_info)
        self.assertTrue("area5" in patches.patch_info)

    def test_include5(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=[],
        )
        self.assertTrue("area1" in patches.patch_info)
        self.assertTrue("area2" in patches.patch_info)
        self.assertTrue("area3" in patches.patch_info)
        self.assertTrue("area4" in patches.patch_info)
        self.assertTrue("area5" in patches.patch_info)

    def test_include6(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=[""],
        )
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertFalse("area5" in patches.patch_info)

    def test_include7(self):
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        poly3 = AnnotationPolygon(
            points=[(1200, 400), (1350, 800), (1800, 1100), (2100, 800), (1400, 200)], label="area3"
        )
        poly4 = AnnotationPolygon(points=[(40, 1100), (40, 2200), (720, 2200), (720, 1100)], label="area4")
        poly5 = AnnotationPolygon(points=[(512, 2304), (0, 3072), (1024, 3072)], label="")
        polygons = [poly1, poly2, poly3, poly4, poly5]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=[""],
        )
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertTrue("" in patches.patch_info)

    def test_exclude1(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            excluded_labels=["area1"],
        )
        self.assertFalse("area1" in patches.patch_info)
        self.assertTrue("area2" in patches.patch_info)
        self.assertTrue("area3" in patches.patch_info)
        self.assertTrue("area4" in patches.patch_info)
        self.assertTrue("area5" in patches.patch_info)

    def test_exclude2(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            excluded_labels=["area5"],
        )
        self.assertTrue("area1" in patches.patch_info)
        self.assertTrue("area2" in patches.patch_info)
        self.assertTrue("area3" in patches.patch_info)
        self.assertTrue("area4" in patches.patch_info)
        self.assertFalse("area5" in patches.patch_info)

    def test_exclude3(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            excluded_labels=["area1", "area2", "area3"],
        )
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertTrue("area4" in patches.patch_info)
        self.assertTrue("area5" in patches.patch_info)

    def test_exclude4(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            excluded_labels=["area1", "area2", "area3", "area4", "area5"],
        )
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertFalse("area5" in patches.patch_info)

    def test_exclude5(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            excluded_labels=[""],
        )
        self.assertTrue("area1" in patches.patch_info)
        self.assertTrue("area2" in patches.patch_info)
        self.assertTrue("area3" in patches.patch_info)
        self.assertTrue("area4" in patches.patch_info)
        self.assertTrue("area5" in patches.patch_info)

    def test_exclude6(self):
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        poly3 = AnnotationPolygon(
            points=[(1200, 400), (1350, 800), (1800, 1100), (2100, 800), (1400, 200)], label="area3"
        )
        poly4 = AnnotationPolygon(points=[(40, 1100), (40, 2200), (720, 2200), (720, 1100)], label="area4")
        poly5 = AnnotationPolygon(points=[(512, 2304), (0, 3072), (1024, 3072)], label="")
        polygons = [poly1, poly2, poly3, poly4, poly5]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            excluded_labels=[""],
        )
        self.assertTrue("area1" in patches.patch_info)
        self.assertTrue("area2" in patches.patch_info)
        self.assertTrue("area3" in patches.patch_info)
        self.assertTrue("area4" in patches.patch_info)
        self.assertFalse("" in patches.patch_info)

    def test_include_exclude1(self):
        with self.assertRaises(ValueError):
            PolygonRegionGridPatches(
                polygon_data=self.polygons,
                wsi_file=self.wsi_file_tif,
                mask_data=self.mask_array,
                patch_size=64,
                level_or_mpp=1,
                patch_stride=1,
                overlap_ratio=1,
                foreground_ratio=0.95,
                included_labels=["area1", "area2", "area3"],
                excluded_labels=["area4", "area5"],
            )

    def test_include_exclude2(self):
        with self.assertRaises(ValueError):
            PolygonRegionGridPatches(
                polygon_data=self.polygons,
                wsi_file=self.wsi_file_tif,
                mask_data=self.mask_array,
                patch_size=64,
                level_or_mpp=1,
                patch_stride=1,
                overlap_ratio=1,
                foreground_ratio=0.95,
                included_labels=[""],
                excluded_labels=[""],
            )

    def test_include_exclude3(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            overlap_ratio=1,
            foreground_ratio=0.95,
            included_labels=[],
            excluded_labels=[],
        )
        self.assertIsNotNone(patches)


class TestPolygonRegionFractionalValues(TestCase):
    """Tests for PolygonRegionRandomPatches with fractional values."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1250), (1250, 1250), (1250, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1300, 1350), (1300, 2700), (2420, 2700)], label="area2")
        self.polygons = [poly1, poly2]

    def test_fractional_values1(self):
        patches = PolygonRegionGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=1,
        )
        output_patch_data_level0 = [
            ((20, 20), (256, 256), "area1"),
            ((276, 20), (256, 256), "area1"),
            ((532, 20), (256, 256), "area1"),
            ((788, 20), (256, 256), "area1"),
            ((20, 276), (256, 256), "area1"),
            ((276, 276), (256, 256), "area1"),
            ((532, 276), (256, 256), "area1"),
            ((788, 276), (256, 256), "area1"),
            ((20, 532), (256, 256), "area1"),
            ((276, 532), (256, 256), "area1"),
            ((532, 532), (256, 256), "area1"),
            ((788, 532), (256, 256), "area1"),
            ((20, 788), (256, 256), "area1"),
            ((276, 788), (256, 256), "area1"),
            ((532, 788), (256, 256), "area1"),
            ((788, 788), (256, 256), "area1"),
            ((1300, 1862), (256, 256), "area2"),
            ((1300, 2118), (256, 256), "area2"),
            ((1556, 2118), (256, 256), "area2"),
            ((1300, 2374), (256, 256), "area2"),
            ((1556, 2374), (256, 256), "area2"),
            ((1812, 2374), (256, 256), "area2"),
        ]
        output_patch_data_mask_level = [
            ((5, 5), (64, 64), "area1"),
            ((69, 5), (64, 64), "area1"),
            ((133, 5), (64, 64), "area1"),
            ((197, 5), (64, 64), "area1"),
            ((5, 69), (64, 64), "area1"),
            ((69, 69), (64, 64), "area1"),
            ((133, 69), (64, 64), "area1"),
            ((197, 69), (64, 64), "area1"),
            ((5, 133), (64, 64), "area1"),
            ((69, 133), (64, 64), "area1"),
            ((133, 133), (64, 64), "area1"),
            ((197, 133), (64, 64), "area1"),
            ((5, 197), (64, 64), "area1"),
            ((69, 197), (64, 64), "area1"),
            ((133, 197), (64, 64), "area1"),
            ((197, 197), (64, 64), "area1"),
            ((325, 465.5), (64, 64), "area2"),
            ((325, 529.5), (64, 64), "area2"),
            ((389, 529.5), (64, 64), "area2"),
            ((325, 593.5), (64, 64), "area2"),
            ((389, 593.5), (64, 64), "area2"),
            ((453, 593.5), (64, 64), "area2"),
        ]
        self.assertEqual(patches.patch_data, output_patch_data_level0)
        self.assertEqual(patches.patch_data_mask_level, output_patch_data_mask_level)

    def test_fractional_values2(self):
        patches = PolygonRegionGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=255,
            level_or_mpp=0,
            foreground_ratio=0.8,
            overlap_ratio=1,
        )
        output_patch_data_level0 = [
            ((20, 20), (255, 255), "area1"),
            ((275, 20), (255, 255), "area1"),
            ((530, 20), (255, 255), "area1"),
            ((785, 20), (255, 255), "area1"),
            ((20, 275), (255, 255), "area1"),
            ((275, 275), (255, 255), "area1"),
            ((530, 275), (255, 255), "area1"),
            ((785, 275), (255, 255), "area1"),
            ((20, 530), (255, 255), "area1"),
            ((275, 530), (255, 255), "area1"),
            ((530, 530), (255, 255), "area1"),
            ((785, 530), (255, 255), "area1"),
            ((20, 785), (255, 255), "area1"),
            ((275, 785), (255, 255), "area1"),
            ((530, 785), (255, 255), "area1"),
            ((785, 785), (255, 255), "area1"),
            ((1300, 1860), (255, 255), "area2"),
            ((1300, 2115), (255, 255), "area2"),
            ((1555, 2115), (255, 255), "area2"),
            ((1300, 2370), (255, 255), "area2"),
            ((1555, 2370), (255, 255), "area2"),
            ((1810, 2370), (255, 255), "area2"),
        ]
        output_patch_data_mask_level = [
            ((5, 5), (63.75, 63.75), "area1"),
            ((68.75, 5), (63.75, 63.75), "area1"),
            ((132.5, 5), (63.75, 63.75), "area1"),
            ((196.25, 5), (63.75, 63.75), "area1"),
            ((5, 68.75), (63.75, 63.75), "area1"),
            ((68.75, 68.75), (63.75, 63.75), "area1"),
            ((132.5, 68.75), (63.75, 63.75), "area1"),
            ((196.25, 68.75), (63.75, 63.75), "area1"),
            ((5, 132.5), (63.75, 63.75), "area1"),
            ((68.75, 132.5), (63.75, 63.75), "area1"),
            ((132.5, 132.5), (63.75, 63.75), "area1"),
            ((196.25, 132.5), (63.75, 63.75), "area1"),
            ((5, 196.25), (63.75, 63.75), "area1"),
            ((68.75, 196.25), (63.75, 63.75), "area1"),
            ((132.5, 196.25), (63.75, 63.75), "area1"),
            ((196.25, 196.25), (63.75, 63.75), "area1"),
            ((325, 465), (63.75, 63.75), "area2"),
            ((325, 528.75), (63.75, 63.75), "area2"),
            ((388.75, 528.75), (63.75, 63.75), "area2"),
            ((325, 592.5), (63.75, 63.75), "area2"),
            ((388.75, 592.5), (63.75, 63.75), "area2"),
            ((452.5, 592.5), (63.75, 63.75), "area2"),
        ]
        self.assertEqual(patches.patch_data, output_patch_data_level0)
        self.assertEqual(patches.patch_data_mask_level, output_patch_data_mask_level)


class TestPolygonRegionNonSquareMaskPatches(TestCase):
    """Tests for PolygonRegionGridPatches with non square mask patches.

    With the arguments provided below mask level patch size will not always be square, e.g.: (6, 6), (7, 6), (6, 7).
    These tests verify if patch validity calculations work properly in such conditions and no valid patches
    are excluded.
    """

    def setUp(self):
        self.wsi_file = make_test_path("wsi/TUPAC-TE-234.svs")
        self.mask_level2 = np.ones((1738, 1957), dtype=bool)
        poly = AnnotationPolygon(points=[(0, 0), (0, 3000), (3000, 3000), (3000, 0)], label="area1")
        self.polygons = [poly]

    def test_non_square_patch_mask1(self):
        patches = PolygonRegionGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_level2,
            polygon_data=self.polygons,
            patch_size=100,
            foreground_ratio=0.9,
            overlap_ratio=0.9,
            level_or_mpp=0,
        )
        self.assertEqual(patches.patch_count, 900)

    def test_non_square_patch_mask2(self):
        patches = PolygonRegionGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_level2,
            polygon_data=self.polygons,
            patch_size=150,
            foreground_ratio=0.9,
            overlap_ratio=0.9,
            level_or_mpp=0,
        )
        self.assertEqual(patches.patch_count, 400)


class TestPolygonRegionGridRoundingPrecision(TestCase):
    """Tests for special cases when lack of float precision leads to wrong results."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.mask_file = make_test_path("mask/TUPAC-TE-234_mask.npy")
        poly1 = AnnotationPolygon(points=[(15000, 15000), (15000, 20000), (20000, 20000), (20000, 15000)], label="")
        self.polygons = [poly1]

    def test_polygon_grid_precision_correct(self):
        patches = PolygonRegionGridPatches(
            wsi_file=self.wsi_file_svs,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            patch_size=512,
            foreground_ratio=0,
            overlap_ratio=1,
            patch_stride=1,
            level_or_mpp=0,
            polygon_buffer=0,
        )
        # this count is correct, calculations require proper rounding when comparing float numbers
        self.assertEqual(patches.patch_count, 81)

    @patch("dplabtools.slides.patches.locations.base.roundfl", round_test)
    def test_polygon_grid_precision_incorrect(self):
        patches = PolygonRegionGridPatches(
            wsi_file=self.wsi_file_svs,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            patch_size=512,
            foreground_ratio=0,
            overlap_ratio=1,
            patch_stride=1,
            level_or_mpp=0,
            polygon_buffer=0,
        )
        # this count is incorrect, float number comparison is purposely wrong
        # openslide counts 64 patches, tiffslide 64 patches
        self.assertNotEqual(patches.patch_count, 81)


class TestPolygonRegionCustomPatches(TestCase):
    """Tests for PolygonRegionCustomPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1500), (1500, 1050), (1500, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800)], label="area2")
        poly3 = AnnotationPolygon(points=[(1000, 500), (1000, 900), (1400, 900), (1400, 500)], label="area3")
        self.polygons = [poly1, poly2, poly3]
        self.points = [(50, 50), (50, 600), (1100, 50), (1100, 600), (1500, 1000), (1600, 2300)]
        self.patch_size = 256
        self.level_or_mpp = 0
        self.foreground_ratio = 0.1
        self.overlap_ratio = 1

    def test_compare_patch_counts1(self):
        patches = PolygonRegionCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            check_polygons=True,
        )
        self.assertEqual(patches.patch_count, 5)

    def test_compare_patch_counts2(self):
        patches = PolygonRegionCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            check_polygons=False,
        )
        self.assertEqual(patches.patch_count, 6)

    def test_compare_patch_counts3(self):
        patches = PolygonRegionCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            check_polygons=True,
            excluded_labels=["area1"],
        )
        self.assertEqual(patches.patch_count, 2)

    def test_compare_patch_data1(self):
        patches = PolygonRegionCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            check_polygons=True,
        )
        result_data = patches.patch_data
        output_data = [
            ((50, 50), (256, 256), "area1"),
            ((50, 600), (256, 256), "area1"),
            ((1100, 50), (256, 256), "area1"),
            ((1600, 2300), (256, 256), "area2"),
            ((1100, 600), (256, 256), "area3"),
        ]
        self.assertEqual(result_data, output_data)

    def test_compare_patch_data2(self):
        patches = PolygonRegionCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            check_polygons=False,
        )
        result_data = patches.patch_data
        output_data = [
            ((50, 50), (256, 256), "area1"),
            ((50, 600), (256, 256), "area1"),
            ((1100, 50), (256, 256), "area1"),
            ((1100, 600), (256, 256), "area1"),
            ((1600, 2300), (256, 256), "area2"),
            ((1100, 600), (256, 256), "area3"),
        ]
        self.assertEqual(result_data, output_data)

    def test_compare_patch_data3(self):
        patches = PolygonRegionCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            check_polygons=True,
            excluded_labels=["area1"],
        )
        result_data = patches.patch_data
        output_data = [
            ((1600, 2300), (256, 256), "area2"),
            ((1100, 600), (256, 256), "area3"),
        ]
        self.assertEqual(result_data, output_data)

    def test_compare_patch_data4(self):
        patches = PolygonRegionCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=self.polygons,
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            check_polygons=True,
            excluded_labels=["area3"],
        )
        result_data = patches.patch_data
        output_data = [
            ((50, 50), (256, 256), "area1"),
            ((50, 600), (256, 256), "area1"),
            ((1100, 50), (256, 256), "area1"),
            ((1100, 600), (256, 256), "area1"),
            ((1600, 2300), (256, 256), "area2"),
        ]
        self.assertEqual(result_data, output_data)
