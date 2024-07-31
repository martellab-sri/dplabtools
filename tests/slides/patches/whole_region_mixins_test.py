# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for whole image based patch classes.

Tested classes:
    WholeImageRandomPatches
    WholeImagePoissonDiskPatches
    WholeImageGridPatches
    WholeImageCustomPatches
    WholeImageInvertedRandomPatches
    WholeImageInvertedPoissonDiskPatches
    WholeImageInvertedGridPatches
    WholeImageInvertedCustomPatches
"""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from dplabtools.slides.patches import (
    WholeImageRandomPatches,
    WholeImagePoissonDiskPatches,
    WholeImageGridPatches,
    WholeImageCustomPatches,
    WholeImageInvertedRandomPatches,
    WholeImageInvertedPoissonDiskPatches,
    WholeImageInvertedGridPatches,
    WholeImageInvertedCustomPatches,
)
from dplabtools.slides.utils import AnnotationPolygon
from dplabtools.slides.patches.locations.regions import WholeImageInvertedPatches

from testutils import make_test_path


def round_test(number, decimal_places=20):
    # purposely hard-coded decimal places, so they would get overwritten in the mock
    return round(number, 20)


class TestWholeImageRandomPatches(TestCase):
    """Tests for WholeImageRandomPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)

    def test_number_of_patches(self):
        patches = WholeImageRandomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            num_patches=13,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 13)

    def test_break_factor(self):
        mask_array = np.zeros((640, 768), dtype=int)
        mask_array[:30, :30] = 1
        with self.assertRaises(ValueError):
            WholeImageRandomPatches(
                wsi_file=self.wsi_file_tif,
                mask_data=mask_array,
                patch_size=256,
                level_or_mpp=0,
                num_patches=15,
                foreground_ratio=0.8,
                overlap_ratio=0.8,
            )

    def test_save_preview_image(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_random_level0.tif")
        patches = WholeImageRandomPatches(
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
        max_red_pixels = ((67 * 67) - (61 * 61) - 4) * 10
        # and when patches perfectly overlap
        min_red_pixels = (67 * 67) - (61 * 61) - 4
        red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(min_red_pixels <= red_pixels_count <= max_red_pixels)


class TestWholeImagePoissonDiskPatches(TestCase):
    """Tests for WholeImagePoissonDiskPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)

    def test_save_preview_image(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_poisson_level0.tif")
        patches = WholeImagePoissonDiskPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=256,
            level_or_mpp=0,
            poisson_spacing=100,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # Find total count of red pixels, at least 20 non overlapping patches should be created
        min_red_pixels = ((67 * 67) - (61 * 61) - 4) * 20
        red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(red_pixels_count >= min_red_pixels)


class TestWholeImageGridPatches(TestCase):
    """Tests for WholeImageGridPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")

    def test_max_stride(self):
        mask_array = np.ones((640, 768), dtype=int)
        with self.assertRaises(ValueError):
            WholeImageGridPatches(
                wsi_file=self.wsi_file_tif,
                mask_data=mask_array,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=10,
                foreground_ratio=0.8,
                overlap_ratio=0.8,
            )

    def test_save_preview_image(self):
        # 1. level 0 patches, mask at level 1
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_grid_level0.tif")
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        # compare images
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_whole_grid_level0.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # compare counts
        self.assertEqual(patches.patch_count, 120)
        # 2. level 1 patches, mask at level 1
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_grid_level1.tif")
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        # compare images
        output_image = Image.open(make_test_path("ref_data/slides/patches/preview/ref_preview_whole_grid_level1.tif"))
        output_image_array = np.asarray(output_image)
        output_image.close()
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        np.testing.assert_equal(result_image_array, output_image_array)
        # compare counts
        self.assertEqual(patches.patch_count, 6)

    def test_compare_patch_counts(self):
        # 1. Mask full (level 1), stride 0.5
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 19 * 23)
        # 2. Mask full (level 1), stride 0.5, overlap_ratio= 0.5, foreground_ratio = 0.1
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            overlap_ratio=0.1,
            foreground_ratio=0.1,
        )
        self.assertEqual(patches.patch_count, 20 * 24)
        # 3. Mask full (level 1), stride 1, patch_size = 128
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 20 * 24)
        # 4. Mask full (level 1), stride 0.5, patch_size = 128
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 39 * 47)
        # 5. Mask full (level 2), stride 1, patch_size = 256
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 10 * 12)
        # 6. Mask full (level 2), stride 1, patch_size = 128
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 20 * 24)
        # 7. Mask full (level 2), stride 1, patch_size = 64
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 40 * 48)
        # 8. Mask full (level 2), stride 1, patch_size = 64, reading at level 1
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 10 * 12)
        # 9. Mask full (level 2), stride 0.5, patch_size = 64, reading at level 1
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 19 * 23)
        # 10. Mask full (level 2), stride 0.25, patch_size = 64, reading at level 1
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=1,
            patch_stride=0.25,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 37 * 45)
        # 11. Mask full (level 2), stride 1, patch_size = 64, reading at level 2
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=2,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 6)
        # 12. Mask full (level 1), stride 1, patch_size = 128, reading at level 2
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=2,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 1)
        # 13. Mask full (level 1), stride 0.5, patch_size = 128, reading at level 2
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=2,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.assertEqual(patches.patch_count, 2)

    def test_compare_patch_data(self):
        # 1.
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=2,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [((0, 0), (128, 128), ""), ((0, 1024), (128, 128), "")]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [((0, 0), (512, 512), ""), ((0, 256), (512, 512), "")]
        self.assertEqual(result_data, output_data)
        # 2.
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=2,
            patch_stride=0.5,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [((0, 0), (128, 128), ""), ((0, 1024), (128, 128), "")]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [((0, 0), (128, 128), ""), ((0, 64), (128, 128), "")]
        self.assertEqual(result_data, output_data)
        # 3.
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=2,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [((0, 0), (128, 128), "")]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [((0, 0), (128, 128), "")]
        self.assertEqual(result_data, output_data)
        # 4.
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=128,
            level_or_mpp=2,
            patch_stride=0.5,
            weak_label="label1",
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [((0, 0), (128, 128), "label1"), ((0, 1024), (128, 128), "label1")]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [((0, 0), (128, 128), "label1"), ((0, 64), (128, 128), "label1")]
        self.assertEqual(result_data, output_data)
        # 5.
        mask_array = np.ones((160, 192), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_array,
            patch_size=64,
            level_or_mpp=2,
            patch_stride=1,
            weak_label="label2",
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        result_data = patches.patch_data
        output_data = [
            ((0, 0), (64, 64), "label2"),
            ((1024, 0), (64, 64), "label2"),
            ((0, 1024), (64, 64), "label2"),
            ((1024, 1024), (64, 64), "label2"),
            ((0, 2048), (64, 64), "label2"),
            ((1024, 2048), (64, 64), "label2"),
        ]
        self.assertEqual(result_data, output_data)
        # mask level
        result_data = patches.patch_data_mask_level
        output_data = [
            ((0, 0), (64, 64), "label2"),
            ((64, 0), (64, 64), "label2"),
            ((0, 64), (64, 64), "label2"),
            ((64, 64), (64, 64), "label2"),
            ((0, 128), (64, 64), "label2"),
            ((64, 128), (64, 64), "label2"),
        ]
        self.assertEqual(result_data, output_data)


class TestWholeImageGridPatchesLocations(TestCase):
    """Tests if grid patch locations are computed properly for larger images."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.mask = np.ones((1738, 1957))  # level2 mask

    def get_patches_level0(self, patch_size, patch_stride):
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_svs,
            mask_data=self.mask,
            foreground_ratio=1,
            overlap_ratio=1,
            patch_size=patch_size,
            level_or_mpp=0,
            patch_stride=patch_stride,
        )
        return patches

    def get_patches_level1(self, patch_size, patch_stride):
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_svs,
            mask_data=self.mask,
            foreground_ratio=1,
            overlap_ratio=1,
            patch_size=patch_size,
            level_or_mpp=1,
            patch_stride=patch_stride,
        )
        return patches

    def assess_patches_level0(self, patches, patch_size):
        # must be pixel perfect
        for pd in patches.patch_data:
            location_x, location_y = pd[0]
            self.assertEqual(location_x % patch_size, 0)
            self.assertEqual(location_y % patch_size, 0)

    def assess_patches_level1(self, patches, patch_size):
        # one pixel off allowed
        for pd in patches.patch_data:
            location_x, location_y = pd[0]
            div_rx = location_x % patch_size
            div_ry = location_y % patch_size
            self.assertIn(div_rx, (0, 1))
            self.assertIn(div_ry, (0, 1))

    def test_patch_locations1(self):
        patch_size = 256
        patch_stride = 1
        patches = self.get_patches_level0(patch_size, patch_stride)
        self.assess_patches_level0(patches, patch_size)
        self.assertEqual(patches.patch_count, 108 * 122)

    def test_patch_locations2(self):
        patch_size = 333
        patch_stride = 1
        patches = self.get_patches_level0(patch_size, patch_stride)
        self.assess_patches_level0(patches, patch_size)
        self.assertEqual(patches.patch_count, 83 * 94)

    def test_patch_locations3(self):
        patch_size = 65
        patch_stride = 1
        patches = self.get_patches_level0(patch_size, patch_stride)
        self.assess_patches_level0(patches, patch_size)
        self.assertEqual(patches.patch_count, 427 * 481)

    def test_patch_locations4(self):
        patch_size = 256
        patch_stride = 0.5
        patches = self.get_patches_level0(patch_size, patch_stride)
        self.assess_patches_level0(patches, 128)
        self.assertEqual(patches.patch_count, 52488)

    def test_patch_locations5(self):
        patch_size = 256
        patch_stride = 0.25
        patches = self.get_patches_level0(patch_size, patch_stride)
        self.assess_patches_level0(patches, 64)
        self.assertEqual(patches.patch_count, 209466)

    def test_patch_locations6(self):
        patch_size = 707
        patch_stride = 1
        patches = self.get_patches_level0(patch_size, patch_stride)
        self.assess_patches_level0(patches, patch_size)
        self.assertEqual(patches.patch_count, 39 * 44)

    def test_patch_locations7(self):
        patch_size = 1504
        patch_stride = 0.25
        patches = self.get_patches_level0(patch_size, patch_stride)
        self.assess_patches_level0(patches, 376)
        self.assertEqual(patches.patch_count, 5600)

    def test_patch_locations8(self):
        # For this particular image (TUPAC-TE-234.svs) using mask at level2 and computing patches at level1
        # will never produce pixel perfect results. This is because downsampling factor ratio between adjacent layers
        # is not consistant:
        # - level[0].downsample: '1'
        # - level[1].downsample: '4.0001438021282709'
        # - level[2].downsample: '16.003898842372223'
        # however: level[2].downsample/level[1].downsample = 4.000830879 [not 4.0001438...]
        # Validating results must allow at least one pixel shift in level0 location coordinates.
        patch_size = 1504
        patch_stride = 1
        patches = self.get_patches_level1(patch_size, patch_stride)
        self.assess_patches_level1(patches, 1504 * 4)
        self.assertEqual(patches.patch_count, 20)

    def test_patch_locations9(self):
        patch_size = 711
        patch_stride = 1
        patches = self.get_patches_level1(patch_size, patch_stride)
        self.assess_patches_level1(patches, 711 * 4)
        self.assertEqual(patches.patch_count, 99)

    def test_patch_locations10(self):
        patch_size = 711
        patch_stride = 0.5
        patches = self.get_patches_level1(patch_size, patch_stride)
        self.assess_patches_level1(patches, 711 * 2)
        self.assertEqual(patches.patch_count, 378)


class TestWholeImageWeakLabel(TestCase):
    """Tests for weak label argument present in whole image classes."""

    def test_weak_label(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            weak_label="abc",
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        result_label = patches.patch_labels
        output_label = ["abc"]
        self.assertEqual(result_label, output_label)


class TestWholeImageFullOverlapCheck(TestCase):
    """Tests for "_full_overlap_check".

    When using whole image classes for patch locations, no patch should exceed image borders when
    overlap_ratio is 1. This condition is imposed by "_is_overlap_check_required" even when
    "_full_overlap_check" is False.
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        self.patch_size = 200
        self.level_or_mpp = 1
        self.foreground_ratio = 0.01
        self.overlap_ratio = 1

    def get_result_tuple(self, patch_data):
        location_x = []
        location_y = []
        max_x = 640 - self.patch_size
        max_y = 768 - self.patch_size
        for _data in patch_data:
            # mask is at level 1, so divide by 4
            x = round(_data[0][0] / 4)
            y = round(_data[0][1] / 4)
            check_x = bool(max_x >= x >= 0)
            check_y = bool(max_y >= y >= 0)
            location_x.append(check_x)
            location_y.append(check_y)
        return (location_x, location_y)

    def test_random_full_overlap_check(self):
        patches = WholeImageRandomPatches(
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
        patches = WholeImagePoissonDiskPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            poisson_spacing=30,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        result_x, result_y = self.get_result_tuple(patches.patch_data)
        self.assertTrue(all(result_x))
        self.assertTrue(all(result_y))

    def test_grid_full_overlap_check(self):
        patches = WholeImageGridPatches(
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


class TestWholeImageCustomPatches(TestCase):
    """Tests for WholeImageCustomPatches.

    Tests for all custom classes were added way after non-custom classes, so they do not need as extensive testing
    and also they use a slightly different approach.
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        self.points = [(30, 30), (30, 600), (1100, 30), (1100, 600), (1500, 1000)]
        self.patch_size = 256
        self.level_or_mpp = 0
        self.overlap_ratio = 1

    def test_save_preview_image1(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_custom_level0a.tif")
        patches = WholeImageCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            points=self.points,
            foreground_ratio=0.5,
            overlap_ratio=self.overlap_ratio,
        )
        patches.save_preview_image(result_test_image_tif, level_or_minsize=1, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # Find total count of red pixels when patches do not overlap
        max_red_pixels = ((67 * 67) - (61 * 61) - 4) * 5
        # and when patches perfectly overlap
        min_red_pixels = (67 * 67) - (61 * 61) - 4
        red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(min_red_pixels <= red_pixels_count <= max_red_pixels)

    def test_save_preview_image2(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_custom_level0b.tif")
        patches = WholeImageCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            points=self.points,
            foreground_ratio=0.7,
            overlap_ratio=self.overlap_ratio,
        )
        patches.save_preview_image(result_test_image_tif, level_or_minsize=1, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # Find total count of red pixels when patches do not overlap
        max_red_pixels = ((67 * 67) - (61 * 61) - 4) * 2
        # and when patches perfectly overlap
        min_red_pixels = (67 * 67) - (61 * 61) - 4
        red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(min_red_pixels <= red_pixels_count <= max_red_pixels)

    def test_compare_patch_counts1(self):
        patches = WholeImageCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            points=self.points,
            foreground_ratio=0.5,
            overlap_ratio=self.overlap_ratio,
        )
        self.assertEqual(patches.patch_count, 5)

    def test_compare_patch_counts2(self):
        patches = WholeImageCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            points=self.points,
            foreground_ratio=0.7,
            overlap_ratio=self.overlap_ratio,
        )
        self.assertEqual(patches.patch_count, 2)

    def test_compare_patch_data1(self):
        patches = WholeImageCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            points=self.points,
            foreground_ratio=0.5,
            overlap_ratio=self.overlap_ratio,
        )
        result_data = patches.patch_data
        output_data = [
            ((30, 30), (256, 256), ""),
            ((30, 600), (256, 256), ""),
            ((1100, 30), (256, 256), ""),
            ((1100, 600), (256, 256), ""),
            ((1500, 1000), (256, 256), ""),
        ]
        self.assertEqual(result_data, output_data)

    def test_compare_patch_data2(self):
        patches = WholeImageCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            points=self.points,
            foreground_ratio=0.7,
            overlap_ratio=self.overlap_ratio,
        )
        result_data = patches.patch_data
        output_data = [
            ((30, 30), (256, 256), ""),
            ((1500, 1000), (256, 256), ""),
        ]
        self.assertEqual(result_data, output_data)


class TestWholeImageInvertedRandomPatches(TestCase):
    """Tests for WholeImageInvertedRandomPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        self.patch_size = 256
        self.level_or_mpp = 0
        self.foreground_ratio = 0.8
        self.overlap_ratio = 1
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800), (2500, 1500)], label="area2")
        self.polygons = [poly1, poly2]

    def test_save_preview_image(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_random_inverted_level0.tif")
        patches = WholeImageInvertedRandomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=self.patch_size,
            num_patches=50,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # 1. Test/count total red pixels
        # Find total count of red pixels when patches do not overlap
        max_red_pixels = ((67 * 67) - (61 * 61) - 4) * 50
        # and when patches perfectly overlap
        min_red_pixels = (67 * 67) - (61 * 61) - 4
        total_image_red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(min_red_pixels <= total_image_red_pixels_count <= max_red_pixels)
        # 2. Test/count red pixels in polygon areas
        polygon1_array = result_image_array[7:254, 7:254, :]
        polygon1_red_pixels_count = np.sum(np.all(polygon1_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon1_red_pixels_count, 0)
        polygon2_array = result_image_array[377:699, 302:622, :]
        polygon2_red_pixels_count = np.sum(np.all(polygon2_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon2_red_pixels_count, 0)
        # 3. Test total patches
        self.assertEqual(patches.patch_count, 50)


class TestWholeImageInvertedPoissonDiskPatches(TestCase):
    """Tests for WholeImageInvertedPoissonDiskPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        self.patch_size = 256
        self.level_or_mpp = 0
        self.foreground_ratio = 0.8
        self.overlap_ratio = 1
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800), (2500, 1500)], label="area2")
        self.polygons = [poly1, poly2]

    def test_save_preview_image(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_poisson_inverted_level0.tif")
        patches = WholeImageInvertedPoissonDiskPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=self.patch_size,
            poisson_spacing=90,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        patches.save_preview_image(result_test_image_tif, thickness=2)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # 1. Test/count total red pixels
        # Find total count of red pixels, at least 10 non overlapping patches should be created
        min_red_pixels = ((67 * 67) - (61 * 61) - 4) * 10
        total_image_red_pixels_count = np.sum(np.all(result_image_array == [255, 0, 0], axis=2))
        self.assertTrue(total_image_red_pixels_count >= min_red_pixels)
        # 2. Test/count red pixels in polygon areas
        polygon1_array = result_image_array[7:254, 7:254, :]
        polygon1_red_pixels_count = np.sum(np.all(polygon1_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon1_red_pixels_count, 0)
        polygon2_array = result_image_array[377:699, 302:622, :]
        polygon2_red_pixels_count = np.sum(np.all(polygon2_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon2_red_pixels_count, 0)


class TestWholeImageInvertedGridPatches(TestCase):
    """Tests for WholeImageInvertedGridPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        self.patch_size = 256
        self.level_or_mpp = 0
        self.foreground_ratio = 0.8
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800), (2500, 1500)], label="area2")
        self.polygons = [poly1, poly2]

    def test_save_preview_image1(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_grid_inverted_level0a.tif")
        overlap_ratio = 1
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=self.patch_size,
            patch_stride=1,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=overlap_ratio,
        )
        patches.save_preview_image(result_test_image_tif)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # 1 Test/count red pixels in polygon areas
        polygon1_array = result_image_array[7:254, 7:254, :]
        polygon1_red_pixels_count = np.sum(np.all(polygon1_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon1_red_pixels_count, 0)
        polygon2_array = result_image_array[377:699, 302:622, :]
        polygon2_red_pixels_count = np.sum(np.all(polygon2_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon2_red_pixels_count, 0)
        # 3. Test total patches
        self.assertEqual(patches.patch_count, 68)

    def test_save_preview_image2(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_grid_inverted_level0b.tif")
        overlap_ratio = 0.5
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=self.patch_size,
            patch_stride=1,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=overlap_ratio,
        )
        patches.save_preview_image(result_test_image_tif)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # 1 Test/count red pixels in polygon areas (some should be present here)
        polygon1_array = result_image_array[7:254, 7:254, :]
        polygon1_red_pixels_count = np.sum(np.all(polygon1_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon1_red_pixels_count, 0)
        polygon2_array = result_image_array[377:699, 302:622, :]
        polygon2_red_pixels_count = np.sum(np.all(polygon2_array == [255, 0, 0], axis=2))
        self.assertGreater(polygon2_red_pixels_count, 0)
        # 3. Test total patches
        self.assertEqual(patches.patch_count, 79)

    def test_save_preview_image3(self):
        result_test_image_tif = make_test_path("saved_data/preview/test_preview_whole_grid_inverted_level0c.tif")
        overlap_ratio = 1
        self.level_or_mpp = 1
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=self.patch_size,
            patch_stride=1,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=overlap_ratio,
        )
        patches.save_preview_image(result_test_image_tif)
        result_image = Image.open(result_test_image_tif)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # 1 Test/count red pixels in polygon areas
        polygon1_array = result_image_array[7:254, 7:254, :]
        polygon1_red_pixels_count = np.sum(np.all(polygon1_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon1_red_pixels_count, 0)
        polygon2_array = result_image_array[377:699, 302:622, :]
        polygon2_red_pixels_count = np.sum(np.all(polygon2_array == [255, 0, 0], axis=2))
        self.assertEqual(polygon2_red_pixels_count, 0)
        # 3. Test total patches
        self.assertEqual(patches.patch_count, 3)


class TestWholeImageInvertedWeakLabel(TestCase):
    """Tests for weak label argument present whole image inverted classes."""

    def test_weak_label(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800), (2500, 1500)], label="area2")
        polygons = [poly1, poly2]
        patches = WholeImageInvertedGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            polygon_data=polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            weak_label="abc",
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        result_label = patches.patch_labels
        output_label = ["abc"]
        self.assertEqual(result_label, output_label)


class TestWholeImageInvertedMergedPolygons(TestCase):
    """Tests when polygons overlap or contain other polygons."""

    def test_merged_polygons_grid1(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(900, 500), (900, 2200), (2200, 2200), (2200, 500)], label="area1")
        poly2 = AnnotationPolygon(points=[(850, 1500), (850, 2800), (2150, 2800), (2150, 1500)], label="area2")
        poly3 = AnnotationPolygon(points=[(950, 550), (950, 1200), (1400, 1200), (1400, 550)], label="area3")
        poly4 = AnnotationPolygon(points=[(1700, 550), (1700, 1200), (2150, 1200), (2150, 550)], label="area4")
        poly5 = AnnotationPolygon(points=[(1730, 590), (1730, 1100), (2100, 1100), (2100, 590)], label="area5")
        poly6 = AnnotationPolygon(points=[(150, 550), (150, 1200), (650, 1200), (650, 550)], label="area6")
        poly7 = AnnotationPolygon(points=[(180, 590), (180, 1100), (620, 1100), (620, 590)], label="area7")
        polygons = [poly1, poly2, poly3, poly4, poly5, poly6, poly7]
        patches = WholeImageInvertedGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            polygon_data=polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
        )
        self.assertEqual(patches.patch_count, 51)

    def test_merged_polygons_grid2(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(950, 550), (950, 1200), (1400, 1200), (1400, 550)], label="area1")
        poly2 = AnnotationPolygon(points=[(1700, 550), (1700, 1200), (2150, 1200), (2150, 550)], label="area2")
        poly3 = AnnotationPolygon(points=[(150, 550), (150, 1200), (650, 1200), (650, 550)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = WholeImageInvertedGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            polygon_data=polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
        )
        self.assertEqual(patches.patch_count, 93)

    def test_merged_polygons_grid3(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(1700, 550), (1700, 1200), (2150, 1200), (2150, 550)], label="area1")
        polygons = [poly1]
        patches = WholeImageInvertedGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            polygon_data=polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
        )
        self.assertEqual(patches.patch_count, 111)

    def test_merged_polygons_grid4(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(150, 550), (150, 1200), (650, 1200), (650, 550)], label="area1")
        poly2 = AnnotationPolygon(points=[(180, 590), (180, 1100), (620, 1100), (620, 590)], label="area2")
        polygons = [poly1, poly2]
        patches = WholeImageInvertedGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_array,
            polygon_data=polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
        )
        self.assertEqual(patches.patch_count, 111)


class TestWholeImageInvertedIncludeExclude(TestCase):
    """Test for include/exclude arguments for inverted classes."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        poly1 = AnnotationPolygon(points=[(950, 550), (950, 1200), (1400, 1200), (1400, 550)], label="area1")
        poly2 = AnnotationPolygon(points=[(1700, 550), (1700, 1200), (2150, 1200), (2150, 550)], label="area2")
        poly3 = AnnotationPolygon(points=[(150, 550), (150, 1200), (650, 1200), (650, 550)], label="area3")
        poly4 = AnnotationPolygon(points=[(150, 1600), (150, 2200), (650, 2200), (650, 1600)], label="area4")
        self.polygons = [poly1, poly2, poly3, poly4]

    def test_include1(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            included_labels=["area3"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 111)

    def test_include2(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            included_labels=["area3", "area4"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 102)

    def test_include3(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            included_labels=["area1", "area3", "area4"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 93)

    def test_include4(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            included_labels=["area1", "area2", "area3", "area4"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 84)

    def test_include5(self):
        poly1 = AnnotationPolygon(points=[(950, 550), (950, 1200), (1400, 1200), (1400, 550)], label="area1")
        poly2 = AnnotationPolygon(points=[(1700, 550), (1700, 1200), (2150, 1200), (2150, 550)], label="area2")
        poly3 = AnnotationPolygon(points=[(150, 550), (150, 1200), (650, 1200), (650, 550)], label="")
        poly4 = AnnotationPolygon(points=[(150, 1600), (150, 2200), (650, 2200), (650, 1600)], label="area4")
        polygons = [poly1, poly2, poly3, poly4]
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            included_labels=["area1", "area2", "", "area4"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 84)

    def test_exclude1(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            excluded_labels=["area3"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 93)

    def test_exclude2(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            excluded_labels=["area3", "area2"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 102)

    def test_exclude3(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            excluded_labels=["area3", "area2", "area1"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 111)

    def test_exclude4(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            excluded_labels=["area3", "area2", "area1", "area4"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("area2" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 120)

    def test_exclude5(self):
        poly1 = AnnotationPolygon(points=[(950, 550), (950, 1200), (1400, 1200), (1400, 550)], label="area1")
        poly2 = AnnotationPolygon(points=[(1700, 550), (1700, 1200), (2150, 1200), (2150, 550)], label="")
        poly3 = AnnotationPolygon(points=[(150, 550), (150, 1200), (650, 1200), (650, 550)], label="area3")
        poly4 = AnnotationPolygon(points=[(150, 1600), (150, 2200), (650, 2200), (650, 1600)], label="area4")
        polygons = [poly1, poly2, poly3, poly4]
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            excluded_labels=["area3", "", "area1", "area4"],
            weak_label="area0",
        )
        self.assertTrue("area0" in patches.patch_info)
        self.assertFalse("area1" in patches.patch_info)
        self.assertFalse("" in patches.patch_info)
        self.assertFalse("area3" in patches.patch_info)
        self.assertFalse("area4" in patches.patch_info)
        self.assertEqual(patches.patch_count, 120)

    def test_include_exclude1(self):
        with self.assertRaises(ValueError):
            WholeImageInvertedGridPatches(
                wsi_file=self.wsi_file_tif,
                mask_data=self.mask_array,
                polygon_data=self.polygons,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=1,
                foreground_ratio=0.8,
                overlap_ratio=1,
                included_labels=["area1", "area2"],
                excluded_labels=["area4", "area3"],
            )

    def test_include_exclude2(self):
        with self.assertRaises(ValueError):
            WholeImageInvertedGridPatches(
                wsi_file=self.wsi_file_tif,
                mask_data=self.mask_array,
                polygon_data=self.polygons,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=1,
                foreground_ratio=0.8,
                overlap_ratio=1,
                included_labels=[""],
                excluded_labels=[""],
            )

    def test_include_exclude3(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=1,
            included_labels=[],
            excluded_labels=[],
        )
        self.assertIsNotNone(patches)


class TestWholeImageInvertedStaticMethods(TestCase):
    """Test static methods present in inverted classes."""

    def setUp(self):
        self.wsi_name = "slide1.svs"

    def test__merge_annotation_polygons_empty(self):
        polygons = []
        merged_polygons = WholeImageInvertedPatches._merge_annotation_polygons(polygons, self.wsi_name)
        self.assertEqual(merged_polygons, polygons)

    @patch("dplabtools.slides.patches.locations.regions.unary_union")
    def test__merge_annotation_polygons_invalid(self, mock_func):
        poly = AnnotationPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="area")
        polygons = [poly]
        mock_func.return_value = Point(10, 10)
        with self.assertRaises(TypeError):
            WholeImageInvertedPatches._merge_annotation_polygons(polygons, self.wsi_name)

    def test__merge_annotation_polygons_one(self):
        poly = AnnotationPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="")
        polygons = [poly]
        merged_polygons = WholeImageInvertedPatches._merge_annotation_polygons(polygons, self.wsi_name)
        output_polygons = Polygon(poly.points)
        result_polygons = Polygon(merged_polygons[0].points)
        self.assertEqual(
            sorted(set(result_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
            sorted(set(output_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
        )

    def test__merge_annotation_polygons_two_same(self):
        poly1 = AnnotationPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="")
        poly2 = AnnotationPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="")
        polygons = [poly1, poly2]
        merged_polygons = WholeImageInvertedPatches._merge_annotation_polygons(polygons, self.wsi_name)
        output_polygons = Polygon(poly1.points)
        result_polygons = Polygon(merged_polygons[0].points)
        self.assertEqual(
            sorted(set(result_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
            sorted(set(output_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
        )

    def test__merge_annotation_polygons_two_different(self):
        poly1 = AnnotationPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="")
        poly2 = AnnotationPolygon(points=[(110, 110), (110, 120), (120, 120), (120, 110)], label="")
        polygons = [poly1, poly2]
        merged_polygons = WholeImageInvertedPatches._merge_annotation_polygons(polygons, self.wsi_name)
        output_polygons = [Polygon(poly1.points), Polygon(poly2.points)]
        result_polygons = [Polygon(merged_polygons[0].points), Polygon(merged_polygons[1].points)]
        self.assertEqual(
            sorted(set(result_polygons[0].exterior.coords), key=lambda x: (x[0], x[1])),
            sorted(set(output_polygons[0].exterior.coords), key=lambda x: (x[0], x[1])),
        )
        self.assertEqual(
            sorted(set(result_polygons[1].exterior.coords), key=lambda x: (x[0], x[1])),
            sorted(set(output_polygons[1].exterior.coords), key=lambda x: (x[0], x[1])),
        )

    def test__merge_annotation_polygons_two_overlapping(self):
        poly1 = AnnotationPolygon(points=[(100, 100), (100, 200), (200, 200), (200, 100)], label="")
        poly2 = AnnotationPolygon(points=[(100, 150), (100, 250), (200, 250), (200, 150)], label="")
        polygons = [poly1, poly2]
        merged_polygons = WholeImageInvertedPatches._merge_annotation_polygons(polygons, self.wsi_name)
        output_polygons = Polygon(
            [(100, 100), (100, 150), (100, 200), (100, 250), (200, 250), (200, 200), (200, 150), (200, 100)]
        )
        result_polygons = Polygon(merged_polygons[0].points)
        self.assertEqual(
            sorted(set(result_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
            sorted(set(output_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
        )

    def test__merge_annotation_polygons_two_contained(self):
        poly1 = AnnotationPolygon(points=[(100, 100), (100, 200), (200, 200), (200, 100)], label="")
        poly2 = AnnotationPolygon(points=[(150, 150), (150, 170), (170, 170), (170, 150)], label="")
        polygons = [poly1, poly2]
        merged_polygons = WholeImageInvertedPatches._merge_annotation_polygons(polygons, self.wsi_name)
        output_polygons = Polygon(poly1.points)
        result_polygons = Polygon(merged_polygons[0].points)
        self.assertEqual(
            sorted(set(result_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
            sorted(set(output_polygons.exterior.coords), key=lambda x: (x[0], x[1])),
        )


class TestWholeImageInvertedFullOverlapCheck(TestCase):
    """Tests for "_full_overlap_check".

    See comment in TestWholeImageFullOverlapCheck class
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_array = np.ones((640, 768), dtype=int)
        self.patch_size = 200
        self.level_or_mpp = 1
        self.foreground_ratio = 0.01
        self.overlap_ratio = 1
        poly1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        poly2 = AnnotationPolygon(points=[(1200, 1500), (1200, 2800), (2500, 2800), (2500, 1500)], label="area2")
        self.polygons = [poly1, poly2]

    def get_result_tuple(self, patch_data):
        location_x = []
        location_y = []
        max_x = 640 - self.patch_size
        max_y = 768 - self.patch_size
        for _data in patch_data:
            # mask is at level 1, so divide by 4
            x = round(_data[0][0] / 4)
            y = round(_data[0][1] / 4)
            check_x = bool(max_x >= x >= 0)
            check_y = bool(max_y >= y >= 0)
            location_x.append(check_x)
            location_y.append(check_y)
        return (location_x, location_y)

    def test_random_full_overlap_check(self):
        patches = WholeImageInvertedRandomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
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
        patches = WholeImageInvertedPoissonDiskPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            poisson_spacing=30,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        result_x, result_y = self.get_result_tuple(patches.patch_data)
        self.assertTrue(all(result_x))
        self.assertTrue(all(result_y))

    def test_grid_full_overlap_check(self):
        patches = WholeImageInvertedGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_array,
            polygon_data=self.polygons,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            patch_stride=1,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        result_x, result_y = self.get_result_tuple(patches.patch_data)
        self.assertTrue(all(result_x))
        self.assertTrue(all(result_y))


class TestWholeImageInvertedGridRoundingPrecision(TestCase):
    """Tests for special cases when lack of float precision leads to wrong results."""

    def setUp(self):
        self.wsi_file_svs = make_test_path("wsi/TUPAC-TE-234.svs")
        self.mask_file = make_test_path("mask/TUPAC-TE-234_mask.npy")
        poly1 = AnnotationPolygon(points=[(15000, 15000), (15000, 20000), (20000, 20000), (20000, 15000)], label="")
        self.polygons = [poly1]

    def test_inverted_grid_precision_correct(self):
        patches = WholeImageInvertedGridPatches(
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
        self.assertEqual(patches.patch_count, 2816)

    @patch("dplabtools.slides.patches.locations.base.roundfl", round_test)
    def test_inverted_grid_precision_incorrect(self):
        patches = WholeImageInvertedGridPatches(
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
        # openslide counts 2616 patches, tiffslide 200 patches
        self.assertNotEqual(patches.patch_count, 2816)


class TestWholeImageInvertedCustomPatches(TestCase):
    """Tests for WholeImageInvertedCustomPatches."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        self.polygon1 = AnnotationPolygon(points=[(20, 20), (20, 1020), (1020, 1020), (1020, 20)], label="area1")
        self.polygon2 = AnnotationPolygon(points=[(20, 1020), (20, 2020), (1020, 2020), (1020, 1020)], label="area2")
        self.points = [(30, 30), (30, 600), (1100, 30), (1100, 600), (1500, 1000)]
        self.patch_size = 256
        self.level_or_mpp = 0
        self.foreground_ratio = 0.5
        self.overlap_ratio = 1

    def test_compare_patch_counts1(self):
        patches = WholeImageInvertedCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=[self.polygon1],
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        self.assertEqual(patches.patch_count, 3)

    def test_compare_patch_counts2(self):
        patches = WholeImageInvertedCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=[self.polygon2],
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        self.assertEqual(patches.patch_count, 5)

    def test_compare_patch_data1(self):
        patches = WholeImageInvertedCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=[self.polygon1],
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
        )
        result_data = patches.patch_data
        output_data = [
            ((1100, 30), (256, 256), ""),
            ((1100, 600), (256, 256), ""),
            ((1500, 1000), (256, 256), ""),
        ]
        self.assertEqual(result_data, output_data)

    def test_compare_patch_data2(self):
        patches = WholeImageInvertedCustomPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            polygon_data=[self.polygon2],
            points=self.points,
            patch_size=self.patch_size,
            level_or_mpp=self.level_or_mpp,
            foreground_ratio=self.foreground_ratio,
            overlap_ratio=self.overlap_ratio,
            weak_label="abc",
        )
        result_data = patches.patch_data
        output_data = [
            ((30, 30), (256, 256), "abc"),
            ((30, 600), (256, 256), "abc"),
            ((1100, 30), (256, 256), "abc"),
            ((1100, 600), (256, 256), "abc"),
            ((1500, 1000), (256, 256), "abc"),
        ]
        self.assertEqual(result_data, output_data)
