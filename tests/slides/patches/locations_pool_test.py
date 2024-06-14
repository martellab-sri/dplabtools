# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for patches pool classes.

Tested classes:
    WholeImageRandomPatchesPool
    WholeImagePoissonDiskPatchesPool
    WholeImageGridPatchesPool
    WholeImageCustomPatchesPool
    WholeImageInvertedRandomPatchesPool
    WholeImageInvertedPoissonDiskPatchesPool
    WholeImageInvertedGridPatchesPool
    WholeImageInvertedCustomPatchesPool
    PolygonRegionRandomPatchesPool
    PolygonRegionPoissonDiskPatchesPool
    PolygonRegionGridPatchesPool
    PolygonRegionCustomPatchesPool
"""

import os
from unittest import TestCase

from PIL import Image
import numpy as np

from dplabtools.slides.patches import (
    WholeImageRandomPatchesPool,
    WholeImagePoissonDiskPatchesPool,
    WholeImageGridPatchesPool,
    WholeImageCustomPatchesPool,
    WholeImageInvertedRandomPatchesPool,
    WholeImageInvertedPoissonDiskPatchesPool,
    WholeImageInvertedGridPatchesPool,
    WholeImageInvertedCustomPatchesPool,
    PolygonRegionRandomPatchesPool,
    PolygonRegionPoissonDiskPatchesPool,
    PolygonRegionGridPatchesPool,
    PolygonRegionCustomPatchesPool,
)
from dplabtools.slides.patches.locations.pools import (
    BasePatchesPool,
    WholeImagePatchesPoolBase,
    WholeImageInvertedPatchesPoolBase,
)
from dplabtools.slides.utils import AnnotationPolygon
from testutils import make_test_path

wsi_tif_list = [
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color1.tif"),
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color1a.tif"),
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color2.tif"),
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color2a.tif"),
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color3.tif"),
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color3a.tif"),
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color4.tif"),
    make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color4a.tif"),
]

mask_data = make_test_path("mask/board-clean-mask.npy")
mask_data_list = [mask_data] * 8

poly1 = AnnotationPolygon(points=[(200, 200), (200, 1200), (1200, 1200), (1200, 200)], label="area1")
poly2 = AnnotationPolygon(points=[(1500, 1500), (1500, 2800), (2500, 2800)], label="area2")
polygons_list = [[poly1, poly2]] * 8


class TestPatchesPoolClassesBasic(TestCase):
    """Basic tests for all patches pool classes."""

    def test_patches_pool_wrong_arg(self):
        patches_args = {
            "patch_size_value": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "num_patches": 30,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        with self.assertRaises(TypeError):
            WholeImageRandomPatchesPool(
                wsi_file_list=wsi_tif_list,
                mask_data_list=mask_data_list,
                patches_args=patches_args,
                proc_num_workers=4,
            )

    def test_patches_pool_class1(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "num_patches": 30,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 30 * 8)
        self.assertEqual(patches_pool.patch_info, {"label1": 30 * 8})
        self.assertEqual(patches_pool.patch_labels, ["label1"])

    def test_patches_pool_class2(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "poisson_spacing": 30,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        patches_pool = WholeImagePoissonDiskPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        # at least 18 patches per image on average
        self.assertGreaterEqual(patches_pool.patch_count, 18 * 8)
        self.assertEqual(patches_pool.patch_info, {"label1": patches_pool.patch_count})
        self.assertEqual(patches_pool.patch_labels, ["label1"])

    def test_patches_pool_class3(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        patches_pool = WholeImageGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 60 * 8)
        self.assertEqual(patches_pool.patch_info, {"label1": 60 * 8})
        self.assertEqual(patches_pool.patch_labels, ["label1"])

    def test_patches_pool_class4(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.95,
            "level_or_mpp": 0,
            "num_patches": 7,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        patches_pool = WholeImageInvertedRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 7 * 8)
        self.assertEqual(patches_pool.patch_info, {"label1": 7 * 8})
        self.assertEqual(patches_pool.patch_labels, ["label1"])

    def test_patches_pool_class5(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.95,
            "level_or_mpp": 0,
            "poisson_spacing": 30,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        patches_pool = WholeImageInvertedPoissonDiskPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertGreaterEqual(patches_pool.patch_count, 9 * 8)
        self.assertEqual(patches_pool.patch_info, {"label1": patches_pool.patch_count})
        self.assertEqual(patches_pool.patch_labels, ["label1"])

    def test_patches_pool_class6(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 1,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        patches_pool = WholeImageInvertedGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 36 * 8)
        self.assertEqual(patches_pool.patch_info, {"label1": 36 * 8})
        self.assertEqual(patches_pool.patch_labels, ["label1"])

    def test_patches_pool_class7(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.8,
            "level_or_mpp": 0,
            "num_patches": 3,
            "polygon_buffer": 0,
        }
        patches_pool = PolygonRegionRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 3 * 2 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 3 * 8, "area2": 3 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_patches_pool_class8(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.2,
            "level_or_mpp": 0,
            "poisson_spacing": 25,
            "polygon_buffer": 0,
        }
        patches_pool = PolygonRegionPoissonDiskPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertGreaterEqual(patches_pool.patch_count, 8 * 3 * 2)
        self.assertIn("area1", patches_pool.patch_info)
        self.assertIn("area2", patches_pool.patch_info)
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_patches_pool_class9(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 8, "area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_patches_pool_class10(self):
        patches_args = {
            "patch_size": 1000,
            "foreground_ratio": 0.99,
            "overlap_ratio": 0.99,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 0)
        self.assertEqual(patches_pool.patch_info, {"area1": 0, "area2": 0})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])


class TestPatchesPoolClassAdvanced(TestCase):
    """Advanced tests for arguments and counts."""

    def test_list_count1(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.8,
            "level_or_mpp": 0,
            "num_patches": [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]],
            "polygon_buffer": 0,
        }
        patches_pool = PolygonRegionRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, (3 + 1) * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 3 * 8, "area2": 1 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_count2(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": [
                [0.1, 0.4],
                [0.1, 0.4],
                [0.1, 0.4],
                [0.1, 0.4],
                [0.1, 0.4],
                [0.1, 0.4],
                [0.1, 0.4],
                [0.1, 0.4],
            ],
            "overlap_ratio": [
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
            ],
            "level_or_mpp": 0,
            "patch_stride": [[1.5, 1], [1.5, 1], [1.5, 1], [1.5, 1], [1.5, 1], [1.5, 1], [1.5, 1], [1.5, 1]],
            "polygon_buffer": [[20, 0], [20, 0], [20, 0], [20, 0], [20, 0], [20, 0], [20, 0], [20, 0]],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, (13 + 4) * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 13 * 8, "area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_count3(self):
        local_wsi_tif_list = [
            make_test_path("wsi/board-multi-layer-no-compression-mpp.tif"),
            make_test_path("wsi/TUPAC-TE-234.svs"),
        ]
        local_mask_data_list = [
            make_test_path("mask/board-clean-mask.npy"),
            make_test_path("mask/TUPAC-TE-234_mask.npy"),
        ]
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.8,
            "overlap_ratio": 0.8,
            "level_or_mpp": 0,
            "patch_stride": 2,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        patches_pool = WholeImageGridPatchesPool(
            wsi_file_list=local_wsi_tif_list,
            mask_data_list=local_mask_data_list,
            patches_args=patches_args,
            proc_num_workers=2,
        )
        self.assertEqual(patches_pool.patch_count, 30 + 679)
        self.assertEqual(patches_pool.patch_info, {"label1": 30 + 679})
        self.assertEqual(patches_pool.patch_labels, ["label1"])

    def test_list_count4(self):
        local_wsi_tif_list = [
            make_test_path("wsi/board-multi-layer-no-compression-mpp.tif"),
            make_test_path("wsi/TUPAC-TE-234.svs"),
            make_test_path("wsi/JP2K-33003-1.svs"),
        ]
        local_mask_data_list = [
            make_test_path("mask/board-clean-mask.npy"),
            make_test_path("mask/TUPAC-TE-234_mask.npy"),
            make_test_path("mask/JP2K-33003-1_mask.npy"),
        ]
        polygon1 = AnnotationPolygon(points=[(200, 200), (200, 1200), (1200, 1200), (1200, 200)], label="area1")
        polygon2 = AnnotationPolygon(points=[(1500, 1500), (1500, 2800), (2500, 2800)], label="area2")
        polygon3 = AnnotationPolygon(points=[(50, 2000), (50, 2700), (700, 2700), (700, 2000)], label="area3")
        polygon4 = AnnotationPolygon(
            points=[(15000, 20000), (15000, 21000), (16000, 21000), (16000, 20000)], label="area1"
        )
        polygon5 = AnnotationPolygon(
            points=[(15000, 22000), (15000, 23000), (16000, 23000), (16000, 22000)], label="area2"
        )
        polygon6 = AnnotationPolygon(
            points=[(15000, 24000), (15000, 25000), (16000, 25000), (16000, 24000)], label="area4"
        )
        polygon7 = AnnotationPolygon(points=[(3500, 12000), (4500, 12000), (4500, 13000), (3500, 13000)], label="area1")
        polygon8 = AnnotationPolygon(points=[(6000, 12000), (7000, 12000), (7000, 13000), (6000, 13000)], label="area4")
        polygon9 = AnnotationPolygon(
            points=[(10000, 12000), (11000, 12000), (11000, 13000), (10000, 13000)], label="area5"
        )
        polygons1 = [polygon1, polygon2, polygon3]
        polygons2 = [polygon4, polygon5, polygon6]
        polygons3 = [polygon7, polygon8, polygon9]
        local_polygons_list = [polygons1, polygons2, polygons3]
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.6,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=local_wsi_tif_list,
            mask_data_list=local_mask_data_list,
            polygon_data_list=local_polygons_list,
            patches_args=patches_args,
            proc_num_workers=2,
        )
        self.assertEqual(patches_pool.patch_count, (7 + 4 + 2) + (15 + 8 + 11) + (6 + 11 + 15))
        self.assertEqual(
            patches_pool.patch_info, {"area1": 7 + 15 + 6, "area2": 4 + 8, "area3": 2, "area4": 11 + 11, "area5": 15}
        )
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2", "area3", "area4", "area5"])

    def test_list_count5(self):
        local_wsi_tif_list = [
            make_test_path("wsi/board-multi-layer-no-compression-mpp.tif"),
            make_test_path("wsi/TUPAC-TE-234.svs"),
            make_test_path("wsi/JP2K-33003-1.svs"),
        ]
        local_mask_data_list = [
            make_test_path("mask/board-clean-mask.npy"),
            make_test_path("mask/TUPAC-TE-234_mask.npy"),
            make_test_path("mask/JP2K-33003-1_mask.npy"),
        ]
        polygon2 = AnnotationPolygon(points=[(1500, 1500), (1500, 2800), (2500, 2800)], label="area2")
        polygon3 = AnnotationPolygon(points=[(50, 2000), (50, 2700), (700, 2700), (700, 2000)], label="area3")
        polygon4 = AnnotationPolygon(
            points=[(15000, 20000), (15000, 21000), (16000, 21000), (16000, 20000)], label="area1"
        )
        polygon5 = AnnotationPolygon(
            points=[(15000, 22000), (15000, 23000), (16000, 23000), (16000, 22000)], label="area2"
        )
        polygon6 = AnnotationPolygon(
            points=[(15000, 24000), (15000, 25000), (16000, 25000), (16000, 24000)], label="area4"
        )
        polygon7 = AnnotationPolygon(points=[(3500, 11000), (4500, 11000), (4500, 12000), (3500, 12000)], label="area1")
        polygon8 = AnnotationPolygon(points=[(6000, 12000), (7000, 12000), (7000, 13000), (6000, 13000)], label="area4")
        polygon9 = AnnotationPolygon(
            points=[(10000, 12000), (11000, 12000), (11000, 13000), (10000, 13000)], label="area5"
        )
        polygon10 = AnnotationPolygon(
            points=[(12000, 10000), (13000, 10000), (13000, 11000), (12000, 11000)], label="area1"
        )

        polygons1 = [polygon2, polygon3]
        polygons2 = [polygon4, polygon5, polygon6]
        polygons3 = [polygon7, polygon8, polygon9, polygon10]
        local_polygons_list = [polygons1, polygons2, polygons3]
        patches_args = {
            "patch_size": 64,
            "foreground_ratio": 0.3,
            "overlap_ratio": 0.8,
            "level_or_mpp": 0,
            "num_patches": [[7, 8], [9, 10, 11], [12, 13, 14, 15]],
            "polygon_buffer": 0,
        }
        patches_pool = PolygonRegionRandomPatchesPool(
            wsi_file_list=local_wsi_tif_list,
            mask_data_list=local_mask_data_list,
            polygon_data_list=local_polygons_list,
            patches_args=patches_args,
            proc_num_workers=2,
        )
        self.assertEqual(patches_pool.patch_count, (7 + 8) + (9 + 10 + 11) + (12 + 13 + 14 + 15))
        self.assertEqual(
            patches_pool.patch_info, {"area1": 9 + 12 + 15, "area2": 7 + 10, "area3": 8, "area4": 11 + 13, "area5": 14}
        )
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2", "area3", "area4", "area5"])
        output_patch_details = [
            ("testdata/wsi/board-multi-layer-no-compression-mpp.tif", {"area2": 7, "area3": 8}),
            ("testdata/wsi/TUPAC-TE-234.svs", {"area1": 9, "area2": 10, "area4": 11}),
            ("testdata/wsi/JP2K-33003-1.svs", {"area1": 27, "area4": 13, "area5": 14}),
        ]
        self.assertEqual(patches_pool.patch_details, output_patch_details)

    def test_list_included_labels1(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "included_labels": ["area1"],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 7 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1"])

    def test_list_included_labels2(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "included_labels": ["area1", "area2"],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 8, "area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_included_labels3(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "included_labels": [["area1", "area2"], [], [], [], [], [], [], []],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 8, "area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_included_labels4(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "included_labels": [["area1", "area2"], [""], [""], [""], [""], [""], [""], ["area1", "area2"]],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 2)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 2, "area2": 4 * 2})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_included_labels5(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "included_labels": [["area1"], ["area2"], ["area1"], ["area2"], ["area1"], ["area2"], ["area1"], ["area2"]],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 4)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 4, "area2": 4 * 4})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_included_labels6(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "included_labels": [["area1"], ["area1"], ["area1"], ["area1"], ["area1"], ["area1"], ["area1"], ["area1"]],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 7 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1"])

    def test_list_excluded_labels1(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": ["area1"],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 4 * 8)
        self.assertEqual(patches_pool.patch_info, {"area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area2"])

    def test_list_excluded_labels2(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": ["area1", "area2"],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 0)
        self.assertEqual(patches_pool.patch_info, {})
        self.assertEqual(patches_pool.patch_labels, [])

    def test_list_excluded_labels3(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": [""],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 8, "area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_excluded_labels4(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": [],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 8)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 8, "area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_excluded_labels5(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": [["area1", "area2"], [], [], [], [], [], [], []],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 7)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 7, "area2": 4 * 7})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_excluded_labels6(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": [["area1", "area2"], [""], [""], [""], [""], [""], [""], [""]],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 7)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 7, "area2": 4 * 7})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_excluded_labels7(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": [
                ["area1", "area2"],
                ["area1", "area2"],
                ["area1", "area2"],
                [],
                ["area1", "area2"],
                ["area1", "area2"],
                ["area1", "area2"],
                ["area1", "area2"],
            ],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 1)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 1, "area2": 4 * 1})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_excluded_labels8(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": [["area1"], ["area2"], ["area1"], ["area2"], ["area1"], ["area2"], ["area1"], ["area2"]],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 11 * 4)
        self.assertEqual(patches_pool.patch_info, {"area1": 7 * 4, "area2": 4 * 4})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])

    def test_list_excluded_labels9(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.5,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "excluded_labels": [["area1"], ["area1"], ["area1"], ["area1"], ["area1"], ["area1"], ["area1"], ["area1"]],
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(patches_pool.patch_count, 4 * 8)
        self.assertEqual(patches_pool.patch_info, {"area2": 4 * 8})
        self.assertEqual(patches_pool.patch_labels, ["area2"])


class TestPatchesPoolStaticMethods(TestCase):
    """Tests for static methods in  base classes."""

    def test__check_patches_params1(self):
        args = {"image_file": "file1", "other_file": "file2"}
        BasePatchesPool._check_patches_params(args)
        self.assertIsNotNone(args)

    def test__check_patches_params2(self):
        args = {"image_file": "file1", "wsi_file": "file2"}
        with self.assertRaises(ValueError):
            BasePatchesPool._check_patches_params(args)

    def test__update_preview_args1(self):
        args = {"image_file": "_preview.xyz", "param1": 111, "param2": 222}
        wsi_file = "/tmp/FILE1.svs"
        output_dict = {"image_file": "FILE1_preview.xyz", "param1": 111, "param2": 222}
        BasePatchesPool._update_preview_args(args, wsi_file)
        self.assertEqual(args, output_dict)

    def test__update_preview_args2(self):
        args = {"image_file": "/tmp/dir/_preview.xyz", "param1": 111, "param2": 222}
        wsi_file = "/tmp/FILE2.svs"
        output_dict = {"image_file": "/tmp/dir/FILE2_preview.xyz", "param1": 111, "param2": 222}
        BasePatchesPool._update_preview_args(args, wsi_file)
        self.assertEqual(args, output_dict)

    def test__update_process_args_whole_image(self):
        args = {"param1": 111, "param2": 222}
        process_data = ["data1", "data2"]
        WholeImagePatchesPoolBase._update_process_args(args, process_data)
        output_dict = {"wsi_file": "data1", "mask_data": "data2", "param1": 111, "param2": 222}
        self.assertEqual(args, output_dict)

    def test__update_process_args_polygon_regions(self):
        args = {"param1": 111, "param2": 222}
        process_data = ["data1", "data2", "data3"]
        WholeImageInvertedPatchesPoolBase._update_process_args(args, process_data)
        output_dict = {"wsi_file": "data1", "mask_data": "data2", "polygon_data": "data3", "param1": 111, "param2": 222}
        self.assertEqual(args, output_dict)

    def test__update_expandable_args(self):
        args = {"param1": 1, "param2": 2}
        process_data_index = 1
        output_dict = {"param1": 1, "param2": 2}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)
        #
        args = {"param1": 1, "param2": ["a", "b", "c", "d"]}
        process_data_index = 2
        output_dict = {"param1": 1, "param2": "c"}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)
        #
        args = {"param1": [1, 2, 3, 4], "param2": ["a", "b", "c", "d"]}
        process_data_index = 1
        output_dict = {"param1": 2, "param2": "b"}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)
        #
        args = {"param1": [1, 2, 3, 4], "param2": ["a", "b", "c", "d"]}
        process_data_index = 3
        output_dict = {"param1": 4, "param2": "d"}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)
        #
        args = {"param1": [[1, 2], [3, 4], [5, 6], [7, 8]], "param2": ["a", "b", "c", "d"]}
        process_data_index = 0
        output_dict = {"param1": [1, 2], "param2": "a"}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)
        #
        args = {"param1": [[1, 2], "abc", [5, 6], "xyz"], "param2": ["aaa", [11, 12], "bbb", [13, 14]]}
        process_data_index = 1
        output_dict = {"param1": "abc", "param2": [11, 12]}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)
        #
        args = {"param1": [1, 2, 3, 4], "param2": ["a", "b", "c", "d"]}
        process_data_index = 2
        with self.assertRaises(ValueError):
            WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 2)
        #
        args = {"included_labels": [1, 2, 3], "param2": 22, "excluded_labels": [4, 5]}
        process_data_index = 1
        output_dict = {"included_labels": [1, 2, 3], "param2": 22, "excluded_labels": [4, 5]}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)
        #
        args = {
            "included_labels": [[1, 2], "abc", [5, 6], "xyz"],
            "param2": 22,
            "excluded_labels": ["aaa", [11, 12], "bbb", [13, 14]],
        }
        process_data_index = 2
        output_dict = {"included_labels": [5, 6], "param2": 22, "excluded_labels": "bbb"}
        WholeImagePatchesPoolBase._update_expandable_args(args, process_data_index, 4)
        self.assertEqual(args, output_dict)


class TestPatchesPoolSavePreviewImage(TestCase):
    """Tests for save_preview_image method."""

    def test_preview_present_default(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "num_patches": 12,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        save_preview_image_args = {
            "image_file": make_test_path("saved_data/pool_preview1/_preview.tif"),
        }
        WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
            mp_chunksize=1,
            save_preview_image_args=save_preview_image_args,
        )
        image1_path = make_test_path("saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color1_preview.tif")
        image1a_path = make_test_path(
            "saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color1a_preview.tif"
        )
        image2_path = make_test_path("saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color2_preview.tif")
        image2a_path = make_test_path(
            "saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color2a_preview.tif"
        )
        image3_path = make_test_path("saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color3_preview.tif")
        image3a_path = make_test_path(
            "saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color3a_preview.tif"
        )
        image4_path = make_test_path("saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color4_preview.tif")
        image4a_path = make_test_path(
            "saved_data/pool_preview1/board-multi-layer-no-compression-mpp-color4a_preview.tif"
        )
        self.assertTrue(os.path.exists(image1_path))
        self.assertTrue(os.path.exists(image1a_path))
        self.assertTrue(os.path.exists(image2_path))
        self.assertTrue(os.path.exists(image2a_path))
        self.assertTrue(os.path.exists(image3_path))
        self.assertTrue(os.path.exists(image3a_path))
        self.assertTrue(os.path.exists(image4_path))
        self.assertTrue(os.path.exists(image4a_path))
        image1 = Image.open(image1_path)
        image1a = Image.open(image1a_path)
        image2 = Image.open(image2_path)
        image2a = Image.open(image2a_path)
        image3 = Image.open(image3_path)
        image3a = Image.open(image3a_path)
        image4 = Image.open(image4_path)
        image4a = Image.open(image4a_path)
        self.assertEqual(image1.width, 160)
        self.assertEqual(image1.height, 192)
        self.assertEqual(image1a.width, 160)
        self.assertEqual(image1a.height, 192)
        self.assertEqual(image2.width, 160)
        self.assertEqual(image2.height, 192)
        self.assertEqual(image2a.width, 160)
        self.assertEqual(image2a.height, 192)
        self.assertEqual(image3.width, 160)
        self.assertEqual(image3.height, 192)
        self.assertEqual(image3a.width, 160)
        self.assertEqual(image3a.height, 192)
        self.assertEqual(image4.width, 160)
        self.assertEqual(image4.height, 192)
        self.assertEqual(image4a.width, 160)
        self.assertEqual(image4a.height, 192)
        image1.close()
        image1a.close()
        image2.close()
        image2a.close()
        image3.close()
        image3a.close()
        image4.close()
        image4a.close()

    def test_preview_present_custom(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "num_patches": 12,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        save_preview_image_args = {
            "image_file": make_test_path("saved_data/pool_preview2/_preview.tif"),
            "level_or_minsize": 1,
            "thickness": 1,
        }
        WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
            mp_chunksize=1,
            save_preview_image_args=save_preview_image_args,
        )
        image1_path = make_test_path("saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color1_preview.tif")
        image1a_path = make_test_path(
            "saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color1a_preview.tif"
        )
        image2_path = make_test_path("saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color2_preview.tif")
        image2a_path = make_test_path(
            "saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color2a_preview.tif"
        )
        image3_path = make_test_path("saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color3_preview.tif")
        image3a_path = make_test_path(
            "saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color3a_preview.tif"
        )
        image4_path = make_test_path("saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color4_preview.tif")
        image4a_path = make_test_path(
            "saved_data/pool_preview2/board-multi-layer-no-compression-mpp-color4a_preview.tif"
        )
        self.assertTrue(os.path.exists(image1_path))
        self.assertTrue(os.path.exists(image1a_path))
        self.assertTrue(os.path.exists(image2_path))
        self.assertTrue(os.path.exists(image2a_path))
        self.assertTrue(os.path.exists(image3_path))
        self.assertTrue(os.path.exists(image3a_path))
        self.assertTrue(os.path.exists(image4_path))
        self.assertTrue(os.path.exists(image4a_path))
        image1 = Image.open(image1_path)
        image1a = Image.open(image1a_path)
        image2 = Image.open(image2_path)
        image2a = Image.open(image2a_path)
        image3 = Image.open(image3_path)
        image3a = Image.open(image3a_path)
        image4 = Image.open(image4_path)
        image4a = Image.open(image4a_path)
        self.assertEqual(image1.width, 640)
        self.assertEqual(image1.height, 768)
        self.assertEqual(image1a.width, 640)
        self.assertEqual(image1a.height, 768)
        self.assertEqual(image2.width, 640)
        self.assertEqual(image2.height, 768)
        self.assertEqual(image2a.width, 640)
        self.assertEqual(image2a.height, 768)
        self.assertEqual(image3.width, 640)
        self.assertEqual(image3.height, 768)
        self.assertEqual(image3a.width, 640)
        self.assertEqual(image3a.height, 768)
        self.assertEqual(image4.width, 640)
        self.assertEqual(image4.height, 768)
        self.assertEqual(image4a.width, 640)
        self.assertEqual(image4a.height, 768)
        image1.close()
        image1a.close()
        image2.close()
        image2a.close()
        image3.close()
        image3a.close()
        image4.close()
        image4a.close()

    def test_preview_wrong_arg(self):
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "num_patches": 12,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }
        save_preview_image_args = {
            "image_file": "_preview.tif",
            "level_or_PINK": 1,
            "thickness": 1,
        }
        with self.assertRaises(TypeError):
            WholeImageRandomPatchesPool(
                wsi_file_list=wsi_tif_list,
                mask_data_list=mask_data_list,
                patches_args=patches_args,
                proc_num_workers=4,
                mp_chunksize=1,
                save_preview_image_args=save_preview_image_args,
            )


class TestPatchesPoolChunksize(TestCase):
    """Tests for passing chunksize argument."""

    def setUp(self):
        self.patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "num_patches": 30,
            "polygon_buffer": 0,
            "weak_label": "label1",
        }

    def test_chunksize1(self):
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            mp_chunksize=1,
        )
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_chunksize2(self):
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            mp_chunksize=2,
        )
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_chunksize3(self):
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            mp_chunksize=4,
        )
        self.assertEqual(len(set(patches_pool.pids)), 2)

    def test_chunksize4(self):
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            mp_chunksize=8,
        )
        self.assertEqual(len(set(patches_pool.pids)), 1)

    def test_chunksize5(self):
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            mp_chunksize=3,
        )
        self.assertEqual(len(set(patches_pool.pids)), 3)


class TestPatchesPoolAsIterator(TestCase):
    """Tests for using patches pool as iterator."""

    def setUp(self):
        self.patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "num_patches": [1, 2, 3, 4, 5, 6, 7, 8],
            "polygon_buffer": 0,
            "weak_label": "label1",
        }

    def test_pool_iterator1(self):
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
        )
        self.assertEqual(len(list(patches_pool)), 8)

    def test_pool_iterator2(self):
        patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
        )
        patches_counts = []
        for patches in patches_pool:
            patches_counts.append(patches.patch_count)
        self.assertEqual(sorted(set(patches_counts)), sorted(set([1, 2, 3, 4, 5, 6, 7, 8])))


class TestPatchesPoolMockObject(TestCase):
    """Tests for checking mock object output validity."""

    def test_pool_mockobject1(self):
        # At least one default parameter should be skipped in patches_args,
        # then it should reappear in patches.param_info
        patches_args = {
            "patch_size": 512,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 1,
            "patch_stride": 1,
            "weak_label": "label5",
        }
        patches_pool = WholeImageGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        # "wsi_file" is excluded from automatic comparison, so not present in dict below
        output_param_info = {
            "patch_stride": 1,
            "mask_data": make_test_path("mask/board-clean-mask.npy"),
            "patch_size": 512,
            "level_or_mpp": 1,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "polygon_buffer": 0,
            "weak_label": "label5",
        }
        for patches in patches_pool:
            self.assertEqual(patches.patch_count, 1)
            self.assertEqual(patches.patch_data, [((0, 0), (512, 512), "label5")])
            self.assertEqual(patches.patch_info, {"label5": 1})
            self.assertEqual(patches.patch_labels, ["label5"])
            # parameter "wsi_file" needs special handling as files names are unique
            result_wsi_file_param = patches.param_info.pop("wsi_file")
            self.assertIn(result_wsi_file_param, wsi_tif_list)
            self.assertEqual(patches.param_info, output_param_info)
            self.assertEqual(patches.level_or_mpp, 1)
            self.assertIn(patches.wsi_file, wsi_tif_list)
            self.assertEqual(patches.class_name, "WholeImageGridPatches")

    def test_pool_mockobject2(self):
        # see comment above in previous test
        patches_args = {
            "patch_size": 50,
            "foreground_ratio": 0.2,
            "polygon_buffer": 0,
            "level_or_mpp": 2,
            "patch_stride": 1,
        }
        patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygons_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )
        # "wsi_file" and "polygon_data" are excluded from automatic comparison,  so not present in dict below
        output_param_info = {
            "patch_stride": 1,
            "mask_data": make_test_path("mask/board-clean-mask.npy"),
            "patch_size": 50,
            "level_or_mpp": 2,
            "foreground_ratio": 0.2,
            "overlap_ratio": 0.95,
            "polygon_buffer": 0,
            "included_labels": [],
            "excluded_labels": [],
            "check_polygons": True,
        }
        for patches in patches_pool:
            self.assertEqual(patches.patch_count, 1)
            self.assertEqual(patches.patch_data, [((200, 200), (50, 50), "area1")])
            self.assertEqual(patches.patch_info, {"area1": 1, "area2": 0})
            self.assertEqual(patches.patch_labels, ["area1", "area2"])
            # parameter "wsi_file" needs special handling as file names are unique
            result_wsi_file_param = patches.param_info.pop("wsi_file")
            self.assertIn(result_wsi_file_param, wsi_tif_list)
            # parameter "polygon_data" needs special handling as it includes AnnotationPolygon objects
            result_polygons_param = patches.param_info.pop("polygon_data")
            self.assertEqual(result_polygons_param[0], poly1)
            self.assertEqual(result_polygons_param[1], poly2)
            self.assertEqual(patches.param_info, output_param_info)
            self.assertEqual(patches.level_or_mpp, 2)
            self.assertIn(patches.wsi_file, wsi_tif_list)
            self.assertEqual(patches.class_name, "PolygonRegionGridPatches")


class TestPatchesPoolProperties(TestCase):
    """Tests for properties in  base classes.

    All current properties tested in other test classes
    """

    pass


class TestCustomPatchesPool(TestCase):
    """Tests for custom patches pool classes."""

    patches_args = {
        "patch_size": 64,
        "foreground_ratio": 0.1,
        "overlap_ratio": 0.1,
        "level_or_mpp": 0,
    }

    def test_pool_custom_whole_image1(self):
        points = [(10, 10), (20, 20)]
        points_list = [points] * 8
        patches_pool = WholeImageCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            points_list=points_list,
        )
        self.assertEqual(patches_pool.patch_count, 2 * 8)
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_pool_custom_whole_image2(self):
        points = [(10, 10), (20, 20), (30, 30), (5000, 5000)]
        points_list = [points] * 8
        patches_pool = WholeImageCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=2,
            points_list=points_list,
        )
        self.assertEqual(patches_pool.patch_count, 3 * 8)
        self.assertEqual(len(set(patches_pool.pids)), 2)

    def test_pool_custom_whole_image3(self):
        points = [(10000, 10000), (20000, 20000)]
        points_list = [points] * 8
        patches_pool = WholeImageCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=1,
            points_list=points_list,
        )
        self.assertEqual(patches_pool.patch_count, 0)
        self.assertEqual(len(set(patches_pool.pids)), 1)

    def test_pool_custom_whole_image4(self):
        points_list = [
            [(10, 10)],
            [(10, 10), (20, 20)],
            [(10, 10), (20, 20), (30, 30)],
            [(10, 10), (20, 20), (30, 30), (40, 40)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80)],
        ]
        patches_pool = WholeImageCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=3,
            points_list=points_list,
        )
        self.assertEqual(patches_pool.patch_count, 36)
        self.assertEqual(len(set(patches_pool.pids)), 3)

    def test_pool_custom_whole_image5(self):
        empty_mask = np.zeros((160, 192), dtype=bool)
        mask_data_list = [mask_data, empty_mask, mask_data, mask_data, mask_data, empty_mask, mask_data, mask_data]
        points_list = [
            [(10, 10)],
            [(10, 10), (20, 20)],
            [(10, 10), (20, 20), (30, 30)],
            [(10, 10), (20, 20), (30, 30), (40, 40)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70)],
            [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (70, 70), (80, 80)],
        ]
        patches_pool = WholeImageCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            points_list=points_list,
        )
        self.assertEqual(patches_pool.patch_count, 36 - 6 - 2)
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_pool_custom_whole_image_inverted1(self):
        polygon = AnnotationPolygon(points=[(200, 200), (200, 1200), (1200, 1200), (1200, 200)], label="area1")
        polygons_list = [[polygon]] * 8
        points = [(10, 10), (20, 20)]
        points_list = [points] * 8
        patches_pool = WholeImageInvertedCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 2 * 8)
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_pool_custom_whole_image_inverted2(self):
        polygon = AnnotationPolygon(points=[(200, 200), (200, 1200), (1200, 1200), (1200, 200)], label="area1")
        polygons_list = [[polygon]] * 8
        points = [(300, 300), (500, 500)]
        points_list = [points] * 8
        patches_pool = WholeImageInvertedCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=3,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 0)
        self.assertEqual(len(set(patches_pool.pids)), 3)

    def test_pool_custom_whole_image_inverted3(self):
        polygon = AnnotationPolygon(points=[(200, 200), (200, 1200), (1200, 1200), (1200, 200)], label="area1")
        polygons_list = [[polygon]] * 8
        points1 = [(10, 10), (20, 20)]
        points2 = [(300, 300), (500, 500)]
        points_list = [points1, points2, points2, points2, points2, points2, points1, points1]
        patches_pool = WholeImageInvertedCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=2,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 2 * 3)
        self.assertEqual(len(set(patches_pool.pids)), 2)

    def test_pool_custom_whole_image_inverted4(self):
        polygon = AnnotationPolygon(points=[(200, 200), (200, 1200), (1200, 1200), (1200, 200)], label="area1")
        polygons_list = [[polygon]] * 8
        points1 = [(10, 10), (20, 20)]
        points2 = [(300, 300), (500, 500)]
        points_list = [points1, points2, points2, points2, points2, points2, points2, points2]
        patches_args = {
            "patch_size": 64,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.1,
            "level_or_mpp": 0,
            "weak_label": "abc",
        }
        patches_pool = WholeImageInvertedCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=2,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 2 * 1)
        self.assertEqual(patches_pool.patch_info, {"abc": 2})
        self.assertEqual(len(set(patches_pool.pids)), 2)

    def test_pool_custom_whole_image_inverted5(self):
        polygon1 = AnnotationPolygon(points=[(200, 200), (200, 1400), (1400, 1400), (1400, 200)], label="area1")
        polygon2 = AnnotationPolygon(points=[(20, 20), (20, 1200), (1200, 1200), (1200, 20)], label="area2")
        polygons_list = [[polygon1], [polygon1], [polygon1], [polygon1], [polygon2], [polygon2], [polygon2], [polygon2]]
        points1 = [(10, 10), (400, 40), (20, 20), (50, 50), (100, 100), (500, 500), (1100, 1100)]
        points2 = [(300, 300), (5000, 5000), (1250, 1250), (1500, 1500)]
        points_list = [points1, points2, points1, points2, points1, points2, points1, points2]
        patches_pool = WholeImageInvertedCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=2,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 4 + 1 + 4 + 1 + 1 + 2 + 1 + 2)
        self.assertEqual(len(set(patches_pool.pids)), 2)

    def test_pool_custom_polygons1(self):
        polygon1 = AnnotationPolygon(points=[(200, 200), (200, 1400), (1400, 1400), (1400, 200)], label="area1")
        polygon2 = AnnotationPolygon(points=[(400, 400), (400, 1800), (1500, 1800), (1500, 400)], label="area2")
        polygons_list = [[polygon1], [polygon1], [polygon1], [polygon1], [polygon2], [polygon2], [polygon2], [polygon2]]
        points1 = [(10, 10), (260, 260), (1050, 1050), (600, 300), (900, 300), (5000, 5000)]
        points2 = [(40, 40), (300, 1300), (800, 800), (1300, 1300), (1400, 1400), (4000, 3000), (1300, 270)]
        points_list = [points1, points2, points1, points2, points1, points2, points1, points2]
        patches_pool = PolygonRegionCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=2,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 3 + 4 + 3 + 4 + 1 + 3 + 1 + 3)
        self.assertEqual(patches_pool.patch_info, {"area1": 3 + 4 + 3 + 4, "area2": 1 + 3 + 1 + 3})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])
        self.assertEqual(len(set(patches_pool.pids)), 2)

    def test_pool_custom_polygons2(self):
        empty_mask = np.zeros((160, 192), dtype=bool)
        mask_data_list = [empty_mask, empty_mask, empty_mask, empty_mask, mask_data, mask_data, mask_data, mask_data]
        polygon1 = AnnotationPolygon(points=[(200, 200), (200, 1400), (1400, 1400), (1400, 200)], label="area1")
        polygon2 = AnnotationPolygon(points=[(400, 400), (400, 1800), (1500, 1800), (1500, 400)], label="area2")
        polygons_list = [[polygon1], [polygon1], [polygon1], [polygon1], [polygon2], [polygon2], [polygon2], [polygon2]]
        points1 = [(10, 10), (260, 260), (1050, 1050), (600, 300), (900, 300), (5000, 5000)]
        points2 = [(40, 40), (300, 1300), (800, 800), (1300, 1300), (1400, 1400), (4000, 3000), (1300, 270)]
        points_list = [points1, points2, points1, points2, points1, points2, points1, points2]
        patches_pool = PolygonRegionCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 0 + 0 + 0 + 0 + 1 + 3 + 1 + 3)
        self.assertEqual(patches_pool.patch_info, {"area1": 0, "area2": 1 + 3 + 1 + 3})
        self.assertEqual(patches_pool.patch_labels, ["area1", "area2"])
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_pool_custom_polygons3(self):
        mask_data_list = [mask_data, mask_data, mask_data, mask_data, mask_data, mask_data, mask_data, mask_data]
        polygon1 = AnnotationPolygon(points=[(200, 200), (200, 1400), (1400, 1400), (1400, 200)], label="area10")
        polygon2 = AnnotationPolygon(points=[(400, 400), (400, 1800), (1500, 1800), (1500, 400)], label="area20")
        polygons_list = [[polygon1], [polygon1], [polygon1], [polygon1], [polygon2], [polygon2], [polygon2], [polygon2]]
        points1 = [(200, 200)]
        points2 = [(1400, 1400)]
        points_list = [points1, points2, points1, points2, points1, points2, points1, points2]
        patches_pool = PolygonRegionCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 1 + 0 + 1 + 0 + 1 + 0 + 1 + 0)
        self.assertEqual(patches_pool.patch_info, {"area10": 1 + 0 + 1 + 0, "area20": 1 + 0 + 1 + 0})
        self.assertEqual(patches_pool.patch_labels, ["area10", "area20"])
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_pool_custom_polygons4(self):
        mask_data_list = [mask_data, mask_data, mask_data, mask_data, mask_data, mask_data, mask_data, mask_data]
        polygon1 = AnnotationPolygon(points=[(200, 200), (200, 500), (500, 500), (500, 200)], label="area100")
        polygon2 = AnnotationPolygon(points=[(1000, 1000), (1000, 1500), (1500, 1500), (1500, 1000)], label="area200")
        polygons_list = [
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
        ]
        points1 = [(200, 200)]
        points2 = [(1000, 1000)]
        points_list = [points1, points2, points1, points2, points1, points2, points1, points2]
        patches_pool = PolygonRegionCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=4,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)
        self.assertEqual(patches_pool.patch_info, {"area100": 1 + 1 + 1 + 1, "area200": 1 + 1 + 1 + 1})
        self.assertEqual(patches_pool.patch_labels, ["area100", "area200"])
        self.assertEqual(len(set(patches_pool.pids)), 4)

    def test_pool_custom_polygons5(self):
        mask_data_list = [mask_data, mask_data, mask_data, mask_data, mask_data, mask_data, mask_data, mask_data]
        polygon1 = AnnotationPolygon(points=[(200, 200), (200, 500), (500, 500), (500, 200)], label="area1000")
        polygon2 = AnnotationPolygon(points=[(1000, 1000), (1000, 1500), (1500, 1500), (1500, 1000)], label="area2000")
        polygons_list = [
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
            [polygon1, polygon2],
        ]
        points1 = [(200, 200), (1000, 1000)]
        points2 = [(1000, 1000), (200, 200)]
        points_list = [points1, points2, points1, points2, points1, points2, points1, points2]
        patches_pool = PolygonRegionCustomPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=self.patches_args,
            proc_num_workers=3,
            points_list=points_list,
            polygon_data_list=polygons_list,
        )
        self.assertEqual(patches_pool.patch_count, 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2)
        self.assertEqual(patches_pool.patch_info, {"area1000": 2 + 2 + 2 + 2, "area2000": 2 + 2 + 2 + 2})
        self.assertEqual(patches_pool.patch_labels, ["area1000", "area2000"])
        self.assertEqual(len(set(patches_pool.pids)), 3)
