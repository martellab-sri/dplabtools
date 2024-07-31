# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for patches mock class.

Tested classes:
    PatchesMock
"""

import os
import glob
from unittest import TestCase

from dplabtools.slides.patches import PatchesMock, DiskPatchExtractor, MemPatchExtractor
from testutils import make_test_path


class TestPatchesMockPropertiesAndDiskExtraction(TestCase):
    """Tests for PatchesMock class properties and disk based extraction using PatchesMock."""

    def setUp(self):
        self.patch_data = [
            ((2048, 1024), (256, 256), "area3"),
            ((256, 256), (256, 256), "area1"),
            ((256, 768), (256, 256), "area1"),
            ((256, 2304), (256, 256), "area2"),
            ((512, 512), (256, 256), "area1"),
            ((256, 1792), (256, 256), "area2"),
        ]
        self.wsi_file = "/tmp/wsi1.svs"

    def test_mock_properties_default(self):
        patches_mock = PatchesMock(
            patch_data=self.patch_data,
            wsi_file=self.wsi_file,
            level_or_mpp=1,
        )
        #
        result_patch_count = patches_mock.patch_count
        self.assertEqual(result_patch_count, len(self.patch_data))
        #
        result_patch_data = patches_mock.patch_data
        self.assertEqual(result_patch_data, self.patch_data)
        #
        result_patch_info = patches_mock.patch_info
        self.assertEqual(result_patch_info, {"area1": 3, "area2": 2, "area3": 1})
        #
        result_patch_labels = patches_mock.patch_labels
        self.assertEqual(result_patch_labels, ["area1", "area2", "area3"])
        #
        result_param_info = patches_mock.param_info
        param_info = {
            "patch_data": self.patch_data,
            "wsi_file": self.wsi_file,
            "level_or_mpp": 1,
            "check_location": True,
            "class_name": None,
            "class_params": {},
            "patch_labels": [],
        }
        self.assertEqual(result_param_info, param_info)
        #
        result_level_or_mpp = patches_mock.level_or_mpp
        self.assertEqual(result_level_or_mpp, 1)
        #
        result_wsi_file = patches_mock.wsi_file
        self.assertEqual(result_wsi_file, "/tmp/wsi1.svs")
        #
        result_class_name = patches_mock.class_name
        self.assertEqual(result_class_name, patches_mock.__class__.__name__)

    def test_mock_properties_custom(self):
        patches_mock = PatchesMock(
            patch_data=self.patch_data,
            wsi_file=self.wsi_file,
            level_or_mpp=1,
            class_name="SomeClass",
            class_params={"param1": 1, "param2": 2, "param3": 3},
        )
        #
        result_patch_count = patches_mock.patch_count
        self.assertEqual(result_patch_count, len(self.patch_data))
        #
        result_patch_data = patches_mock.patch_data
        self.assertEqual(result_patch_data, self.patch_data)
        #
        result_patch_info = patches_mock.patch_info
        self.assertEqual(result_patch_info, {"area1": 3, "area2": 2, "area3": 1})
        #
        result_patch_labels = patches_mock.patch_labels
        self.assertEqual(result_patch_labels, ["area1", "area2", "area3"])
        #
        result_param_info = patches_mock.param_info
        param_info = {"param1": 1, "param2": 2, "param3": 3}
        self.assertEqual(result_param_info, param_info)
        #
        result_level_or_mpp = patches_mock.level_or_mpp
        self.assertEqual(result_level_or_mpp, 1)
        #
        result_wsi_file = patches_mock.wsi_file
        self.assertEqual(result_wsi_file, "/tmp/wsi1.svs")
        #
        result_class_name = patches_mock.class_name
        self.assertEqual(result_class_name, "SomeClass")

    def test_mock_disk_patch_extraction(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        patches_mock = PatchesMock(
            patch_data=self.patch_data,
            wsi_file=wsi_file_tif,
            level_or_mpp=0,
        )
        output_dir = make_test_path("saved_data/patches5")
        DiskPatchExtractor(
            patches=patches_mock,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="",
            filename_separator="_",
            create_subdirs=True,
        )
        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 6)


class TestPatchesMockDuplicateLocations(TestCase):
    """Test cases for duplicate patches/locations."""

    def setUp(self):
        self.wsi_file = "/tmp/wsi1.svs"

    def test_single_duplicated_location(self):
        patch_data = [
            ((512, 512), (256, 256), "area3"),
            ((256, 256), (256, 256), "area1"),
            ((256, 768), (256, 256), "area1"),
            ((256, 2304), (256, 256), "area2"),
            ((512, 512), (256, 256), "area1"),
            ((256, 1792), (256, 256), "area2"),
        ]
        patches_mock = PatchesMock(
            patch_data=patch_data,
            wsi_file=self.wsi_file,
            level_or_mpp=0,
            check_location=False,
        )
        self.assertIsNotNone(patches_mock)
        #
        with self.assertRaises(ValueError):
            PatchesMock(
                patch_data=patch_data,
                wsi_file=self.wsi_file,
                level_or_mpp=0,
                check_location=True,
            )

    def test_multi_duplicated_locations(self):
        patch_data = [
            ((512, 512), (256, 256), "area3"),
            ((256, 256), (256, 256), "area1"),
            ((256, 768), (256, 256), "area1"),
            ((256, 2304), (256, 256), "area2"),
            ((512, 512), (256, 256), "area1"),
            ((256, 256), (256, 256), "area2"),
        ]
        patches_mock = PatchesMock(
            patch_data=patch_data,
            wsi_file=self.wsi_file,
            level_or_mpp=0,
            check_location=False,
        )
        self.assertIsNotNone(patches_mock)
        #
        with self.assertRaises(ValueError):
            PatchesMock(
                patch_data=patch_data,
                wsi_file=self.wsi_file,
                level_or_mpp=0,
                check_location=True,
            )


class TestPatchesMockWithPatchLabels(TestCase):
    """Tests for patch_labels parameter."""

    def setUp(self):
        self.patch_data = [
            ((2048, 1024), (256, 256), "area3"),
            ((256, 256), (256, 256), "area1"),
            ((256, 768), (256, 256), "area1"),
            ((256, 2304), (256, 256), "area2"),
            ((512, 512), (256, 256), "area1"),
            ((256, 1792), (256, 256), "area2"),
        ]
        self.wsi_file = "/tmp/wsi1.svs"

    def test_labels_some_empty(self):
        patches_mock = PatchesMock(
            patch_data=self.patch_data,
            wsi_file=self.wsi_file,
            level_or_mpp=1,
            patch_labels=["area1", "area2", "area3", "area4", "area5"],
        )
        result_patch_count = patches_mock.patch_count
        self.assertEqual(result_patch_count, 3 + 2 + 1)
        result_patch_info = patches_mock.patch_info
        self.assertEqual(result_patch_info, {"area1": 3, "area2": 2, "area3": 1, "area4": 0, "area5": 0})
        result_patch_labels = patches_mock.patch_labels
        self.assertEqual(result_patch_labels, ["area1", "area2", "area3", "area4", "area5"])

    def test_labels_all_empty(self):
        patches_mock = PatchesMock(
            patch_data=[], wsi_file=self.wsi_file, level_or_mpp=1, patch_labels=["area1", "area2", "area3"]
        )
        result_patch_count = patches_mock.patch_count
        self.assertEqual(result_patch_count, 0)
        result_patch_info = patches_mock.patch_info
        self.assertEqual(result_patch_info, {"area1": 0, "area2": 0, "area3": 0})
        result_patch_labels = patches_mock.patch_labels
        self.assertEqual(result_patch_labels, ["area1", "area2", "area3"])

    def test_labels_some_blank(self):
        patch_data = [
            ((2048, 1024), (256, 256), ""),
            ((256, 256), (256, 256), ""),
            ((256, 768), (256, 256), ""),
            ((256, 2304), (256, 256), "area2"),
            ((512, 512), (256, 256), "area1"),
            ((256, 1792), (256, 256), "area2"),
        ]
        patches_mock = PatchesMock(
            patch_data=patch_data, wsi_file=self.wsi_file, level_or_mpp=1, patch_labels=["area1", "area2", "", "area3"]
        )
        result_patch_count = patches_mock.patch_count
        self.assertEqual(result_patch_count, 3 + 1 + 2)
        result_patch_info = patches_mock.patch_info
        self.assertEqual(result_patch_info, {"": 3, "area1": 1, "area2": 2, "area3": 0})
        result_patch_labels = patches_mock.patch_labels
        self.assertEqual(result_patch_labels, ["", "area1", "area2", "area3"])

    def test_labels_all_blank1(self):
        patch_data = [
            ((2048, 1024), (256, 256), ""),
            ((256, 256), (256, 256), ""),
            ((256, 768), (256, 256), ""),
            ((256, 2304), (256, 256), ""),
            ((512, 512), (256, 256), ""),
            ((256, 1792), (256, 256), ""),
        ]
        patches_mock = PatchesMock(
            patch_data=patch_data, wsi_file=self.wsi_file, level_or_mpp=1, patch_labels=["area1", ""]
        )
        result_patch_count = patches_mock.patch_count
        self.assertEqual(result_patch_count, 6)
        result_patch_info = patches_mock.patch_info
        self.assertEqual(result_patch_info, {"area1": 0, "": 6})
        result_patch_labels = patches_mock.patch_labels
        self.assertEqual(result_patch_labels, ["", "area1"])

    def test_labels_all_blank2(self):
        patch_data = [
            ((2048, 1024), (256, 256), ""),
            ((256, 256), (256, 256), ""),
            ((256, 768), (256, 256), ""),
            ((256, 2304), (256, 256), ""),
            ((512, 512), (256, 256), ""),
            ((256, 1792), (256, 256), ""),
        ]
        patches_mock = PatchesMock(
            patch_data=patch_data,
            wsi_file=self.wsi_file,
            level_or_mpp=1,
        )
        result_patch_count = patches_mock.patch_count
        self.assertEqual(result_patch_count, 6)
        result_patch_info = patches_mock.patch_info
        self.assertEqual(result_patch_info, {"": 6})
        result_patch_labels = patches_mock.patch_labels
        self.assertEqual(result_patch_labels, [""])

    def test_labels_invalid(self):
        patch_data = [
            ((2048, 1024), (256, 256), "area1"),
            ((256, 256), (256, 256), "area2"),
            ((256, 768), (256, 256), "area66"),
        ]
        with self.assertRaises(ValueError):
            PatchesMock(
                patch_data=patch_data, wsi_file=self.wsi_file, level_or_mpp=1, patch_labels=["area1", "area2", "area3"]
            )


class TestPatchesMockMemExtraction(TestCase):
    """Tests for memory based extraction using PatchesMock."""

    def test_mock_mem_patch_extraction(self):
        patch_data = [
            ((2048, 1024), (256, 256), "area3"),
            ((256, 256), (256, 256), "area1"),
            ((256, 768), (256, 256), "area1"),
            ((256, 2304), (256, 256), "area2"),
            ((512, 512), (256, 256), "area1"),
            ((256, 1792), (256, 256), "area2"),
        ]
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        patches_mock = PatchesMock(
            patch_data=patch_data,
            wsi_file=wsi_file_tif,
            level_or_mpp=0,
        )
        extractor = MemPatchExtractor(patches=patches_mock, num_workers=4)
        self.assertEqual(len(list(extractor.patch_stream)), 6)
