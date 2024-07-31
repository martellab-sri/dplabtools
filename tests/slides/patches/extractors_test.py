# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for extractor classes.

Tested classes:
    DiskPatchExtractor
    MemPatchExtractor
    MultiResDiskPatchExtractor
    MultiResMemPatchExtractor
"""

import os
import glob
from unittest import TestCase
from unittest.mock import patch
from hashlib import md5
from multiprocessing.pool import ThreadPool

import numpy as np
from PIL import Image

from dplabtools.slides.patches import (
    DiskPatchExtractor,
    MemPatchExtractor,
    MultiResDiskPatchExtractor,
    MultiResMemPatchExtractor,
    WholeImageGridPatches,
    PolygonRegionGridPatches,
    PatchesMock,
)
from dplabtools.slides.utils import AnnotationPolygon
from dplabtools.slides.patches.extractors.base import MultiResBasePatchExtractor
from dplabtools.common import get_random_string
from testutils import make_test_path


class TestDiskPatchExtractorStaticMethods(TestCase):
    """Tests for static methods in DiskPatchExtractor class."""

    def test__check_output_dir(self):
        # exists
        dir_path = "saved_data/tmp"
        _dir = make_test_path(dir_path)
        DiskPatchExtractor._check_output_dir(_dir)
        # does not exist
        _dir = make_test_path(dir_path, "does-not-exist-dir")
        with self.assertRaises(OSError):
            DiskPatchExtractor._check_output_dir(_dir)

    def test__create_output_subdirs(self):
        # must be called twice in a row
        dir_names = ["dir1", "dir2", "dir3"]
        dirs_path = "saved_data/dirs"
        pool_mode = False
        output_dir = make_test_path(dirs_path)
        wsi_file = "wsi.svs"
        DiskPatchExtractor._create_output_subdirs(dir_names, output_dir, pool_mode, wsi_file)
        for _dir in dir_names:
            self.assertTrue(os.path.exists(make_test_path(dirs_path, _dir)))
        DiskPatchExtractor._create_output_subdirs(dir_names, output_dir, pool_mode, wsi_file)
        for _dir in dir_names:
            self.assertTrue(os.path.exists(make_test_path(dirs_path, _dir)))

    def test__create_output_subdirs_empty(self):
        dir_names = ["", "dirdir"]
        dirs_path = "saved_data/dirs"
        pool_mode = False
        output_dir = make_test_path(dirs_path)
        wsi_file = "wsi.svs"
        with self.assertRaises(ValueError):
            DiskPatchExtractor._create_output_subdirs(dir_names, output_dir, pool_mode, wsi_file)

    def test__create_output_subdirs_pool(self):
        # must be called twice in a row
        # pool_mode is enabled for Pool disk extractor classes only
        dir_names = ["dir1a", "dir2a", "dir3a"]
        dirs_path = "saved_data/dirs"
        output_dir = make_test_path(dirs_path)
        pool_mode = True
        wsi_file = "wsi.svs"
        DiskPatchExtractor._create_output_subdirs(dir_names, output_dir, pool_mode, wsi_file)
        for _dir in dir_names:
            self.assertTrue(os.path.exists(make_test_path(dirs_path, _dir)))
        DiskPatchExtractor._create_output_subdirs(dir_names, output_dir, pool_mode, wsi_file)
        for _dir in dir_names:
            self.assertTrue(os.path.exists(make_test_path(dirs_path, _dir)))

    def test__create_manifest_dir(self):
        # must be called twice in a row
        output_dir = make_test_path("saved_data/manifest_dir")
        manifest_dir_name = "manifest_dir"
        pool_mode = False
        manifest_dir = DiskPatchExtractor._create_manifest_dir(output_dir, manifest_dir_name, pool_mode)
        self.assertEqual(manifest_dir, os.path.join(output_dir, manifest_dir_name))
        manifest_dir = DiskPatchExtractor._create_manifest_dir(output_dir, manifest_dir_name, pool_mode)
        self.assertEqual(manifest_dir, os.path.join(output_dir, manifest_dir_name))

    def test__create_manifest_dir_pool(self):
        # must be called twice in a row
        # pool_mode is enabled for Pool disk extractor classes only
        output_dir = make_test_path("saved_data/manifest_dir_a")
        manifest_dir_name = "manifest_dir_a"
        pool_mode = True
        manifest_dir = DiskPatchExtractor._create_manifest_dir(output_dir, manifest_dir_name, pool_mode)
        self.assertEqual(manifest_dir, os.path.join(output_dir, manifest_dir_name))
        manifest_dir = DiskPatchExtractor._create_manifest_dir(output_dir, manifest_dir_name, pool_mode)
        self.assertEqual(manifest_dir, os.path.join(output_dir, manifest_dir_name))

    def test__create_manifest_file(self):
        # must be called twice in a row to raise OSError
        manifest_dir = make_test_path("saved_data/manifest_file")
        manifest_file_name = "manitest_test_file.txt"
        wsi_name = "file1.svs"
        manifest_file = DiskPatchExtractor._create_manifest_file(manifest_dir, manifest_file_name, wsi_name)
        self.assertTrue(os.path.exists(manifest_file))
        with self.assertRaises(OSError):
            DiskPatchExtractor._create_manifest_file(manifest_dir, manifest_file_name, wsi_name)

    def test__get_extractor_class_params(self):
        base_params = {"p3": "abc", "p5": 111, "p1": "xyz", "kwargs": 555}
        base_child_params = {"p9": "ABC", "p10": "XYZ", "kwargs": 333}
        abstract_params = {"p4": 222, "p2": 333, "p6": "def", "self": 444}
        mixin_params = {"p7": 999, "self": 333, "p8": "opt", "kwargs": 222}
        output_params = [
            ("p1", "xyz"),
            ("p10", "XYZ"),
            ("p2", 333),
            ("p3", "abc"),
            ("p4", 222),
            ("p5", 111),
            ("p6", "def"),
            ("p7", 999),
            ("p8", "opt"),
            ("p9", "ABC"),
        ]
        result_params = DiskPatchExtractor._get_extractor_class_params(
            base_params, base_child_params, abstract_params, mixin_params
        )
        self.assertEqual(result_params, output_params)

    def test__get_patch_file_data(self):
        wsi_id = "12345"
        patch = ((123, 456), (128, 128), "label1")
        filename_comment = "testcomment"
        manifest_id = "abc123"
        output_data = ["12345", "abc123", "testcomment", "label1", "x123", "y456"]
        result_data = DiskPatchExtractor._get_patch_file_data(wsi_id, patch, filename_comment, manifest_id)
        self.assertEqual(result_data, output_data)
        #
        wsi_id = "12345"
        patch = ((123, 456), (128, 128), "")
        filename_comment = "testcomment"
        manifest_id = "abc123"
        output_data = ["12345", "abc123", "testcomment", "x123", "y456"]
        result_data = DiskPatchExtractor._get_patch_file_data(wsi_id, patch, filename_comment, manifest_id)
        self.assertEqual(result_data, output_data)
        #
        wsi_id = "12345"
        patch = ((123, 456), (128, 128), "")
        filename_comment = ""
        manifest_id = "abc123"
        output_data = ["12345", "abc123", "x123", "y456"]
        result_data = DiskPatchExtractor._get_patch_file_data(wsi_id, patch, filename_comment, manifest_id)
        self.assertEqual(result_data, output_data)
        #
        wsi_id = "12345"
        patch = ((123, 456), (128, 128), "label2")
        filename_comment = ""
        manifest_id = "abc123"
        output_data = ["12345", "abc123", "label2", "x123", "y456"]
        result_data = DiskPatchExtractor._get_patch_file_data(wsi_id, patch, filename_comment, manifest_id)
        self.assertEqual(result_data, output_data)
        #
        wsi_id = "12345"
        patch = ((-325, -756), (128, 128), "label2")
        filename_comment = ""
        manifest_id = "abc123"
        output_data = ["12345", "abc123", "label2", "x-325", "y-756"]
        result_data = DiskPatchExtractor._get_patch_file_data(wsi_id, patch, filename_comment, manifest_id)
        self.assertEqual(result_data, output_data)

    def test__get_patch_file_name(self):
        patch_file_data = ["abc", "123", "def", "456", "xyz"]
        filename_separator = "---"
        image_type = "tif"
        output_value = "abc---123---def---456---xyz.tif"
        result_value = DiskPatchExtractor._get_patch_file_name(patch_file_data, filename_separator, image_type)
        self.assertEqual(result_value, output_value)

    def test_get_patch_file_path(self):
        patch_file_name = "abc-123-def.tif"
        label = ""
        create_subdirs = False
        output_dir = "/tmp/data/"
        output_value = "/tmp/data/abc-123-def.tif"
        result_value = DiskPatchExtractor._get_patch_file_path(patch_file_name, label, create_subdirs, output_dir)
        self.assertEqual(result_value, output_value)
        #
        patch_file_name = "abc-123-def.tif"
        label = ""
        create_subdirs = True
        output_dir = "/tmp/data/"
        output_value = "/tmp/data/abc-123-def.tif"
        result_value = DiskPatchExtractor._get_patch_file_path(patch_file_name, label, create_subdirs, output_dir)
        self.assertEqual(result_value, output_value)
        #
        patch_file_name = "abc-123-def.tif"
        label = "xyz"
        create_subdirs = False
        output_dir = "/tmp/data/"
        output_value = "/tmp/data/abc-123-def.tif"
        result_value = DiskPatchExtractor._get_patch_file_path(patch_file_name, label, create_subdirs, output_dir)
        self.assertEqual(result_value, output_value)
        #
        patch_file_name = "abc-123-def.tif"
        label = "xyz"
        create_subdirs = True
        output_dir = "/tmp/data/"
        output_value = "/tmp/data/xyz/abc-123-def.tif"
        result_value = DiskPatchExtractor._get_patch_file_path(patch_file_name, label, create_subdirs, output_dir)
        self.assertEqual(result_value, output_value)
        #
        patch_file_name = "abc-123-def.tif"
        label = "xyz"
        create_subdirs = True
        output_dir = "/tmp/data/"
        patchset_dir = "set"
        output_value = "/tmp/data/xyz/set/abc-123-def.tif"
        result_value = DiskPatchExtractor._get_patch_file_path(
            patch_file_name, label, create_subdirs, output_dir, patchset_dir
        )
        self.assertEqual(result_value, output_value)
        #
        patch_file_name = "abc-123-def.tif"
        label = "xyz"
        create_subdirs = False
        output_dir = "/tmp/data/"
        patchset_dir = "set"
        output_value = "/tmp/data/set/abc-123-def.tif"
        result_value = DiskPatchExtractor._get_patch_file_path(
            patch_file_name, label, create_subdirs, output_dir, patchset_dir
        )
        self.assertEqual(result_value, output_value)

    def test__get_saved_patch_count_one_dir(self):
        output_dir = make_test_path("saved_data/counts")
        image_type = "png"
        manifest_id = "ABC123DEF456"
        valid_files_created_count = 9
        # create random valid files first
        for _ in range(valid_files_created_count):
            file_name = get_random_string(5) + manifest_id + get_random_string(5) + "." + image_type
            f = open(os.path.join(output_dir, file_name), "w")
            f.close()
        # create a number of other non valid files
        for _ in range(11):
            file_name = get_random_string(15) + "." + image_type
            f = open(os.path.join(output_dir, file_name), "w")
            f.close()
        # count valid files
        result_value = DiskPatchExtractor._get_saved_patch_count(output_dir, manifest_id, image_type)
        self.assertEqual(result_value, valid_files_created_count)

    def test__get_saved_patch_count_subdirs(self):
        # same as above, but with subdirectories
        output_dir = make_test_path("saved_data/counts_dirs")
        image_type = "jpg"
        manifest_id = "123ABC456DEF"
        valid_files_created_count = 7
        subdirs = ["data1", "data2", "data3"]
        for _dir in subdirs:
            os.mkdir(os.path.join(output_dir, _dir))
        for _dir in subdirs:
            for _ in range(valid_files_created_count):
                file_name = get_random_string(5) + manifest_id + get_random_string(5) + "." + image_type
                f = open(os.path.join(output_dir, _dir, file_name), "w")
                f.close()
        result_value = DiskPatchExtractor._get_saved_patch_count(output_dir, manifest_id, image_type)
        self.assertEqual(result_value, valid_files_created_count * 3)

    def test__indent_line(self):
        left_str = "Abc"
        right_str = "123"
        output_value = "Abc                     123\n"
        result_value = DiskPatchExtractor._indent_line(left_str, right_str)
        self.assertEqual(result_value, output_value)


class TestDiskPatchExtractorProperties(TestCase):
    """Add future class property tests here."""

    def non_test_patch_data(self):
        # This is a base property, tested in:
        # - TestDiskPatchExtractorSavingDataPolygonRegion
        # - TestDiskPatchExtractorIncludeExcludeLabels
        pass

    def non_test_patch_labels(self):
        # This is a base property, tested in:
        # - TestDiskPatchExtractorSavingDataPolygonRegion
        # - TestDiskPatchExtractorIncludeExcludeLabels
        # - TestMultiResDiskPatchExtractorPatchSet
        pass

    def non_test_patch_count(self):
        # This is a base property, tested in multiple places
        pass

    def non_test_manifest_id(self):
        # This property is tested in TestDiskPatchExtractorManifest
        pass

    def non_test_patchset_counter(self):
        # This property is tested in TestMultiResDiskPatchExtractorPatchSet
        pass


class TestDiskPatchExtractorManifest(TestCase):
    """Tests for manifest file creation and content."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.wsi_id = "board-multi-layer-no-compression-mpp"
        self.mask_file = make_test_path("mask/board-clean-mask-inner.npy")
        self.output_dir = make_test_path("saved_data/manifests")

    def test_manifest_file1(self):
        output_manifest_file = make_test_path("ref_data/slides/patches/extract/manifests/manifest1.txt")
        output_manifest_id = "t8mzd2i2"
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.88,
            overlap_ratio=0.77,
            weak_label="XYZ",
            polygon_buffer=0,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=self.output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            filename_separator="__",
            create_subdirs=True,
        )
        result_manifest_id = extractor.manifest_id
        result_manifest_file = os.path.join(
            self.output_dir, "manifests", "manifest__%s__%s.txt" % (result_manifest_id, self.wsi_id)
        )
        with open(result_manifest_file, "r") as f:
            result_manifest_lines = f.readlines()
        with open(output_manifest_file, "r") as f:
            output_manifest_lines = f.readlines()
        result_manifest_lines = [line.replace(result_manifest_id, output_manifest_id) for line in result_manifest_lines]
        self.assertEqual(result_manifest_lines[0], output_manifest_lines[0])  # id
        # initial lines will not match perfectly
        self.assertEqual(result_manifest_lines[1][:24], output_manifest_lines[1][:24])  # path
        self.assertEqual(result_manifest_lines[2][:24], output_manifest_lines[2][:24])  # date
        self.assertEqual(result_manifest_lines[3][:24], output_manifest_lines[3][:24])  # hostname
        self.assertEqual(result_manifest_lines[4][:50], output_manifest_lines[4][:50])  # separator
        self.assertEqual(sorted(result_manifest_lines[5:]), sorted(output_manifest_lines[5:]))  # all others

    def test_manifest_file2(self):
        output_manifest_file = make_test_path("ref_data/slides/patches/extract/manifests/manifest2.txt")
        output_manifest_id = "ha6v38ok"
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=0.89,
            overlap_ratio=0.66,
            polygon_buffer=0,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=self.output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            filename_separator="__",
            create_subdirs=True,
        )
        result_manifest_id = extractor.manifest_id
        result_manifest_file = os.path.join(
            self.output_dir, "manifests", "manifest__%s__%s.txt" % (result_manifest_id, self.wsi_id)
        )
        with open(result_manifest_file, "r") as f:
            result_manifest_lines = f.readlines()
        with open(output_manifest_file, "r") as f:
            output_manifest_lines = f.readlines()
        result_manifest_lines = [line.replace(result_manifest_id, output_manifest_id) for line in result_manifest_lines]
        self.assertEqual(result_manifest_lines[0], output_manifest_lines[0])  # id
        # initial lines will not match perfectly
        self.assertEqual(result_manifest_lines[1][:24], output_manifest_lines[1][:24])  # path
        self.assertEqual(result_manifest_lines[2][:24], output_manifest_lines[2][:24])  # date
        self.assertEqual(result_manifest_lines[3][:24], output_manifest_lines[3][:24])  # hostname
        self.assertEqual(result_manifest_lines[4][:50], output_manifest_lines[4][:50])  # separator
        self.assertEqual(sorted(result_manifest_lines[5:]), sorted(output_manifest_lines[5:]))  # all others

    def test_manifest_file3(self):
        output_manifest_file = make_test_path("ref_data/slides/patches/extract/manifests/manifest3.txt")
        output_manifest_id = "cbfly4wx"
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.84,
            overlap_ratio=0.74,
            weak_label="ABC",
            polygon_buffer=0,
        )
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=self.output_dir,
            num_workers=3,
            levels_or_mpps=[0, 0.5, 0.75],
            resampling_mode="tile",
            image_type="tif",
            create_subdirs=True,
        )
        result_manifest_id = extractor.manifest_id
        result_manifest_file = os.path.join(
            self.output_dir, "manifests", "manifest__%s__%s.txt" % (result_manifest_id, self.wsi_id)
        )
        with open(result_manifest_file, "r") as f:
            result_manifest_lines = f.readlines()
        with open(output_manifest_file, "r") as f:
            output_manifest_lines = f.readlines()
        result_manifest_lines = [line.replace(result_manifest_id, output_manifest_id) for line in result_manifest_lines]
        self.assertEqual(result_manifest_lines[0], output_manifest_lines[0])  # id
        # initial lines will not match perfectly
        self.assertEqual(result_manifest_lines[1][:24], output_manifest_lines[1][:24])  # path
        self.assertEqual(result_manifest_lines[2][:24], output_manifest_lines[2][:24])  # date
        self.assertEqual(result_manifest_lines[3][:24], output_manifest_lines[3][:24])  # hostname
        self.assertEqual(result_manifest_lines[4][:50], output_manifest_lines[4][:50])  # separator
        self.assertEqual(sorted(result_manifest_lines[5:]), sorted(output_manifest_lines[5:]))  # all others

    def test_manifest_file4(self):
        output_manifest_file = make_test_path("ref_data/slides/patches/extract/manifests/manifest4.txt")
        output_manifest_id = "qqv7k7ko"
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area4")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area5")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area6")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=0.81,
            overlap_ratio=0.61,
            polygon_buffer=0,
        )
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=self.output_dir,
            num_workers=3,
            levels_or_mpps=[0, 0.55, 0.75],
            resampling_mode="tile",
            image_type="tif",
            create_subdirs=True,
        )
        result_manifest_id = extractor.manifest_id
        result_manifest_file = os.path.join(
            self.output_dir, "manifests", "manifest__%s__%s.txt" % (result_manifest_id, self.wsi_id)
        )
        with open(result_manifest_file, "r") as f:
            result_manifest_lines = f.readlines()
        with open(output_manifest_file, "r") as f:
            output_manifest_lines = f.readlines()
        result_manifest_lines = [line.replace(result_manifest_id, output_manifest_id) for line in result_manifest_lines]
        self.assertEqual(result_manifest_lines[0], output_manifest_lines[0])  # id
        # initial lines will not match perfectly
        self.assertEqual(result_manifest_lines[1][:24], output_manifest_lines[1][:24])  # path
        self.assertEqual(result_manifest_lines[2][:24], output_manifest_lines[2][:24])  # date
        self.assertEqual(result_manifest_lines[3][:24], output_manifest_lines[3][:24])  # hostname
        self.assertEqual(result_manifest_lines[4][:50], output_manifest_lines[4][:50])  # separator
        self.assertEqual(sorted(result_manifest_lines[5:]), sorted(output_manifest_lines[5:]))  # all others


class TestDiskPatchExtractorSavingDataWholeImage(TestCase):
    """Tests for saving patches using WholeImageGridPatches class."""

    def test_save_whole_image_patches(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_file = make_test_path("mask/board-clean-mask.npy")
        output_dir = make_test_path("saved_data/patches1")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/patches1")
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=1,
            overlap_ratio=1,
            weak_label="label1",
        )
        DiskPatchExtractor(
            patches=self.patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=True,
        )
        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 60)
        # calculate checksums for reference patches
        ref_patch_path = os.path.join(ref_patch_dir, "**", "*" + "*.tif")
        ref_patch_list = glob.glob(ref_patch_path, recursive=True)
        ref_patch_md5_list = []
        for patch_file in ref_patch_list:
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                ref_patch_md5_list.append(patch_checksum)
        # examine extracted patches
        for patch_file in result_patch_list:
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)


class TestDiskPatchExtractorSavingDataPolygonRegion(TestCase):
    """Tests for saving patches using PolygonRegionGridPatches class."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")

    def test_save_polygon_patches_subdirs(self):
        output_dir = make_test_path("saved_data/patches2")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/patches2")
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=True,
        )
        # check extracted files individually for each label
        patch_counter = 0
        for label in patches.patch_labels:
            # count files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), patches.patch_info[label])
            patch_counter += len(result_patch_list)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            ref_patch_md5_list = []
            for patch_file in ref_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    ref_patch_md5_list.append(patch_checksum)
            # examine extracted patches
            for patch_file in result_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)
        # total patch count for all labels based on individual counts
        self.assertEqual(patch_counter, 12 + 6 + 3)
        # total patch count based on extractor patch_count property
        self.assertEqual(extractor.patch_count, 12 + 6 + 3)
        # extractor patch_data property test
        self.assertEqual(extractor.patch_data, patches.patch_data)
        # extractor patch_labels property test
        self.assertEqual(extractor.patch_labels, ["area1", "area2", "area3"])

    def test_save_polygon_patches_no_subdirs(self):
        output_dir = make_test_path("saved_data/patches13")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/patches3")
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=False,
        )
        # check extracted files individually for each label (but label subdirs are not created)
        for label in patches.patch_labels:
            # count files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 0)
        # count all files (labels not present as directories)
        label_dir = ""
        count_path = os.path.join(output_dir, label_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), extractor.patch_count)
        # calculate checksums for reference patches
        ref_patch_path = os.path.join(ref_patch_dir, label_dir, "**", "*" + "*.tif")
        ref_patch_list = glob.glob(ref_patch_path, recursive=True)
        ref_patch_md5_list = []
        for patch_file in ref_patch_list:
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                ref_patch_md5_list.append(patch_checksum)
        # examine extracted patches
        for patch_file in result_patch_list:
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)


class TestDiskPatchExtractorFailedCount(TestCase):
    """Test for OSError raised when patch file count does not match the expected number of patches."""

    def test_patch_file_count_fail(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        output_dir = make_test_path("saved_data/patches3")
        # two locations are identical, so only 5 patch files will be saved
        patch_data = [
            ((256, 256), (256, 256), "area1"),
            ((256, 256), (256, 256), "area1"),
            ((512, 512), (256, 256), "area1"),
            ((256, 1792), (256, 256), "area2"),
            ((256, 2304), (256, 256), "area2"),
            ((2048, 1024), (256, 256), "area3"),
        ]
        patches = PatchesMock(patch_data=patch_data, wsi_file=wsi_file_tif, level_or_mpp=0, check_location=False)
        with self.assertRaises(OSError):
            DiskPatchExtractor(
                patches=patches,
                output_dir=output_dir,
                num_workers=4,
                image_type="tif",
                filename_comment="testcomment",
                create_subdirs=True,
            )


class TestDiskPatchExtractorDuplicatePatches(TestCase):
    """Tests for checking if duplicate patches are eliminated properly.

    OSError will be raised if duplicate patches are not excluded and saved.
    check_polygns must be set to False here.
    """

    def test_exclude_duplicate_patches(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_file = make_test_path("mask/board-clean-mask.npy")
        output_dir = make_test_path("saved_data/patches6")
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 1],
            foreground_ratio=1,
            overlap_ratio=1,
            check_polygons=False,
        )
        DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=True,
        )


class TestDiskPatchExtractorDownsampling(TestCase):
    """Tests for downsampling modes."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")

    def test_downsampling_wsi(self):
        output_dir = make_test_path("saved_data/patches19")
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area1")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=512,
            level_or_mpp=0.5,  # must be MPP, not level
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=False,
            resampling_mode="wsi",
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), extractor.patch_count)

    def test_downsampling_tile(self):
        output_dir = make_test_path("saved_data/patches20")
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area1")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=512,
            level_or_mpp=0.5,  # must be MPP, not level
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=False,
            resampling_mode="tile",
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), extractor.patch_count)

    def test_downsampling_default(self):
        output_dir = make_test_path("saved_data/patches21")
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area1")
        polygons = [poly]
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=512,
            level_or_mpp=0.5,  # must be MPP, not level
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        with self.assertRaises(ValueError):
            DiskPatchExtractor(
                patches=patches,
                output_dir=output_dir,
                num_workers=4,
                image_type="tif",
                filename_comment="testcomment",
                create_subdirs=False,
            )


class TestDiskPatchExtractorPoolMode(TestCase):
    """Tests for pool mode.

    Tests for checking if pool mode will suppress error messages when multiple extractor objects
    are created in parallel (which would cause directory creation problems).
    """

    @patch("os.path.exists")
    def test_create_output_subdirs(self, mock_func):
        num_threads = 5
        dir_names = ["dir111", "dir222", "dir333"]
        output_dir = make_test_path("saved_data/tmp")
        wsi_file = "wsi.svs"
        mock_func.return_value = False

        def thread_function(args_list):
            DiskPatchExtractor._create_output_subdirs(args_list[0], args_list[1], args_list[2], args_list[3])

        # fail, directory already exists
        pool_mode = False
        args = [dir_names, output_dir, pool_mode, wsi_file]
        args_imap = [args for x in range(num_threads)]
        with self.assertRaises(FileExistsError):
            with ThreadPool(num_threads) as tpool:
                for _ in tpool.imap_unordered(thread_function, args_imap):
                    pass

        # success, error suppressed
        pool_mode = True
        args = [dir_names, output_dir, pool_mode, wsi_file]
        args_imap = [args for x in range(num_threads)]
        with ThreadPool(num_threads) as tpool:
            for _ in tpool.imap_unordered(thread_function, args_imap):
                pass

    @patch("os.path.exists")
    def test_create_manifest_dir(self, mock_func):
        num_threads = 5
        output_dir = make_test_path("saved_data/tmp")
        manifest_dir_name = "_manifest_"
        mock_func.return_value = False

        def thread_function(args_list):
            DiskPatchExtractor._create_manifest_dir(args_list[0], args_list[1], args_list[2])

        # fail, directory already exists
        pool_mode = False
        args = [output_dir, manifest_dir_name, pool_mode]
        args_imap = [args for x in range(num_threads)]
        with self.assertRaises(FileExistsError):
            with ThreadPool(num_threads) as tpool:
                for _ in tpool.imap_unordered(thread_function, args_imap):
                    pass

        # success, error suppressed
        pool_mode = True
        args = [output_dir, manifest_dir_name, pool_mode]
        args_imap = [args for x in range(num_threads)]
        with ThreadPool(num_threads) as tpool:
            for _ in tpool.imap_unordered(thread_function, args_imap):
                pass


class TestDiskPatchExtractorIncludeExcludeLabels(TestCase):
    """Test if include/exclude labels work with DiskPatchExtractor.

    More cases are tested for MemPatchExtractor.
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        self.polygons = [poly1, poly2, poly3]

    def test_included_labels(self):
        output_dir = make_test_path("saved_data/patches25")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=True,
            included_labels=["area2"],
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 6)
        self.assertEqual(extractor.patch_count, 6)
        # extractor patch_data + filtering property test
        output_data = list(single_patch for single_patch in patches.patch_data if single_patch[-1] == "area2")
        self.assertEqual(extractor.patch_data, output_data)
        # extractor patch_labels property test
        self.assertEqual(extractor.patch_labels, ["area2"])

    def test_excluded_labels(self):
        output_dir = make_test_path("saved_data/patches26")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            create_subdirs=True,
            excluded_labels=["area2"],
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 15)
        self.assertEqual(extractor.patch_count, 15)


class TestMemPatchExtractor(TestCase):
    """Tests for MemPatchExtractor class."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")

    def test_patch_stream_no_inference(self):
        """Save memory patches and compare them to disk patches."""
        output_dir = make_test_path("saved_data/patches4")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/patches2")
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        os.mkdir(os.path.join(output_dir, "area1"))
        os.mkdir(os.path.join(output_dir, "area2"))
        os.mkdir(os.path.join(output_dir, "area3"))
        extractor = MemPatchExtractor(
            patches=patches,
            num_workers=4,
        )
        # save memory patches (must use sequential names)
        counter = 0
        for patch_tuple in extractor.patch_stream:
            if counter == 0:
                # test tuple size only once
                self.assertEqual(len(patch_tuple), 2)
            counter += 1
            image = patch_tuple[0]
            label = patch_tuple[1]
            image_file = "image" + str(counter) + ".tif"
            image_path = os.path.join(output_dir, label, image_file)
            image.save(image_path)
        # check extracted files individually for each label
        patch_counter = 0
        for label in patches.patch_labels:
            # count files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), patches.patch_info[label])
            patch_counter += len(result_patch_list)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            ref_patch_md5_list = []
            for patch_file in ref_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    ref_patch_md5_list.append(patch_checksum)
            # examine extracted patches
            for patch_file in result_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)
        # total patch count for all labels based on individual counts
        self.assertEqual(patch_counter, 12 + 6 + 3)
        # total extracted patches as property
        self.assertEqual(extractor.patch_count, 12 + 6 + 3)

    def test_patch_stream_inference(self):
        """Test one in memory patch."""
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(
            patches=patches,
            num_workers=4,
            inference_mode=True,
        )
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list[0]), 3)
        self.assertIsInstance(patch_list[0][0], Image.Image)
        self.assertIsInstance(patch_list[0][1], str)
        self.assertIsInstance(patch_list[0][2], int)

    def test_downsampling_wsi(self):
        """Test one in memory patch."""
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0.3,  # must be MPP, not level
            patch_stride=[0.5],
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, resampling_mode="wsi")
        patch_list = list(extractor.patch_stream)
        self.assertTrue(len(patch_list) > 0)

    def test_downsampling_tile(self):
        """Test one in memory patch."""
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0.3,  # must be MPP, not level
            patch_stride=[0.5],
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, resampling_mode="tile")
        patch_list = list(extractor.patch_stream)
        self.assertTrue(len(patch_list) > 0)

    def test_downsampling_default(self):
        """Test one in memory patch."""
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0.3,  # must be MPP, not level
            patch_stride=[0.5],
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        extractor = MemPatchExtractor(
            patches=patches,
            num_workers=4,
        )
        with self.assertRaises(ValueError):
            list(extractor.patch_stream)

    def test_included_labels1(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=["area1", "area3"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 12 + 3)
        self.assertEqual(extractor.patch_count, 12 + 3)

    def test_included_labels2(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=["area1"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 12)
        self.assertEqual(extractor.patch_count, 12)

    def test_included_labels3(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=["area1", "area2", "area3"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 12 + 6 + 3)
        self.assertEqual(extractor.patch_count, 12 + 6 + 3)

    def test_included_labels4(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=["area3"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 3)
        self.assertEqual(extractor.patch_count, 3)

    def test_included_labels5(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=[])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 12 + 6 + 3)
        self.assertEqual(extractor.patch_count, 12 + 6 + 3)

    def test_included_labels6(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=[""])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 0)
        self.assertEqual(extractor.patch_count, 0)

    def test_included_labels7(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=[""])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 6)
        self.assertEqual(extractor.patch_count, 6)

    def test_excluded_labels1(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, excluded_labels=["area1"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 6 + 3)
        self.assertEqual(extractor.patch_count, 6 + 3)

    def test_excluded_labels2(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, excluded_labels=["area3"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 12 + 6)
        self.assertEqual(extractor.patch_count, 12 + 6)

    def test_excluded_labels3(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, excluded_labels=["area1", "area3"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 6)
        self.assertEqual(extractor.patch_count, 6)

    def test_excluded_labels4(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, excluded_labels=["area1", "area2", "area3"])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 0)
        self.assertEqual(extractor.patch_count, 0)

    def test_excluded_labels5(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, excluded_labels=[""])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 12 + 6 + 3)
        self.assertEqual(extractor.patch_count, 12 + 6 + 3)

    def test_excluded_labels6(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, excluded_labels=[""])
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list), 6 + 3)
        self.assertEqual(extractor.patch_count, 6 + 3)

    def test_included_excluded_labels1(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        with self.assertRaises(ValueError):
            MemPatchExtractor(patches=patches, num_workers=4, included_labels=["area1"], excluded_labels=["area2"])

    def test_included_excluded_labels2(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        with self.assertRaises(ValueError):
            MemPatchExtractor(patches=patches, num_workers=4, included_labels=[""], excluded_labels=[""])

    def test_included_excluded_labels3(self):
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        polygons = [poly1, poly2, poly3]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MemPatchExtractor(patches=patches, num_workers=4, included_labels=[], excluded_labels=[])
        self.assertIsNotNone(extractor)


class TestMultiResMemPatchExtractor(TestCase):
    """Tests for MultiResMemPatchExtractor class."""

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        self.polygons = [poly1, poly2, poly3]

    def test_patch_stream_saved_no_downsampling(self):
        """Save memory patches and compare them to disk patches."""
        output_dir = make_test_path("saved_data/patches7")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/multires1")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        os.mkdir(os.path.join(output_dir, "area1"))
        os.mkdir(os.path.join(output_dir, "area2"))
        os.mkdir(os.path.join(output_dir, "area3"))
        extractor = MultiResMemPatchExtractor(patches=patches, num_workers=4, levels_or_mpps=[0, 1])
        # save memory patches (must use sequential names)
        counter = 0
        for patch_tuple in extractor.patch_stream:
            if counter == 0:
                # test tuple size only once
                self.assertEqual(len(patch_tuple), 2)
            for patch_data in patch_tuple:
                counter += 1
                image = patch_data[0]
                label = patch_data[1]
                image_file = "image" + str(counter) + ".tif"
                image_path = os.path.join(output_dir, label, image_file)
                image.save(image_path)
        # check extracted files individually for each label
        patch_counter = 0
        for label in patches.patch_labels:
            # count files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 2 * patches.patch_info[label])
            patch_counter += len(result_patch_list)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            ref_patch_md5_list = []
            for patch_file in ref_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    ref_patch_md5_list.append(patch_checksum)
            # examine extracted patches
            for patch_file in result_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)
        # total patch count for all labels and all patch sets
        self.assertEqual(patch_counter, 12 * 2 + 6 * 2 + 3 * 2)
        # total extracted patches as property
        self.assertEqual(extractor.patch_count, 12 * 2 + 6 * 2 + 3 * 2)

    def test_patch_stream_no_downsampling_inference(self):
        """Test one in memory patch set."""
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area1")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResMemPatchExtractor(patches=patches, num_workers=4, levels_or_mpps=[0], inference_mode=True)
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list[0]), 1)
        self.assertEqual(len(patch_list[0][0]), 3)
        self.assertIsInstance(patch_list[0][0][0], Image.Image)
        self.assertIsInstance(patch_list[0][0][1], str)
        self.assertIsInstance(patch_list[0][0][2], int)

    def test_patch_stream_order_and_scaling(self):
        # test one patch set only to make results predictable
        output_dir = make_test_path("saved_data/patches8")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/multires2")
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (800, 1600)], label="area1")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=4,
            levels_or_mpps=[0, 0.5, 0.75, 1, 1.25],
            resampling_mode="wsi",
        )
        # process one in-memory patch (with multiple resolutions)
        counter = 0
        label = None
        for patch_tuple in extractor.patch_stream:
            for patch_data in patch_tuple:
                counter += 1
                image = patch_data[0]
                label = patch_data[1]
                self.assertEqual(label, "area1")
                image_file = "image" + str(counter) + ".tif"
                image_path = os.path.join(output_dir, image_file)
                image.save(image_path)
        for i in range(1, 6):
            ref_patch_path = os.path.join(ref_patch_dir, "image" + str(i) + ".tif")
            saved_patch_path = os.path.join(output_dir, "image" + str(i) + ".tif")
            with open(ref_patch_path, "rb") as ref_file_img:
                ref_image = Image.open(ref_file_img)
                ref_image_array = np.asarray(ref_image)
            with open(saved_patch_path, "rb") as saved_file_img:
                saved_image = Image.open(saved_file_img)
                saved_image_array = np.asarray(saved_image)
            np.testing.assert_equal(ref_image_array, saved_image_array)

    def test_patch_stream_saved_wsi_downsampling(self):
        """Save memory patches and compare them to disk patches."""
        output_dir = make_test_path("saved_data/patches9")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/multires3")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        os.mkdir(os.path.join(output_dir, "area1"))
        os.mkdir(os.path.join(output_dir, "area2"))
        os.mkdir(os.path.join(output_dir, "area3"))
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=4,
            levels_or_mpps=[0.3, 0.4, 0.6, 0.7, 0.8],  # must be MPPs, not levels
            resampling_mode="wsi",
        )
        # save memory patches (must use sequential names)
        counter = 0
        for patch_tuple in extractor.patch_stream:
            for patch_data in patch_tuple:
                counter += 1
                image = patch_data[0]
                label = patch_data[1]
                image_file = "image" + str(counter) + ".tif"
                image_path = os.path.join(output_dir, label, image_file)
                image.save(image_path)
        # check extracted files individually for each label
        patch_counter = 0
        for label in patches.patch_labels:
            # count files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 5 * patches.patch_info[label])
            patch_counter += len(result_patch_list)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            ref_patch_md5_list = []
            for patch_file in ref_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    ref_patch_md5_list.append(patch_checksum)
            # examine extracted patches
            for patch_file in result_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)
        # total patch count for all labels and all patch sets
        self.assertEqual(patch_counter, 12 * 5 + 6 * 5 + 3 * 5)

    def test_patch_stream_tile_downsampling(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=4,
            levels_or_mpps=[0.3, 0.4, 0.6, 0.7, 0.8],  # must be MPPs, not levels
            resampling_mode="tile",
        )
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list) * 5, 12 * 5 + 6 * 5 + 3 * 5)

    def test_patch_stream_default_downsampling(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=4,
            levels_or_mpps=[0.3, 0.4, 0.6, 0.7, 0.8],  # must be MPPs, not levels
        )
        with self.assertRaises(ValueError):
            list(extractor.patch_stream)

    def test_patch_stream_order_and_scale_reversed(self):
        # test one patch set only to make results predictable
        output_dir = make_test_path("saved_data/patches10")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/multires4")
        poly = AnnotationPolygon(points=[(256, 1280), (256, 2816), (2400, 1700)], label="area2")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.5,
            overlap_ratio=0.8,
        )
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=4,
            levels_or_mpps=[1.0, 0.8, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25],
            resampling_mode="wsi",
        )
        # process one in-memory patch (with multiple resolutions)
        counter = 0
        label = None
        for patch_tuple in extractor.patch_stream:
            for patch_data in patch_tuple:
                counter += 1
                image = patch_data[0]
                label = patch_data[1]
                self.assertEqual(label, "area2")
                image_file = "image" + str(counter) + ".tif"
                image_path = os.path.join(output_dir, image_file)
                image.save(image_path)
        for i in range(1, 9):
            ref_patch_path = os.path.join(ref_patch_dir, "image" + str(i) + ".tif")
            saved_patch_path = os.path.join(output_dir, "image" + str(i) + ".tif")
            with open(ref_patch_path, "rb") as ref_file_img:
                ref_image = Image.open(ref_file_img)
                ref_image_array = np.asarray(ref_image)
            with open(saved_patch_path, "rb") as saved_file_img:
                saved_image = Image.open(saved_file_img)
                saved_image_array = np.asarray(saved_image)
            np.testing.assert_equal(ref_image_array, saved_image_array)

    def test_included_labels(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=4,
            levels_or_mpps=[0.28, 0.30, 0.32],
            resampling_mode="tile",
            included_labels=["area2"],
        )
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list) * 3, 6 * 3)
        self.assertEqual(extractor.patch_count, 6 * 3)

    def test_excluded_labels(self):
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=4,
            levels_or_mpps=[0.28, 0.30, 0.32],
            resampling_mode="tile",
            excluded_labels=["area1", "area3"],
        )
        patch_list = list(extractor.patch_stream)
        self.assertEqual(len(patch_list) * 3, 6 * 3)
        self.assertEqual(extractor.patch_count, 6 * 3)


class TestMultiResDiskPatchExtractorStaticMethods(TestCase):
    """Tests for MultiResdiskPatchExtractor class."""

    def test__get_extra_mpps(self):
        levels_or_mpps = [1, 2, 3, 4, 5, 6]
        output_mpps = []
        result_mpps = MultiResBasePatchExtractor._get_extra_mpps(levels_or_mpps)
        self.assertEqual(result_mpps, output_mpps)
        #
        levels_or_mpps = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
        output_mpps = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
        result_mpps = MultiResBasePatchExtractor._get_extra_mpps(levels_or_mpps)
        self.assertEqual(result_mpps, output_mpps)
        #
        levels_or_mpps = [1.1, 2.2, 3, 4, 5.5, 6]
        output_mpps = [1.1, 2.2, 5.5]
        result_mpps = MultiResBasePatchExtractor._get_extra_mpps(levels_or_mpps)
        self.assertEqual(result_mpps, output_mpps)
        #
        levels_or_mpps = []
        output_mpps = []
        result_mpps = MultiResBasePatchExtractor._get_extra_mpps(levels_or_mpps)
        self.assertEqual(result_mpps, output_mpps)


class TestMultiResDiskPatchExtractor(TestCase):
    """Tests for MultiResdiskPatchExtractor class.

    Only need to test methods/features which are not shared with DiskPatchExtractor.
    """

    def setUp(self):
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        poly1 = AnnotationPolygon(points=[(256, 0), (256, 768), (2304, 768), (2304, 0)], label="area1")
        poly2 = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area2")
        poly3 = AnnotationPolygon(points=[(1920, 896), (1920, 2650), (2304, 2400), (2304, 900)], label="area3")
        self.polygons = [poly1, poly2, poly3]

    @patch("dplabtools.slides.patches.extractors.mixins.MultiResDiskPatchExtractorMixin._global_patchset_counter", None)
    def test_save_patches_subdirs_and_others(self):
        # mock is used to allow testing comment suffix (provides predictable patch order and names)
        output_dir = make_test_path("saved_data/patches11")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/multires5")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=1,
            levels_or_mpps=[0, 0.5, 0.75],
            resampling_mode="wsi",
            image_type="tif",
            create_subdirs=True,
        )
        # check extracted files individually for each label
        patch_counter = 0
        for label in patches.patch_labels:
            # count files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 3 * patches.patch_info[label])
            patch_counter += len(result_patch_list)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            ref_patch_md5_list = []
            for patch_file in ref_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    ref_patch_md5_list.append(patch_checksum)
            # examine extracted patches
            for patch_file in result_patch_list:
                with open(patch_file, "rb") as file_img:
                    patch_checksum = md5(file_img.read()).hexdigest()
                    self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)
        # total patch count for all labels and all patch sets
        self.assertEqual(patch_counter, 12 * 3 + 6 * 3 + 3 * 3)
        #
        # also test auto-generated comment suffix here:
        #
        comment_suffix_test_dir = os.path.join(output_dir, "area1/set1")
        # test existance of 3 files with partial names corresponding to levels_or_mpps
        file_path = os.path.join(comment_suffix_test_dir, "*" + "level0_area1" + "*.tif")
        file_list1 = glob.glob(file_path)
        self.assertEqual(len(file_list1), 1)
        #
        file_path = os.path.join(comment_suffix_test_dir, "*" + "mpp0.5_area1" + "*.tif")
        file_list2 = glob.glob(file_path)
        self.assertEqual(len(file_list2), 1)
        #
        file_path = os.path.join(comment_suffix_test_dir, "*" + "mpp0.75_area1" + "*.tif")
        file_list3 = glob.glob(file_path)
        self.assertEqual(len(file_list3), 1)
        #
        # also test patch coordinates embedded in file names
        #
        file_string = file_list1[0].replace(".tif", "")
        xy_data = file_string.split("_")[-2:]
        xy = (xy_data[0][1:], xy_data[1][1:])
        self.assertEqual((int(xy[0]), int(xy[1])), (512, 0))
        #
        file_string = file_list2[0].replace(".tif", "")
        xy_data = file_string.split("_")[-2:]
        xy = (xy_data[0][1:], xy_data[1][1:])
        self.assertEqual((int(xy[0]), int(xy[1])), (384, -128))
        #
        file_string = file_list3[0].replace(".tif", "")
        xy_data = file_string.split("_")[-2:]
        xy = (xy_data[0][1:], xy_data[1][1:])
        self.assertEqual((int(xy[0]), int(xy[1])), (256, -256))

    def test_save_patches_no_subdirs(self):
        output_dir = make_test_path("saved_data/patches12")
        ref_patch_dir = make_test_path("ref_data/slides/patches/extract/multires6")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            levels_or_mpps=[0, 1],
            image_type="tif",
            create_subdirs=False,
        )
        # check extracted files individually for each label (but label subdirs are not created)
        for label in patches.patch_labels:
            # count files (should not find anything using label)
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 0)

        # count all files (labels not present as directories)
        label_dir = ""
        count_path = os.path.join(output_dir, label_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), extractor.patch_count)
        # calculate checksums for reference patches
        ref_patch_path = os.path.join(ref_patch_dir, label_dir, "**", "*" + "*.tif")
        ref_patch_list = glob.glob(ref_patch_path, recursive=True)
        ref_patch_md5_list = []
        for patch_file in ref_patch_list:
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                ref_patch_md5_list.append(patch_checksum)
        # examine extracted patches
        for patch_file in result_patch_list:
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)

    def test_wsi_downsampling(self):
        output_dir = make_test_path("saved_data/patches22")
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area1")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=512,
            level_or_mpp=0.5,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            levels_or_mpps=[0.3, 0.35, 0.4],  # must be MPPs, not levels
            image_type="tif",
            create_subdirs=False,
            resampling_mode="wsi",
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), extractor.patch_count)

    def test_tile_downsampling(self):
        output_dir = make_test_path("saved_data/patches23")
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area1")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=512,
            level_or_mpp=0.5,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=4,
            levels_or_mpps=[0.35, 0.4],  # must be MPPs, not levels
            image_type="tif",
            create_subdirs=False,
            resampling_mode="tile",
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), extractor.patch_count)

    def test_default_downsampling(self):
        output_dir = make_test_path("saved_data/patches24")
        poly = AnnotationPolygon(points=[(128, 1280), (128, 2816), (1600, 2816)], label="area1")
        polygons = [poly]
        patches = PolygonRegionGridPatches(
            polygon_data=polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=512,
            level_or_mpp=0.5,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        with self.assertRaises(ValueError):
            MultiResDiskPatchExtractor(
                patches=patches,
                output_dir=output_dir,
                num_workers=4,
                levels_or_mpps=[0.55, 0.65],  # must be MPPs, not levels
                image_type="tif",
                create_subdirs=False,
            )

    def test_included_labels(self):
        output_dir = make_test_path("saved_data/patches27")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=1,
            levels_or_mpps=[0.28, 0.30, 0.32],
            resampling_mode="tile",
            image_type="tif",
            create_subdirs=True,
            included_labels=["area2", "area3"],
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 6 * 3 + 3 * 3)
        self.assertEqual(extractor.patch_count, 6 * 3 + 3 * 3)

    def test_excluded_labels(self):
        output_dir = make_test_path("saved_data/patches28")
        patches = PolygonRegionGridPatches(
            polygon_data=self.polygons,
            wsi_file=self.wsi_file_tif,
            mask_data=self.mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=[1, 0.5, 0.5],
            foreground_ratio=1,
            overlap_ratio=1,
        )
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=1,
            levels_or_mpps=[0.28, 0.30, 0.32],
            resampling_mode="tile",
            image_type="tif",
            create_subdirs=True,
            excluded_labels=["area1"],
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 6 * 3 + 3 * 3)
        self.assertEqual(extractor.patch_count, 6 * 3 + 3 * 3)


class TestMultiResDiskPatchExtractorPatchSet(TestCase):
    """Tests for different combinations of patch set directory handling."""

    def setUp(self):
        wsi_file1 = make_test_path("wsi/JP2K-33003-1.svs")
        wsi_file2 = make_test_path("wsi/TUPAC-TE-234.svs")
        self.wsi_files = [wsi_file1, wsi_file2]
        self.patch_data = [
            ((5256, 5256), (256, 256), "area1"),
            ((5256, 5256), (256, 256), "area1"),
            ((5512, 5512), (256, 256), "area1"),
            ((7256, 6792), (256, 256), "area2"),
            ((8256, 9304), (256, 256), "area2"),
            ((12048, 11024), (256, 256), "area3"),
        ]

    def test_multires_global_counter_none_subdirs(self):
        output_dir = make_test_path("saved_data/patches14")

        for wsi_file in self.wsi_files:
            patches_mock = PatchesMock(
                patch_data=self.patch_data, wsi_file=wsi_file, level_or_mpp=0, check_location=False
            )
            extractor = MultiResDiskPatchExtractor(
                patches=patches_mock,
                output_dir=output_dir,
                num_workers=1,
                levels_or_mpps=[0, 1],
                image_type="png",
                filename_comment="",
                create_subdirs=True,
                global_counter=None,
            )
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area1/JP2K-33003-1__set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area1/JP2K-33003-1__set2")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area1/JP2K-33003-1__set3")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area1/TUPAC-TE-234__set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area1/TUPAC-TE-234__set2")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area1/TUPAC-TE-234__set3")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area2/JP2K-33003-1__set4")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area2/JP2K-33003-1__set5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area2/TUPAC-TE-234__set4")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area2/TUPAC-TE-234__set5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area3/JP2K-33003-1__set6")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "area3/TUPAC-TE-234__set6")))
        # extractor patch_labels property test (distinct values)
        self.assertEqual(extractor.patch_labels, ["area1", "area2", "area3"])

    def test_multires_global_counter_none_no_subdirs(self):
        output_dir = make_test_path("saved_data/patches15")
        for wsi_file in self.wsi_files:
            patches_mock = PatchesMock(
                patch_data=self.patch_data, wsi_file=wsi_file, level_or_mpp=0, check_location=False
            )
            MultiResDiskPatchExtractor(
                patches=patches_mock,
                output_dir=output_dir,
                num_workers=1,
                levels_or_mpps=[0, 1],
                image_type="png",
                filename_comment="",
                create_subdirs=False,
                global_counter=None,
            )
        self.assertTrue(os.path.exists(os.path.join(output_dir, "JP2K-33003-1__set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "JP2K-33003-1__set2")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "JP2K-33003-1__set3")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "TUPAC-TE-234__set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "TUPAC-TE-234__set2")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "TUPAC-TE-234__set3")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "JP2K-33003-1__set4")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "JP2K-33003-1__set5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "TUPAC-TE-234__set4")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "TUPAC-TE-234__set5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "JP2K-33003-1__set6")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "TUPAC-TE-234__set6")))

    @patch("dplabtools.slides.patches.extractors.mixins.MultiResDiskPatchExtractorMixin._global_patchset_counter", None)
    def test_multires_global_counter_defaulf(self):
        output_dir = make_test_path("saved_data/patches16")
        for wsi_file in self.wsi_files:
            patches_mock = PatchesMock(
                patch_data=self.patch_data, wsi_file=wsi_file, level_or_mpp=0, check_location=False
            )
            MultiResDiskPatchExtractor(
                patches=patches_mock,
                output_dir=output_dir,
                num_workers=1,
                levels_or_mpps=[0, 1],
                image_type="png",
                filename_comment="",
                create_subdirs=False,
            )
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set2")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set3")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set4")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set5")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set6")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set7")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set8")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set9")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set10")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set11")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set12")))

    @patch("dplabtools.slides.patches.extractors.mixins.MultiResDiskPatchExtractorMixin._global_patchset_counter", None)
    def test_multires_global_counter_custom(self):
        output_dir = make_test_path("saved_data/patches17")
        for wsi_file in self.wsi_files:
            patches_mock = PatchesMock(
                patch_data=self.patch_data, wsi_file=wsi_file, level_or_mpp=0, check_location=False
            )
            extractor = MultiResDiskPatchExtractor(
                patches=patches_mock,
                output_dir=output_dir,
                num_workers=1,
                levels_or_mpps=[0, 1],
                image_type="png",
                filename_comment="",
                create_subdirs=False,
                global_counter=50,
            )
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set50")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set51")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set52")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set53")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set54")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set55")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set56")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set57")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set58")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set59")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set60")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "set61")))
        self.assertEqual(extractor.patchset_counter, 61)

    def test_multires_two_identical_ids(self):
        # this should raise "directory already exists" error
        wsi_file_1 = make_test_path("wsi/JP2K-33003-1.svs")
        wsi_file_2 = make_test_path("wsi/JP2K-33003-1.svs")
        self.wsi_files = [wsi_file_1, wsi_file_2]
        output_dir = make_test_path("saved_data/patches18")
        with self.assertRaises(OSError):
            for wsi_file in self.wsi_files:
                patches_mock = PatchesMock(
                    patch_data=self.patch_data, wsi_file=wsi_file, level_or_mpp=0, check_location=False
                )
                MultiResDiskPatchExtractor(
                    patches=patches_mock,
                    output_dir=output_dir,
                    num_workers=1,
                    levels_or_mpps=[0, 1],
                    image_type="png",
                    filename_comment="",
                    create_subdirs=True,
                    global_counter=None,
                )


class TestDiskMemExtractorsChunksize(TestCase):
    """Tests for multiprocessing chunksize argument in all relevant extractor classes."""

    def setUp(self):
        mask_data = np.zeros((160, 192))
        mask_data[64:64, 96:96] = 1
        self.wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.output_dir = make_test_path("saved_data/tmp")
        self.patches = WholeImageGridPatches(
            wsi_file=self.wsi_file_tif,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.5,
            overlap_ratio=1,
            weak_label="",
        )

    def test_class1a_chunksize_int(self):
        """Test DiskPatchExtractor."""
        e = DiskPatchExtractor(
            patches=self.patches,
            output_dir=self.output_dir,
            num_workers=4,
            image_type="tif",
            filename_comment="testcomment",
            filename_separator="__",
            mp_chunksize=2,
        )
        self.assertIsNotNone(e)

    def test_class1b_chunksize_none(self):
        """Test DiskPatchExtractor."""
        with self.assertRaises(TypeError):
            DiskPatchExtractor(
                patches=self.patches,
                output_dir=self.output_dir,
                num_workers=4,
                image_type="tif",
                filename_comment="testcomment",
                filename_separator="__",
                mp_chunksize=None,
            )

    def test_class2a_chunksize_int(self):
        """Test MultiResDiskPatchExtractor."""
        e = MultiResDiskPatchExtractor(
            patches=self.patches,
            output_dir=self.output_dir,
            num_workers=4,
            levels_or_mpps=[0, 1],
            image_type="tif",
            filename_comment="testcomment",
            filename_separator="__",
            mp_chunksize=2,
        )
        self.assertIsNotNone(e)

    def test_class2b_chunksize_none(self):
        """Test MultiResDiskPatchExtractor."""
        with self.assertRaises(TypeError):
            MultiResDiskPatchExtractor(
                patches=self.patches,
                output_dir=self.output_dir,
                num_workers=4,
                levels_or_mpps=[0, 1],
                image_type="tif",
                filename_comment="testcomment",
                filename_separator="__",
                mp_chunksize=None,
            )

    def test_class3a_chunksize_int(self):
        """Test MemPatchExtractor."""
        e = MemPatchExtractor(
            patches=self.patches,
            num_workers=4,
            mp_chunksize=2,
        )
        self.assertIsNotNone(e)

    def test_class3b_chunksize_none(self):
        """Test MemPatchExtractor."""
        with self.assertRaises(TypeError):
            MemPatchExtractor(
                patches=self.patches,
                num_workers=4,
                mp_chunksize=None,
            )

    def test_class4a_chunksize_int(self):
        """Test MultiResMemPatchExtractor."""
        e = MultiResMemPatchExtractor(
            patches=self.patches,
            num_workers=4,
            levels_or_mpps=[0, 1],
            mp_chunksize=2,
        )
        self.assertIsNotNone(e)

    def test_class4b_chunksize_none(self):
        """Test MultiResMemPatchExtractor."""
        with self.assertRaises(TypeError):
            MultiResMemPatchExtractor(
                patches=self.patches,
                num_workers=4,
                levels_or_mpps=[0, 1],
                mp_chunksize=None,
            )
