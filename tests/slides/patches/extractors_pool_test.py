# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for extractor pool classes.

Tested classes:
    MemPatchExtractorPool
    MultiResMemPatchExtractorPool
    DiskPatchExtractorPool
    MultiResDiskPatchExtractorPool
"""

import os
import glob
from hashlib import md5
from unittest import TestCase

from dplabtools.slides.patches import (
    WholeImageRandomPatches,
    WholeImagePoissonDiskPatches,
    WholeImageGridPatches,
    PatchesMock,
    MemPatchExtractorPool,
    MultiResMemPatchExtractorPool,
    DiskPatchExtractorPool,
    MultiResDiskPatchExtractorPool,
    WholeImageGridPatchesPool,
)
from testutils import make_test_path


class TestMemPatchExtractorPoolStaticMethods(TestCase):
    """Static methods common to all pool extractor classes."""

    def test_get_process_pool_data(self):
        wsi_file1 = "/tmp/wsi1.svs"
        patch_data1 = [
            ((100, 200), (256, 256), "area1"),
            ((300, 400), (128, 128), "area2"),
        ]
        patches1 = PatchesMock(
            patch_data=patch_data1,
            wsi_file=wsi_file1,
            level_or_mpp=1,
            check_location=False,
            class_name="SomeClass1",
            class_params={"param1": 1, "param2": 2},
        )
        wsi_file2 = "/tmp/wsi2.svs"
        patch_data2 = [
            ((500, 600), (128, 128), "area3"),
            ((700, 800), (256, 256), "area4"),
        ]
        patches2 = PatchesMock(
            patch_data=patch_data2,
            wsi_file=wsi_file2,
            level_or_mpp=2,
            check_location=False,
            class_name="SomeClass2",
            class_params={"param3": 3, "param4": 4},
        )
        patches_pool = [patches1, patches2]
        result_list = MemPatchExtractorPool._get_process_pool_data(patches_pool)
        output_list = [
            (
                "/tmp/wsi1.svs",
                [
                    ((100, 200), (256, 256), "area1"),
                    ((300, 400), (128, 128), "area2"),
                ],
                1,
                "SomeClass1",
                {"param1": 1, "param2": 2},
            ),
            (
                "/tmp/wsi2.svs",
                [
                    ((500, 600), (128, 128), "area3"),
                    ((700, 800), (256, 256), "area4"),
                ],
                2,
                "SomeClass2",
                {"param3": 3, "param4": 4},
            ),
        ]
        self.assertEqual(result_list, output_list)


class TestMemPatchExtractorPool(TestCase):
    """Tests for in memory extractor pool class."""

    def setUp(self):
        self.labels = ["color1", "color2", "color3", "color4"]
        self.wsi_tif_list = [
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color1.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color2.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color3.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color4.tif"),
        ]
        self.mask_file = make_test_path("mask/board-clean-mask-inner.npy")
        self.ref_patch_dir = make_test_path("ref_data/slides/patches/extract/pool")

    def test_pool_extractor_4proc_1thread(self):
        output_dir = make_test_path("saved_data/pool1a")
        [os.mkdir(os.path.join(output_dir, label)) for label in self.labels]
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MemPatchExtractorPool(
            patches_pool=patches_pool,
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 4)
        self.assertEqual(len(extractor_pool.patch_list), 160)
        self.assertEqual(extractor_pool.patch_count, 160)

        counter = 0
        for single_patch in extractor_pool.patch_list:
            counter += 1
            patch_image = single_patch[0]
            patch_label = single_patch[1]
            patch_file_name = "patch%s.tif" % counter
            patch_file_path = os.path.join(output_dir, patch_label, patch_file_name)
            patch_image.save(patch_file_path)

        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 40)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 40)
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

    def test_pool_extractor_2proc_2thread(self):
        output_dir = make_test_path("saved_data/pool1b")
        [os.mkdir(os.path.join(output_dir, label)) for label in self.labels]
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MemPatchExtractorPool(
            patches_pool=patches_pool,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 2)
        self.assertEqual(len(extractor_pool.patch_list), 160)
        self.assertEqual(extractor_pool.patch_count, 160)

        counter = 0
        for single_patch in extractor_pool.patch_list:
            counter += 1
            patch_image = single_patch[0]
            patch_label = single_patch[1]
            patch_file_name = "patch%s.tif" % counter
            patch_file_path = os.path.join(output_dir, patch_label, patch_file_name)
            patch_image.save(patch_file_path)

        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 40)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 40)
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

    def test_pool_extractor_wsi_downsampling(self):
        patches_pool = []
        patch_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0.3,  # must be MPP, not level
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patch_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        extractor_pool = MemPatchExtractorPool(
            patches_pool=patches_pool,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            resampling_mode="wsi",
        )
        self.assertEqual(len(extractor_pool.patch_list), patch_counter)
        self.assertEqual(extractor_pool.patch_count, patch_counter)

    def test_pool_extractor_tile_downsampling(self):
        patches_pool = []
        patch_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0.35,  # must be MPP, not level
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patch_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        extractor_pool = MemPatchExtractorPool(
            patches_pool=patches_pool,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            resampling_mode="tile",
        )
        self.assertEqual(len(extractor_pool.patch_list), patch_counter)
        self.assertEqual(extractor_pool.patch_count, patch_counter)

    def test_pool_extractor_default_downsampling(self):
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0.4,  # must be MPP, not level
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        with self.assertRaises(ValueError):
            MemPatchExtractorPool(
                patches_pool=patches_pool,
                thread_num_workers=2,
                proc_num_workers=2,
                thread_mp_chunksize=1,
                proc_mp_chunksize=1,
            )


class TestMultiResMemPatchExtractorPool(TestCase):
    """Tests for in memory multi resolution extractor pool class."""

    def setUp(self):
        self.labels = ["color1", "color2", "color3", "color4"]
        self.wsi_tif_list = [
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color1.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color2.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color3.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color4.tif"),
        ]
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        self.ref_patch_dir = make_test_path("ref_data/slides/patches/extract/pool_multires")

    def test_pool_extractor_4proc_1thread(self):
        output_dir = make_test_path("saved_data/pool2a")
        [os.mkdir(os.path.join(output_dir, label)) for label in self.labels]
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=patches_pool,
            levels_or_mpps=[2, 1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 4)
        self.assertEqual(len(extractor_pool.patch_list), 4)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

        counter = 0
        for single_multires_patch in extractor_pool.patch_list:
            for single_patch in single_multires_patch:
                counter += 1
                patch_image = single_patch[0]
                patch_label = single_patch[1]
                patch_file_name = "patch%s.tif" % counter
                patch_file_path = os.path.join(output_dir, patch_label, patch_file_name)
                patch_image.save(patch_file_path)

        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 3)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 3)
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

            # Test if parameters have been properly passed to MultiResMemPatchExtractor:
            #
            # levels_or_mpps - tested when counting extracted patches

    def test_pool_extractor_2proc_2thread(self):
        output_dir = make_test_path("saved_data/pool2b")
        [os.mkdir(os.path.join(output_dir, label)) for label in self.labels]
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                overlap_ratio=1,
                foreground_ratio=0.5,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=patches_pool,
            levels_or_mpps=[2, 1, 0],
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 2)
        self.assertEqual(len(extractor_pool.patch_list), 4)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

        counter = 0
        for single_multires_patch in extractor_pool.patch_list:
            for single_patch in single_multires_patch:
                counter += 1
                patch_image = single_patch[0]
                patch_label = single_patch[1]
                patch_file_name = "patch%s.tif" % counter
                patch_file_path = os.path.join(output_dir, patch_label, patch_file_name)
                patch_image.save(patch_file_path)

        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 3)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 3)
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

    def test_pool_extractor_wsi_downsampling(self):
        patches_pool = []
        patchset_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patchset_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=patches_pool,
            levels_or_mpps=[2, 0.6, 0.4],  # must be MPPs, not levels (mixed is fine)
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            resampling_mode="wsi",
        )
        self.assertEqual(len(extractor_pool.patch_list), patchset_counter)
        self.assertEqual(len(extractor_pool.patch_list[0]), 3)
        self.assertEqual(len(extractor_pool.patch_list[1]), 3)
        self.assertEqual(len(extractor_pool.patch_list[2]), 3)
        self.assertEqual(len(extractor_pool.patch_list[3]), 3)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

    def test_pool_extractor_tile_downsampling(self):
        patches_pool = []
        patchset_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patchset_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=patches_pool,
            levels_or_mpps=[2, 0.6, 0.4],  # must be MPPs, not levels (mixed is fine)
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            resampling_mode="tile",
        )
        self.assertEqual(len(extractor_pool.patch_list), patchset_counter)
        self.assertEqual(len(extractor_pool.patch_list[0]), 3)
        self.assertEqual(len(extractor_pool.patch_list[1]), 3)
        self.assertEqual(len(extractor_pool.patch_list[2]), 3)
        self.assertEqual(len(extractor_pool.patch_list[3]), 3)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

    def test_pool_extractor_default_downsampling(self):
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        with self.assertRaises(ValueError):
            MultiResMemPatchExtractorPool(
                patches_pool=patches_pool,
                levels_or_mpps=[2, 0.6, 0.4],  # must be MPPs, not levels (mixed is fine)
                thread_num_workers=1,
                proc_num_workers=4,
                thread_mp_chunksize=1,
                proc_mp_chunksize=1,
            )


class TestDiskPatchExtractorPool(TestCase):
    """Tests for disk extractor pool class."""

    def setUp(self):
        self.labels = ["color1", "color2", "color3", "color4"]
        self.wsi_tif_list = [
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color1.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color2.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color3.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color4.tif"),
        ]
        self.mask_file = make_test_path("mask/board-clean-mask-inner.npy")
        self.ref_patch_dir = make_test_path("ref_data/slides/patches/extract/pool")
        self.image_type = "tif"
        self.filename_comment = "dext"
        self.filename_separator = "___"

    def test_pool_extractor_4proc_1thread(self):
        output_dir = make_test_path("saved_data/pool3a")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = DiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=True,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 4)
        self.assertEqual(extractor_pool.patch_count, 160)

        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 40)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 40)
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

            # Test if parameters have been properly passed to DiskPatchExtractor:
            #
            # output dir - tested when counting extracted patches
            #
            # image_type - tested when counting extracted patches
            #
            # filename_comment
            #
            one_file_name = os.path.basename(result_patch_list[0])
            self.assertTrue(one_file_name.count(self.filename_comment) > 0)
            #
            # filename_separator
            #
            self.assertTrue(one_file_name.count(self.filename_separator) > 0)
            #
            # create_subdirs - tested when counting extracted patches (only True)

    def test_pool_extractor_2proc_2thread(self):
        output_dir = make_test_path("saved_data/pool3b")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = DiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=True,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 2)
        self.assertEqual(extractor_pool.patch_count, 160)

        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 40)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 40)
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

    def test_pool_extractor_2proc_2thread_no_subdirs(self):
        output_dir = make_test_path("saved_data/pool3c")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = DiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 2)
        self.assertEqual(extractor_pool.patch_count, 160)

        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 160)
        # calculate checksums for reference patches
        ref_patch_path = os.path.join(self.ref_patch_dir, "**", "*" + "*.tif")
        ref_patch_list = glob.glob(ref_patch_path, recursive=True)
        self.assertEqual(len(ref_patch_list), 160)
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

    def test_pool_extractor_wsi_downsampling(self):
        output_dir = make_test_path("saved_data/pool6a")
        patches_pool = []
        patch_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0.37,  # must be MPP, not level
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patch_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        DiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=True,
            resampling_mode="wsi",
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), patch_counter)

    def test_pool_extractor_tile_downsampling(self):
        output_dir = make_test_path("saved_data/pool6b")
        patches_pool = []
        patch_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0.32,  # must be MPP, not level
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patch_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        DiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=True,
            resampling_mode="tile",
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), patch_counter)

    def test_pool_extractor_default_downsampling(self):
        output_dir = make_test_path("saved_data/pool6c")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=256,
                level_or_mpp=0.345,  # must be MPP, not level
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        with self.assertRaises(ValueError):
            DiskPatchExtractorPool(
                patches_pool=patches_pool,
                output_dir=output_dir,
                image_type=self.image_type,
                thread_num_workers=1,
                proc_num_workers=4,
                thread_mp_chunksize=1,
                proc_mp_chunksize=1,
                filename_comment=self.filename_comment,
                filename_separator=self.filename_separator,
                create_subdirs=True,
            )


class TestMultiResDiskPatchExtractorPool(TestCase):
    """Tests for disk multi resolution extractor pool class."""

    def setUp(self):
        MultiResDiskPatchExtractorPool.reset_global_counter()
        self.labels = ["color1", "color2", "color3", "color4"]
        self.wsi_tif_list = [
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color1.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color2.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color3.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color4.tif"),
        ]
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        self.ref_patch_dir = make_test_path("ref_data/slides/patches/extract/pool_multires")
        self.image_type = "tif"
        self.filename_comment = "dext"
        self.filename_separator = "___"

    def test_pool_extractor_4proc_1thread(self):
        output_dir = make_test_path("saved_data/pool4a")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[2, 1, 0],
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=True,
            global_counter=200002,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 4)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

        global_counter_set_dirs_output = {"set200002", "set200003", "set200004", "set200005"}
        global_counter_set_dirs_result = set()
        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 3)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 3)
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

            # test if parameters have been properly passed to MultiResDiskPatchExtractor
            #
            # output dir - tested when counting extracted patches
            #
            # levels_or_mpps - tested when counting extracted patches
            #
            # image_type - tested when counting extracted patches
            #
            # filename_comment
            #
            one_file_path = result_patch_list[0]
            one_file_name = os.path.basename(one_file_path)
            self.assertTrue(one_file_name.count(self.filename_comment) > 0)
            #
            # filename_separator
            #
            self.assertTrue(one_file_name.count(self.filename_separator) > 0)
            #
            # create_subdirs - tested when counting extracted patches (only True)
            #
            # global_counter
            #
            global_counter_set_dir = one_file_path.split("/")[-2]
            global_counter_set_dirs_result.add(global_counter_set_dir)

        self.assertTrue(global_counter_set_dirs_output == global_counter_set_dirs_result)

    def test_pool_extractor_2proc_2thread(self):
        output_dir = make_test_path("saved_data/pool4b")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[2, 1, 0],
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=True,
            global_counter=100001,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 2)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

        global_counter_set_dirs_output = {"set100001", "set100002", "set100003", "set100004"}
        global_counter_set_dirs_result = set()
        for label in self.labels:
            # count extracted files
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 3)
            # calculate checksums for reference patches
            ref_patch_path = os.path.join(self.ref_patch_dir, label, "**", "*" + "*.tif")
            ref_patch_list = glob.glob(ref_patch_path, recursive=True)
            self.assertEqual(len(ref_patch_list), 3)
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
            # test global_counter again
            #
            one_file_path = result_patch_list[0]
            global_counter_set_dir = one_file_path.split("/")[-2]
            global_counter_set_dirs_result.add(global_counter_set_dir)

        self.assertTrue(global_counter_set_dirs_output == global_counter_set_dirs_result)

    def test_pool_extractor_2proc_2thread_no_subdirs(self):
        output_dir = make_test_path("saved_data/pool4c")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[2, 1, 0],
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
            global_counter=300003,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 2)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

        global_counter_set_dirs_output = {"set300003", "set300004", "set300005", "set300006"}
        global_counter_set_dirs_result = set()
        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 12)
        # calculate checksums for reference patches
        ref_patch_path = os.path.join(self.ref_patch_dir, "**", "*" + "*.tif")
        ref_patch_list = glob.glob(ref_patch_path, recursive=True)
        self.assertEqual(len(ref_patch_list), 12)
        ref_patch_md5_list = []
        for patch_file in ref_patch_list:
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                ref_patch_md5_list.append(patch_checksum)
        # examine extracted patches
        for patch_file in result_patch_list:
            global_counter_set_dir = patch_file.split("/")[-2]
            global_counter_set_dirs_result.add(global_counter_set_dir)
            with open(patch_file, "rb") as file_img:
                patch_checksum = md5(file_img.read()).hexdigest()
                self.assertTrue(patch_checksum in ref_patch_md5_list, patch_checksum)
        # test global_counter again
        self.assertTrue(global_counter_set_dirs_output == global_counter_set_dirs_result)

    def test_pool_extractor_2proc_2thread_no_subdirs_no_counter(self):
        MultiResDiskPatchExtractorPool.remove_global_counter()
        output_dir = make_test_path("saved_data/pool4d")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[2, 1, 0],
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
            global_counter=None,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 2)
        self.assertEqual(extractor_pool.patch_count, 4 * 3)
        self.assertEqual(extractor_pool.patchset_count, 4)

        self.assertTrue(os.path.exists(os.path.join(output_dir, "board-multi-layer-no-compression-mpp-color1__set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "board-multi-layer-no-compression-mpp-color2__set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "board-multi-layer-no-compression-mpp-color3__set1")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "board-multi-layer-no-compression-mpp-color4__set1")))

    def test_pool_extractor_8proc_1thread_no_subdirs_many_sets(self):
        """Test if there is no shared variable conflict when using more cores and more sets."""
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
        mask_file = make_test_path("mask/board-clean-mask-inner.npy")
        labels = ["color1", "color1", "color2", "color2", "color3", "color3", "color4", "color4"]
        output_dir = make_test_path("saved_data/pool4e")
        patches_pool = []
        for index, wsi_file in enumerate(wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=mask_file,
                patch_size=64,
                level_or_mpp=2,
                patch_stride=0.25,
                foreground_ratio=0.25,
                overlap_ratio=1,
                weak_label=labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[2, 1],
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=8,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
            global_counter=400004,
        )
        self.assertEqual(len(set(extractor_pool.pids)), 8)
        self.assertEqual(extractor_pool.patch_count, 35 * 8 * 2)
        self.assertEqual(extractor_pool.patchset_count, 35 * 8)

        global_counter_set_dirs_output = {"set" + str(x) for x in range(400004, 400004 + 35 * 8)}
        global_counter_set_dirs_result = set()
        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 35 * 8 * 2)
        # examine extracted patches (set dir only)
        for patch_file in result_patch_list:
            global_counter_set_dir = patch_file.split("/")[-2]
            global_counter_set_dirs_result.add(global_counter_set_dir)
        # test global_counter again
        self.assertTrue(global_counter_set_dirs_output == global_counter_set_dirs_result)

    def test_pool_extractor_wsi_downsampling(self):
        output_dir = make_test_path("saved_data/pool7a")
        patches_pool = []
        patchset_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patchset_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[0.36, 0.42, 0.54],  # must be MPPs, not levels
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
            resampling_mode="wsi",
            global_counter=None,
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), patchset_counter * 3)
        self.assertEqual(extractor_pool.patch_count, patchset_counter * 3)
        self.assertEqual(extractor_pool.patchset_count, patchset_counter)

    def test_pool_extractor_tile_downsampling(self):
        output_dir = make_test_path("saved_data/pool7b")
        patches_pool = []
        patchset_counter = 0
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patchset_counter += grid_patches.patch_count
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[0.38, 0.44, 0.57],  # must be MPPs, not levels
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
            resampling_mode="tile",
            global_counter=None,
        )
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), patchset_counter * 3)
        self.assertEqual(extractor_pool.patch_count, patchset_counter * 3)
        self.assertEqual(extractor_pool.patchset_count, patchset_counter)

    def test_pool_extractor_default_downsampling(self):
        output_dir = make_test_path("saved_data/pool7c")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        with self.assertRaises(ValueError):
            MultiResDiskPatchExtractorPool(
                patches_pool=patches_pool,
                output_dir=output_dir,
                levels_or_mpps=[0.39, 0.45, 0.58],  # must be MPPs, not levels
                image_type=self.image_type,
                thread_num_workers=2,
                proc_num_workers=2,
                thread_mp_chunksize=1,
                proc_mp_chunksize=1,
                filename_comment=self.filename_comment,
                filename_separator=self.filename_separator,
                create_subdirs=False,
                global_counter=None,
            )

    def test_pool_extractor_patchset_count_fail(self):
        output_dir = make_test_path("saved_data/pool4f")
        patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            patches_pool.append(grid_patches)

        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=patches_pool,
            output_dir=output_dir,
            levels_or_mpps=[2, 1, 0],
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=2,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=True,
            global_counter=100,
        )
        extractor_pool._patch_count = 10
        with self.assertRaises(ValueError):
            _ = extractor_pool.patchset_count


class TestDiskMemExtractorPoolsChunksize(TestCase):
    """Tests for multiprocessing chunksize argument in all relevant extractor pool classes."""

    def setUp(self):
        self.labels = ["color1", "color2", "color3", "color4"]
        self.wsi_tif_list = [
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color1.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color2.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color3.tif"),
            make_test_path("wsi/pool/board-multi-layer-no-compression-mpp-color4.tif"),
        ]
        self.mask_file = make_test_path("mask/board-clean-mask.npy")
        self.output_dir = make_test_path("saved_data/pool5")
        self.image_type = "tif"
        self.filename_comment = "dext"
        self.filename_separator = "___"

        self.patches_pool = []
        for index, wsi_file in enumerate(self.wsi_tif_list):
            grid_patches = WholeImageGridPatches(
                wsi_file=wsi_file,
                mask_data=self.mask_file,
                patch_size=128,
                level_or_mpp=2,
                patch_stride=1,
                foreground_ratio=0.5,
                overlap_ratio=1,
                weak_label=self.labels[index],
            )
            self.patches_pool.append(grid_patches)

    def test_class1a_chunksize_int(self):
        """Test MemPatchExtractorPool."""
        extractor_pool = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=1,
            proc_mp_chunksize=2,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class1b_chunksize_none(self):
        """Test MemPatchExtractorPool."""
        with self.assertRaises(TypeError):
            MemPatchExtractorPool(
                patches_pool=self.patches_pool,
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=1,
                proc_mp_chunksize=None,
            )

    def test_class1c_chunksize_int(self):
        """Test MemPatchExtractorPool."""
        extractor_pool = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=2,
            proc_mp_chunksize=1,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class1d_chunksize_none(self):
        """Test MemPatchExtractorPool."""
        with self.assertRaises(TypeError):
            MemPatchExtractorPool(
                patches_pool=self.patches_pool,
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=None,
                proc_mp_chunksize=1,
            )

    def test_class2a_chunksize_int(self):
        """Test MultiResMemPatchExtractorPool."""
        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=self.patches_pool,
            levels_or_mpps=[2, 1, 0],
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=1,
            proc_mp_chunksize=2,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class2b_chunksize_none(self):
        """Test MultiResMemPatchExtractorPool."""
        with self.assertRaises(TypeError):
            MultiResMemPatchExtractorPool(
                patches_pool=self.patches_pool,
                levels_or_mpps=[2, 1, 0],
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=1,
                proc_mp_chunksize=None,
            )

    def test_class2c_chunksize_int(self):
        """Test MultiResMemPatchExtractorPool."""
        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=self.patches_pool,
            levels_or_mpps=[2, 1, 0],
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=2,
            proc_mp_chunksize=1,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class2d_chunksize_none(self):
        """Test MultiResMemPatchExtractorPool."""
        with self.assertRaises(TypeError):
            MultiResMemPatchExtractorPool(
                patches_pool=self.patches_pool,
                levels_or_mpps=[2, 1, 0],
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=None,
                proc_mp_chunksize=1,
            )

    def test_class3a_chunksize_int(self):
        """Test DiskPatchExtractorPool."""
        extractor_pool = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=self.output_dir,
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=1,
            proc_mp_chunksize=2,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class3b_chunksize_none(self):
        """Test DiskPatchExtractorPool."""
        with self.assertRaises(TypeError):
            DiskPatchExtractorPool(
                patches_pool=self.patches_pool,
                output_dir=self.output_dir,
                image_type=self.image_type,
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=1,
                proc_mp_chunksize=None,
                filename_comment=self.filename_comment,
                filename_separator=self.filename_separator,
                create_subdirs=False,
            )

    def test_class3c_chunksize_int(self):
        """Test DiskPatchExtractorPool."""
        extractor_pool = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=self.output_dir,
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=2,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class3d_chunksize_none(self):
        """Test DiskPatchExtractorPool."""
        with self.assertRaises(TypeError):
            DiskPatchExtractorPool(
                patches_pool=self.patches_pool,
                output_dir=self.output_dir,
                image_type=self.image_type,
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=None,
                proc_mp_chunksize=1,
                filename_comment=self.filename_comment,
                filename_separator=self.filename_separator,
                create_subdirs=False,
            )

    def test_class4a_chunksize_int(self):
        """Test MultiResDiskPatchExtractorPool."""
        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=self.output_dir,
            levels_or_mpps=[2, 1, 0],
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=1,
            proc_mp_chunksize=2,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
            global_counter=1,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class4b_chunksize_none(self):
        """Test MultiResDiskPatchExtractorPool."""
        with self.assertRaises(TypeError):
            MultiResDiskPatchExtractorPool(
                patches_pool=self.patches_pool,
                output_dir=self.output_dir,
                levels_or_mpps=[2, 1, 0],
                image_type=self.image_type,
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=1,
                proc_mp_chunksize=None,
                filename_comment=self.filename_comment,
                filename_separator=self.filename_separator,
                create_subdirs=False,
                global_counter=2,
            )

    def test_class4c_chunksize_int(self):
        """Test MultiResDiskPatchExtractorPool."""
        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=self.output_dir,
            levels_or_mpps=[2, 1, 0],
            image_type=self.image_type,
            thread_num_workers=1,
            proc_num_workers=1,
            thread_mp_chunksize=2,
            proc_mp_chunksize=1,
            filename_comment=self.filename_comment,
            filename_separator=self.filename_separator,
            create_subdirs=False,
            global_counter=3,
        )
        self.assertIsNotNone(extractor_pool)

    def test_class4d_chunksize_none(self):
        """Test MultiResDiskPatchExtractorPool."""
        with self.assertRaises(TypeError):
            MultiResDiskPatchExtractorPool(
                patches_pool=self.patches_pool,
                output_dir=self.output_dir,
                levels_or_mpps=[2, 1, 0],
                image_type=self.image_type,
                thread_num_workers=1,
                proc_num_workers=1,
                thread_mp_chunksize=None,
                proc_mp_chunksize=1,
                filename_comment=self.filename_comment,
                filename_separator=self.filename_separator,
                create_subdirs=False,
                global_counter=4,
            )


class TestPatchExtractorPoolDifferentPatchesClasses(TestCase):
    """Tests for different patches classes and mocks inside pool extractors."""

    def setUp(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_file = make_test_path("mask/board-clean-mask-inner.npy")
        self.image_type = "tif"
        self.grid_patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=2,
            foreground_ratio=0.88,
            overlap_ratio=0.77,
            weak_label="label1",
        )
        self.random_patches = WholeImageRandomPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            num_patches=10,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
            weak_label="label2",
        )
        self.poisson_patches = WholeImagePoissonDiskPatches(
            wsi_file=wsi_file_tif,
            mask_data=mask_file,
            patch_size=256,
            level_or_mpp=0,
            poisson_spacing=20,
            foreground_ratio=0.2,
            overlap_ratio=0.8,
            weak_label="label3",
        )
        self.patches_pool = [self.grid_patches, self.random_patches, self.poisson_patches]
        patches_mock1 = PatchesMock(
            patch_data=self.grid_patches.patch_data,
            wsi_file=self.grid_patches.wsi_file,
            level_or_mpp=self.grid_patches.level_or_mpp,
            check_location=False,
        )
        patches_mock2 = PatchesMock(
            patch_data=self.random_patches.patch_data,
            wsi_file=self.random_patches.wsi_file,
            level_or_mpp=self.random_patches.level_or_mpp,
            check_location=False,
        )
        patches_mock3 = PatchesMock(
            patch_data=self.poisson_patches.patch_data,
            wsi_file=self.poisson_patches.wsi_file,
            level_or_mpp=self.poisson_patches.level_or_mpp,
            check_location=False,
        )
        self.patches_mock_pool = [patches_mock1, patches_mock2, patches_mock3]

    def test_three_patches_classes_manifest_files(self):
        output_dir = make_test_path("saved_data/pool8a")
        extractor_pool = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=4,
            proc_num_workers=1,
            create_subdirs=True,
        )
        # test manifest IDs
        self.assertEqual(len(set(extractor_pool.manifest_ids)), 3)
        self.assertEqual(len(extractor_pool.manifest_ids[0]), 8)
        self.assertEqual(len(extractor_pool.manifest_ids[1]), 8)
        self.assertEqual(len(extractor_pool.manifest_ids[2]), 8)
        # test manifest file values specific to patches classes
        # 1.
        manifest_id = extractor_pool.manifest_ids[0]
        manifest_path = os.path.join(output_dir, "**", "*" + manifest_id + "*.txt")
        manifest_file_list = glob.glob(manifest_path, recursive=True)
        self.assertEqual(len(manifest_file_list), 1)
        with open(manifest_file_list[0]) as f:
            all_lines = f.readlines()
        patches_class = all_lines[5].split(":")[1].strip()
        patches_specific_param = all_lines[12].split(":")[0].strip()
        self.assertEqual(patches_class, "WholeImageGridPatches")
        self.assertEqual(patches_specific_param, "patch_stride")
        # 2.
        manifest_id = extractor_pool.manifest_ids[1]
        manifest_path = os.path.join(output_dir, "**", "*" + manifest_id + "*.txt")
        manifest_file_list = glob.glob(manifest_path, recursive=True)
        self.assertEqual(len(manifest_file_list), 1)
        with open(manifest_file_list[0]) as f:
            all_lines = f.readlines()
        patches_class = all_lines[5].split(":")[1].strip()
        patches_specific_param = all_lines[10].split(":")[0].strip()
        self.assertEqual(patches_class, "WholeImageRandomPatches")
        self.assertEqual(patches_specific_param, "num_patches")
        # 3.
        manifest_id = extractor_pool.manifest_ids[2]
        manifest_path = os.path.join(output_dir, "**", "*" + manifest_id + "*.txt")
        manifest_file_list = glob.glob(manifest_path, recursive=True)
        self.assertEqual(len(manifest_file_list), 1)
        with open(manifest_file_list[0]) as f:
            all_lines = f.readlines()
        patches_class = all_lines[5].split(":")[1].strip()
        patches_specific_param = all_lines[12].split(":")[0].strip()
        self.assertEqual(patches_class, "WholeImagePoissonDiskPatches")
        self.assertEqual(patches_specific_param, "poisson_spacing")

    def test_three_patches_classes_patch_count(self):
        # 1. disk extraction
        output_dir = make_test_path("saved_data/pool8b")
        extractor_pool1 = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=4,
            create_subdirs=True,
        )
        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        disk_patch_list = glob.glob(count_path, recursive=True)
        # 2. memory extraction
        extractor_pool2 = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=4,
            proc_num_workers=2,
        )
        mem_patch_list = list(extractor_pool2.patch_list)
        # 3. compare counts
        self.assertEqual(len(disk_patch_list), len(mem_patch_list))
        # 4. compare counts as property
        self.assertEqual(extractor_pool1.patch_count, extractor_pool2.patch_count)

    def test_three_patches_classes_patch_count_with_two_mocks(self):
        # 1. disk extraction
        output_dir = make_test_path("saved_data/pool8c")
        extractor_pool1 = DiskPatchExtractorPool(
            patches_pool=self.patches_mock_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=4,
            create_subdirs=True,
        )
        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        disk_patch_list = glob.glob(count_path, recursive=True)
        # 2. memory extraction
        extractor_pool2 = MemPatchExtractorPool(
            patches_pool=self.patches_mock_pool,
            thread_num_workers=4,
            proc_num_workers=2,
        )
        mem_patch_list = list(extractor_pool2.patch_list)
        # 3. compare counts
        self.assertEqual(len(disk_patch_list), len(mem_patch_list))
        # 4. compare counts as property
        self.assertEqual(extractor_pool1.patch_count, extractor_pool2.patch_count)

    def test_three_patches_classes_patch_count_with_one_mock(self):
        # 1. disk extraction
        output_dir = make_test_path("saved_data/pool8d")
        extractor_pool1 = DiskPatchExtractorPool(
            patches_pool=self.patches_mock_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=4,
            create_subdirs=True,
        )
        # count extracted files
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        disk_patch_list = glob.glob(count_path, recursive=True)
        # 2. memory extraction
        extractor_pool2 = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=4,
            proc_num_workers=2,
        )
        mem_patch_list = list(extractor_pool2.patch_list)
        # 3. compare counts
        self.assertEqual(len(disk_patch_list), len(mem_patch_list))
        # 4. compare counts as property
        self.assertEqual(extractor_pool1.patch_count, extractor_pool2.patch_count)

    def test_three_patches_classes_manifest_files_with_one_mock(self):
        def compare_manifests(manifest1_id, manifest2_id, output_dir1, output_dir2):
            manifest_path = os.path.join(output_dir1, "**", "*" + manifest1_id + "*.txt")
            manifest_file_list = glob.glob(manifest_path, recursive=True)
            self.assertEqual(len(manifest_file_list), 1)
            with open(manifest_file_list[0]) as f:
                manifest1_lines = f.readlines()
            manifest1_lines = [line.replace(manifest1_id, "0123456789") for line in manifest1_lines]
            manifest1_lines = [line.replace(output_dir1, "output_dir") for line in manifest1_lines]
            #
            manifest_path = os.path.join(output_dir2, "**", "*" + manifest2_id + "*.txt")
            manifest_file_list = glob.glob(manifest_path, recursive=True)
            self.assertEqual(len(manifest_file_list), 1)
            with open(manifest_file_list[0]) as f:
                manifest2_lines = f.readlines()
            manifest2_lines = [line.replace(manifest2_id, "0123456789") for line in manifest2_lines]
            manifest2_lines = [line.replace(output_dir2, "output_dir") for line in manifest2_lines]
            # some lines will not match perfectly
            self.assertEqual(manifest1_lines[1][:24], manifest2_lines[1][:24])
            self.assertEqual(manifest1_lines[2][:24], manifest2_lines[2][:24])
            self.assertEqual(sorted(manifest1_lines[3:]), sorted(manifest2_lines[3:]))

        patches_mock1 = PatchesMock(
            patch_data=self.grid_patches.patch_data,
            wsi_file=self.grid_patches.wsi_file,
            level_or_mpp=self.grid_patches.level_or_mpp,
            check_location=False,
            class_name=self.grid_patches.class_name,
            class_params=self.grid_patches.param_info,
        )
        patches_mock2 = PatchesMock(
            patch_data=self.random_patches.patch_data,
            wsi_file=self.random_patches.wsi_file,
            level_or_mpp=self.random_patches.level_or_mpp,
            check_location=False,
            class_name=self.random_patches.class_name,
            class_params=self.random_patches.param_info,
        )
        patches_mock3 = PatchesMock(
            patch_data=self.poisson_patches.patch_data,
            wsi_file=self.poisson_patches.wsi_file,
            level_or_mpp=self.poisson_patches.level_or_mpp,
            check_location=False,
            class_name=self.poisson_patches.class_name,
            class_params=self.poisson_patches.param_info,
        )
        patches_mock_pool = [patches_mock1, patches_mock2, patches_mock3]
        # 1. disk extraction with mocks
        output_dir1 = make_test_path("saved_data/pool8e")
        extractor_pool1 = DiskPatchExtractorPool(
            patches_pool=patches_mock_pool,
            output_dir=output_dir1,
            image_type=self.image_type,
            thread_num_workers=4,
            proc_num_workers=1,
            create_subdirs=True,
        )
        manifest1_ids = extractor_pool1.manifest_ids
        # count extracted files
        count_path = os.path.join(output_dir1, "**", "*" + "*.tif")
        disk1_patch_list = glob.glob(count_path, recursive=True)
        # 2. disk extraction without mocks
        output_dir2 = make_test_path("saved_data/pool8f")
        extractor_pool2 = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir2,
            image_type=self.image_type,
            thread_num_workers=4,
            proc_num_workers=1,
            create_subdirs=True,
        )
        manifest2_ids = extractor_pool2.manifest_ids
        # count extracted files
        count_path = os.path.join(output_dir2, "**", "*" + "*.tif")
        disk2_patch_list = glob.glob(count_path, recursive=True)
        # 3. compare counts
        self.assertEqual(len(disk1_patch_list), len(disk2_patch_list))
        # 4. compare counts as property
        self.assertEqual(extractor_pool1.patch_count, extractor_pool2.patch_count)
        # 5. compare content
        compare_manifests(manifest1_ids[0], manifest2_ids[0], output_dir1, output_dir2)
        compare_manifests(manifest1_ids[1], manifest2_ids[1], output_dir1, output_dir2)
        compare_manifests(manifest1_ids[2], manifest2_ids[2], output_dir1, output_dir2)


class TestPatchExtractorPoolsIncludedExcludedLabels(TestCase):
    """Tests for included/excluded labels in pool extractors."""

    def setUp(self):
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
        mask_file = make_test_path("mask/board-clean-mask-inner.npy")
        labels = ["color1", "color1", "color2", "color2", "color3", "color3", "color4", "color4"]
        self.image_type = "tif"
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 0,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "weak_label": labels,
        }
        mask_data_list = [mask_file] * 8
        self.patches_pool = WholeImageGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )

    def test_mem_patch_extractor_pool_default(self):
        extractor_pool = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
        )
        self.assertEqual(len(extractor_pool.patch_list), self.patches_pool.patch_count)
        result_labels = sorted(set([patch[1] for patch in extractor_pool.patch_list]))
        self.assertEqual(result_labels, sorted(set(["color1", "color2", "color3", "color4"])))
        counts_color1 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color1"])
        counts_color2 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color2"])
        counts_color3 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color3"])
        counts_color4 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color4"])
        self.assertEqual(counts_color1, 80)
        self.assertEqual(counts_color2, 80)
        self.assertEqual(counts_color3, 80)
        self.assertEqual(counts_color4, 80)
        self.assertEqual(extractor_pool.patch_count, 80 * 4)

    def test_mem_patch_extractor_pool_include(self):
        included_labels = ["color1", "color3", "color2"]
        extractor_pool = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=2,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            included_labels=included_labels,
        )
        self.assertEqual(len(extractor_pool.patch_list), 240)
        result_labels = sorted(set([patch[1] for patch in extractor_pool.patch_list]))
        self.assertEqual(result_labels, sorted(set(["color1", "color3", "color2"])))
        counts_color1 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color1"])
        counts_color2 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color2"])
        counts_color3 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color3"])
        counts_color4 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color4"])
        self.assertEqual(counts_color1, 80)
        self.assertEqual(counts_color2, 80)
        self.assertEqual(counts_color3, 80)
        self.assertEqual(counts_color4, 0)
        self.assertEqual(extractor_pool.patch_count, 80 * 3)

    def test_mem_patch_extractor_pool_exclude(self):
        excluded_labels = ["color1", "color4"]
        extractor_pool = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=2,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            excluded_labels=excluded_labels,
        )
        self.assertEqual(len(extractor_pool.patch_list), 160)
        result_labels = sorted(set([patch[1] for patch in extractor_pool.patch_list]))
        self.assertEqual(result_labels, sorted(set(["color2", "color3"])))
        counts_color1 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color1"])
        counts_color2 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color2"])
        counts_color3 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color3"])
        counts_color4 = len([patch for patch in extractor_pool.patch_list if patch[1] == "color4"])
        self.assertEqual(counts_color1, 0)
        self.assertEqual(counts_color2, 80)
        self.assertEqual(counts_color3, 80)
        self.assertEqual(counts_color4, 0)
        self.assertEqual(extractor_pool.patch_count, 80 * 2)

    def test_mem_patch_extractor_pool_exclude_all(self):
        excluded_labels = ["color1", "color2", "color3", "color4"]
        extractor_pool = MemPatchExtractorPool(
            patches_pool=self.patches_pool,
            thread_num_workers=2,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            excluded_labels=excluded_labels,
        )
        self.assertEqual(len(extractor_pool.patch_list), 0)
        self.assertEqual(extractor_pool.patch_count, 0)

    def test_disk_patch_extractor_pool_default(self):
        output_dir = make_test_path("saved_data/pool9a")
        extractor_pool = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=4,
            create_subdirs=True,
        )
        # count extracted files by labels
        for label in ["color1", "color2", "color3", "color4"]:
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 80)
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), self.patches_pool.patch_count)
        self.assertEqual(extractor_pool.patch_count, self.patches_pool.patch_count)

    def test_disk_patch_extractor_pool_include(self):
        output_dir = make_test_path("saved_data/pool9b")
        included_labels = ["color2", "color4"]
        extractor_pool = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=4,
            create_subdirs=True,
            included_labels=included_labels,
        )
        # count extracted files by labels
        for label in included_labels:
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 80)
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 80 * 2)
        # count as property
        self.assertEqual(extractor_pool.patch_count, 80 * 2)

    def test_disk_patch_extractor_pool_exclude(self):
        output_dir = make_test_path("saved_data/pool9c")
        excluded_labels = ["color2", "color3"]
        extractor_pool = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=4,
            create_subdirs=True,
            excluded_labels=excluded_labels,
        )
        # count extracted files by labels
        for label in ["color1", "color4"]:
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 80)
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 80 * 2)
        # count as property
        self.assertEqual(extractor_pool.patch_count, 80 * 2)

    def test_disk_patch_extractor_pool_exclude_all(self):
        output_dir = make_test_path("saved_data/pool9g")
        excluded_labels = ["color1", "color2", "color3", "color4"]
        extractor_pool = DiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            thread_num_workers=2,
            proc_num_workers=4,
            create_subdirs=True,
            excluded_labels=excluded_labels,
        )
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 0)
        # count as property
        self.assertEqual(extractor_pool.patch_count, 0)


class TestMultiResPatchExtractorPoolsIncludedExcludedLabels(TestCase):
    """Tests for included/excluded labels in multi resolution pool extractors."""

    def setUp(self):
        MultiResDiskPatchExtractorPool.reset_global_counter()
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
        mask_file = make_test_path("mask/board-clean-mask-inner.npy")
        labels = ["color1", "color1", "color2", "color2", "color3", "color3", "color4", "color4"]
        self.image_type = "tif"
        patches_args = {
            "patch_size": 256,
            "foreground_ratio": 0.1,
            "overlap_ratio": 0.9,
            "level_or_mpp": 1,
            "patch_stride": 1,
            "polygon_buffer": 0,
            "weak_label": labels,
        }
        mask_data_list = [mask_file] * 8
        self.patches_pool = WholeImageGridPatchesPool(
            wsi_file_list=wsi_tif_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=4,
        )

    def test_multires_mem_patch_extractor_pool_default(self):
        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=self.patches_pool,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
        )
        self.assertEqual(len(extractor_pool.patch_list), self.patches_pool.patch_count)
        result_labels0 = sorted(set([patches[0][1] for patches in extractor_pool.patch_list]))
        self.assertEqual(result_labels0, sorted(set(["color1", "color2", "color3", "color4"])))
        result_labels1 = sorted(set([patches[1][1] for patches in extractor_pool.patch_list]))
        self.assertEqual(result_labels1, sorted(set(["color1", "color2", "color3", "color4"])))
        counts_color1 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color1"])
        counts_color2 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color2"])
        counts_color3 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color3"])
        counts_color4 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color4"])
        self.assertEqual(counts_color1, 8)
        self.assertEqual(counts_color2, 8)
        self.assertEqual(counts_color3, 8)
        self.assertEqual(counts_color4, 8)
        self.assertEqual(extractor_pool.patch_count, 8 * 4 * 2)
        self.assertEqual(extractor_pool.patchset_count, 8 * 4)

    def test_multires_mem_patch_extractor_pool_include(self):
        included_labels = ["color4"]
        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=self.patches_pool,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            included_labels=included_labels,
        )
        self.assertEqual(len(extractor_pool.patch_list), 8)
        result_labels0 = sorted(set([patches[0][1] for patches in extractor_pool.patch_list]))
        self.assertEqual(result_labels0, sorted(set(["color4"])))
        result_labels1 = sorted(set([patches[1][1] for patches in extractor_pool.patch_list]))
        self.assertEqual(result_labels1, sorted(set(["color4"])))
        counts_color1 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color1"])
        counts_color2 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color2"])
        counts_color3 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color3"])
        counts_color4 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color4"])
        self.assertEqual(counts_color1, 0)
        self.assertEqual(counts_color2, 0)
        self.assertEqual(counts_color3, 0)
        self.assertEqual(counts_color4, 8)
        self.assertEqual(extractor_pool.patch_count, 8 * 1 * 2)
        self.assertEqual(extractor_pool.patchset_count, 8)

    def test_multires_mem_patch_extractor_pool_exclude(self):
        excluded_labels = ["color3"]
        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=self.patches_pool,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            excluded_labels=excluded_labels,
        )
        self.assertEqual(len(extractor_pool.patch_list), 8 * 3)
        result_labels0 = sorted(set([patches[0][1] for patches in extractor_pool.patch_list]))
        self.assertEqual(result_labels0, sorted(set(["color1", "color2", "color4"])))
        result_labels1 = sorted(set([patches[1][1] for patches in extractor_pool.patch_list]))
        self.assertEqual(result_labels1, sorted(set(["color1", "color2", "color4"])))
        counts_color1 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color1"])
        counts_color2 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color2"])
        counts_color3 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color3"])
        counts_color4 = len([patch for patch in extractor_pool.patch_list if patch[0][1] == "color4"])
        self.assertEqual(counts_color1, 8)
        self.assertEqual(counts_color2, 8)
        self.assertEqual(counts_color3, 0)
        self.assertEqual(counts_color4, 8)
        self.assertEqual(extractor_pool.patch_count, 8 * 3 * 2)
        self.assertEqual(extractor_pool.patchset_count, 8 * 3)

    def test_multires_mem_patch_extractor_pool_exclude_all(self):
        excluded_labels = ["color1", "color2", "color3", "color4"]
        extractor_pool = MultiResMemPatchExtractorPool(
            patches_pool=self.patches_pool,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            excluded_labels=excluded_labels,
        )
        self.assertEqual(len(extractor_pool.patch_list), 0)
        self.assertEqual(extractor_pool.patch_count, 0)
        self.assertEqual(extractor_pool.patchset_count, 0)

    def test_multires_disk_patch_extractor_pool_default(self):
        output_dir = make_test_path("saved_data/pool9d")
        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            create_subdirs=True,
            global_counter=1,
        )
        # count extracted files by labels
        for label in ["color1", "color2", "color3", "color4"]:
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 8 * 2)
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), self.patches_pool.patch_count * 2)
        # counts as properties
        self.assertEqual(extractor_pool.patch_count, self.patches_pool.patch_count * 2)
        self.assertEqual(extractor_pool.patchset_count, self.patches_pool.patch_count)

    def test_multires_disk_patch_extractor_pool_include(self):
        output_dir = make_test_path("saved_data/pool9e")
        included_labels = ["color1", "color4"]
        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            included_labels=included_labels,
            create_subdirs=True,
            global_counter=1,
        )
        # count extracted files by labels
        for label in ["color4", "color1"]:
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 8 * 2)
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 8 * 2 * 2)
        # counts as properties
        self.assertEqual(extractor_pool.patch_count, 8 * 2 * 2)
        self.assertEqual(extractor_pool.patchset_count, 8 * 2)

    def test_multires_disk_patch_extractor_pool_exclude(self):
        output_dir = make_test_path("saved_data/pool9f")
        excluded_labels = ["color4", "color1", "color3"]
        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            excluded_labels=excluded_labels,
            create_subdirs=True,
            global_counter=1,
        )
        # count extracted files by labels
        for label in ["color2"]:
            count_path = os.path.join(output_dir, label, "**", "*" + "*.tif")
            result_patch_list = glob.glob(count_path, recursive=True)
            self.assertEqual(len(result_patch_list), 8 * 2)
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 8 * 2)
        # counts as properties
        self.assertEqual(extractor_pool.patch_count, 8 * 2)
        self.assertEqual(extractor_pool.patchset_count, 8)

    def test_multires_disk_patch_extractor_pool_exclude_all(self):
        output_dir = make_test_path("saved_data/pool9h")
        excluded_labels = ["color1", "color2", "color3", "color4"]
        extractor_pool = MultiResDiskPatchExtractorPool(
            patches_pool=self.patches_pool,
            output_dir=output_dir,
            image_type=self.image_type,
            levels_or_mpps=[1, 0],
            thread_num_workers=1,
            proc_num_workers=4,
            thread_mp_chunksize=1,
            proc_mp_chunksize=1,
            excluded_labels=excluded_labels,
            create_subdirs=True,
            global_counter=1,
        )
        # count extracted files total
        count_path = os.path.join(output_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 0)
        # counts as properties
        self.assertEqual(extractor_pool.patch_count, 0)
        self.assertEqual(extractor_pool.patchset_count, 0)
