# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for WSI datasets.

Tested classes:
    WSIDataset
    WSIMultiResDataset
"""

import os
import glob
from hashlib import md5
from unittest import TestCase

from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as tf

from dplabtools.slides.patches import WholeImageGridPatches
from dplabtools.slides.processing import WSIDataset, WSIMultiResDataset
from dplabtools.common import get_random_string
from testutils import make_test_path

WSIDataset.save_patches_image_type = "tif"
WSIMultiResDataset.save_patches_image_type = "tif"


class TestWSIDatasetProperties(TestCase):
    """Tests for TestWSIDataset class properties (shared with WSIMultiResDataset)."""

    def test_property_patches(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_mask = make_test_path("mask/board-clean-mask-inner.npy")
        patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=wsi_mask,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        wsi_dataset = WSIDataset(patches=patches)
        wsi_dataset.worker_init()
        self.assertEqual(type(wsi_dataset.patches), type(patches))
        self.assertEqual(wsi_dataset.patches.patch_count, 40)


class TestWSIDataset(TestCase):
    """Tests for TestWSIDataset class methods."""

    _blank_image = Image.new(size=(256, 256), mode="RGB", color="black")
    _blank_image_tensor = tf.to_tensor(_blank_image)

    def setUp(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_mask = make_test_path("mask/board-clean-mask-inner.npy")
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=wsi_mask,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )

    @staticmethod
    def transform_function(image):
        return TestWSIDataset._blank_image_tensor

    def test___getitem_notransform(self):
        wsi_dataset = WSIDataset(patches=self.patches)
        # torch dataloaders call worker init automatically, without torch it must be called manually
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 40)
        ref_patch_dir = make_test_path("ref_data/slides/processing/dataset1")
        for wsi_data_elem in wsi_dataset_list:
            patch_image_data, x_level0, y_level0 = wsi_data_elem
            self.assertIsInstance(x_level0, int)
            self.assertIsInstance(y_level0, int)
            ref_file_name = "board-multi-layer-no-compression_x%d_y%d.tif" % (x_level0, y_level0)
            ref_file_path = os.path.join(ref_patch_dir, ref_file_name)
            output_image = Image.open(ref_file_path)
            output_image_tensor = tf.to_tensor(output_image)
            output_image.close()
            result_image_tensor = patch_image_data
            self.assertTrue(torch.equal(result_image_tensor, output_image_tensor))

    def test___getitem_transform(self):
        wsi_dataset = WSIDataset(patches=self.patches, transform_fn=TestWSIDataset.transform_function)
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 40)
        output_image_tensor = TestWSIDataset._blank_image_tensor
        for wsi_data_elem in wsi_dataset_list:
            patch_image_data, x_level0, y_level0 = wsi_data_elem
            self.assertIsInstance(x_level0, int)
            self.assertIsInstance(y_level0, int)
            result_image_tensor = patch_image_data
            self.assertTrue(torch.equal(result_image_tensor, output_image_tensor))

    def test__save_patches(self):
        patches_dir = make_test_path("saved_data/dataset1")
        wsi_dataset = WSIDataset(patches=self.patches, save_patches_dir=patches_dir)
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 40)
        # calculate checksums for reference patches
        ref_patch_dir = make_test_path("ref_data/slides/processing/dataset1")
        ref_patch_path = os.path.join(ref_patch_dir, "**", "*" + "*.tif")
        ref_patch_list = glob.glob(ref_patch_path, recursive=True)
        ref_patch_md5_list = []
        for patch_file in ref_patch_list:
            with open(patch_file, "rb") as file_img:
                ref_patch_checksum = md5(file_img.read()).hexdigest()
                ref_patch_md5_list.append(ref_patch_checksum)
        # count and examine saved patches
        count_path = os.path.join(patches_dir, "**", "*" + "*.tif")
        result_patch_list = glob.glob(count_path, recursive=True)
        self.assertEqual(len(result_patch_list), 40)
        for patch_file in result_patch_list:
            with open(patch_file, "rb") as file_img:
                result_patch_checksum = md5(file_img.read()).hexdigest()
                self.assertIn(result_patch_checksum, ref_patch_md5_list)

    def test__resampling_mode_wsi(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_mask = make_test_path("mask/board-clean-mask-inner.npy")
        patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=wsi_mask,
            patch_size=256,
            level_or_mpp=0.3,  # must be MPP, not level
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.8,
        )
        wsi_dataset = WSIDataset(patches=patches, resampling_mode="wsi", extra_mpps=[0.3])
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 44)

    def test_dataloader(self):
        wsi_dataset = WSIDataset(patches=self.patches)
        wsi_dataloader = torch.utils.data.DataLoader(
            wsi_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=3,
            pin_memory=False,
            worker_init_fn=wsi_dataset.worker_init,
        )
        self.assertEqual(len(wsi_dataloader), 10)
        ref_patch_dir = make_test_path("ref_data/slides/processing/dataset1")
        for (image_data, x_level0_data, y_level0_data) in wsi_dataloader:
            self.assertEqual(len(image_data), 4)
            batch_num_images = len(x_level0_data)
            self.assertEqual(batch_num_images, 4)
            self.assertEqual(len(y_level0_data), 4)
            for i in range(batch_num_images):
                patch_image_data = image_data[i]
                x_level0 = x_level0_data[i]
                y_level0 = y_level0_data[i]
                ref_file_name = "board-multi-layer-no-compression_x%d_y%d.tif" % (x_level0, y_level0)
                ref_file_path = os.path.join(ref_patch_dir, ref_file_name)
                output_image = Image.open(ref_file_path)
                output_image_tensor = tf.to_tensor(output_image)
                result_image_tensor = patch_image_data
                output_image.close()
                self.assertTrue(torch.equal(result_image_tensor, output_image_tensor))

    def test_dataloader_zero_workers(self):
        wsi_dataset = WSIDataset(patches=self.patches, zero_workers=True)
        wsi_dataloader = torch.utils.data.DataLoader(
            wsi_dataset,
            batch_size=5,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.assertEqual(len(wsi_dataloader), 8)
        ref_patch_dir = make_test_path("ref_data/slides/processing/dataset1")
        for (image_data, x_level0_data, y_level0_data) in wsi_dataloader:
            self.assertEqual(len(image_data), 5)
            batch_num_images = len(x_level0_data)
            self.assertEqual(batch_num_images, 5)
            self.assertEqual(len(y_level0_data), 5)
            for i in range(batch_num_images):
                patch_image_data = image_data[i]
                x_level0 = x_level0_data[i]
                y_level0 = y_level0_data[i]
                ref_file_name = "board-multi-layer-no-compression_x%d_y%d.tif" % (x_level0, y_level0)
                ref_file_path = os.path.join(ref_patch_dir, ref_file_name)
                output_image = Image.open(ref_file_path)
                output_image_tensor = tf.to_tensor(output_image)
                result_image_tensor = patch_image_data
                output_image.close()
                self.assertTrue(torch.equal(result_image_tensor, output_image_tensor))


class TestWSIMultiResDataset(TestCase):
    """Tests for WSIMultiResDataset class methods."""

    _blank_image = Image.new(size=(256, 256), mode="RGB", color="black")
    _blank_image_tensor = tf.to_tensor(_blank_image)
    _blank_image_tensor_list = [_blank_image_tensor, _blank_image_tensor, _blank_image_tensor]

    @classmethod
    def setUpClass(cls):
        ref_patch_md5_list = []
        ref_patch_dir = make_test_path("ref_data/slides/processing/dataset2")
        ref_patch_path = os.path.join(ref_patch_dir, "**", "*" + "*.tif")
        ref_patch_list = sorted(glob.glob(ref_patch_path, recursive=True))
        chunk_size = 3
        for i in range(0, len(ref_patch_list), chunk_size):
            ref_patches = ref_patch_list[i : i + chunk_size]
            ref_patches_md5 = []
            for ref_patch_file in ref_patches:
                with open(ref_patch_file, "rb") as file_img:
                    ref_patch_checksum = md5(file_img.read()).hexdigest()
                    ref_patches_md5.append(ref_patch_checksum)
            ref_patch_md5_list.append(ref_patches_md5)
        cls._ref_patch_md5_list = ref_patch_md5_list

    def setUp(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_mask = make_test_path("mask/board-clean-mask-inner.npy")
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=wsi_mask,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )

    @staticmethod
    def transform_function_multires(images):
        return TestWSIMultiResDataset._blank_image_tensor_list

    def test___getitem_notransform(self):
        patches_dir = make_test_path("saved_data/dataset2a")
        wsi_dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0, 0.5, 0.625], resampling_mode="tile")
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 40)
        self.assertEqual(len(self._ref_patch_md5_list), 40)
        # count and examine retrieved patches
        for wsi_data_elem in wsi_dataset_list:
            dir_name = get_random_string(12)
            dir_path = os.path.join(patches_dir, dir_name)
            os.mkdir(dir_path)
            patch_image_data, x_level0, y_level0 = wsi_data_elem
            self.assertIsInstance(x_level0, int)
            self.assertIsInstance(y_level0, int)
            result_image0_tensor = patch_image_data[0]
            result_image1_tensor = patch_image_data[1]
            result_image2_tensor = patch_image_data[2]
            result_image0_path = os.path.join(patches_dir, dir_name, "image0.tif")
            result_image1_path = os.path.join(patches_dir, dir_name, "image1.tif")
            result_image2_path = os.path.join(patches_dir, dir_name, "image2.tif")
            torchvision.utils.save_image(result_image0_tensor, result_image0_path)
            torchvision.utils.save_image(result_image1_tensor, result_image1_path)
            torchvision.utils.save_image(result_image2_tensor, result_image2_path)
            result_patches_md5 = []
            for result_image_file in [result_image0_path, result_image1_path, result_image2_path]:
                with open(result_image_file, "rb") as file_img:
                    result_patch_checksum = md5(file_img.read()).hexdigest()
                    result_patches_md5.append(result_patch_checksum)
            self.assertIn(result_patches_md5, self._ref_patch_md5_list)

    def test___getitem_transform(self):
        wsi_dataset = WSIMultiResDataset(
            patches=self.patches,
            transform_fn=TestWSIMultiResDataset.transform_function_multires,
            levels_or_mpps=[0, 0.5, 0.625],
            resampling_mode="tile",
        )
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 40)
        output_image_tensor = TestWSIMultiResDataset._blank_image_tensor
        for wsi_data_elem in wsi_dataset_list:
            patch_image_data, x_level0, y_level0 = wsi_data_elem
            self.assertIsInstance(x_level0, int)
            self.assertIsInstance(y_level0, int)
            result_image0_tensor = patch_image_data[0]
            result_image1_tensor = patch_image_data[1]
            result_image2_tensor = patch_image_data[2]
            self.assertTrue(torch.equal(result_image0_tensor, output_image_tensor))
            self.assertTrue(torch.equal(result_image1_tensor, output_image_tensor))
            self.assertTrue(torch.equal(result_image2_tensor, output_image_tensor))

    def test_save_patches(self):
        patches_dir = make_test_path("saved_data/dataset2b")
        wsi_dataset = WSIMultiResDataset(
            patches=self.patches,
            transform_fn=TestWSIMultiResDataset.transform_function_multires,
            levels_or_mpps=[0, 0.5, 0.625],
            resampling_mode="tile",
            save_patches_dir=patches_dir,
        )
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 40)
        # count and examine saved patches
        count_path = os.path.join(patches_dir, "**", "*" + "*.tif")
        result_patch_list = sorted(glob.glob(count_path, recursive=True))
        self.assertEqual(len(result_patch_list), 40 * 3)
        chunk_size = 3
        for i in range(0, len(result_patch_list), chunk_size):
            result_patches_md5 = []
            result_patches = result_patch_list[i : i + chunk_size]
            with open(result_patches[0], "rb") as file_img:
                result_patch_checksum = md5(file_img.read()).hexdigest()
                result_patches_md5.append(result_patch_checksum)
            with open(result_patches[1], "rb") as file_img:
                result_patch_checksum = md5(file_img.read()).hexdigest()
                result_patches_md5.append(result_patch_checksum)
            with open(result_patches[2], "rb") as file_img:
                result_patch_checksum = md5(file_img.read()).hexdigest()
                result_patches_md5.append(result_patch_checksum)
            self.assertIn(result_patches_md5, self._ref_patch_md5_list)

    def test__resampling_mode_wsi(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_mask = make_test_path("mask/board-clean-mask-inner.npy")
        patches = WholeImageGridPatches(
            wsi_file=wsi_file_tif,
            mask_data=wsi_mask,
            patch_size=256,
            level_or_mpp=0.3,  # must be MPP, not level
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.8,
        )
        wsi_dataset = WSIMultiResDataset(
            patches=patches, levels_or_mpps=[0.3, 0.35], resampling_mode="wsi", extra_mpps=[0.3, 0.35]
        )
        wsi_dataset.worker_init()
        wsi_dataset_list = list(wsi_dataset)
        self.assertEqual(len(wsi_dataset_list), 44)

    def test_dataloader(self):
        patches_dir = make_test_path("saved_data/dataset2c")
        wsi_dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0, 0.5, 0.625], resampling_mode="tile")
        wsi_dataloader = torch.utils.data.DataLoader(
            wsi_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            worker_init_fn=wsi_dataset.worker_init,
        )
        self.assertEqual(len(wsi_dataloader), 10)  # 10 = num_images/batch_size
        # calculate checksums for retrieved patches
        for (image_data, x_level0_data, y_level0_data) in wsi_dataloader:
            self.assertIsInstance(image_data, list)
            self.assertEqual(len(image_data), 3)
            batch_num_images = len(image_data[0])
            self.assertEqual(batch_num_images, 4)
            self.assertEqual(len(x_level0_data), 4)
            self.assertEqual(len(y_level0_data), 4)
            for i in range(batch_num_images):
                self.assertIsInstance(int(x_level0_data[i]), int)
                self.assertIsInstance(int(y_level0_data[i]), int)
                dir_name = get_random_string(12)
                dir_path = os.path.join(patches_dir, dir_name)
                os.mkdir(dir_path)
                patch_image0_tensor = image_data[0][i]  # level 0
                patch_image1_tensor = image_data[1][i]  # MPP 0.5
                patch_image2_tensor = image_data[2][i]  # MPP 0.625
                result_image0_path = os.path.join(patches_dir, dir_name, "image0.tif")
                result_image1_path = os.path.join(patches_dir, dir_name, "image1.tif")
                result_image2_path = os.path.join(patches_dir, dir_name, "image2.tif")
                torchvision.utils.save_image(patch_image0_tensor, result_image0_path)
                torchvision.utils.save_image(patch_image1_tensor, result_image1_path)
                torchvision.utils.save_image(patch_image2_tensor, result_image2_path)
                result_patches_md5 = []
                for result_image_file in [result_image0_path, result_image1_path, result_image2_path]:
                    with open(result_image_file, "rb") as file_img:
                        result_patch_checksum = md5(file_img.read()).hexdigest()
                        result_patches_md5.append(result_patch_checksum)
                self.assertIn(result_patches_md5, self._ref_patch_md5_list)
