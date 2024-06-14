# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for WSI inference.

Tested classes:
    WSIInference
"""

import os
from unittest import TestCase

from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2

from dplabtools.slides import GenericSlide
from dplabtools.slides.patches import WholeImageGridPatches
from dplabtools.slides.processing import WSIDataset, WSIMultiResDataset, WSIInference

from testutils import make_test_path

# ************ Helper classes ************


class DummyFixedClassificationModelOneClass(torch.nn.Module):
    """Return tensor with value 123 for each data sample."""

    fixed_value = 123

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        return torch.full((batch_size, 1), self.fixed_value)


class DummyVariableClassificationModelOneClass(torch.nn.Module):
    """Return tensor with value 32 multiplied by variable factor for each data sample."""

    fixed_value = 32
    _mul_factor = 0

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        self._mul_factor += 2
        return torch.full((batch_size, 1), self.fixed_value * self._mul_factor)


class DummyVariableClassificationModelOneClassForStride(torch.nn.Module):
    """Return tensor with value 10! multiplied by variable factor for each data sample.

    This guarantees average value to be an integer, but only when total number of patches is < 10
    """

    factorial_of_ten = 3628800
    _mul_factor = 0

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        self._mul_factor += 2
        return torch.full((batch_size, 1), self.factorial_of_ten * self._mul_factor)


class DummyFunctionClassifierOneClass:
    """Return one value multiplied by fixed factor."""

    fixed_value = 17

    @staticmethod
    def action_fn(value):
        return value * DummyFunctionClassifierOneClass.fixed_value


class DummyFixedClassificationModelThreeClasses(torch.nn.Module):
    """Return tensor with values 93 for each data sample."""

    fixed_value = 93

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        return torch.full((batch_size, 3), self.fixed_value)


class DummyFunctionClassifierThreeClasses:
    """Return three values multiplied by fixed factors."""

    fixed_values = [111, 222, 333]

    @staticmethod
    def action_fn(values):
        values[:, 0] = values[:, 0] * DummyFunctionClassifierThreeClasses.fixed_values[0]
        values[:, 1] = values[:, 1] * DummyFunctionClassifierThreeClasses.fixed_values[1]
        values[:, 2] = values[:, 2] * DummyFunctionClassifierThreeClasses.fixed_values[2]
        return values


class DummyFunctionClassifierThreeClassesSoftmax:
    """Return three fixed values after softmax.

    Those values will yield (0, 224,  30) after converting to uint8 values.
    """

    fixed_values = [111, 117, 115]

    @staticmethod
    def action_fn(values):
        values[:, 0] = DummyFunctionClassifierThreeClassesSoftmax.fixed_values[0]
        values[:, 1] = DummyFunctionClassifierThreeClassesSoftmax.fixed_values[1]
        values[:, 2] = DummyFunctionClassifierThreeClassesSoftmax.fixed_values[2]
        return torch.nn.functional.softmax(values.float(), dim=1)


class DummyFixedClassificationModelOneClassMultiRes(torch.nn.Module):
    """Return tensor with value 57 for each data sample."""

    fixed_value = 57

    def __init__(self, batch_size, *args, **kwargs):
        self._batch_size = batch_size
        super().__init__(*args, **kwargs)

    def forward(self, batch_data0, batch_data1, batch_data2):
        # batch size checks
        batch_size0 = batch_data0.shape[0]
        batch_size1 = batch_data1.shape[0]
        batch_size2 = batch_data2.shape[0]
        TestCase().assertEqual(batch_size0, self._batch_size)
        TestCase().assertEqual(batch_size1, self._batch_size)
        TestCase().assertEqual(batch_size2, self._batch_size)
        # correct order of patches can be determined by how many white pixels they contain
        for img_index in range(batch_size0):
            image0 = batch_data0[img_index]
            image1 = batch_data1[img_index]
            image2 = batch_data2[img_index]
            image0_array = image0.numpy().transpose(1, 2, 0)
            image0_white_pixels_count = (image0_array == (255, 255, 255)).all(axis=-1).sum()
            image1_array = image1.numpy().transpose(1, 2, 0)
            image1_white_pixels_count = (image1_array == (255, 255, 255)).all(axis=-1).sum()
            image2_array = image2.numpy().transpose(1, 2, 0)
            image2_white_pixels_count = (image2_array == (255, 255, 255)).all(axis=-1).sum()
            TestCase().assertTrue(image0_white_pixels_count < image1_white_pixels_count)
            TestCase().assertTrue(image1_white_pixels_count < image2_white_pixels_count)
        return torch.full((batch_size0, 1), self.fixed_value)


class DummyFixedSegmentationModelOneClass(torch.nn.Module):
    """Return tensor with value 49.5 for each data sample.

    fixed_value must be float, not integer. With array of identical int values cv2.resize will error out:
    cv2.error: OpenCV(4.8.0) /io/opencv/modules/imgproc/src/resize.cpp:3940: error:
    (-215:Assertion failed) func != 0 in function 'resize'
    """

    fixed_value = 49.5

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        return torch.full((batch_size, 1, 256, 256), self.fixed_value)


class DummyFixedSegmentationModelThreeClasses(torch.nn.Module):
    """Return tensor with value 72.5 for each data sample."""

    fixed_value = 72.5

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        return torch.full((batch_size, 3, 256, 256), self.fixed_value)


class DummyVariableSegmentationModelOneClass(torch.nn.Module):
    """Return tensor with value 83 multiplied by variable factor for each data sample."""

    fixed_value = 83
    _mul_factor = 0

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        self._mul_factor += 2
        return torch.full((batch_size, 1, 256, 256), self.fixed_value * self._mul_factor)


class DummyVariableSegmentationModelOneClassForInterpolation(torch.nn.Module):
    """Return tensor with value 67.5 multiplied by variable factor for each data sample.

    Additionally, returned tensor should NOT be uniform.
    """

    fixed_value = 67.5
    _mul_factor = 0

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        self._mul_factor += 2
        res_tensor = torch.full((batch_size, 1, 256, 256), self.fixed_value * self._mul_factor)
        res_tensor[:, :, 100:150, 120:170] = 101.5
        return res_tensor


class DummyVariableSegmentationModelOneClassForStride(torch.nn.Module):
    """Return tensor with value 10! multiplied by variable factor for each data sample."""

    factorial_of_ten = float(3628800)
    _mul_factor = 0

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        self._mul_factor += 2
        return torch.full((batch_size, 1, 256, 256), self.factorial_of_ten * self._mul_factor)


class DummyVariableSegmentationModelOneClassForStrideSpecialMods(torch.nn.Module):
    """Like above, but uniquely modifies first and last patch to test transposition."""

    factorial_of_ten = float(3628800)
    _mul_factor = 0

    def __init__(self, *args, **kwargs):
        self._patch_counter = 0
        super().__init__(*args, **kwargs)

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        self._mul_factor += 2
        self._patch_counter += 1
        output = torch.full((batch_size, 1, 256, 256), self.factorial_of_ten * self._mul_factor)
        output[0, 0, 15, 70] = 59
        output[8, 0, 161, 42] = 31
        return output


class DummyFunctionClassifierThreeClassesForSegmentation:
    """Return array with values multiplied by fixed factors."""

    fixed_values = [111, 222, 333]

    @staticmethod
    def action_fn(values):
        values[:, 0, :, :] = values[:, 0, :, :] * DummyFunctionClassifierThreeClassesForSegmentation.fixed_values[0]
        values[:, 1, :, :] = values[:, 1, :, :] * DummyFunctionClassifierThreeClassesForSegmentation.fixed_values[1]
        values[:, 2, :, :] = values[:, 2, :, :] * DummyFunctionClassifierThreeClassesForSegmentation.fixed_values[2]
        return values


class DummyModelWrongOutput(torch.nn.Module):
    """Return unsupported 2D tensor with values 29 for each data sample."""

    fixed_value = 29

    def forward(self, batch_data):
        batch_size = batch_data.shape[0]
        return torch.full((batch_size, 2, 6), self.fixed_value)


# ************ Test cases ************


class TestWSIInferenceStaticMethods(TestCase):
    """Tests for static methods in WSIInference class."""

    def test__get_image_in_range_full(self):
        probs_array = np.full((3, 50, 70), np.nan)
        probs_array[0:, :, :] = 0.2
        probs_array[1:, :, :] = 0.4
        probs_array[2:, :, :] = 0.8
        result_image_class0 = WSIInference._get_image(0, probs_array)
        result_image_class1 = WSIInference._get_image(1, probs_array)
        result_image_class2 = WSIInference._get_image(2, probs_array)
        result_image_array_class0 = np.asarray(result_image_class0)
        result_image_array_class1 = np.asarray(result_image_class1)
        result_image_array_class2 = np.asarray(result_image_class2)
        output_image_array_class0 = np.asarray(Image.new(size=(50, 70), mode="L", color=51))
        output_image_array_class1 = np.asarray(Image.new(size=(50, 70), mode="L", color=102))
        output_image_array_class2 = np.asarray(Image.new(size=(50, 70), mode="L", color=204))
        np.testing.assert_equal(result_image_array_class0, output_image_array_class0)
        np.testing.assert_equal(result_image_array_class1, output_image_array_class1)
        np.testing.assert_equal(result_image_array_class2, output_image_array_class2)

    def test__get_image_in_range_partial(self):
        probs_array = np.full((3, 50, 70), np.nan)
        probs_array[0, 5:20, 30:40] = 0.2
        probs_array[1, 10:25, 35:45] = 0.4
        probs_array[2, 15:30, 40:50] = 0.8
        result_image_class0 = WSIInference._get_image(0, probs_array)
        result_image_class1 = WSIInference._get_image(1, probs_array)
        result_image_class2 = WSIInference._get_image(2, probs_array)
        result_image_array_class0 = np.asarray(result_image_class0)
        result_image_array_class1 = np.asarray(result_image_class1)
        result_image_array_class2 = np.asarray(result_image_class2)
        #
        output_image_class0 = Image.new(size=(50, 70), mode="L", color=0)
        output_draw0 = ImageDraw.Draw(output_image_class0)
        output_draw0.rectangle((5, 30, 20 - 1, 40 - 1), fill=51)
        output_image_array_class0 = np.asarray(output_image_class0)
        output_image_class1 = Image.new(size=(50, 70), mode="L", color=0)
        output_draw1 = ImageDraw.Draw(output_image_class1)
        output_draw1.rectangle((10, 35, 25 - 1, 45 - 1), fill=102)
        output_image_array_class1 = np.asarray(output_image_class1)
        output_image_class2 = Image.new(size=(50, 70), mode="L", color=0)
        output_draw2 = ImageDraw.Draw(output_image_class2)
        output_draw2.rectangle((15, 40, 30 - 1, 50 - 1), fill=204)
        output_image_array_class2 = np.asarray(output_image_class2)
        #
        np.testing.assert_equal(result_image_array_class0, output_image_array_class0)
        np.testing.assert_equal(result_image_array_class1, output_image_array_class1)
        np.testing.assert_equal(result_image_array_class2, output_image_array_class2)

    def test__get_image_call_check_array_range(self):
        # test if range checking is called inside get_image
        probs_array = np.full((3, 50, 70), 100)
        with self.assertRaises(ValueError):
            WSIInference._get_image(0, probs_array)

    def test__check_array_range(self):
        probs_array = np.full((3, 20, 20), np.nan)
        WSIInference._check_array_range(probs_array)
        #
        probs_array = np.full((3, 20, 20), np.nan)
        probs_array[2, 2, 2] = 1
        WSIInference._check_array_range(probs_array)
        #
        probs_array = np.full((3, 20, 20), np.nan)
        probs_array[2, 2, 2] = -1
        with self.assertRaises(ValueError):
            WSIInference._check_array_range(probs_array)
        #
        probs_array = np.full((3, 20, 20), 0)
        WSIInference._check_array_range(probs_array)
        #
        probs_array = np.full((3, 20, 20), 1)
        WSIInference._check_array_range(probs_array)
        #
        probs_array = np.full((3, 20, 20), -1)
        with self.assertRaises(ValueError):
            WSIInference._check_array_range(probs_array)
        #
        probs_array = np.full((3, 20, 20), 10)
        with self.assertRaises(ValueError):
            WSIInference._check_array_range(probs_array)


class TestWSIInferenceProperties(TestCase):
    """Tests for properties in WSIInference class."""

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        model = DummyFixedClassificationModelOneClass()
        classifier = DummyFunctionClassifierOneClass.action_fn
        self.dataset = WSIDataset(patches=self.patches)
        self.inference = WSIInference(
            model=model,
            classifier=classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )

    def test_classes_array(self):
        self.inference.process_dataset(self.dataset)
        self.assertEqual(self.inference.classes_array.shape, (1, 160, 192))

    def test_torch_device(self):
        self.inference.process_dataset(self.dataset)
        self.assertEqual(str(self.inference.torch_device), "cpu")


class TestWSIInferenceClassificationFixedOneClassNoStride1(TestCase):
    """Tests for classification model with one class.

    One worker, batch size is not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyFixedClassificationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        output_probs_array[:, 64:128, 192:256] = self.patch_output_value
        output_probs_array[:, 128:192, 256:320] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level_from_size(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=100,  # level 2
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2_no_classifier(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=None,
            level_or_minsize=2,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        output_probs_array[:, 16:32, 48:64] = self.model.fixed_value
        output_probs_array[:, 32:48, 64:80] = self.model.fixed_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2_no_classifier_with_cuda(self):
        if torch.cuda.is_available():
            dataset = WSIDataset(patches=self.patches)
            inference = WSIInference(
                model=self.model,
                classifier=None,
                level_or_minsize=2,
                num_classes=1,
                num_workers=1,
                batch_size=5,
                use_cuda=True,
            )
            inference.process_dataset(dataset)
            result_probs_array = inference.classes_array
            self.assertEqual(result_probs_array.shape, (1, 160, 192))
            output_probs_array = np.full((1, 160, 192), np.nan)
            output_probs_array[:, 16:32, 48:64] = self.model.fixed_value
            output_probs_array[:, 32:48, 64:80] = self.model.fixed_value
            np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassificationFixedOneClassNoStride2(TestCase):
    """Tests for classification model with one class.

    Two workers, batch size is not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyFixedClassificationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=2,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        output_probs_array[:, 64:128, 192:256] = self.patch_output_value
        output_probs_array[:, 128:192, 256:320] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=2,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassficationFixedThreeClassesNoStride1(TestCase):
    """Tests for classification model with three classes.

    One worker, batch size not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyFixedClassificationModelThreeClasses()
        self.classifier = DummyFunctionClassifierThreeClasses.action_fn
        self.patch_output_values = (
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[0],
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[1],
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[2],
        )

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 2560, 3072))
        output_probs_array = np.full((3, 2560, 3072), np.nan)
        output_probs_array[0, 256:512, 768:1024] = self.patch_output_values[0]
        output_probs_array[1, 256:512, 768:1024] = self.patch_output_values[1]
        output_probs_array[2, 256:512, 768:1024] = self.patch_output_values[2]
        output_probs_array[0, 512:768, 1024:1280] = self.patch_output_values[0]
        output_probs_array[1, 512:768, 1024:1280] = self.patch_output_values[1]
        output_probs_array[2, 512:768, 1024:1280] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 640, 768))
        output_probs_array = np.full((3, 640, 768), np.nan)
        output_probs_array[0, 64:128, 192:256] = self.patch_output_values[0]
        output_probs_array[1, 64:128, 192:256] = self.patch_output_values[1]
        output_probs_array[2, 64:128, 192:256] = self.patch_output_values[2]
        output_probs_array[0, 128:192, 256:320] = self.patch_output_values[0]
        output_probs_array[1, 128:192, 256:320] = self.patch_output_values[1]
        output_probs_array[2, 128:192, 256:320] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 160, 192))
        output_probs_array = np.full((3, 160, 192), np.nan)
        output_probs_array[0, 16:32, 48:64] = self.patch_output_values[0]
        output_probs_array[1, 16:32, 48:64] = self.patch_output_values[1]
        output_probs_array[2, 16:32, 48:64] = self.patch_output_values[2]
        output_probs_array[0, 32:48, 64:80] = self.patch_output_values[0]
        output_probs_array[1, 32:48, 64:80] = self.patch_output_values[1]
        output_probs_array[2, 32:48, 64:80] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassficationFixedThreeClassesNoStride2(TestCase):
    """Tests for classification model with three classes.

    Two workers, batch size not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyFixedClassificationModelThreeClasses()
        self.classifier = DummyFunctionClassifierThreeClasses.action_fn
        self.patch_output_values = (
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[0],
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[1],
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[2],
        )

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 2560, 3072))
        output_probs_array = np.full((3, 2560, 3072), np.nan)
        output_probs_array[0, 256:512, 768:1024] = self.patch_output_values[0]
        output_probs_array[1, 256:512, 768:1024] = self.patch_output_values[1]
        output_probs_array[2, 256:512, 768:1024] = self.patch_output_values[2]
        output_probs_array[0, 512:768, 1024:1280] = self.patch_output_values[0]
        output_probs_array[1, 512:768, 1024:1280] = self.patch_output_values[1]
        output_probs_array[2, 512:768, 1024:1280] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=3,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 640, 768))
        output_probs_array = np.full((3, 640, 768), np.nan)
        output_probs_array[0, 64:128, 192:256] = self.patch_output_values[0]
        output_probs_array[1, 64:128, 192:256] = self.patch_output_values[1]
        output_probs_array[2, 64:128, 192:256] = self.patch_output_values[2]
        output_probs_array[0, 128:192, 256:320] = self.patch_output_values[0]
        output_probs_array[1, 128:192, 256:320] = self.patch_output_values[1]
        output_probs_array[2, 128:192, 256:320] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=3,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 160, 192))
        output_probs_array = np.full((3, 160, 192), np.nan)
        output_probs_array[0, 16:32, 48:64] = self.patch_output_values[0]
        output_probs_array[1, 16:32, 48:64] = self.patch_output_values[1]
        output_probs_array[2, 16:32, 48:64] = self.patch_output_values[2]
        output_probs_array[0, 32:48, 64:80] = self.patch_output_values[0]
        output_probs_array[1, 32:48, 64:80] = self.patch_output_values[1]
        output_probs_array[2, 32:48, 64:80] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassificationFixedOneClassWithStride1(TestCase):
    """Tests for classification model with one class.

    One worker, batch size not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((640, 768), dtype=np.uint8)  # level1
        mask_data[64:128, 192:256] = 1  # patch #16
        mask_data[128:192, 256:320] = 1  # patch #29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        self.model = DummyFixedClassificationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = self.patch_output_value
        output_probs_array[:, 384:512, 1024:1280] = self.patch_output_value
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        # patch #16
        output_probs_array[:, 64:128, 192:256] = self.patch_output_value
        # patch #28
        output_probs_array[:, 128:160, 192:256] = self.patch_output_value
        output_probs_array[:, 128:192, 224:256] = self.patch_output_value
        # patch #17
        output_probs_array[:, 64:128, 256:288] = self.patch_output_value
        output_probs_array[:, 96:128, 256:320] = self.patch_output_value
        # patch #29
        output_probs_array[:, 128:192, 256:320] = self.patch_output_value
        # patch #41
        output_probs_array[:, 192:224, 256:320] = self.patch_output_value
        # patch #30
        output_probs_array[:, 128:192, 320:352] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        # patch #16
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        # patch #28
        output_probs_array[:, 32:40, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 56:64] = self.patch_output_value
        # patch #17
        output_probs_array[:, 16:32, 64:72] = self.patch_output_value
        output_probs_array[:, 24:32, 64:80] = self.patch_output_value
        # patch #29
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        # patch #41
        output_probs_array[:, 48:56, 64:80] = self.patch_output_value
        # patch #30
        output_probs_array[:, 32:48, 80:88] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassificationFixedOneClassWithStride2(TestCase):
    """Tests for classification model with one class.

    Two workers, batch size not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((640, 768), dtype=np.uint8)  # level1
        mask_data[64:128, 192:256] = 1  # patch #16
        mask_data[128:192, 256:320] = 1  # patch #29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        self.model = DummyFixedClassificationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value  # 123*17

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = self.patch_output_value
        output_probs_array[:, 384:512, 1024:1280] = self.patch_output_value
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        # patch #16
        output_probs_array[:, 64:128, 192:256] = self.patch_output_value
        # patch #28
        output_probs_array[:, 128:160, 192:256] = self.patch_output_value
        output_probs_array[:, 128:192, 224:256] = self.patch_output_value
        # patch #17
        output_probs_array[:, 64:128, 256:288] = self.patch_output_value
        output_probs_array[:, 96:128, 256:320] = self.patch_output_value
        # patch #29
        output_probs_array[:, 128:192, 256:320] = self.patch_output_value
        # patch #41
        output_probs_array[:, 192:224, 256:320] = self.patch_output_value
        # patch #30
        output_probs_array[:, 128:192, 320:352] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        # patch #16
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        # patch #28
        output_probs_array[:, 32:40, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 56:64] = self.patch_output_value
        # patch #17
        output_probs_array[:, 16:32, 64:72] = self.patch_output_value
        output_probs_array[:, 24:32, 64:80] = self.patch_output_value
        # patch #29
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        # patch #41
        output_probs_array[:, 48:56, 64:80] = self.patch_output_value
        # patch #30
        output_probs_array[:, 32:48, 80:88] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassificationVariableOneClassNoStride1(TestCase):
    """Tests for classification model with one class.

    One worker, batch size is relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyVariableClassificationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0_batchsize1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 4
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batchsize2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=2,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassificationVariableOneClassNoStride2(TestCase):
    """Tests for classification model with one class.

    Two workers, batch size is relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyVariableClassificationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0_batchsize1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 4
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batchsize2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=2,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassificationVariableOneClassWithStride(TestCase):
    """Tests for classification model with one class.

    One worker, batch size is relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((640, 768), dtype=np.uint8)  # level1
        mask_data[64:128, 192:256] = 1  # patch #16
        mask_data[128:192, 256:320] = 1  # patch #29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        self.model = DummyVariableClassificationModelOneClassForStride()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.factorial_of_ten * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0_batch_size10(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=10,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value * 2
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = self.patch_output_value * 2
        output_probs_array[:, 384:512, 1024:1280] = self.patch_output_value * 2
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value * 2
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value * 2
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batch_size8(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=8,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value * 2
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = self.patch_output_value * 2
        output_probs_array[:, 384:512, 1024:1280] = self.patch_output_value * 2
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value * 2
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value * 4
        # bottom part of patch #29 overlaps with 2 other patches, one was processed with different mul_factor
        # avarage value of 3 patches must be computed
        output_probs_array[:, 512:768, 1152:1280] = (
            self.patch_output_value * 2 + self.patch_output_value * 2 + self.patch_output_value * 4
        ) / 3
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batch_size5(self):
        # many patches overlap and are processed in 2 batches
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value * 2
        # patch #17
        output_probs_array[:, 256:384, 1024:1152] = self.patch_output_value * 2
        output_probs_array[:, 384:512, 1024:1152] = (
            self.patch_output_value * 2 + self.patch_output_value * 2 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 384:512, 1152:1280] = self.patch_output_value * 4
        # patch #29
        output_probs_array[:, 512:640, 1024:1152] = (
            self.patch_output_value * 2
            + self.patch_output_value * 2
            + self.patch_output_value * 4
            + self.patch_output_value * 4
        ) / 4
        output_probs_array[:, 640:768, 1024:1152] = (
            self.patch_output_value * 2 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 512:640, 1152:1280] = (
            self.patch_output_value * 4 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 640:768, 1152:1280] = (
            self.patch_output_value * 4 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value * 4
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value * 4
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1_batch_size3(self):
        # many patches overlap and are processed in 3 batches at level 1
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=3,
            batch_size=3,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        # patch #16
        output_probs_array[:, 64:128, 192:224] = self.patch_output_value * 2
        output_probs_array[:, 64:96, 224:256] = self.patch_output_value * 2
        output_probs_array[:, 96:128, 224:256] = (
            self.patch_output_value * 2
            + self.patch_output_value * 2
            + self.patch_output_value * 2
            + self.patch_output_value * 4
        ) / 4
        # patch #28
        output_probs_array[:, 128:160, 192:224] = self.patch_output_value * 2
        output_probs_array[:, 128:160, 224:256] = (
            self.patch_output_value * 2 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 160:192, 224:256] = self.patch_output_value * 4
        # patch #17
        output_probs_array[:, 64:96, 256:288] = self.patch_output_value * 2
        output_probs_array[:, 96:128, 256:288] = (
            self.patch_output_value * 2 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 96:128, 288:320] = self.patch_output_value * 4
        # patch #29
        output_probs_array[:, 128:160, 256:288] = (
            self.patch_output_value * 4
            + self.patch_output_value * 4
            + self.patch_output_value * 4
            + self.patch_output_value * 6
        ) / 4
        output_probs_array[:, 160:192, 256:288] = (
            self.patch_output_value * 4 + self.patch_output_value * 6 + self.patch_output_value * 6
        ) / 3
        output_probs_array[:, 128:160, 288:320] = (
            self.patch_output_value * 4 + self.patch_output_value * 6 + self.patch_output_value * 6
        ) / 3
        output_probs_array[:, 160:192, 288:320] = (
            self.patch_output_value * 6 + self.patch_output_value * 6 + self.patch_output_value * 6
        ) / 3
        # patch #41
        output_probs_array[:, 192:224, 256:320] = self.patch_output_value * 6
        # patch #30
        output_probs_array[:, 128:192, 320:352] = self.patch_output_value * 6
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceClassificationFixedOneClassNoStrideMultiRes(TestCase):
    """Test if patches are fed in the right order and batches have the right size/shape."""

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[64:80, 96:112] = 1  # patch55
        mask_data[80:96, 112:128] = 1  # patch68
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.classifier = None

    def test_patches_are_ordered_one_worker(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.35, 0.45], resampling_mode="tile")
        batch_size = 2
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=batch_size,
            use_cuda=False,
        )
        inference.process_dataset(dataset)

    def test_patches_are_ordered_two_workers(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.35, 0.45], resampling_mode="tile")
        batch_size = 1
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)


class TestWSIInferenceClassificationFixedOneClassWithStrideMultiRes(TestCase):
    """Same as above, but with stride."""

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[64:80, 96:112] = 1  # patch55
        mask_data[80:96, 112:128] = 1  # patch68
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.25,
            foreground_ratio=0.7,
            overlap_ratio=0.5,
        )
        self.classifier = None

    def test_patches_multires_one_worker(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.30, 0.85], resampling_mode="tile")
        batch_size = 2
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=batch_size,
            use_cuda=False,
        )
        inference.process_dataset(dataset)

    def test_patches_multires_two_workers_small_batch(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.30, 0.85], resampling_mode="tile")
        batch_size = 1
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=batch_size,
            use_cuda=False,
        )
        inference.process_dataset(dataset)

    def test_patches_multires_two_workers_larger_batch(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.30, 0.85], resampling_mode="tile")
        batch_size = 4
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=batch_size,
            use_cuda=False,
        )
        inference.process_dataset(dataset)

    def test_patches_multires_four_workers_smaller_batch(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.30, 0.85], resampling_mode="tile")
        batch_size = 1
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=4,
            batch_size=batch_size,
            use_cuda=False,
        )
        inference.process_dataset(dataset)

    def test_patches_multires_four_workers_small_batch(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.30, 0.85], resampling_mode="tile")
        batch_size = 2
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=4,
            batch_size=batch_size,
            use_cuda=False,
        )
        inference.process_dataset(dataset)

    def test_patches_multires_four_workers_larger_batch(self):
        dataset = WSIMultiResDataset(patches=self.patches, levels_or_mpps=[0.25, 0.30, 0.85], resampling_mode="tile")
        batch_size = 4
        model = DummyFixedClassificationModelOneClassMultiRes(batch_size)
        inference = WSIInference(
            model=model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=4,
            batch_size=batch_size,
            use_cuda=False,
        )
        inference.process_dataset(dataset)


class TestWSIInferenceSegmentationFixedOneClassNoStride(TestCase):
    """Tests for segmentation model with one class.

    One worker, batch size not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyFixedSegmentationModelOneClass()
        self.classifier = None
        self.patch_output_value = self.model.fixed_value

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        output_probs_array[:, 64:128, 192:256] = self.patch_output_value
        output_probs_array[:, 128:192, 256:320] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2_two_workers(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=2,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceSegmentationFixedThreeClassesNoStride(TestCase):
    """Tests for segmentation model with three classes.

    One worker, batch size not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyFixedSegmentationModelThreeClasses()
        self.classifier = DummyFunctionClassifierThreeClassesForSegmentation.action_fn
        self.patch_output_values = (
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[0],
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[1],
            self.model.fixed_value * DummyFunctionClassifierThreeClasses.fixed_values[2],
        )

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 2560, 3072))
        output_probs_array = np.full((3, 2560, 3072), np.nan)
        output_probs_array[0, 256:512, 768:1024] = self.patch_output_values[0]
        output_probs_array[1, 256:512, 768:1024] = self.patch_output_values[1]
        output_probs_array[2, 256:512, 768:1024] = self.patch_output_values[2]
        output_probs_array[0, 512:768, 1024:1280] = self.patch_output_values[0]
        output_probs_array[1, 512:768, 1024:1280] = self.patch_output_values[1]
        output_probs_array[2, 512:768, 1024:1280] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 640, 768))
        output_probs_array = np.full((3, 640, 768), np.nan)
        output_probs_array[0, 64:128, 192:256] = self.patch_output_values[0]
        output_probs_array[1, 64:128, 192:256] = self.patch_output_values[1]
        output_probs_array[2, 64:128, 192:256] = self.patch_output_values[2]
        output_probs_array[0, 128:192, 256:320] = self.patch_output_values[0]
        output_probs_array[1, 128:192, 256:320] = self.patch_output_values[1]
        output_probs_array[2, 128:192, 256:320] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 160, 192))
        output_probs_array = np.full((3, 160, 192), np.nan)
        output_probs_array[0, 16:32, 48:64] = self.patch_output_values[0]
        output_probs_array[1, 16:32, 48:64] = self.patch_output_values[1]
        output_probs_array[2, 16:32, 48:64] = self.patch_output_values[2]
        output_probs_array[0, 32:48, 64:80] = self.patch_output_values[0]
        output_probs_array[1, 32:48, 64:80] = self.patch_output_values[1]
        output_probs_array[2, 32:48, 64:80] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2_two_workers(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=3,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (3, 160, 192))
        output_probs_array = np.full((3, 160, 192), np.nan)
        output_probs_array[0, 16:32, 48:64] = self.patch_output_values[0]
        output_probs_array[1, 16:32, 48:64] = self.patch_output_values[1]
        output_probs_array[2, 16:32, 48:64] = self.patch_output_values[2]
        output_probs_array[0, 32:48, 64:80] = self.patch_output_values[0]
        output_probs_array[1, 32:48, 64:80] = self.patch_output_values[1]
        output_probs_array[2, 32:48, 64:80] = self.patch_output_values[2]
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceSegmentationFixedOneClassWithStride(TestCase):
    """Tests for segmentation model with three classes.

    One worker, batch size not relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((640, 768), dtype=np.uint8)  # level1
        mask_data[64:128, 192:256] = 1  # patch #16
        mask_data[128:192, 256:320] = 1  # patch #29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        self.model = DummyFixedSegmentationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = self.patch_output_value
        output_probs_array[:, 384:512, 1024:1280] = self.patch_output_value
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        # patch #16
        output_probs_array[:, 64:128, 192:256] = self.patch_output_value
        # patch #28
        output_probs_array[:, 128:160, 192:256] = self.patch_output_value
        output_probs_array[:, 128:192, 224:256] = self.patch_output_value
        # patch #17
        output_probs_array[:, 64:128, 256:288] = self.patch_output_value
        output_probs_array[:, 96:128, 256:320] = self.patch_output_value
        # patch #29
        output_probs_array[:, 128:192, 256:320] = self.patch_output_value
        # patch #41
        output_probs_array[:, 192:224, 256:320] = self.patch_output_value
        # patch #30
        output_probs_array[:, 128:192, 320:352] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        # patch #16
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        # patch #28
        output_probs_array[:, 32:40, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 56:64] = self.patch_output_value
        # patch #17
        output_probs_array[:, 16:32, 64:72] = self.patch_output_value
        output_probs_array[:, 24:32, 64:80] = self.patch_output_value
        # patch #29
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        # patch #41
        output_probs_array[:, 48:56, 64:80] = self.patch_output_value
        # patch #30
        output_probs_array[:, 32:48, 80:88] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level2_two_workers(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=1,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 160, 192))
        output_probs_array = np.full((1, 160, 192), np.nan)
        # patch #16
        output_probs_array[:, 16:32, 48:64] = self.patch_output_value
        # patch #28
        output_probs_array[:, 32:40, 48:64] = self.patch_output_value
        output_probs_array[:, 32:48, 56:64] = self.patch_output_value
        # patch #17
        output_probs_array[:, 16:32, 64:72] = self.patch_output_value
        output_probs_array[:, 24:32, 64:80] = self.patch_output_value
        # patch #29
        output_probs_array[:, 32:48, 64:80] = self.patch_output_value
        # patch #41
        output_probs_array[:, 48:56, 64:80] = self.patch_output_value
        # patch #30
        output_probs_array[:, 32:48, 80:88] = self.patch_output_value
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceSegmentationVariableOneClassNoStride1(TestCase):
    """Tests for segmentation model with one class.

    One worker, batch size is relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyVariableSegmentationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0_batchsize1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 4
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batchsize2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=2,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceSegmentationVariableOneClassNoStride2(TestCase):
    """Tests for segmentation model with one class.

    Two workers, batch size is relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.model = DummyVariableSegmentationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0_batchsize1(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=1,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 4
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batchsize2(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=2,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceSegmentationVariableOneClassWithStride(TestCase):
    """Tests for segmentation model with one class.

    One worker, batch size is relevant.
    """

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((640, 768), dtype=np.uint8)  # level1
        mask_data[64:128, 192:256] = 1  # patch #16
        mask_data[128:192, 256:320] = 1  # patch #29
        self.patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=0.5,
            foreground_ratio=0.5,
            overlap_ratio=0.5,
        )
        self.model = DummyVariableSegmentationModelOneClassForStride()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        self.patch_output_value = self.model.factorial_of_ten * DummyFunctionClassifierOneClass.fixed_value

    def test_probs_array_level0_batch_size10(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=10,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value * 2
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = self.patch_output_value * 2
        output_probs_array[:, 384:512, 1024:1280] = self.patch_output_value * 2
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value * 2
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value * 2
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batch_size8(self):
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=8,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value * 2
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = self.patch_output_value * 2
        output_probs_array[:, 384:512, 1024:1280] = self.patch_output_value * 2
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = self.patch_output_value * 2
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value * 2
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value * 4
        # bottom part of patch #29 overlaps with 2 other patches, one was processed with different mul_factor
        # avarage value of 3 patches must be computed
        output_probs_array[:, 512:768, 1152:1280] = (
            self.patch_output_value * 2 + self.patch_output_value * 2 + self.patch_output_value * 4
        ) / 3
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batch_size5(self):
        # many patches overlap and are processed in 2 batches
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=2,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = self.patch_output_value * 2
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = self.patch_output_value * 2
        output_probs_array[:, 512:768, 896:1024] = self.patch_output_value * 2
        # patch #17
        output_probs_array[:, 256:384, 1024:1152] = self.patch_output_value * 2
        output_probs_array[:, 384:512, 1024:1152] = (
            self.patch_output_value * 2 + self.patch_output_value * 2 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 384:512, 1152:1280] = self.patch_output_value * 4
        # patch #29
        output_probs_array[:, 512:640, 1024:1152] = (
            self.patch_output_value * 2
            + self.patch_output_value * 2
            + self.patch_output_value * 4
            + self.patch_output_value * 4
        ) / 4
        output_probs_array[:, 640:768, 1024:1152] = (
            self.patch_output_value * 2 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 512:640, 1152:1280] = (
            self.patch_output_value * 4 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 640:768, 1152:1280] = (
            self.patch_output_value * 4 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = self.patch_output_value * 4
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = self.patch_output_value * 4
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level1_batch_size3(self):
        # many patches overlap and are processed in 3 batches at level 1
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=1,
            num_workers=3,
            batch_size=3,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 640, 768))
        output_probs_array = np.full((1, 640, 768), np.nan)
        # patch #16
        output_probs_array[:, 64:128, 192:224] = self.patch_output_value * 2
        output_probs_array[:, 64:96, 224:256] = self.patch_output_value * 2
        output_probs_array[:, 96:128, 224:256] = (
            self.patch_output_value * 2
            + self.patch_output_value * 2
            + self.patch_output_value * 2
            + self.patch_output_value * 4
        ) / 4
        # patch #28
        output_probs_array[:, 128:160, 192:224] = self.patch_output_value * 2
        output_probs_array[:, 128:160, 224:256] = (
            self.patch_output_value * 2 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 160:192, 224:256] = self.patch_output_value * 4
        # patch #17
        output_probs_array[:, 64:96, 256:288] = self.patch_output_value * 2
        output_probs_array[:, 96:128, 256:288] = (
            self.patch_output_value * 2 + self.patch_output_value * 4 + self.patch_output_value * 4
        ) / 3
        output_probs_array[:, 96:128, 288:320] = self.patch_output_value * 4
        # patch #29
        output_probs_array[:, 128:160, 256:288] = (
            self.patch_output_value * 4
            + self.patch_output_value * 4
            + self.patch_output_value * 4
            + self.patch_output_value * 6
        ) / 4
        output_probs_array[:, 160:192, 256:288] = (
            self.patch_output_value * 4 + self.patch_output_value * 6 + self.patch_output_value * 6
        ) / 3
        output_probs_array[:, 128:160, 288:320] = (
            self.patch_output_value * 4 + self.patch_output_value * 6 + self.patch_output_value * 6
        ) / 3
        output_probs_array[:, 160:192, 288:320] = (
            self.patch_output_value * 6 + self.patch_output_value * 6 + self.patch_output_value * 6
        ) / 3
        # patch #41
        output_probs_array[:, 192:224, 256:320] = self.patch_output_value * 6
        # patch #30
        output_probs_array[:, 128:192, 320:352] = self.patch_output_value * 6
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_probs_array_level0_batch_size10_specialmod(self):
        # many patches overlap and are processed in 1 batch
        model = DummyVariableSegmentationModelOneClassForStrideSpecialMods()
        classifier = DummyFunctionClassifierOneClass.action_fn
        patch_output_value = model.factorial_of_ten * DummyFunctionClassifierOneClass.fixed_value
        dataset = WSIDataset(patches=self.patches)
        inference = WSIInference(
            model=model,
            classifier=classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=10,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        result_probs_array = inference.classes_array
        self.assertEqual(result_probs_array.shape, (1, 2560, 3072))
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        # patch #16
        output_probs_array[:, 256:512, 768:1024] = patch_output_value * 2
        # patch #28
        output_probs_array[:, 512:640, 768:1024] = patch_output_value * 2
        output_probs_array[:, 512:768, 896:1024] = patch_output_value * 2
        # patch #17
        output_probs_array[:, 256:512, 1024:1152] = patch_output_value * 2
        output_probs_array[:, 384:512, 1024:1280] = patch_output_value * 2
        # patch #29
        output_probs_array[:, 512:768, 1024:1280] = patch_output_value * 2
        # patch #41
        output_probs_array[:, 768:896, 1024:1280] = patch_output_value * 2
        # patch #30
        output_probs_array[:, 512:768, 1280:1408] = patch_output_value * 2
        # special mods
        output_probs_array[:, 256 + 70, 768 + 15] = 59 * DummyFunctionClassifierOneClass.fixed_value
        output_probs_array[:, 512 + 42, 1152 + 161] = 31 * DummyFunctionClassifierOneClass.fixed_value
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceSegmentationFixedOneClassNoStrideWithTrimming(TestCase):
    """Tests for segmentation model with one class.

    Trimming - some patches exceed image borders and need to be resized and trimmed.
    """

    # xy_array1 = None

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_data = np.zeros((2560, 3072), dtype=np.uint8)  # level0
        self.model = DummyFixedSegmentationModelOneClass()
        self.classifier = DummyFunctionClassifierOneClass.action_fn
        # self.patch_output_value = self.model.fixed_value * DummyFunctionClassifierOneClass.fixed_value

    # def tearDown(self):
    #    WSIInference.set_interpolation_method(cv2.INTER_LINEAR)

    def test_patch_trim_x(self):
        self.mask_data[1792:2560, 1280:2048] = 1  # patches: 90-92, 102-104, 114-116
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_data,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.3,
        )
        dataset = WSIDataset(patches=patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        # test non-np.nan elements only
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 1792:2560, 1280:2304] = True  # 2304 = 1280 + 1024
        result_probs_array = inference.classes_array
        result_probs_array = np.where(~np.isnan(result_probs_array), True, np.nan)
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_patch_trim_y(self):
        self.mask_data[768:1536, 2304:3072] = 1  # patches: 46-48, 58-60, 70-72
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_data,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.3,
        )
        dataset = WSIDataset(patches=patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        # test non-nan elements only
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 768:1792, 2304:3072] = True  # 1792 = 768 + 1024
        result_probs_array = inference.classes_array
        result_probs_array = np.where(~np.isnan(result_probs_array), True, np.nan)
        np.testing.assert_equal(result_probs_array, output_probs_array)

    def test_patch_trim_xy(self):
        self.mask_data[1792:2560, 2304:3072] = 1  # patches: 94-96, 106-108, 118-120
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_data,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.3,
        )
        dataset = WSIDataset(patches=patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        # test non-np.nan elements only
        output_probs_array = np.full((1, 2560, 3072), np.nan)
        output_probs_array[:, 1792:2560, 2304:3072] = True  # same as mask
        result_probs_array = inference.classes_array
        result_probs_array = np.where(~np.isnan(result_probs_array), True, np.nan)
        np.testing.assert_equal(result_probs_array, output_probs_array)


class TestWSIInferenceSegmentationVariableOneClassNoStrideWithTrimmingIdenticalInterpolation(TestCase):
    """Tests for segmentation model with one class.

    - Trimming must be present to test interpolation method.
    - Methods in this class must be run in the order they are written.
    """

    xy_array1 = None

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_data = np.zeros((2560, 3072), dtype=np.uint8)  # level0
        self.mask_data[1792:2560, 2304:3072] = 1  # patches: 94-96, 106-108, 118-120
        self.model = DummyVariableSegmentationModelOneClassForInterpolation()
        self.classifier = DummyFunctionClassifierOneClass.action_fn

    def tearDown(self):
        WSIInference.set_interpolation_method(cv2.INTER_LINEAR)

    def test_patch_trim_xy_interpolation_default(self):
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_data,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.3,
        )
        dataset = WSIDataset(patches=patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        self.__class__.xy_array1 = inference.classes_array

    def test_patch_trim_xy_interpolation_other_same(self):
        WSIInference.set_interpolation_method(cv2.INTER_LINEAR)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_data,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.3,
        )
        dataset = WSIDataset(patches=patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        xy_array2 = inference.classes_array
        np.testing.assert_equal(self.xy_array1, xy_array2)


class TestWSIInferenceSegmentationVariableOneClassNoStrideWithTrimmingDifferentInterpolation(TestCase):
    """Tests for segmentation model with one class.

    - Trimming must be present to test interpolation method.
    - Methods in this class must be run in the order they are written.
    """

    xy_array1 = None

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        self.mask_data = np.zeros((2560, 3072), dtype=np.uint8)  # level0
        self.mask_data[1792:2560, 2304:3072] = 1  # patches: 94-96, 106-108, 118-120
        self.model = DummyVariableSegmentationModelOneClassForInterpolation()
        self.classifier = DummyFunctionClassifierOneClass.action_fn

    def tearDown(self):
        WSIInference.set_interpolation_method(cv2.INTER_LINEAR)

    def test_patch_trim_xy_interpolation_default(self):
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_data,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.3,
        )
        dataset = WSIDataset(patches=patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        self.__class__.xy_array1 = inference.classes_array

    def test_patch_trim_xy_interpolation_other_different(self):
        WSIInference.set_interpolation_method(cv2.INTER_CUBIC)
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=self.mask_data,
            patch_size=256,
            level_or_mpp=1,
            patch_stride=1,
            foreground_ratio=0.3,
            overlap_ratio=0.3,
        )
        dataset = WSIDataset(patches=patches)
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=1,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset)
        xy_array2 = inference.classes_array
        self.assertFalse(np.array_equal(self.xy_array1, xy_array2))


class TestWSIInferenceRaisedErrors(TestCase):
    """Tests for raising errors."""

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.dataset = WSIDataset(patches=patches)

    def test_incorrect_number_of_classes(self):
        model = DummyFixedClassificationModelThreeClasses()
        classifier = DummyFunctionClassifierThreeClassesSoftmax.action_fn
        inference = WSIInference(
            model=model,
            classifier=classifier,
            level_or_minsize=0,
            num_classes=5,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        with self.assertRaises(ValueError):
            inference.process_dataset(self.dataset)

    def test_unsupported_output(self):
        model = DummyModelWrongOutput()
        classifier = None
        inference = WSIInference(
            model=model,
            classifier=classifier,
            level_or_minsize=0,
            num_classes=2,
            num_workers=1,
            batch_size=3,
            use_cuda=False,
        )
        with self.assertRaises(ValueError):
            inference.process_dataset(self.dataset)


class TestWSIInferenceFileSaving(TestCase):
    """Tests for file saving."""

    def setUp(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        dataset = WSIDataset(patches=patches)
        model = DummyFixedClassificationModelThreeClasses()
        classifier = DummyFunctionClassifierThreeClassesSoftmax.action_fn
        self.inference = WSIInference(
            model=model,
            classifier=classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        self.inference.process_dataset(dataset)

    def test_save_classes_array(self):
        result_array_all_classes = make_test_path("saved_data/inference/data_all.npz")
        self.inference.save_classes_array(result_array_all_classes)
        # read reference values
        output_array_file_all = make_test_path("ref_data/slides/processing/inference/data_all.npz")
        output_array_all = np.load(output_array_file_all)["data"]
        # read saved values
        result_array_all = np.load(result_array_all_classes)["data"]
        # compare values
        np.testing.assert_equal(result_array_all, output_array_all)

    def test_save_class_array(self):
        result_array_file_class0 = make_test_path("saved_data/inference/data_class0.npz")
        result_array_file_class1 = make_test_path("saved_data/inference/data_class1.npz")
        result_array_file_class2 = make_test_path("saved_data/inference/data_class2.npz")
        self.inference.save_class_array(0, result_array_file_class0)
        self.inference.save_class_array(1, result_array_file_class1)
        self.inference.save_class_array(2, result_array_file_class2)
        # read reference values
        output_array_file_class0 = make_test_path("ref_data/slides/processing/inference/data_class0.npz")
        output_array_file_class1 = make_test_path("ref_data/slides/processing/inference/data_class1.npz")
        output_array_file_class2 = make_test_path("ref_data/slides/processing/inference/data_class2.npz")
        output_array_class0 = np.load(output_array_file_class0)["data"]
        output_array_class1 = np.load(output_array_file_class1)["data"]
        output_array_class2 = np.load(output_array_file_class2)["data"]
        # read saved values
        result_array_class0 = np.load(result_array_file_class0)["data"]
        result_array_class1 = np.load(result_array_file_class1)["data"]
        result_array_class2 = np.load(result_array_file_class2)["data"]
        # compare values
        np.testing.assert_equal(result_array_class0, output_array_class0)
        np.testing.assert_equal(result_array_class1, output_array_class1)
        np.testing.assert_equal(result_array_class2, output_array_class2)

    def test_save_class_png(self):
        result_image_file_class0 = make_test_path("saved_data/inference/image_class0.png")
        result_image_file_class1 = make_test_path("saved_data/inference/image_class1.png")
        result_image_file_class2 = make_test_path("saved_data/inference/image_class2.png")
        self.inference.save_class_png(0, result_image_file_class0)
        self.inference.save_class_png(1, result_image_file_class1)
        self.inference.save_class_png(2, result_image_file_class2)
        # read reference values
        output_image_class0 = Image.open(make_test_path("ref_data/slides/processing/inference/image_class0.png"))
        output_image_array_class0 = np.asarray(output_image_class0)
        output_image_class0.close()
        output_image_class1 = Image.open(make_test_path("ref_data/slides/processing/inference/image_class1.png"))
        output_image_array_class1 = np.asarray(output_image_class1)
        output_image_class1.close()
        output_image_class2 = Image.open(make_test_path("ref_data/slides/processing/inference/image_class2.png"))
        output_image_array_class2 = np.asarray(output_image_class2)
        output_image_class2.close()
        # read saved values
        result_image_class0 = Image.open(result_image_file_class0)
        result_image_array_class0 = np.asarray(result_image_class0)
        result_image_class0.close()
        result_image_class1 = Image.open(result_image_file_class1)
        result_image_array_class1 = np.asarray(result_image_class1)
        result_image_class1.close()
        result_image_class2 = Image.open(result_image_file_class2)
        result_image_array_class2 = np.asarray(result_image_class2)
        result_image_class2.close()
        # compare values
        np.testing.assert_equal(result_image_array_class0, output_image_array_class0)
        np.testing.assert_equal(result_image_array_class1, output_image_array_class1)
        np.testing.assert_equal(result_image_array_class2, output_image_array_class2)


class TestWSIInferenceFileSavingTif(TestCase):
    """Tests for file saving as TIF."""

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        self.dataset = WSIDataset(patches=patches)
        self.model = DummyFixedClassificationModelThreeClasses()
        self.classifier = DummyFunctionClassifierThreeClassesSoftmax.action_fn

    def test_save_class_tif_level0(self):
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(self.dataset)
        result_image_file_class0 = make_test_path("saved_data/inference/image_level0_class0.tif")
        result_image_file_class1 = make_test_path("saved_data/inference/image_level0_class1.tif")
        result_image_file_class2 = make_test_path("saved_data/inference/image_level0_class2.tif")
        inference.save_class_tif(0, result_image_file_class0, self.wsi_file, jpeg_compression=False)
        inference.save_class_tif(1, result_image_file_class1, self.wsi_file, jpeg_compression=False)
        inference.save_class_tif(2, result_image_file_class2, self.wsi_file, jpeg_compression=False)
        # read reference values
        output_image_file_class0 = make_test_path("ref_data/slides/processing/inference/image_level0_class0.tif")
        output_image_class0 = Image.open(output_image_file_class0)
        output_image_array_class0 = np.asarray(output_image_class0)
        output_image_class0.close()
        output_image_file_class1 = make_test_path("ref_data/slides/processing/inference/image_level0_class1.tif")
        output_image_class1 = Image.open(output_image_file_class1)
        output_image_array_class1 = np.asarray(output_image_class1)
        output_image_class1.close()
        output_image_file_class2 = make_test_path("ref_data/slides/processing/inference/image_level0_class2.tif")
        output_image_class2 = Image.open(output_image_file_class2)
        output_image_array_class2 = np.asarray(output_image_class2)
        output_image_class2.close()
        # read saved values
        result_image_class0 = Image.open(result_image_file_class0)
        result_image_array_class0 = np.asarray(result_image_class0)
        result_image_class0.close()
        result_image_class1 = Image.open(result_image_file_class1)
        result_image_array_class1 = np.asarray(result_image_class1)
        result_image_class1.close()
        result_image_class2 = Image.open(result_image_file_class2)
        result_image_array_class2 = np.asarray(result_image_class2)
        result_image_class2.close()
        # compare values
        np.testing.assert_equal(result_image_array_class0, output_image_array_class0)
        np.testing.assert_equal(result_image_array_class1, output_image_array_class1)
        np.testing.assert_equal(result_image_array_class2, output_image_array_class2)
        # compare MPP
        result_slide_class0 = GenericSlide(wsi_file=result_image_file_class0)
        result_slide_class0_mpp = result_slide_class0.mpp_data
        output_slide_class0 = GenericSlide(wsi_file=output_image_file_class0)
        output_slide_class0_mpp = output_slide_class0.mpp_data
        result_slide_class1 = GenericSlide(wsi_file=result_image_file_class1)
        result_slide_class1_mpp = result_slide_class1.mpp_data
        output_slide_class1 = GenericSlide(wsi_file=output_image_file_class1)
        output_slide_class1_mpp = output_slide_class1.mpp_data
        result_slide_class2 = GenericSlide(wsi_file=result_image_file_class2)
        result_slide_class2_mpp = result_slide_class2.mpp_data
        output_slide_class2 = GenericSlide(wsi_file=output_image_file_class2)
        output_slide_class2_mpp = output_slide_class2.mpp_data
        self.assertEqual(result_slide_class0_mpp, output_slide_class0_mpp)
        self.assertEqual(result_slide_class1_mpp, output_slide_class1_mpp)
        self.assertEqual(result_slide_class2_mpp, output_slide_class2_mpp)

    def test_save_class_tif_level1(self):
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=1,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(self.dataset)
        result_image_file_class0 = make_test_path("saved_data/inference/image_level1_class0.tif")
        result_image_file_class1 = make_test_path("saved_data/inference/image_level1_class1.tif")
        result_image_file_class2 = make_test_path("saved_data/inference/image_level1_class2.tif")
        inference.save_class_tif(0, result_image_file_class0, self.wsi_file, jpeg_compression=False)
        inference.save_class_tif(1, result_image_file_class1, self.wsi_file, jpeg_compression=False)
        inference.save_class_tif(2, result_image_file_class2, self.wsi_file, jpeg_compression=False)
        # read reference values
        output_image_file_class0 = make_test_path("ref_data/slides/processing/inference/image_level1_class0.tif")
        output_image_class0 = Image.open(output_image_file_class0)
        output_image_array_class0 = np.asarray(output_image_class0)
        output_image_class0.close()
        output_image_file_class1 = make_test_path("ref_data/slides/processing/inference/image_level1_class1.tif")
        output_image_class1 = Image.open(output_image_file_class1)
        output_image_array_class1 = np.asarray(output_image_class1)
        output_image_class1.close()
        output_image_file_class2 = make_test_path("ref_data/slides/processing/inference/image_level1_class2.tif")
        output_image_class2 = Image.open(output_image_file_class2)
        output_image_array_class2 = np.asarray(output_image_class2)
        output_image_class2.close()
        # read saved values
        result_image_class0 = Image.open(result_image_file_class0)
        result_image_array_class0 = np.asarray(result_image_class0)
        result_image_class0.close()
        result_image_class1 = Image.open(result_image_file_class1)
        result_image_array_class1 = np.asarray(result_image_class1)
        result_image_class1.close()
        result_image_class2 = Image.open(result_image_file_class2)
        result_image_array_class2 = np.asarray(result_image_class2)
        result_image_class2.close()
        # compare values
        np.testing.assert_equal(result_image_array_class0, output_image_array_class0)
        np.testing.assert_equal(result_image_array_class1, output_image_array_class1)
        np.testing.assert_equal(result_image_array_class2, output_image_array_class2)
        # compare MPP
        result_slide_class0 = GenericSlide(wsi_file=result_image_file_class0)
        result_slide_class0_mpp = result_slide_class0.mpp_data
        output_slide_class0 = GenericSlide(wsi_file=output_image_file_class0)
        output_slide_class0_mpp = output_slide_class0.mpp_data
        result_slide_class1 = GenericSlide(wsi_file=result_image_file_class1)
        result_slide_class1_mpp = result_slide_class1.mpp_data
        output_slide_class1 = GenericSlide(wsi_file=output_image_file_class1)
        output_slide_class1_mpp = output_slide_class1.mpp_data
        result_slide_class2 = GenericSlide(wsi_file=result_image_file_class2)
        result_slide_class2_mpp = result_slide_class2.mpp_data
        output_slide_class2 = GenericSlide(wsi_file=output_image_file_class2)
        output_slide_class2_mpp = output_slide_class2.mpp_data
        self.assertEqual(result_slide_class0_mpp, output_slide_class0_mpp)
        self.assertEqual(result_slide_class1_mpp, output_slide_class1_mpp)
        self.assertEqual(result_slide_class2_mpp, output_slide_class2_mpp)

    def test_save_class_tif_level2(self):
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=2,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(self.dataset)
        result_image_file_class0 = make_test_path("saved_data/inference/image_level2_class0.tif")
        result_image_file_class1 = make_test_path("saved_data/inference/image_level2_class1.tif")
        result_image_file_class2 = make_test_path("saved_data/inference/image_level2_class2.tif")
        inference.save_class_tif(0, result_image_file_class0, self.wsi_file, jpeg_compression=False)
        inference.save_class_tif(1, result_image_file_class1, self.wsi_file, jpeg_compression=False)
        inference.save_class_tif(2, result_image_file_class2, self.wsi_file, jpeg_compression=False)
        # read reference values
        output_image_file_class0 = make_test_path("ref_data/slides/processing/inference/image_level2_class0.tif")
        output_image_class0 = Image.open(output_image_file_class0)
        output_image_array_class0 = np.asarray(output_image_class0)
        output_image_class0.close()
        output_image_file_class1 = make_test_path("ref_data/slides/processing/inference/image_level2_class1.tif")
        output_image_class1 = Image.open(output_image_file_class1)
        output_image_array_class1 = np.asarray(output_image_class1)
        output_image_class1.close()
        output_image_file_class2 = make_test_path("ref_data/slides/processing/inference/image_level2_class2.tif")
        output_image_class2 = Image.open(output_image_file_class2)
        output_image_array_class2 = np.asarray(output_image_class2)
        output_image_class2.close()
        # read saved values
        result_image_class0 = Image.open(result_image_file_class0)
        result_image_array_class0 = np.asarray(result_image_class0)
        result_image_class0.close()
        result_image_class1 = Image.open(result_image_file_class1)
        result_image_array_class1 = np.asarray(result_image_class1)
        result_image_class1.close()
        result_image_class2 = Image.open(result_image_file_class2)
        result_image_array_class2 = np.asarray(result_image_class2)
        result_image_class2.close()
        # compare values
        np.testing.assert_equal(result_image_array_class0, output_image_array_class0)
        np.testing.assert_equal(result_image_array_class1, output_image_array_class1)
        np.testing.assert_equal(result_image_array_class2, output_image_array_class2)
        # compare MPP
        result_slide_class0 = GenericSlide(wsi_file=result_image_file_class0)
        result_slide_class0_mpp = result_slide_class0.mpp_data
        output_slide_class0 = GenericSlide(wsi_file=output_image_file_class0)
        output_slide_class0_mpp = output_slide_class0.mpp_data
        result_slide_class1 = GenericSlide(wsi_file=result_image_file_class1)
        result_slide_class1_mpp = result_slide_class1.mpp_data
        output_slide_class1 = GenericSlide(wsi_file=output_image_file_class1)
        output_slide_class1_mpp = output_slide_class1.mpp_data
        result_slide_class2 = GenericSlide(wsi_file=result_image_file_class2)
        result_slide_class2_mpp = result_slide_class2.mpp_data
        output_slide_class2 = GenericSlide(wsi_file=output_image_file_class2)
        output_slide_class2_mpp = output_slide_class2.mpp_data
        self.assertEqual(result_slide_class0_mpp, output_slide_class0_mpp)
        self.assertEqual(result_slide_class1_mpp, output_slide_class1_mpp)
        self.assertEqual(result_slide_class2_mpp, output_slide_class2_mpp)

    def test_save_class_tif_level0_compression(self):
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(self.dataset)
        result_image_file_class1 = make_test_path("saved_data/inference/image_level0_class1_compressed.tif")
        inference.save_class_tif(1, result_image_file_class1, self.wsi_file, jpeg_compression=True)
        self.assertTrue(os.path.getsize(result_image_file_class1) < 250000)

    def test_save_class_tif_downsample_factor(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp-clean.tif")
        inference = WSIInference(
            model=self.model,
            classifier=self.classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(self.dataset)
        # save df=1
        result_image_file_class = make_test_path("saved_data/inference/image_level0_class1_df1.tif")
        inference.save_class_tif(1, result_image_file_class, self.wsi_file, downsample_factor=1, jpeg_compression=False)
        # read saved image
        result_image = Image.open(result_image_file_class)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/inference/image_level0_class1_df1.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_file_class)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, wsi_slide.mpp_data)
        # save df=4
        result_image_file_class = make_test_path("saved_data/inference/image_level0_class1_df4.tif")
        inference.save_class_tif(1, result_image_file_class, self.wsi_file, downsample_factor=4, jpeg_compression=False)
        # read saved image
        result_image = Image.open(result_image_file_class)
        result_image_array = np.asarray(result_image)
        result_image.close()
        # read reference image
        output_image_path = make_test_path("ref_data/slides/processing/inference/image_level0_class1_df4.tif")
        output_image = Image.open(output_image_path)
        output_image_array = np.asarray(output_image)
        output_image.close()
        # compare
        np.testing.assert_equal(result_image_array, output_image_array)
        wsi_slide = GenericSlide(wsi_file=wsi_file_tif)
        result_slide = GenericSlide(wsi_file=result_image_file_class)
        output_slide = GenericSlide(wsi_file=output_image_path)
        np.testing.assert_equal(result_slide.mpp_data, output_slide.mpp_data)
        np.testing.assert_equal(result_slide.mpp_data, (wsi_slide.mpp_data[0] * 4, wsi_slide.mpp_data[1] * 4))


class TestWSIInferenceFileSavingOutOfRange(TestCase):
    """Tests for checking range of values when saving images."""

    def setUp(self):
        self.wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        patches = WholeImageGridPatches(
            wsi_file=self.wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        dataset = WSIDataset(patches=patches)
        model = DummyFixedClassificationModelThreeClasses()
        classifier = DummyFunctionClassifierThreeClasses.action_fn
        self.inference = WSIInference(
            model=model,
            classifier=classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        self.inference.process_dataset(dataset)

    def test_save_class_png(self):
        result_image_file_class0 = make_test_path("saved_data/inference/image_range_class0.png")
        with self.assertRaises(ValueError):
            self.inference.save_class_png(0, result_image_file_class0)

    def test_save_class_tif(self):
        result_image_file_class0 = make_test_path("saved_data/inference/image_range_class0.tif")
        with self.assertRaises(ValueError):
            self.inference.save_class_tif(0, result_image_file_class0, self.wsi_file)


class TestWSIInferenceTwoDatasets(TestCase):
    """Tests for running inference on two images (datasets) in one go."""

    def test_twodatasets(self):
        wsi_file = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        mask_data = np.zeros((160, 192), dtype=np.uint8)  # level2
        mask_data[16:32, 48:64] = 1  # patch16
        mask_data[32:48, 64:80] = 1  # patch29
        patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_data,
            patch_size=256,
            level_or_mpp=0,
            patch_stride=1,
            foreground_ratio=0.8,
            overlap_ratio=0.8,
        )
        dataset1 = WSIDataset(patches=patches)
        dataset2 = WSIDataset(patches=patches)
        model = DummyFixedClassificationModelThreeClasses()
        classifier = DummyFunctionClassifierThreeClasses.action_fn
        inference = WSIInference(
            model=model,
            classifier=classifier,
            level_or_minsize=0,
            num_classes=3,
            num_workers=1,
            batch_size=5,
            use_cuda=False,
        )
        inference.process_dataset(dataset1)
        inference_results1 = inference.classes_array
        inference.process_dataset(dataset2)
        inference_results2 = inference.classes_array
        np.testing.assert_equal(inference_results1, inference_results2)
        self.assertNotEqual(id(inference_results1), id(inference_results2))
