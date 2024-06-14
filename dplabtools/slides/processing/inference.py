# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for running inference on WSIs using PyTorch models."""

import math
import random

import numpy as np
import torch
from torch.backends import cudnn
from PIL import Image
import cv2

from dplabtools.slides import GenericSlide
from dplabtools.slides.utils.wsi import (
    get_wsi_downsample_factor,
    compute_wsi_resolution_data,
    get_level_or_level,
    get_level_and_mpp,
    get_target_resample_factor,
)
from dplabtools.slides.utils.image import save_tif_image_with_resolution

# PyTorch optimization which may not be supported by some models
cudnn.benchmark = True


class WSIInference:
    """Class for running inference on WSIs using PyTorch models."""

    # In segmentation models, if inference output size is different from level 0 WSI size, each patch processed by
    # model must be scaled down to match the inference level patch. This is easy to do in classification, where
    # model output for each patch is represented by a number, however in segmentation tasks model output must
    # be treated as an image and resized using one of the interpolation methods.
    _interpolation_method = cv2.INTER_LINEAR

    def __init__(
        self, *, model, classifier, level_or_minsize, num_classes, num_workers, batch_size, use_cuda=True, seed=None
    ):
        """Class for running inference on WSIs using PyTorch models.

        Parameters
        ----------
        model : callable
            PyTorch model properly initialized.

        classifier : callable
            PyTorch model of function capable of processing model's output.

        level_or_minsize : int
            WSI level or minimal desired size (in pixels) of inference output array.

        num_classes : int
            Number of classes present in model output.

        num_workers : int
            Number of worker processes used in data loading.

        batch_size : int
            Number of samples per batch to load into GPU.

        use_cuda : bool, default=True
            Declaration whether model will be using CUDA/GPU for processing or not, `False` will indicate pure CPU
            processing.

        seed : int or float, optional
            Custom seed for random number generators.
        """
        self._model = None
        self._classifier = None
        self._level_or_minsize = level_or_minsize
        self._num_classes = num_classes
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._use_cuda = use_cuda
        self._seed = seed
        self._torch_device = None
        self._pin_memory = None
        self._probs_array = None
        self._patch_size = None
        self._output_str = "classifier" if classifier else "model"
        self._set_torch()
        self._set_seed()
        self._set_model(model)
        self._set_classifier(classifier)

    def _set_torch(self):
        if torch.cuda.is_available() and self._use_cuda:
            self._torch_device = torch.device("cuda")
            self._pin_memory = True
        else:
            self._torch_device = torch.device("cpu")
            self._pin_memory = False

    def _set_seed(self):
        if self._seed:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            if self._torch_device == "cuda":
                torch.cuda.manual_seed_all(self._seed)

    def _set_model(self, model):
        self._model = model

    def _set_classifier(self, classifier):
        self._classifier = classifier

    def process_dataset(self, wsi_dataset):
        """Compute model/classifier output values for the whole WSI dataset.

        Parameters
        ----------
        wsi_dataset : WsiDataset or WsiMultiResDataset
            Dataset representing patches from one WSI.
        """
        self._patch_size = wsi_dataset.patches.patch_size
        wsi_dataloader = torch.utils.data.DataLoader(
            wsi_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            worker_init_fn=wsi_dataset.worker_init,
        )
        cumulative_probs, cumulative_counts = self._get_model_probs(wsi_dataloader, wsi_dataset)
        self._probs_array = self._get_final_probs(cumulative_probs, cumulative_counts)

    def _get_model_probs(self, wsi_dataloader, wsi_dataset):

        # helper function
        def get_patch_scale_factor(patches_level_or_mpp, inference_downsample_factor, wsi_slide):
            patch_level, patch_mpp = get_level_and_mpp(patches_level_or_mpp)
            target_resample_factor = get_target_resample_factor(
                patch_level, patch_mpp, wsi_slide.level_downsamples, wsi_slide.level_mpp_values
            )
            return inference_downsample_factor / target_resample_factor

        wsi_level = get_level_or_level(wsi_dataset.patches.wsi_slide, self._level_or_minsize)
        inference_downsample_factor = wsi_dataset.patches.wsi_slide.level_downsamples[wsi_level]
        # scale ratio from patches level_or_mpp to inference level
        patch_scale_factor = get_patch_scale_factor(
            wsi_dataset.patches.level_or_mpp, inference_downsample_factor, wsi_dataset.patches.wsi_slide
        )
        # scale ratio from level 0 to inference level
        location_scale_factor = get_target_resample_factor(
            wsi_level,
            None,
            wsi_dataset.patches.wsi_slide.level_downsamples,
            wsi_dataset.patches.wsi_slide.level_mpp_values,
        )
        inference_width, inference_height = wsi_dataset.patches.wsi_slide.level_dimensions[wsi_level]
        inference_patch_size = round(self._patch_size / patch_scale_factor)

        with torch.no_grad():
            cumulative_probs = np.zeros([self._num_classes, inference_width, inference_height], dtype=float)
            cumulative_counts = np.zeros([self._num_classes, inference_width, inference_height], dtype=int)
            # loop through all batches
            for (image_data, x_level0_data, y_level0_data) in wsi_dataloader:
                # get model output:
                # - multi resolution patches
                if isinstance(image_data, list):
                    images = tuple(image.to(device=self._torch_device, dtype=torch.float) for image in image_data)
                    model_output = self._model(*images)
                # - single resolution patch
                else:
                    image = image_data.to(device=self._torch_device, dtype=torch.float)
                    model_output = self._model(image)
                # run classifier on model output
                if self._classifier:
                    classifier_output = self._classifier(model_output)
                else:
                    classifier_output = model_output
                if classifier_output.shape[1] != self._num_classes:
                    raise ValueError(
                        "Incorrect number of classes in %s output: %d" % (self._output_str, classifier_output.shape[1])
                    )
                classifier_output = classifier_output.cpu()
                classifier_output = classifier_output.data.numpy()
                classifier_output_shape_len = len(classifier_output.shape)
                # loop through all images in one batch
                for i, _ in enumerate(x_level0_data):
                    # use floor instead of round to prevent white pixel lines between tiles on heatmaps
                    inference_patch_x_min = math.floor(x_level0_data[i] / location_scale_factor)
                    inference_patch_x_max = math.floor(x_level0_data[i] / location_scale_factor) + inference_patch_size
                    inference_patch_y_min = math.floor(y_level0_data[i] / location_scale_factor)
                    inference_patch_y_max = math.floor(y_level0_data[i] / location_scale_factor) + inference_patch_size
                    # loop through all classes
                    for c in range(self._num_classes):
                        # select previously saved output values into cumulative_probs_inference_patch
                        # (this is on inference level)
                        cumulative_probs_inference_patch = cumulative_probs[
                            c, inference_patch_x_min:inference_patch_x_max, inference_patch_y_min:inference_patch_y_max
                        ]
                        # add new output values to values already saved:
                        # - classification output
                        if classifier_output_shape_len == 2:
                            cumulative_probs_inference_patch += classifier_output[i, c]
                        # - segmentation output
                        elif classifier_output_shape_len == 4:
                            # In segmentation it may happen that patch size will exceed size of probability array
                            # allocated for that patch, this occurs when extracted patch exceeds image boundaries.
                            # Then raw_inference_patch may need trimming before it can be added to the cumulative array.
                            raw_inference_patch = classifier_output[i, c]
                            # inference patch must be resized if required
                            if raw_inference_patch.shape != (inference_patch_size, inference_patch_size):
                                raw_inference_patch = cv2.resize(
                                    raw_inference_patch,
                                    dsize=(inference_patch_size, inference_patch_size),
                                    interpolation=self.interpolation_method,
                                )
                            # and transposed (which must be applied AFTER resizing)
                            raw_inference_patch = raw_inference_patch.transpose()
                            # check if raw_inference_patch needs trimming
                            slice_limit_x = raw_inference_patch.shape[0]
                            slice_limit_y = raw_inference_patch.shape[1]
                            if cumulative_probs_inference_patch.shape[0] < raw_inference_patch.shape[0]:
                                slice_limit_x = cumulative_probs_inference_patch.shape[0]
                            if cumulative_probs_inference_patch.shape[1] < raw_inference_patch.shape[1]:
                                slice_limit_y = cumulative_probs_inference_patch.shape[1]
                            cumulative_probs_inference_patch += raw_inference_patch[0:slice_limit_x, 0:slice_limit_y]
                        else:
                            raise ValueError(
                                "Unsupported %s output shape length: %d"
                                % (self._output_str, len(classifier_output.shape))
                            )

                        # assign updated output values to inference level array
                        cumulative_probs[
                            c, inference_patch_x_min:inference_patch_x_max, inference_patch_y_min:inference_patch_y_max
                        ] = cumulative_probs_inference_patch
                        # update counters as well
                        cumulative_counts[
                            c, inference_patch_x_min:inference_patch_x_max, inference_patch_y_min:inference_patch_y_max
                        ] += 1

        return cumulative_probs, cumulative_counts

    @staticmethod
    def _get_final_probs(cumulative_probs, cumulative_counts):
        """Calculate final probabilities per pixel/regions.

        Output array will include NaN (not a number) values for the pixels/regions which have not been sent for
        processing by model (cumulative_counts=0 are replaced with np.nan).

        np.errstate context manager is required to suppress warning:
            "RuntimeWarning: invalid value encountered in true_divide"
        This happens because division is evaluated before np.where, more details: SO: #27842884, #25087769
        """
        with np.errstate(invalid="ignore"):
            probs = np.where(cumulative_counts != 0, cumulative_probs / cumulative_counts, np.nan)
        return probs

    def save_classes_array(self, array_file):
        """Save probabilities for all classes as a compressed NumPy array.

        Parameters
        ----------
        array_file : str
            File name or path for saving the NPZ file.
        """
        np.savez_compressed(array_file, data=self._probs_array)

    def save_class_array(self, class_index, array_file):
        """Save probabilities for one class as a compressed NumPy array.

        Parameters
        ----------
        class_index : int
            Index of class to be saved.

        array_file : str
            File name or path for saving the NPZ file.
        """
        np.savez_compressed(array_file, data=self._probs_array[class_index])

    def save_class_png(self, class_index, png_file):
        """Save probabilities as a PNG image.

        Parameters
        ----------
        class_index : int
            Index of class to be saved.

        png_file : str
            File name or path for saving the PNG file.
        """
        image = self._get_image(class_index, self._probs_array)
        image.save(png_file, "PNG")

    def save_class_tif(self, class_index, tif_file, wsi_file, downsample_factor=None, jpeg_compression=True):
        """Save probabilities as a TIF image with resolution information embedded.

        Parameters
        ----------
        class_index : int
            Index of class to be saved.

        tif_file : str
            File name or path for saving the PNG file.

        wsi_file : str
            WSI file name or path used in the inference (for extracting resolution information).

        downsample_factor : float, optional
            Downsample factor used for resolution information. If not provided, then the value will be determined based
            on ``level_or_minsize`` inference parameter.

        jpeg_compression : bool, default=True
            Whether internal JPEG compression should be used or not.

        """
        wsi_slide = GenericSlide(wsi_file=wsi_file)
        image = self._get_image(class_index, self._probs_array)
        image_rgb_data = np.asarray(image.convert("RGB"))
        if not downsample_factor:
            downsample_factor = get_wsi_downsample_factor(wsi_slide, self._probs_array[class_index].shape)
        resolution_data = compute_wsi_resolution_data(wsi_slide, downsample_factor)
        save_tif_image_with_resolution(image_rgb_data, tif_file, resolution_data, jpeg_compression)

    @classmethod
    def set_interpolation_method(cls, interpolation_method):
        """Set interpolation method used for patch resizing in segmentation models output.

        Parameters
        ----------
        interpolation_method : cv2 enum, default=cv2.INTER_LINEAR
            Interpolation method to be used.
            Available methods: https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html
        """
        cls._interpolation_method = interpolation_method

    @staticmethod
    def _get_image(class_index, probs_array):
        probs_array_transposed = np.transpose(probs_array, (0, 2, 1))
        probs = probs_array_transposed[class_index]
        probs = np.nan_to_num(probs, copy=False, nan=0.0)
        WSIInference._check_array_range(probs)
        image = Image.fromarray(np.uint8(probs * 255))
        return image

    @staticmethod
    def _check_array_range(in_array):
        if np.any((in_array < 0) | (in_array > 1)):
            raise ValueError("Array values out of range [0-1]")

    @property
    def classes_array(self):
        """Return probabilities array for all classes."""
        return self._probs_array

    @property
    def torch_device(self):
        """Return torch device name."""
        return self._torch_device

    @property
    def interpolation_method(self):
        """Return `interpolation_method` value set by ``set_interpolation_method`` (default=cv2.INTER_LINEAR)."""
        return self._interpolation_method
