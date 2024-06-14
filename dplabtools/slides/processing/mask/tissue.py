# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Mask created based on tissue processing."""

import math

import numpy as np
import cv2
from skimage import color
import skimage.morphology as skm
from scipy.ndimage import binary_fill_holes

from dplabtools.slides.utils.wsi import get_wsi_level_array
from dplabtools.slides.processing.mask.base import BaseMask


class WSITissueMask(BaseMask):
    """Class for creating WSI masks based on tissue processing."""

    def __init__(
        self,
        *,
        mode="hsv",
        color_threshold=0.1,
        remove_small_holes_ratio=0.1,
        remove_small_objects_ratio=0.1,
        remove_all_holes=False,
        close_fill_kernel_size=0,
        **kwargs,
    ):
        """Create a WSITissueMask object.

        Parameters
        ----------
        mode : str, default="hsv"
            One of the three modes: ``hsv``, ``lab``, ``otsu``.

        color_threshold : float, default=0.1
            Threshold value used by ``hsv`` and ``lab`` modes.

        remove_small_holes_ratio : float, dafault=0.1
            Determines the maximum size of small holes to be removed in the mask.

        remove_small_objects_ratio : float, dafault=0.1
            Determines the maximum size of small objects to be removed in the mask.

        remove_all_holes : bool, default=False
            Whether to remove all holes in the mask or not.

        close_fill_kernel_size : int, default=0
            Kernel size for smoothing the mask.
        """
        self._mode = mode
        self._color_threshold = color_threshold
        self._remove_small_holes_ratio = remove_small_holes_ratio
        self._remove_small_objects_ratio = remove_small_objects_ratio
        self._remove_all_holes = remove_all_holes
        self._close_fill_kernel_size = close_fill_kernel_size
        super().__init__(**kwargs)

    def _process_wsi_file(self, wsi_file):
        wsi_level_array = get_wsi_level_array(self._wsi_slide, self._level)
        raw_mask = self._create_mask(wsi_level_array, self._mode, self._color_threshold)
        self._mask_array_transposed = self._transform_mask(
            raw_mask,
            self._level,
            self._remove_small_holes_ratio,
            self._remove_small_objects_ratio,
            self._remove_all_holes,
            self._close_fill_kernel_size,
        )
        self._mask_array = np.transpose(self._mask_array_transposed)

    @staticmethod
    def _create_mask(image_array, mode, threshold):
        if mode == "lab":
            lab = color.rgb2lab(image_array)
            mean = np.mean(lab[..., 1])
            lab = lab[..., 1] > (1 + threshold) * mean
            mask = lab.astype(bool)
        elif mode == "hsv":
            hsv = color.rgb2hsv(image_array)
            mean = np.mean(hsv[..., 1])
            hsv = hsv[..., 1] > (1 + threshold) * mean
            mask = hsv.astype(bool)
        elif mode == "otsu":
            otsu = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            _, otsu = cv2.threshold(otsu, 0, 255, cv2.THRESH_OTSU)
            mask = ~otsu.astype(bool)
        else:
            raise ValueError("Unknown mask mode: %s" % mode)
        return mask

    @staticmethod
    def _transform_mask(
        mask, level, remove_small_holes_ratio, remove_small_objects_ratio, remove_all_holes, close_fill_kernel_size
    ):

        # level_factor represents aproximate number of pixels in 256x256 patch, scaled based on the level value
        # Level   |  level_factor (approx.)
        # ---------------------------------
        # 0       |         65536
        # 1       |          4096
        # 2       |           256
        # 3       |            16
        # 4       |             1
        level_factor = 256 * 256 * math.exp(-2.772 * level)

        # remove all binary holes
        if remove_all_holes:
            mask = binary_fill_holes(mask)

        # remove holes with threshold
        if remove_small_holes_ratio:
            max_allowed_area = math.ceil(remove_small_holes_ratio * level_factor)
            mask = skm.remove_small_holes(mask, connectivity=1, area_threshold=max_allowed_area)

        # dilate/close with kernel
        if close_fill_kernel_size:
            kernel = np.ones((close_fill_kernel_size, close_fill_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = mask.astype(bool)

        # remove small objects with threshold
        if remove_small_objects_ratio:
            min_allowed_size = math.ceil(remove_small_objects_ratio * level_factor)
            mask = skm.remove_small_objects(mask, min_size=min_allowed_size)

        return mask.astype(bool)
