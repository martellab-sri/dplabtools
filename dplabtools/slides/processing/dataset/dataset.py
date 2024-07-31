# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""WSI dataset classes."""

import os

import torchvision.transforms.functional as tf

from dplabtools.slides.processing.dataset.base import BaseDataset
from dplabtools.slides.utils.wsi import get_basepatch_mpp
from dplabtools.slides.utils.multires import get_scaled_center_based_locations


class WSIDataset(BaseDataset):
    """Class for patch extraction during inference."""

    def __getitem__(self, index):
        """Return one patch (as tensor) along with level 0 coordinates."""
        x_level0, y_level0 = self._patch_list[index][0]
        patch_image = self._worker_wsi_slide.get_region(
            (x_level0, y_level0), self._patches.level_or_mpp, (self._patch_size, self._patch_size)
        )
        if self._transform_fn:
            patch_image_data = self._transform_fn(patch_image)
        else:
            patch_image_data = tf.to_tensor(patch_image)
        if self._save_patches_dir:
            self._save_patches(patch_image, (x_level0, y_level0), image_type=self.save_patches_image_type)
        return patch_image_data, x_level0, y_level0

    def _save_patches(self, patch_image, location, image_type):
        x_level0, y_level0 = location
        file_name = self._patch_name % ("", x_level0, y_level0, image_type)
        file_path = os.path.join(self._save_patches_dir, file_name)
        patch_image.save(file_path)


class WSIMultiResDataset(BaseDataset):
    """Class for multi resolution patch extraction during inference."""

    def __init__(self, *, levels_or_mpps, **kwargs):
        """Init method for WSIMultiResDataset.

        Parameters
        ----------
        levels_or_mpps : list of level_or_mpp values
            Numbers representing WSI levels or MPP values for multi resolution patches.
        """
        self._levels_or_mpps = levels_or_mpps
        super().__init__(**kwargs)
        self._basepatch_mpp = get_basepatch_mpp(self._patches.level_or_mpp, self._level_mpp_values)
        self._level_or_mpp_loop = list(enumerate(self._levels_or_mpps))

    def __getitem__(self, index):
        """Return patches (as list of tensors) along with level 0 coordinates."""
        x_level0, y_level0 = self._patch_list[index][0]
        patch_images = []
        patch_image_data = []
        scaled_locations = get_scaled_center_based_locations(
            self._basepatch_mpp,
            (x_level0, y_level0),
            self._levels_or_mpps,
            (self._patch_size, self._patch_size),
            self._level_mpp_values,
        )
        for counter, level_or_mpp in self._level_or_mpp_loop:
            patch_image = self._worker_wsi_slide.get_region(
                scaled_locations[counter], level_or_mpp, (self._patch_size, self._patch_size)
            )
            patch_images.append(patch_image)

        if self._transform_fn:
            patch_image_data = self._transform_fn(patch_images)
        else:
            patch_image_data = [tf.to_tensor(image) for image in patch_images]

        if self._save_patches_dir:
            self._save_patches(patch_images, (x_level0, y_level0), image_type=self.save_patches_image_type)
        return patch_image_data, x_level0, y_level0

    def _save_patches(self, patch_images, location, image_type):
        x_level0, y_level0 = location
        dir_name = self._patchset_name % (x_level0, y_level0)
        dir_path = os.path.join(self._save_patches_dir, dir_name)
        os.mkdir(dir_path)
        counter = 0
        for patch_image in patch_images:
            counter += 1
            file_name = self._patch_name % (str(counter), x_level0, y_level0, image_type)
            file_path = os.path.join(dir_path, file_name)
            patch_image.save(file_path)
