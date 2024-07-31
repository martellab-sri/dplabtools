# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for WSI datasets."""

import os
from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from dplabtools.slides import GenericSlide


class BaseDataset(ABC, Dataset):
    """Base class for the WSI dataset classes."""

    save_patches_image_type = "png"
    """Class attribute for setting the image type when ``save_patches_dir`` is defined."""

    def __init__(
        self,
        *,
        patches,
        transform_fn=None,
        resampling_mode=None,
        extra_mpps=[],
        zero_workers=False,
        save_patches_dir=None
    ):
        """Init function for the base class.

        Parameters
        ----------
        patches : object
            Object representing one of the patch location classes.

        transform_fn : callable, optional
            A user-defined image transformation that will be called on each patch extracted via ``get_region``,
            this transformation should run its own to-tensor conversion. If no transformation is provided,
            the image objects will be converted to tensors using `to_tensor` from the `torchvision` package.

        resampling_mode : str, optional
            One of two supported down/up-sampling methods: ``wsi`` or ``tile``.

        extra_mpps : list of float, optional
            List of MPP values for ``wsi`` resampling mode.

        zero_workers : bool, default=False
            Set to `True` if dataloaders using the dataset will have `num_workers` set to 0.

        save_patches_dir : str, optional
            Directory for saving the extracted patches, should only be used for troubleshooting inference problems.
            The type of saved image files can be changed using ``save_patches_image_type`` class attribute,
            the default image type is PNG.
        """
        self._patches = patches
        self._transform_fn = transform_fn
        self._resampling_mode = resampling_mode
        self._extra_mpps = extra_mpps
        self._save_patches_dir = save_patches_dir
        self._patch_name = "patch%s_x%d_y%d.%s"
        self._patchset_name = "patchset_x%d_y%d"
        self._patch_list = self._patches.patch_data
        self._wsi_file = self._patches.wsi_file
        self._patch_size = self._patches.patch_size
        self._level_mpp_values = self._patches.wsi_slide.level_mpp_values
        self._worker_wsi_slide = None
        if zero_workers:
            self._worker_wsi_slide = GenericSlide(
                wsi_file=self._wsi_file, resampling_mode=self._resampling_mode, extra_mpps=self._extra_mpps
            )
        if self._save_patches_dir and not os.path.exists(self._save_patches_dir):
            os.mkdir(self._save_patches_dir)

    def worker_init(self, *args):
        """Init worker processes.

        This function will be called by torch DataLoader to make sure that the slide object is not shared between
        processes.

        Method param *args must be present here, even when not used. Otherwise this torch error:
            "TypeError: worker_init() takes 1 positional argument but 2 were given"
        """
        self._worker_wsi_slide = GenericSlide(
            wsi_file=self._wsi_file, resampling_mode=self._resampling_mode, extra_mpps=self._extra_mpps
        )

    def __len__(self):
        """Return lenght of dataset."""
        return len(self._patch_list)

    @abstractmethod
    def __getitem__(self, idx):
        """Return one dataset item."""
        pass

    @abstractmethod
    def _save_patches(self):
        """Save patches into directory."""
        pass

    @property
    def patches(self):
        """Return the ``patches`` object used to build the dataset."""
        return self._patches
