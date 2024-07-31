# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Mask related utilities for slides."""

import numpy as np

from dplabtools.common import roundfl


def get_mask_bounding_box(mask_array):
    """Return bounding box for mask array."""
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    if any(rows) and any(cols):
        x_min, x_max = np.where(rows)[0][[0, -1]]
        y_min, y_max = np.where(cols)[0][[0, -1]]
    else:
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
    return (x_min, y_min, x_max, y_max)


def get_mask_pixel_step(patch_size, patch_stride, downsample_factor, max_stride=5):
    """Calculate mask pixel step for stride based operations."""
    mask_pixel_step = patch_stride * patch_size / downsample_factor
    if roundfl(mask_pixel_step * downsample_factor) < 1:
        raise ValueError("Patch stride value is too small (%f), calculated pixel step at level 0 is < 1" % patch_stride)
    if patch_stride > max_stride:
        raise ValueError(
            "Patch stride value is suspiciously large (%f), current max allowed value is %d"
            % (patch_stride, max_stride)
        )
    return mask_pixel_step
