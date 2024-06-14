# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""WSI related utilities for slides."""

import os
import importlib
from functools import lru_cache

from skimage import io
from PIL import Image

LEVEL_MINSIZE_SPLIT = 99


@lru_cache
def _get_resize_filter_value(resize_filter):
    resize_filter_module = importlib.import_module("PIL.Image")
    resize_filter_object = getattr(resize_filter_module, "Resampling")
    resize_filter_value = resize_filter_object[resize_filter]
    return resize_filter_value


def get_wsi_name(wsi_file):
    """Return slide file name from file path."""
    wsi_name = os.path.basename(wsi_file)
    return wsi_name


def get_wsi_id(wsi_file):
    """Return slide id from file path.

    id = file name without extension
    """
    wsi_name = os.path.basename(wsi_file)
    wsi_id = os.path.splitext(wsi_name)[0]
    return wsi_id


def get_wsi_level_image(wsi_slide, level):
    """Return complete WSI level as image."""
    level_image = wsi_slide.get_region((0, 0), level, wsi_slide.level_dimensions[level], skip_padding=True)
    return level_image


def get_wsi_level_array(wsi_slide, level):
    """Return complete WSI level as array."""
    level_array = wsi_slide.get_region_array((0, 0), level, wsi_slide.level_dimensions[level])
    return level_array


def get_wsi_level_zero_array(wsi_file):
    """Fast reading of level0 image as array using multithreading."""
    return io.imread(wsi_file)


def find_wsi_level(wsi_slide, array_shape):
    """Return level number for given shape/dimensions."""
    level_found = None
    for level_id, level_size in reversed(list(enumerate(wsi_slide.level_dimensions))):
        if level_size == array_shape:
            level_found = level_id
            break
    if level_found is None:
        raise ValueError(
            "Could not find matching WSI level for width=%d and height=%d" % (array_shape[0], array_shape[1])
        )
    return level_found


def get_wsi_downsample_factor(wsi_slide, array_shape):
    """Return embedded downsample factor for given shape/dimensions."""
    level = find_wsi_level(wsi_slide, array_shape)
    downsample_factor = wsi_slide.level_downsamples[level]
    return downsample_factor


def compute_wsi_resolution_data(wsi_slide, downsample_factor):
    """Compute resolution data required when saving tif files with resolution information embedded."""
    mpp_x_slide, mpp_y_slide = wsi_slide.mpp_data
    # 10000 is conversion from microns to centimeters
    mpp_x_scaled = 10000 / (mpp_x_slide * downsample_factor)
    mpp_y_scaled = 10000 / (mpp_y_slide * downsample_factor)
    resolution_data = (mpp_x_scaled, mpp_y_scaled, "CENTIMETER")
    return resolution_data


def find_nearest_wsi_level(wsi_slide, minsize, allow_level_zero=False):
    """Return the highest level number matching conditions: level.width >= minsize and level.height >= minsize."""
    dimensions = wsi_slide.level_dimensions
    level_found = 0
    for level_id, level_size in reversed(list(enumerate(dimensions))):
        level_size_x, level_size_y = level_size
        if level_size_x >= minsize and level_size_y >= minsize:
            level_found = level_id
            break
    if not allow_level_zero and level_found == 0:
        raise ValueError(
            "Could not find matching level for size %s. Existing levels: %s, allow_level_zero is %s."
            % (minsize, str(dimensions), str(allow_level_zero))
        )
    return level_found


def get_resampled_wsi_images(wsi_file, size_list, resize_filter="LANCZOS"):
    """Return list of resampled level zero images in RGB format.

    - This resamplig operation will always use level zero as base.
    - Resampling process will require significant amount of available RAM.
    - resize_filter docs: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
    """
    resized_images = []
    image_array = io.imread(wsi_file)
    resize_filter_value = _get_resize_filter_value(resize_filter)
    # convert to RGB:
    image_array = image_array[:, :, 0:3]
    image = Image.fromarray(image_array)
    for one_size in size_list:
        resized_image = image.resize(one_size, resample=resize_filter_value)
        resized_images.append(resized_image)
    del image_array, image
    return resized_images


def get_resampled_tiles(wsi_tile, size_list, resize_filter="LANCZOS"):
    """Return list of downsampled tiles."""
    resized_tiles = []
    resize_filter_value = _get_resize_filter_value(resize_filter)
    for one_size in size_list:
        resized_tile = wsi_tile.resize(one_size, resample=resize_filter_value)
        resized_tiles.append(resized_tile)
    return resized_tiles


def get_level_and_mpp(level_or_mpp):
    """Return tuple (level, MPP) information extracted from one variable.

    Preferred way of using result information in the code is (as mpp can never be zero):
        if mpp:
            ...
        else:
            ...

    i.e. always check mpp, as level is allowed to be zero.
    """
    level = None
    mpp = None
    if level_or_mpp == 0 and isinstance(level_or_mpp, float):
        raise ValueError("Invalid MPP value: %s" % str(level_or_mpp))
    if isinstance(level_or_mpp, float):
        mpp = level_or_mpp
    elif isinstance(level_or_mpp, int):
        level = level_or_mpp
    else:
        raise ValueError("Invalid MPP or level value: %s" % str(level_or_mpp))
    return (level, mpp)


def get_basepatch_mpp(basepatch_level_or_mpp, level_mpp_values):
    """Return MPP value used to compute base patches."""
    basepatch_mpp = None
    level, mpp = get_level_and_mpp(basepatch_level_or_mpp)
    if mpp:
        basepatch_mpp = basepatch_level_or_mpp
    else:
        basepatch_mpp = level_mpp_values[basepatch_level_or_mpp]
    return basepatch_mpp


def get_level_or_level(wsi_slide, level_or_minsize):
    """Return level or level computed from minsize."""
    level = level_or_minsize
    if level_or_minsize > LEVEL_MINSIZE_SPLIT:
        level = find_nearest_wsi_level(wsi_slide, level_or_minsize, allow_level_zero=True)
    return level


def get_target_resample_factor(patch_level, patch_mpp, level_downsamples, level_mpp_values):
    """Calculate target resample factor in patch computing.

    target resample factor is: ratio between level zero and level/mpp of patches to be retrieved
    """
    target_resample_factor = None
    if patch_level is not None:
        target_resample_factor = level_downsamples[patch_level]
    elif patch_mpp is not None:
        target_resample_factor = patch_mpp / level_mpp_values[0]
    return target_resample_factor
