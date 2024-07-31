# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Image related utilities for slides."""

from PIL import Image, ImageColor
import numpy as np
import tifffile as tf
from matplotlib import colors


def save_tif_image_with_resolution(image_data, tif_file, resolution_data, jpeg_compression=True):
    """Save array data to tif file with resolution information embedded."""
    # - resolution is called pixel spacing in Sedeen
    # - saving using RGBA data will work only if jpeg compression is not used (generates much larger file)
    # - with jpeg compression image data must be in RGB format
    # - only one level is saved here (level0)
    compression = "jpeg" if jpeg_compression else None
    with tf.TiffWriter(tif_file) as tif:
        options = dict(tile=(256, 256), photometric="rgb", compression=compression, resolution=resolution_data)
        tif.write(image_data, **options)


def pad_image_zero_background(image, background_color=[255, 255, 255]):
    """Replace fixed black (zero) background with a new background color, only for rows/columns which are full of 0s."""
    background_changed = False
    # array copying is required by recent versions of Pillow, when behaviour was changed,
    # older Pillow versions do not require it, details:
    # https://github.com/python-pillow/Pillow/issues/6581
    image_array = np.copy(np.asarray(image))
    # find all rows and columns with all zeros
    rows = np.any(image_array, axis=1)
    cols = np.any(image_array, axis=0)
    rows_with_zeros = np.where(~rows.any(axis=1))[0]
    columns_with_zeros = np.where(~cols.any(axis=1))[0]
    # change background
    if len(rows_with_zeros):
        image_array[rows_with_zeros[0] : rows_with_zeros[-1] + 1] = background_color
        background_changed = True
    if len(columns_with_zeros):
        image_array[:, columns_with_zeros[0] : columns_with_zeros[-1] + 1] = background_color
        background_changed = True
    if background_changed:
        image = Image.fromarray(image_array)
    return image


def get_transparency(alpha):
    """Convert alpha value to RBG space value."""
    return round(255 * alpha)


def get_color_hex(color):
    """Convert color string to hex value."""
    if color.startswith("#") and len(color) == 7:
        color_hex = color
    else:
        color_hex = colors.cnames[color]
    return color_hex


def get_color_rgb(color_hex):
    """Convert hex color to RGB tuple."""
    return ImageColor.getcolor(color_hex, "RGB")
