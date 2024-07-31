# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Implementation of method calls for AbstractLib using openslide."""

from ctypes import c_uint32
import openslide
import numpy as np

from dplabtools.slides.libs.abstractlib import AbstractLib


class OpenSlideExtended(openslide.OpenSlide):
    """Extension class providing additional functionality."""

    def _read_region_array(self, slide, location, level, size):
        """Read image region as numpy array without converting to Pillow Image object.

        Based on original openslide function: lowlevel.read_region
        Should only be used for extracting large regions, not for single patches.
        """
        x, y = location
        w, h = size
        buff = (w * h * c_uint32)()
        openslide.lowlevel._read_region(slide, buff, x, y, level, w, h)
        openslide.lowlevel._convert.argb2rgba(buff)
        # Steps:
        # - read from buffer and reshape array (w and h are swapped to match output shape from tiffslide)
        # - add new dimension and split each int32 value into 4 int8 values
        region_array = np.frombuffer(buff, dtype="uint32").reshape(h, w)
        region_array = region_array[:, :, np.newaxis].view("uint8")
        return region_array


class SlideLib(AbstractLib):
    """Implementation of abstract methods in AbstractLib using openslide."""

    def _init_slide(self, wsi_file):
        self._slide = OpenSlideExtended(wsi_file)

    def _get_magnification(self):
        magnification = self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        return int(magnification)

    def _get_level_dimensions(self):
        dimensions = self._slide.level_dimensions
        return dimensions

    def _get_level_downsamples(self):
        level_downsamples = self._slide.level_downsamples
        return level_downsamples

    def _get_mpp_data(self):
        slide_mmp_x = float(self._slide.properties[openslide.PROPERTY_NAME_MPP_X])
        slide_mmp_y = float(self._slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        return (slide_mmp_x, slide_mmp_y)

    def _get_region(self, location, level, size):
        region = self._slide.read_region(location, level, size)
        return region

    def _get_region_array(self, location, level, size):
        region = self._slide._read_region_array(self._slide._osr, location, level, size)
        return region

    def _get_property(self, name):
        property_value = self._slide.properties[name]
        return property_value

    def _get_thumbnail_image(self):
        try:
            thumbnail = self._slide.associated_images["thumbnail"]
        except KeyError:
            thumbnail = self._get_fixed_thumbnail_image((800, 800))
        return thumbnail

    def _get_fixed_thumbnail_image(self, size):
        image = self._slide.get_thumbnail(size)
        return image

    def _get_label_image(self):
        label = self._slide.associated_images["label"]
        return label

    def _get_all_properties(self):
        all_properties = self._slide.properties
        return all_properties

    def _get_libname(self):
        return "openslide"
