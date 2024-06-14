# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Implementation of method calls for AbstractLib using tiffslide in openslide mode."""

import tiffslide as openslide

from dplabtools.slides.libs.abstractlib import AbstractLib


class SlideLib(AbstractLib):
    """Implementation of abstract methods in AbstractLib using tiffslide in openslide mode."""

    def _init_slide(self, wsi_file):
        self._slide = openslide.OpenSlide(wsi_file)

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
        raise NotImplementedError("Feature not supported with Pillow")

    def _get_property(self, name):
        property_value = self._slide.properties[name]
        return property_value

    def _get_thumbnail_image(self):
        thumbnail = self._slide.associated_images["thumbnail"]
        return thumbnail

    def _get_label_image(self):
        label = self._slide.associated_images["label"]
        return label

    def _get_all_properties(self):
        all_properties = self._slide.properties
        return all_properties

    def _get_libname(self):
        return "tiffopenslide"
