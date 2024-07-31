# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Implementation of method calls for AbstractLib using Pillow."""

from PIL import Image

from dplabtools.slides.libs.abstractlib import AbstractLib


class SlideLib(AbstractLib):
    """Implementation of abstract methods in AbstractLib using Pillow."""

    def _init_slide(self, wsi_file):
        self._slide = Image.open(wsi_file)
        # required for multi-threading
        self._slide.load()

    def _get_magnification(self):
        return None

    def _get_level_dimensions(self):
        dimensions = ((self._slide.width, self._slide.height),)
        return dimensions

    def _get_level_downsamples(self):
        return [1.0]

    def _get_mpp_data(self):
        raise ValueError

    def _get_region(self, location, level, size):
        region = self._slide.crop((location[0], location[1], location[0] + size[0], location[1] + size[1]))
        return region

    def _get_region_array(self, location, level, size):
        raise NotImplementedError("Feature not supported with Pillow")

    def _get_property(self, name):
        return None

    def _get_thumbnail_image(self):
        thumbnail = self._slide.resize((800, 800))
        return thumbnail

    def _get_label_image(self):
        return None

    def _get_all_properties(self):
        return None

    def _get_libname(self):
        return "pillow"
