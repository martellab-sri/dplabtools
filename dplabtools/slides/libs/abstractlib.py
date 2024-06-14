# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""One class module for slide reading abstraction."""

from abc import ABC, abstractmethod


class AbstractLib(ABC):
    """Abstract class for methods to be implemented by WSI reading libraries as child classes."""

    @abstractmethod
    def _init_slide(self, wsi_file):
        pass

    @abstractmethod
    def _get_magnification(self):
        pass

    @abstractmethod
    def _get_level_dimensions(self):
        pass

    @abstractmethod
    def _get_level_downsamples(self):
        pass

    @abstractmethod
    def _get_mpp_data(self):
        pass

    @abstractmethod
    def _get_region(self, location, level, size):
        pass

    @abstractmethod
    def _get_region_array(self, location, level, size):
        pass

    @abstractmethod
    def _get_property(self, name):
        pass

    @abstractmethod
    def _get_thumbnail_image(self):
        pass

    @abstractmethod
    def _get_label_image(self):
        pass

    @abstractmethod
    def _get_all_properties(self):
        pass

    @abstractmethod
    def _get_libname(self):
        pass
