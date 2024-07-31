# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Module with function calls to WSI reading libraries.

There are three levels where WSI reading code can be placed and executed:

1. Lowest level: these are modules specific to WSI reading libraries, e.g. libopenslide.py. Only code specific to
   one particular library should be included there.

2. Medium level: it's this very file (libcalls.py) - it's where functionality common to all WSI reading libraries
   (openslide, tiffslide, etc.) should be placed. Also helper functions for "_lib*" functions should be placed here,
   e.g. _read_mpp_custom

3. Highest level: baseslide.py - this is where public methods and properties should be located, however, they should
   not make direct calls to WSI specific libraries (openslide, tiffslide) at the lowest level. The highest level
   should only communicate with those libraries using libcalls.py i.e. the medium level.
"""

from dplabtools.slides.libs.mppfunctions import mpp_functions


class LibCallsMixin:
    """Mixin class to be combined with BaseSlide and SlideLib."""

    def _lib_read_region(self, location, level, size):
        """Get WSI region - wrapper call to dedicated WSI reading library."""
        return self._get_region(location, level, size)

    def _lib_read_region_array(self, location, level, size):
        """Get WSI region - wrapper call to dedicated WSI reading library."""
        return self._get_region_array(location, level, size)

    def _lib_get_magnification(self):
        """Get WSI magnification - wrapper call to dedicated WSI reading library."""
        try:
            magnification = self._get_magnification()
        except (KeyError, TypeError):
            magnification = None
            self._print_out_wsi("Magnification is not available")
        return magnification

    def _lib_get_level_dimensions(self):
        """Get WSI level dimensions - wrapper call to dedicated WSI reading library."""
        return self._get_level_dimensions()

    def _lib_get_level_downsamples(self):
        """Get WSI level downsamples - wrapper call to dedicated WSI reading library."""
        return self._get_level_downsamples()

    def _lib_get_mpp_data(self):
        """Get WSI mpp data - wrapper call to dedicated WSI reading library and custom MPP reading functions."""
        # Helper function
        def _read_mpp_custom():
            """Run custom functions for reading MPP data."""
            mpp_data = (None, None)
            for mpp_func in mpp_functions:
                mpp_data = mpp_func(self)
                if all(mpp_data):
                    break
            return mpp_data

        mpp_data = (None, None)
        try:
            mpp_data = self._get_mpp_data()
            # If no MPP data is present openslide raises KeyError, tiffslide returns None which will raise TypeError
            # Pillow by design will raise ValueError
        except (KeyError, TypeError, ValueError):
            mpp_data = _read_mpp_custom()

        if not all(mpp_data):
            self._print_out_wsi(
                "MPP data is not available, external MPP value can be set using 'GenericSlide.set_external_mpp'"
            )
        else:
            # some scanners (e.g. Phillips) report slightly different values for X and Y axes (numerical representation)
            # for instance (file "tumor_075.tif"):
            #       openslide.mpp-x: '0.22632099999999999'
            #       openslide.mpp-y: '0.22631600000000002'
            if mpp_data[0] != mpp_data[1]:
                mpp_data = (
                    round(mpp_data[0], self.mpp_round_decimal_places),
                    round(mpp_data[1], self.mpp_round_decimal_places),
                )
            if mpp_data[0] != mpp_data[1]:
                raise ValueError(self._wsi_msg("MPP data is not supported"))
        return mpp_data

    def _lib_get_property(self, name):
        """Get WSI property - wrapper call to dedicated WSI reading library.

        Forced conversion to string is required as different libraries return properties as different data types.
        """
        try:
            property_value = self._get_property(name)
        except KeyError as exc:
            raise ValueError(self._wsi_msg("Property %s does not exist" % name)) from exc
        return str(property_value)

    def _lib_get_level_mpp_values(self):
        """Get WSI level MPP values.

        This is not a wrapper function, as no similar function exists in WSI libraries.  It computes MPP values
        for all WSI pyramid levels.
        """
        mpp_data = self._mpp_data
        if all(mpp_data):
            level_mpp_values = tuple((mpp_data[0] * downsample_factor for downsample_factor in self._level_downsamples))
        else:
            level_mpp_values = tuple((None for downsample_factor in self._level_downsamples))
        return level_mpp_values

    def _lib_get_thumbnail_image(self):
        """Get WSI thumbnail image - wrapper call to dedicated WSI reading library."""
        return self._get_thumbnail_image()

    def _lib_get_label_image(self):
        """Get WSI thumbnail image - wrapper call to dedicated WSI reading library."""
        try:
            label = self._get_label_image()
        except KeyError:
            label = None
        if label is None:
            self._print_out_wsi("Embedded label image is not available")
        return label

    def _lib_get_all_properties(self):
        """Get all properties of WSI image."""
        return self._get_all_properties()

    def _lib_get_libname(self):
        """Get WSI reading library name."""
        return self._get_libname()
