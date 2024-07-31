# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for WSI reading functionality."""

from functools import cached_property
from collections import OrderedDict

from dplabtools.common import print_out, roundfl
from dplabtools.slides.utils.wsi import (
    get_wsi_name,
    get_wsi_id,
    get_resampled_wsi_level_images,
    get_wsi_level_array,
    get_wsi_level_zero_array,
    get_resampled_tiles,
    get_level_and_mpp,
    get_target_resample_factor,
    find_best_resample_wsi_level,
)
from dplabtools.slides.utils.image import pad_image_zero_background


class BaseSlide:
    """Base class for GenericSlide, it must be expanded by SlideLib (child of AbstractLib) and LibCallsMixin."""

    # Stores user-set MPP value, when embedded MPP data is not available or missing (e.g. flat image files)
    _external_mpp = None

    # Number of decimal places used in MPP data value rounding
    _mpp_round_decimal_places = 5

    # When checking MPP range or upsampling images, use this value as the lower MPP boundary
    _range_min_mpp = 0.001

    # When embedded magnification is not available, use this value for range checking
    _range_max_magnification = 40

    # Pillow filter used in resizing WSI images or patches
    # docs: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
    _resampling_filter = "LANCZOS"

    # Accuracy of converting MPP values to WSI levels (maximum absolute value)
    _mpp_level_margin = 0.003

    # Proximity to image border that will trigger background padding of the extracted region.
    _padding_margin_pixels = 10

    # If True, resampling will always use level zero as base. If False, the closest level will be used.
    _level_zero_resampling = True

    def __init__(self, *, wsi_file, resampling_mode=None, extra_mpps=[]):
        """Create a GenericSlide object for reading WSIs.

        Parameters
        ----------
        wsi_file : str
            WSI file name or path.

        resampling_mode : str, optional
            One of the two supported resampling methods: ``wsi`` or ``tile``.

        extra_mpps : list of float, optional
            List of MPP values for extra pyramid levels, required when using ``resampling mode="wsi"``.


        -!-
        Dev. notes
        ----------
        Other variables in the class:
            - self._extra_mpps_created - MPP values used to create extra levels, those values may differ from extra_mpps
            - self._max_mpp - maximum allowed MPP value
            - self._resampled_images_dict - dictionary with Pillow images created in memory during resampling
            - self._mpp_wsi_level_cache - dictionary with level values corresponding to MPP values e.g.:
              {0.252 : 0, 1.008: 1, 3.05: -1}
            - self._resample_cache - dictionary with calculated values used in resampling, more details in
              _add_to_resample_cache
        -!-
        """
        self._slide = None
        self._wsi_file = wsi_file
        self._wsi_name = get_wsi_name(wsi_file)
        self._wsi_id = get_wsi_id(wsi_file)
        self._resampling_mode = resampling_mode
        self._extra_mpps = extra_mpps
        self._extra_mpps_created = []
        self._max_mpp = None
        self._resampled_images_dict = OrderedDict()
        self._mpp_wsi_level_cache = {}
        self._resample_cache = {}
        self._base_init()

    def _base_init(self):
        """Start internal initialization."""
        self._init_slide(self._wsi_file)
        self._level_downsamples = self._lib_get_level_downsamples()
        self._mpp_data = self._lib_get_mpp_data()
        self._level_mpp_values = self._lib_get_level_mpp_values()
        self._level_dimensions = self._lib_get_level_dimensions()
        self._level_zero_dimensions = self._level_dimensions[0]
        self._init_mpp()

    def _init_mpp(self):
        """Init MPP related processing and resampling.

        Resample WSI if at least one mpp value does not match any WSI levels
        """
        self._check_resampling_args(self._resampling_mode, self._extra_mpps, self._wsi_name)
        if self._resampling_mode:
            self._check_mpp_data(self._mpp_data, self._wsi_name)
            self._max_mpp = self._get_max_mpp_for_range(
                self._level_mpp_values, self.magnification, self.range_max_magnification, self._wsi_name
            )
            if self._resampling_mode == "wsi":
                self._check_extra_mpp_types(self._extra_mpps, self._wsi_name)
                self._check_extra_mpp_unique(self._extra_mpps, self._wsi_name)
                self._check_mpp_range(self._extra_mpps, self.range_min_mpp, self._max_mpp, self._wsi_name)
                mpp_wsi_levels = self._add_mpp_levels_to_cache(self._extra_mpps)
                if -1 in mpp_wsi_levels:
                    self._create_resample_cache(self._extra_mpps)
                    self._resample_slide(mpp_wsi_levels)
                else:
                    self._skip_resampling(mpp_wsi_levels)

    @staticmethod
    def _check_resampling_args(resampling_mode, extra_mpps, wsi_name):
        if resampling_mode not in (None, "wsi", "tile"):
            raise ValueError("%s: Invalid resampling mode: %s" % (wsi_name, resampling_mode))
        if resampling_mode == "wsi" and not extra_mpps:
            raise ValueError("%s: Missing MPP values for WSI resampling mode" % wsi_name)
        if resampling_mode == "tile" and extra_mpps:
            raise ValueError("%s: Superfluous MPP values for TILE resampling mode" % wsi_name)
        if resampling_mode is None and extra_mpps:
            raise ValueError("%s: Superfluous MPP values" % wsi_name)

    @staticmethod
    def _check_mpp_data(mpp_data, wsi_name):
        """Check if MPP data is present."""
        if not all(mpp_data):
            raise ValueError("%s: Resampling not possible, MPP data is not available" % wsi_name)

    @staticmethod
    def _get_max_mpp_for_range(level_mpp_values, magnification, range_max_magnification, wsi_name):
        """Compute max allowed MPP value for range checking.

        Use default or user provided magnification if embedded magnification value is not available.
        """
        if not magnification:
            magnification = range_max_magnification
            print_out(
                "%s: Setting max magnification to %d in MPP range checking [GenericSlide.range_max_magnification]"
                % (wsi_name, magnification)
            )
        max_mpp = roundfl(level_mpp_values[0] * magnification)
        return max_mpp

    @staticmethod
    def _check_extra_mpp_types(mpp_list, wsi_name):
        """Check if all MPP values are float numbers."""
        if any(not isinstance(mpp, float) for mpp in mpp_list):
            raise ValueError("%s: One or more MPP values %s is not of float type" % (wsi_name, mpp_list))

    @staticmethod
    def _check_extra_mpp_unique(mpp_list, wsi_name):
        """Check if all MPP values are unique."""
        if len(mpp_list) != len(set(mpp_list)):
            raise ValueError("%s: MPP values %s are not unique" % (wsi_name, mpp_list))

    @staticmethod
    def _check_mpp_range(mpp_values, min_mpp, max_mpp, wsi_name):
        """Check if MPP values are within allowed range."""
        for mpp_value in mpp_values:
            if mpp_value < min_mpp or mpp_value > max_mpp:
                raise ValueError("%s: MPP value is out of range [%s - %s]" % (wsi_name, min_mpp, max_mpp))

    def _add_mpp_levels_to_cache(self, extra_mpp_values):
        """Add MPP values to internal dictionary based cache (MPP is key, level is value)."""
        levels = []
        for mpp_value in extra_mpp_values:
            level = self._find_mpp_wsi_level(self._level_mpp_values, mpp_value, self.mpp_level_margin)
            self._mpp_wsi_level_cache[mpp_value] = level
            levels.append(level)
        return levels

    @staticmethod
    def _find_mpp_wsi_level(level_mpp_values, mpp_value, mpp_level_margin):
        """Return a potential WSI level that matches the desired MPP value (or -1 for no match)."""
        level = -1
        if all(level_mpp_values):
            for num_level, num_mpp_value in enumerate(level_mpp_values):
                if roundfl(abs(num_mpp_value - mpp_value)) < mpp_level_margin:
                    level = num_level
                    break
        return level

    def _create_resample_cache(self, mpps):
        """Cache all necessary values for WSI mode resampling."""
        for mpp in mpps:
            self._add_to_resample_cache(mpp)

    def _resample_slide(self, mpp_wsi_levels):
        """Start WSI resampling."""
        mpp_for_resampling = []
        mpp_skipped = []
        # exclude MPP values which match WSI levels
        for level, mpp in zip(mpp_wsi_levels, self._extra_mpps):
            if level == -1:
                mpp_for_resampling.append(mpp)
            else:
                mpp_skipped.append(mpp)

        if mpp_skipped:
            self._print_out_wsi("MPP values matching WSI levels (resampling skipped): %s" % mpp_skipped)
        self._print_out_wsi("Resampling file at MPP=%s" % mpp_for_resampling)
        self._extra_mpps_created = mpp_for_resampling
        resampling_batches = self._get_resampling_batches(mpp_for_resampling, self._resample_cache)

        for level, mpps in resampling_batches.items():
            level_array = self._get_wsi_level_array(self, level)
            resample_size_list = self._get_batch_resample_size_list(
                self._level_dimensions[level], mpps, self._resample_cache
            )
            resampled_images = self._get_resampled_wsi_level_images(
                level_array, resample_size_list, self.resampling_filter
            )
            for mpp, image in zip(mpps, resampled_images):
                self._resampled_images_dict[mpp] = image

    @staticmethod
    def _get_resampling_batches(mpps, resample_cache):
        resampling_batches = {}
        for mpp in mpps:
            level = resample_cache[str(mpp)]["L"]
            if level not in resampling_batches:
                resampling_batches[level] = []
            resampling_batches[level].append(mpp)
        return resampling_batches

    @staticmethod
    def _get_wsi_level_array(wsi_slide, level):
        if level == 0:
            wsi_level_array = get_wsi_level_zero_array(wsi_slide.slide_file)
        else:
            wsi_level_array = get_wsi_level_array(wsi_slide, level)
        return wsi_level_array

    @staticmethod
    def _get_batch_resample_size_list(level_size, mpps, resample_cache):
        batch_resample_size_list = []
        for mpp in mpps:
            base_resample_factor = resample_cache[str(mpp)]["B"]
            width = round(level_size[0] / base_resample_factor)
            height = round(level_size[1] / base_resample_factor)
            batch_resample_size_list.append((width, height))
        return batch_resample_size_list

    @staticmethod
    def _get_resampled_wsi_level_images(level_array, resample_size_list, resampling_filter):
        """Get list of resampled wsi levels as images."""
        return get_resampled_wsi_level_images(level_array, resample_size_list, resize_filter=resampling_filter)

    def _skip_resampling(self, levels):
        """Skip resampling with optional user notification."""
        self._print_out_wsi(
            "Resampling skipped, all requested MPP values %s match WSI levels %s [MPP margin is %s]"
            % (self._extra_mpps, levels, self.mpp_level_margin)
        )

    def get_property(self, name):
        """Return the embedded property value from a WSI.

        Parameters
        ----------
        name : str
            Property name.
        """
        return self._lib_get_property(name)

    def get_region_array(self, location, level, size):
        """Return the specified WSI region as a NumPy array.

        Parameters
        ----------
        location : tuple
            (x, y) pair of `int` values indicating the WSI region (top left corner).
        level : int
            Value indicating the WSI level.
        size : tuple
            (width, height) pair of `int` values indicating the WSI region size.

        Notes
        -----
        - only native WSI level values are supported here, no MPPs (float numbers)
        - region padding is not supported
        - additional checks (e.g. location) are also not supported
        - the purpose of this function is to retrieve LARGE regions as NumPy arrays, not small patches
        - the alpha channel is removed to match ``get_region`` output format
        """
        region = self._lib_read_region_array(location, level, size)[:, :, 0:3]
        return region

    def get_region(self, location, level_or_mpp, size, skip_padding=False):
        """Return the specified WSI region as an RGB Pillow image.

        Parameters
        ----------
        location : tuple
            (x, y) pair of `int` values indicating the WSI region (top left corner).
        level_or_mpp : int or float
            Value indicating the WSI level or the target MPP value for the region.
        size : tuple
            (width, height) pair of `int` values indicating the WSI region size.
        skip_padding : bool, default=False
            Whether to skip padding the region, if it partially exceeds the image area.


        -!-
        Dev. notes
        ----------
        - for WSI resampling padding should be applied at the end
        - for TILE resampling padding must be applied before resizing, otherwise not all black pixel stripes
          will be properly removed
        - since the order of operations is different for WSI and TILE resampling, there will be cases when
          those methods produce different results for the same image
        -!-
        """
        region = None
        tile_padded = False
        self._check_location(location, level_or_mpp, size)
        level, mpp = get_level_and_mpp(level_or_mpp)
        # If MPP is provided, check if MPP matches any cached level number. If not, add it to cache
        if mpp:
            if mpp in self._mpp_wsi_level_cache:
                level = self._mpp_wsi_level_cache[mpp]
            else:
                level = self._add_mpp_levels_to_cache([mpp])[0]
        # If level is available, use it
        if level >= 0:
            region = self._lib_read_region(location, level, size).convert("RGB")
        elif self._resampling_mode and mpp:
            # In WSI resampling mode check if requested MPP is available
            if self._resampling_mode == "wsi":
                self._check_mpp_match(mpp)
                region = self._get_resampled_region(location, size, mpp)  # already in RGB format
            # In TILE resampling mode check if MPP range is valid
            elif self._resampling_mode == "tile":
                self._check_mpp_range([mpp], self._range_min_mpp, self._max_mpp, self._wsi_name)
                # Calculate base level patch based on mpp value, and then resize it
                resample_cache = self._get_resample_cache(mpp)
                base_resample_factor = resample_cache["B"]
                base_resample_level = resample_cache["L"]
                base_level_size = tuple(round(s * base_resample_factor) for s in size)
                base_level_region = self._lib_read_region(location, base_resample_level, base_level_size).convert("RGB")
                if not skip_padding:
                    base_level_region = self._pad_region(base_level_region, location, mpp, size)
                    tile_padded = True
                region = self._get_resampled_tile(base_level_region, size, self.resampling_filter)
        else:
            raise ValueError(
                self._wsi_msg('Resampling is not enabled. Initiate GenericSlide with "resampling_mode=[wsi|tile]"')
            )
        # padding if not already done
        if not skip_padding and not tile_padded:
            region = self._pad_region(region, location, level_or_mpp, size)
        return region

    def _check_location(self, location, level_or_mpp, size):
        """Check if location is within image dimensions or area."""
        # location exceeds image size
        if location[0] > self._level_zero_dimensions[0] or location[1] > self._level_zero_dimensions[1]:
            raise ValueError(
                self._wsi_msg(
                    "Attempted to read region at location %s which exceeds slide level 0 dimensions %s"
                    % (location, self._level_zero_dimensions)
                )
            )
        # location is negative, must check if bottom left corner is still within image area
        # and if the entire image is not spanned by extracted region in any dimension
        if location[0] < 0 or location[1] < 0:
            level0_resample_factor = self._get_resample_cache(level_or_mpp)["Z"]
            scaled_size = (round(size[0] * level0_resample_factor), round(size[1] * level0_resample_factor))
            if not (location[0] + scaled_size[0] > 0 and location[1] + scaled_size[1] > 0):
                raise ValueError(
                    self._wsi_msg(
                        "Attempted to read region at location %s and of level0 scaled size %s which is not within "
                        "level 0 area" % (location, scaled_size)
                    )
                )
            if (
                location[0] < 0
                and scaled_size[0] > self._level_zero_dimensions[0]
                or location[1] < 0
                and scaled_size[1] > self._level_zero_dimensions[1]
            ):
                raise ValueError(
                    self._wsi_msg(
                        "Attempted to read region at location %s and of level0 scaled size %s, which spans the "
                        "whole image area in at least one dimension" % (location, scaled_size)
                    )
                )

    def _get_resample_cache(self, level_or_mpp):
        """Get calculated resampling values for given level_or_mpp.

        - dictionary keys are hashed as strings to make both 1 (int) and 1.0 (float) work
        """
        if str(level_or_mpp) in self._resample_cache:
            resample_cache = self._resample_cache[str(level_or_mpp)]
        else:
            resample_cache = self._add_to_resample_cache(level_or_mpp)
        return resample_cache

    def _add_to_resample_cache(self, level_or_mpp):
        """Add level_or_mpp value to internal dictionary based cache.

        - dictionary keys are hashed as strings to make both 1 (int) and 1.0 (float) work
        - cache structure is two-dimensional:
            - key is level_or_mpp converted to string
            - value is another dictionary with the following keys: "B", "L", "Z" (see below)
        """
        level, mpp = get_level_and_mpp(level_or_mpp)
        if mpp:
            base_resample_level = self._get_base_resample_level(self, mpp, self._level_zero_resampling)
        else:
            base_resample_level = level
        base_resample_factor = get_target_resample_factor(
            level, mpp, self._level_downsamples, self._level_mpp_values, base_resample_level
        )
        level0_resample_factor = get_target_resample_factor(
            level, mpp, self._level_downsamples, self._level_mpp_values, 0
        )

        self._resample_cache[str(level_or_mpp)] = {
            "B": base_resample_factor,  # "B-ase"  - base level resampling factor (int)
            "L": base_resample_level,  # "L-evel" - base level used for resampling (float)
            "Z": level0_resample_factor,  # "Z-ero"  - level zero resampling factor (float)
        }
        return self._resample_cache[str(level_or_mpp)]

    @staticmethod
    def _get_base_resample_level(wsi_slide, mpp, level_zero_resampling):
        """Get level used for resampling based on user settings."""
        if level_zero_resampling:
            base_level = 0
        else:
            base_level = find_best_resample_wsi_level(wsi_slide, mpp)
        return base_level

    def _check_mpp_match(self, mpp_value):
        """Check if image can be read using provided MPP value."""
        if not self._extra_mpps_created:
            raise ValueError(
                self._wsi_msg(
                    "MPP value %s does not match any existing levels %s [MPP margin is %s]. "
                    'Initiate GenericSlide with "extra_mpps=[%s]"'
                    % (mpp_value, self._level_mpp_values, self.mpp_level_margin, mpp_value)
                )
            )
        if mpp_value not in self._extra_mpps_created:
            raise ValueError(
                self._wsi_msg(
                    "MPP value %s does not match any existing levels %s [MPP margin is %s], "
                    "including MPP values used during resampling %s"
                    % (mpp_value, self._level_mpp_values, self.mpp_level_margin, self._extra_mpps_created)
                )
            )

    def _get_resampled_region(self, location, size, mpp):
        """Read patch from resampled pyramid level identified by MPP.

        Location uses level 0 coordinates.
        """
        level0_resample_factor = self._get_resample_cache(mpp)["Z"]
        resampled_location = (
            round(location[0] / level0_resample_factor),
            round(location[1] / level0_resample_factor),
        )
        region_rect = (
            resampled_location[0],
            resampled_location[1],
            resampled_location[0] + size[0],
            resampled_location[1] + size[1],
        )
        region = self._resampled_images_dict[mpp].crop(region_rect)
        return region

    def _pad_region(self, region, location, level_or_mpp, size):
        """Pad extracted region with zero background (white pixels).

        There will ALWAYS be rounding errors when using resample factors and scaling level coordinates,
        padding_margin_pixels is introduced to overcome that issue. If patch edge is closer than padding_margin_pixels
        to image border, a padding attempt action will be executed (which may modify patch or not).
        If location is negative, then padding action will be always executed and will always modify patch data.

        Steps (when location is not negative):
        - determine level0_resample_factor
        - use it to scale size of patch to level 0
        - check if padding is necessary

        Location check should be always called before padding!
        """
        if location[0] < 0 or location[1] < 0:
            self._print_out_wsi("Padding background of extracted region: location=%s" % str(location))
            region = pad_image_zero_background(region)
        else:
            level0_resample_factor = self._get_resample_cache(level_or_mpp)["Z"]
            scaled_size = (round(size[0] * level0_resample_factor), round(size[1] * level0_resample_factor))
            if (
                location[0] + scaled_size[0] + self.padding_margin_pixels > self._level_zero_dimensions[0]
                or location[1] + scaled_size[1] + self.padding_margin_pixels > self._level_zero_dimensions[1]
            ):
                self._print_out_wsi(
                    "Attempting to pad background of extracted region: location=%s, "
                    "requested patch size=%s, level0 scaled patch size=%s" % (location, size, scaled_size)
                )
                region = pad_image_zero_background(region)
        return region

    @staticmethod
    def _get_resampled_tile(base_level_region, size, resampling_filter):
        """Return resized tile/region."""
        resampled_tile = get_resampled_tiles(base_level_region, [size], resize_filter=resampling_filter)
        return resampled_tile[0]

    def _print_out_wsi(self, message):
        """Print message with WSI prefix."""
        print_out(self._wsi_msg(message))

    def _wsi_msg(self, message):
        """Add prefix with WSI file name."""
        return "%s: %s" % (self._wsi_name, message)

    def check_mpp_data(self, mpp):
        """Check if MPP data is present.

        This is public method calling internal private method
        """
        self._check_mpp_data(mpp, self._wsi_name)

    def check_mpp_range(self, mpp):
        """Check if MPP value is within allowed range.

        This is a public method calling internal private method
        """
        if self._max_mpp:
            max_mpp = self._max_mpp
        else:
            max_mpp = self._get_max_mpp_for_range(
                self.level_mpp_values, self.magnification, self.range_max_magnification, self._wsi_name
            )
        self._check_mpp_range([mpp], self.range_min_mpp, max_mpp, self._wsi_name)

    @classmethod
    def set_external_mpp(cls, external_mpp):
        """Set the externally provided MPP value.

        This value will be used only if MPP data is not embedded in a WSI or cannot be retrieved using any other
        implemented methods. When MPP data is available natively, any value set using this method will be ignored.

        Parameters
        ----------
        external_mpp : float
            MPP value to be used.


        -!-
        Dev. notes
        ----------
        mppfunctions.py will have a list of other MPP reading methods.

        -!-
        """
        cls._external_mpp = external_mpp

    @classmethod
    def set_mpp_round_decimal_places(cls, mpp_round_decimal_places):
        """Set the number of decimal places used in MPP value rounding.

        Some scanners may report slightly different values for X and Y axes, in such cases rounding is necessary.
        No rounding will take place if MPP values for X and Y axes match.

        Parameters
        ----------
        mpp_round_decimal_places : int, default=5
            Number of decimal places value to be used.
        """
        cls._mpp_round_decimal_places = mpp_round_decimal_places

    @classmethod
    def set_range_min_mpp(cls, range_min_mpp):
        """Set the minimum MPP value used in MPP range checking during region reading or image upsampling.

        Parameters
        ----------
        range_min_mpp : float, default=0.001
            Minimum MPP value to be used.
        """
        cls._range_min_mpp = range_min_mpp

    @classmethod
    def set_range_max_magnification(cls, range_max_magnification):
        """Set the maximum magnification value for MPP range checking.

        This value will be used only if the embedded magnification information is not available.

        Parameters
        ----------
        range_max_magnification : int, default=40
            Maximum magnification value to be used.
        """
        cls._range_max_magnification = range_max_magnification

    @classmethod
    def set_resampling_filter(cls, resampling_filter):
        """Set the resampling filter value for image downsampling and upsampling.

        Available filter values:
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters

        Parameters
        ----------
        resampling_filter : str, default="LANCZOS"
            Resampling filter value to be used.
        """
        cls._resampling_filter = resampling_filter

    @classmethod
    def set_mpp_level_margin(cls, mpp_level_margin):
        """Set the accuracy margin value when converting MPP values to WSI levels.

        The provided MPP value will be considered an equivalent of the WSI level, if their computed difference is less
        than ``mpp_level_margin``.

        Parameters
        ----------
        mpp_level_margin : float, default=0.003
            Margin value to be used.
        """
        cls._mpp_level_margin = mpp_level_margin

    @classmethod
    def set_padding_margin_pixels(cls, padding_margin_pixels):
        """Set the number of pixels as proximity to image borders to trigger the potential image background padding process.

        Parameters
        ----------
        padding_margin_pixels : int, default=10
            Number of pixels value to be used.
        """
        cls._padding_margin_pixels = padding_margin_pixels

    @classmethod
    def set_level_zero_resampling(cls, level_zero_resampling):
        """Set the base WSI level for image resampling.

        If `True` (default), then the WSI level zero will be used for resampling, otherwise the optimal level will be
        found. Level zero will provide the best data quality, but at the same time the poorest performance.

        Parameters
        ----------
        level_zero_resampling : bool, default=True
            Use level zero or other level for WSI resampling.
        """
        cls._level_zero_resampling = level_zero_resampling

    @cached_property
    def magnification(self):
        """Return the WSI magnification value."""
        return self._lib_get_magnification()

    @property
    def level_dimensions(self):
        """Return the dimension values for all native levels in the WSI."""
        return self._level_dimensions

    @cached_property
    def level_dimensions_extra(self):
        """Return the MPP-indexed dimension values for additional WSI levels created during resampling."""
        level_dimensions = OrderedDict(
            {mpp: (img.width, img.height) for mpp, img in self._resampled_images_dict.items()}
        )
        return level_dimensions

    @property
    def level_downsamples(self):
        """Return the downsample factor values for all native levels in the WSI."""
        return self._level_downsamples

    @cached_property
    def level_resamples_extra(self):
        """Return the MPP-indexed downsample/upsample factor values for additional WSI levels created during resampling."""
        level_resamples = OrderedDict(
            {
                float(ld_key): ld_value["Z"]
                for ld_key, ld_value in self._resample_cache.items()
                if float(ld_key) in self._extra_mpps_created
            }
        )
        return level_resamples

    @property
    def mpp_data(self):
        """Return the MPP data embedded in the WSI."""
        return self._mpp_data

    @property
    def level_mpp_values(self):
        """Return the MPP values for all native levels in the WSI."""
        return self._level_mpp_values

    @cached_property
    def level_mpp_values_extra(self):
        """Return the MPP values for additional WSI levels created during resampling."""
        return tuple(self._extra_mpps_created)

    @cached_property
    def level_count(self):
        """Return the number of native levels in the WSI."""
        return len(self._level_dimensions)

    @cached_property
    def level_count_extra(self):
        """Return the number of additional WSI levels created during resampling."""
        return len(self._resampled_images_dict)

    @cached_property
    def thumbnail_image(self):
        """Return the embedded WSI thumbnail image."""
        return self._lib_get_thumbnail_image()

    @cached_property
    def label_image(self):
        """Return the embedded WSI label image."""
        return self._lib_get_label_image()

    @cached_property
    def all_properties(self):
        """Return all embedded properties from the WSI."""
        return self._lib_get_all_properties()

    @cached_property
    def lib_name(self):
        """Return the current WSI reading library name."""
        return self._lib_get_libname()

    @property
    def level_images_extra(self):
        """Return MPP-indexed images of additional WSI levels created during resampling."""
        return self._resampled_images_dict

    @property
    def slide_file(self):
        """Return the WSI file name or path."""
        return self._wsi_file

    @property
    def slide_name(self):
        """Return the WSI file name."""
        return self._wsi_name

    @property
    def slide_id(self):
        """Return the WSI file id (file name without extension)."""
        return self._wsi_id

    @property
    def slide_object(self):
        """Return the object representing the WSI reading library."""
        return self._slide

    @property
    def external_mpp(self):
        """Return the `external MPP` value set by ``set_external_mpp``."""
        return self._external_mpp

    @property
    def mpp_round_decimal_places(self):
        """Return the `mpp_round_decimal_places` value set by ``set_mpp_round_decimal_places`` (default=5)."""
        return self._mpp_round_decimal_places

    @property
    def range_min_mpp(self):
        """Return the `range_min_mpp` value set by ``set_range_min_mpp`` (default=0.001)."""
        return self._range_min_mpp

    @property
    def range_max_magnification(self):
        """Return the `range_max_magnification` value set by ``set_range_max_magnification`` (default=40)."""
        return self._range_max_magnification

    @property
    def resampling_filter(self):
        """Return the `resampling_filter` value set by ``set_resampling_filter`` (default="LANCZOS")."""
        return self._resampling_filter

    @property
    def mpp_level_margin(self):
        """Return the `mpp_level_margin` value set by ``set_mpp_level_margin`` (default=0.003)."""
        return self._mpp_level_margin

    @property
    def padding_margin_pixels(self):
        """Return the `padding_margin_pixels` value set by ``set_padding_margin_pixels`` (default=10)."""
        return self._padding_margin_pixels

    @property
    def level_zero_resampling(self):
        """Return the `level_zero_resampling` value set by ``set_level_zero_resampling`` (default=True)."""
        return self._level_zero_resampling

    @property
    def _prop_mpp_wsi_level_cache(self):
        """Return the internal cache variable - for testing only."""
        return self._mpp_wsi_level_cache

    @property
    def _prop_resample_cache(self):
        """Return the internal cache variable - for testing only."""
        return self._resample_cache
