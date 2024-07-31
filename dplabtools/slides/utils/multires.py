# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Multi-resolution related utilities."""

from dplabtools.slides.utils.wsi import get_level_and_mpp


def get_scaled_center_based_locations(
    base_patch_mpp, base_topleft_location, levels_or_mpps, size, slide_level_mpp_values
):
    """Compute location for new patches based on their MPP values.

    New patches will have different X/Y cordinates, but the same center as base patches (provided as
    base_topleft_location).

    Calculations are not optimized (for code clarity), basic steps:
    - find patch centers on level0 image
    - scale center coordinates to desired patch level
    - compute top/left corner at patch level
    - scale back computed top/left corner from patch to level0
    """
    scaled_locations = []
    level0_mpp = slide_level_mpp_values[0]
    scaling_factor_base = base_patch_mpp / level0_mpp
    for level_or_mpp in levels_or_mpps:
        patch_level, patch_mpp = get_level_and_mpp(level_or_mpp)
        if not patch_mpp:
            patch_mpp = slide_level_mpp_values[patch_level]
        scaling_factor_patch = patch_mpp / level0_mpp
        level0_top_left_x = base_topleft_location[0]
        level0_top_left_y = base_topleft_location[1]
        level0_size_x = size[0] * scaling_factor_base
        level0_size_y = size[1] * scaling_factor_base
        level0_center_x = level0_top_left_x + level0_size_x / 2
        level0_center_y = level0_top_left_y + level0_size_y / 2
        scaled_center_x = level0_center_x / scaling_factor_patch
        scaled_center_y = level0_center_y / scaling_factor_patch
        scaled_location_x = scaled_center_x - size[0] / 2
        scaled_location_y = scaled_center_y - size[1] / 2
        scaled_location_x_at_level0 = scaled_location_x * scaling_factor_patch
        scaled_location_y_at_level0 = scaled_location_y * scaling_factor_patch
        scaled_locations.append((round(scaled_location_x_at_level0), round(scaled_location_y_at_level0)))
    return scaled_locations
