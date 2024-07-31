# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Module with custom MPP reading functions.

- functions from this module are passed to BaseSlide class (foundation for GenericSlide) via variable mpp_functions
- each function should have descriptive name and use only one argument: wsi_slide
- if needed, exception catching should be implemented inside functions
- on failure each function should return (None, None)
- order of processing is determined by mpp_functions list order
"""

from dplabtools.common import print_out


def compute_tiff_mpp_in_centimeters(wsi_slide):
    """Compute MPP data when unit is centimeters and X/Y resolutions are provided."""
    mpp_data = (None, None)
    try:
        unit = wsi_slide.get_property("tiff.ResolutionUnit")
        res_x_str = wsi_slide.get_property("tiff.XResolution")
        res_y_str = wsi_slide.get_property("tiff.YResolution")
    except (ValueError, KeyError):
        unit = res_x_str = res_y_str = None

    if unit and unit.upper() == "CENTIMETER" and len(res_x_str) and len(res_y_str):
        try:
            res_x_num = float(res_x_str)
            res_y_num = float(res_y_str)
            mpp_data = (10000 / res_x_num, 10000 / res_y_num)
        except (ValueError, ZeroDivisionError):
            pass
    return mpp_data


def force_external_mpp_value(wsi_slide):
    """Force externally provided MPP value.

    This function should be included as last in mpp_functions list.
    """
    mpp_data = (None, None)
    if wsi_slide.external_mpp is not None:
        print_out("%s: Using externally provided MPP value of %s" % (wsi_slide.slide_name, str(wsi_slide.external_mpp)))
        mpp_data = (wsi_slide.external_mpp, wsi_slide.external_mpp)
    return mpp_data


mpp_functions = [
    compute_tiff_mpp_in_centimeters,
    force_external_mpp_value,
]
