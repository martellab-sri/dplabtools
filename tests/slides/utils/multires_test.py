# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions included in slides.utils.multires."""

from unittest import TestCase

from dplabtools.slides.utils.multires import get_scaled_center_based_locations


class TestUtilsMultiRes(TestCase):
    """Tests for functions included in slides.utils.multires."""

    def test_get_scaled_center_based_locations(self):
        basepatch_mpp = 0.25
        base_topleft_location = (512, 768)
        levels_or_mpps = [0, 0.5, 0.75, 1, 4.0, 0.8]
        size = (256, 512)
        slide_level_mpp_values = (0.25, 1.0, 4.0)
        output_locations = [(512, 768), (384, 512), (256, 256), (128, 0), (-1408, -3072), (230, 205)]
        result_locations = get_scaled_center_based_locations(
            basepatch_mpp, base_topleft_location, levels_or_mpps, size, slide_level_mpp_values
        )
        self.assertEqual(result_locations, output_locations)
