# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for MPP reading module using flat image files."""

from unittest import TestCase

from dplabtools.slides import GenericSlide
import dplabtools.slides.libs.mppfunctions as mppfunctions
from testutils import make_test_path


class TestMPPComputeFunctionsFlatImage(TestCase):
    """Test all MPP computing functions included in module."""

    def test_compute_tiff_mpp_in_centimeters(self):
        wsi_file_tif = make_test_path("wsi/board-flat.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        output_mpp = (None, None)
        result_mpp = mppfunctions.compute_tiff_mpp_in_centimeters(wsi_slide_tif)
        self.assertEqual(result_mpp, output_mpp)
