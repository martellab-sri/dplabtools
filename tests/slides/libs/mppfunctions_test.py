# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for MPP reading module using WSI files."""

from unittest import TestCase

from dplabtools.slides import GenericSlide
import dplabtools.slides.libs.mppfunctions as mppfunctions
from testutils import make_test_path


class TestForceMPPExternalLast(TestCase):
    """Test if force_external_mpp_value function is exported as last."""

    def test_force_external_mpp_is_last(self):
        last_function = mppfunctions.mpp_functions[-1]
        self.assertEqual(last_function.__name__, "force_external_mpp_value")


class TestMPPComputeFunctions(TestCase):
    """Test all MPP computing functions included in module."""

    def test_compute_tiff_mpp_in_centimeters(self):
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression-mpp.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        output_mpp = (0.25, 0.25)
        result_mpp = mppfunctions.compute_tiff_mpp_in_centimeters(wsi_slide_tif)
        self.assertEqual(result_mpp, output_mpp)
        #
        wsi_file_tif = make_test_path("wsi/TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        output_mpp = (0.5, 0.5)
        result_mpp = mppfunctions.compute_tiff_mpp_in_centimeters(wsi_slide_tif)
        self.assertEqual(result_mpp, output_mpp)
        #
        wsi_file_tif = make_test_path("wsi/board-multi-layer-no-compression.tif")
        wsi_slide_tif = GenericSlide(wsi_file=wsi_file_tif)
        output_mpp = (None, None)
        result_mpp = mppfunctions.compute_tiff_mpp_in_centimeters(wsi_slide_tif)
        self.assertEqual(result_mpp, output_mpp)
