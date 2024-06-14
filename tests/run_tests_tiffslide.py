# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Main script to run all tests using tiffslide library.

NOTES: import order is important here.
"""

import sys

sys.path.append("..")
sys.path.append(".")

import dplabtools.config
from testutils import make_log_path

dplabtools.config.slide_library = "tiffslide"
dplabtools.config.print_messages = make_log_path("tiffslide_screen_log.txt")

from testcommon import run_test_suite

if __name__ == "__main__":
    directory = "."
    pattern = "*_test.py"
    run_test_suite(pattern, directory)
