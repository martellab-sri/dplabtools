# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Utility functions used in different places during testing."""

import os

from testconsts import testdata_dir, saveddata_dir, logs_dir


def make_test_path(path, *args):
    return os.path.join(testdata_dir, path, *args)


def make_log_path(log_file):
    return os.path.join(testdata_dir, saveddata_dir, logs_dir, log_file)
