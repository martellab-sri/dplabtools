# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test configuration variables used in various test modules."""

# If True, tests marked as slow will be skipped. Default: False
fast_tests_only = False

# If True, files created during testing will be automatically deleted. Default: True
# Use it only for debugging with one slide library at a time (i.e. when not running "run_all_tests.sh")
delete_created_files = True

# Integer value passed to unittest library, controls screen messages. Default: 1
screen_verbosity = 1
