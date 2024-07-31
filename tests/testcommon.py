# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Plumbing code for running tests with different slide reader libraries."""

import os
import shutil
from unittest import TestLoader, TextTestRunner, TestCase, TestResult, skipIf

from dplabtools.config import print_messages as config_log_file
from dplabtools.config import slide_library as config_lib_name
from dplabtools._version import __version__ as lib_version
from testconsts import testdata_dir, saveddata_dir, refdata_dir, logs_dir
from testconfig import fast_tests_only, delete_created_files, screen_verbosity
from testutils import make_test_path


# **** VARIABLES ****

# list of directories inside "saved_data" dir that are required for writing test results
saveddata_subdirs = [
    "counts",
    "counts_dirs",
    "dirs",
    "libs",
    "logs",
    "manifests",
    "manifest_dir",
    "manifest_file",
    "patches1",
    "patches2",
    "patches3",
    "patches4",
    "patches5",
    "patches6",
    "patches7",
    "patches8",
    "patches9",
    "patches10",
    "patches11",
    "patches12",
    "patches13",
    "patches14",
    "patches15",
    "patches16",
    "patches17",
    "patches18",
    "patches19",
    "patches20",
    "patches21",
    "patches22",
    "patches23",
    "patches24",
    "patches25",
    "patches26",
    "patches27",
    "patches28",
    "patches29",
    "patches30",
    "patches31",
    "pool1a",
    "pool1b",
    "pool2a",
    "pool2b",
    "pool3a",
    "pool3b",
    "pool3c",
    "pool4a",
    "pool4b",
    "pool4c",
    "pool4d",
    "pool4e",
    "pool4f",
    "pool5",
    "pool6a",
    "pool6b",
    "pool6c",
    "pool7a",
    "pool7b",
    "pool7c",
    "pool8a",
    "pool8b",
    "pool8c",
    "pool8d",
    "pool8e",
    "pool8f",
    "pool9a",
    "pool9b",
    "pool9c",
    "pool9d",
    "pool9e",
    "pool9f",
    "pool9g",
    "pool9h",
    "preview",
    "pool_preview1",
    "pool_preview2",
    "dataset1",
    "dataset2a",
    "dataset2b",
    "dataset2c",
    "inference1",
    "inference2a",
    "inference2b",
    "inference2c",
    "inference3",
    "masks",
    "heatmaps1",
    "heatmaps2",
    "print",
    "tmp",
    "utils",
    "annotations1",
    "annotations2",
    "annotations3",
]


# ****  FUNCTIONS ****


def run_test_suite(pattern, directory):
    """Run all tests for one slide reader library."""
    print("Version %s, running tests using %s" % (lib_version, config_lib_name))
    suite = TestLoader().discover(directory, pattern=pattern)
    if isinstance(config_log_file, str):
        if os.path.exists(config_log_file):
            os.remove(config_log_file)
        screen_messages_test = TestScreenMessages(config_log_file, methodName="test_screen_message_log")
        suite.addTest(screen_messages_test)
    last_test = TestLastLibName(config_lib_name, methodName="test_last_lib_name")
    suite.addTest(last_test)
    TextTestRunner(verbosity=screen_verbosity).run(suite)


def run_before_tests(self):
    """Set hook for running the code before all tests start.

    Docs: https://docs.python.org/3/library/unittest.html#unittest.TestResult.startTestRun
    """
    # create directory for saving test data
    saved_data_dir = os.path.join(testdata_dir, saveddata_dir)
    if not os.path.exists(saved_data_dir):
        os.mkdir(saved_data_dir)
    # create subdirectories for test data
    for _dir_name in saveddata_subdirs:
        _dir_path = os.path.join(saved_data_dir, _dir_name)
        if not os.path.exists(_dir_path):
            os.mkdir(_dir_path)


def run_after_tests(self):
    """Set hook for running the code after all tests end.

    Docs: https://docs.python.org/3/library/unittest.html#unittest.TestResult.stopTestRun
    """
    # delete files created during testing
    if delete_created_files:
        saved_data_dir = os.path.join(testdata_dir, saveddata_dir)
        if os.path.exists(saved_data_dir):
            shutil.rmtree(saved_data_dir)
            os.mkdir(saved_data_dir)


# **** CLASSES ****


class TestLastLibName(TestCase):
    """Check if slide library was not dynamically changed during testing.

    Initial value (passed to init) is compared to the value retrieved in the last test.
    """

    def __init__(self, lib_name, **kwargs):
        """Init method."""
        super().__init__(**kwargs)
        self.initial_lib_name = lib_name

    def test_last_lib_name(self):
        """Run actual test."""
        from dplabtools.config import slide_library as last_lib_name

        self.assertEqual(last_lib_name, self.initial_lib_name)


@skipIf(fast_tests_only, "This test must be skipped if any of tests with screen messages is skipped")
class TestScreenMessages(TestCase):
    """Check if screen messages are the same.

    Compare screen messages created during testing to reference screen messages.
    """

    def __init__(self, log_file, **kwargs):
        """Init method."""
        super().__init__(**kwargs)
        self.log_file = log_file

    def test_screen_message_log(self):
        """Run actual test."""
        result_log_file = self.log_file
        log_file_name = os.path.basename(self.log_file)
        output_log_file = make_test_path(refdata_dir, logs_dir, log_file_name)
        with open(output_log_file, "r") as output_log:
            output_log_list = output_log.readlines()
        with open(result_log_file, "r") as result_log:
            result_log_list = result_log.readlines()
        self.assertEqual(sorted(result_log_list), sorted(output_log_list))


# **** RUNTIME SETTINGS ****

setattr(TestResult, "startTestRun", run_before_tests)
setattr(TestResult, "stopTestRun", run_after_tests)
