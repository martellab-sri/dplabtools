# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions included in "common" namespace."""

import io
from unittest import TestCase
from unittest.mock import patch
from contextlib import redirect_stdout

from dplabtools.common import get_random_string, print_out, roundfl
from testutils import make_test_path

log_file_path = make_test_path("saved_data/print/log.txt")


class TestCommonFunctions(TestCase):
    """Tests for functions included in "common"."""

    def test_get_random_string(self):
        result_value = get_random_string(9)
        self.assertEqual(len(result_value), 9)

    @patch("dplabtools.common.common.print_messages", True)
    def test_print_out_true(self):
        f = io.StringIO()
        with redirect_stdout(f):
            print_out("ABC")
        result_value = f.getvalue()
        self.assertEqual(result_value, "ABC\n")

    @patch("dplabtools.common.common.print_messages", False)
    def test_print_out_false(self):
        f = io.StringIO()
        with redirect_stdout(f):
            print_out("ABC")
        result_value = f.getvalue()
        self.assertEqual(result_value, "")

    @patch("dplabtools.common.common.print_messages", log_file_path)
    def test_print_out_log(self):
        f = io.StringIO()
        with redirect_stdout(f):
            print_out("ABC")
        result_value = f.getvalue()
        with open(log_file_path) as flog:
            result_value = flog.readlines()
        self.assertEqual(result_value, ["ABC\n"])

    @patch("dplabtools.common.common.print_messages", 1)
    def test_print_out_invalid(self):
        with self.assertRaises(ValueError):
            print_out("ABC")

    def test_roundfl(self):
        # passing integer value
        result_value = roundfl(11)
        self.assertEqual(result_value, 11)
        # default number of decimal places (10)
        result_value = roundfl(1.199999999999999999999)
        self.assertEqual(result_value, 1.2)
        # default number of decimal places (10)
        result_value = roundfl(1.11111111111111)
        self.assertEqual(result_value, 1.1111111111)
        # custom number of decimal places
        result_value = roundfl(3.66666666666666666666, 7)
        self.assertEqual(result_value, 3.6666667)
        # custom number of decimal places
        result_value = roundfl(5.888888888888888888, 3)
        self.assertEqual(result_value, 5.889)
