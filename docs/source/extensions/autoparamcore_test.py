# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Tests for AutoParamCore class.

run as: python3 -m unittest autoparamcore_test.py
"""

import re
import unittest

from autoparamcore import AutoParamCore


class TestAutoParamCoreMethods(unittest.TestCase):
    def test__get_docstring_data(self):
        params = ["param_one", "param_two"]
        paths = ["testclassdir.testclass.SomeTestClass", "testclassdir.testclass.SomeTestClass"]
        output = [
            ("param_one", "type_one", "param_one description"),
            ("param_two", "type_two", "param_two description"),
        ]
        result = AutoParamCore._get_docstring_data(params, paths)
        self.assertEqual(result, output)

    def test__split_path1(self):
        path = "abc.def.ghi.Class"
        output = ("abc.def.ghi", "Class")
        result = AutoParamCore._split_path(path)
        self.assertEqual(result, output)

    def test__split_path2(self):
        path = "abcdefghiClass"
        with self.assertRaises(ValueError):
            AutoParamCore._split_path(path)

    def test__get_class_params_data1(self):
        class_name = "TmpClass"
        class_params_text = """
        Some dummy test class

        Parameters
        ----------
        param1 : param1 type
            param1 description

        param2 : param2 type
            param2 description

        param3 : param3 type
            param3 description
        """
        output = [
            ("param1", "param1 type", "param1 description"),
            ("param2", "param2 type", "param2 description"),
            ("param3", "param3 type", "param3 description"),
        ]
        result = AutoParamCore._get_class_params_data(class_name, class_params_text)
        self.assertEqual(result, output)

    def test__get_class_params_data2(self):
        class_name = "TmpClass"
        class_params_text = """
        Some dummy test class

        Parameters
        ----------
        param1 : param1 type
            param1 description
            Line2

        param2 : param2 type
            param2 description
            Line 2

        param3 : param3 type
            param3 description
            Line 2.
        """
        output = [
            ("param1", "param1 type", "param1 description Line2"),
            ("param2", "param2 type", "param2 description Line 2"),
            ("param3", "param3 type", "param3 description Line 2."),
        ]
        result = AutoParamCore._get_class_params_data(class_name, class_params_text)
        self.assertEqual(result, output)

    def test__get_class_params_data3(self):
        class_name = "TmpClass"
        class_params_text = """
        Some dummy test class

        Other text.
        """
        with self.assertRaises(ValueError):
            AutoParamCore._get_class_params_data(class_name, class_params_text)

    def test__get_class_params_data4(self):
        class_name = "TmpClass"
        class_params_text = """
        Some dummy test class

        Parameters
        ----------

        param1 : param1 type
            param1 description
            Line2
        """
        with self.assertRaises(ValueError):
            AutoParamCore._get_class_params_data(class_name, class_params_text)

    def test__get_class_params_data5(self):
        class_name = "TmpClass"
        class_params_text = """
        Some dummy test class

        Parameters
        ----------
        param1
            param1 description
            Line2
        """
        with self.assertRaises(ValueError):
            AutoParamCore._get_class_params_data(class_name, class_params_text)

    def test__get_class_params_data6(self):
        class_name = "TmpClass"
        class_params_text = """
        Some dummy test class

        Parameters
        ----------
        param1 : type1

        param2 : type2
            param2 description
        """
        with self.assertRaises(ValueError):
            AutoParamCore._get_class_params_data(class_name, class_params_text)

    def test__get_class_params_data7(self):
        class_name = "TmpClass"
        class_params_text = """
        Some dummy test class

        Parameters
        ----------
        param1 :
            param1 description
            Line2
        """
        with self.assertRaises(ValueError):
            AutoParamCore._get_class_params_data(class_name, class_params_text)

    def test__find_param_data1(self):
        param_name = "def"
        class_params_data = [("abc", 1, 2), ("def", 3, 4), ("ghi", 5, 6)]
        output = ("def", 3, 4)
        result = AutoParamCore._find_param_data(param_name, class_params_data)
        self.assertEqual(result, output)

    def test__find_param_data2(self):
        param_name = "xyz"
        class_params_data = [("abc", 1, 2), ("def", 3, 4), ("ghi", 5, 6)]
        output = None
        result = AutoParamCore._find_param_data(param_name, class_params_data)
        self.assertEqual(result, output)

    def test__create_docstring(self):
        docstring_data = [("abc", "1", "2"), ("def", "3", "4"), ("ghi", "5", "6")]
        output = """ .. class:: dplabtoolshiddenclass_1c5bb6e9dc37442aac84284c88a4cfbe
                    :noindex:
                    :param 1 abc: 2
                    :param 3 def: 4
                    :param 5 ghi: 6
        """
        result = AutoParamCore._create_docstring(docstring_data)
        result = re.sub(r"dplabtoolshiddenclass[\_]\w{32}", "class", result)
        result = result.strip().replace("\n", "").replace(" ", "")
        output = re.sub(r"dplabtoolshiddenclass[\_]\w{32}", "class", output)
        output = output.strip().replace("\n", "").replace(" ", "")
        self.assertEqual(result, output)


class TestAutoParamCoreProperties(unittest.TestCase):
    def test_property1(self):
        params = ["param_one", "param_two"]
        paths = ["testclassdir.testclass.SomeTestClass", "testclassdir.testclass.SomeTestClass"]
        autoparam = AutoParamCore(params, paths)
        self.assertNotEqual(autoparam.docstring.find("dplabtoolshiddenclass"), -1)
