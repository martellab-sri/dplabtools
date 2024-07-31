# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


from unittest import TestCase

from dplabtools.slides.annotations.mappers.base import BaseMapper


class BaseMapperExtended(BaseMapper):
    """Helper class with static methods."""

    @staticmethod
    def key_value_fn1(**kwargs):
        return "VALUE1"

    @staticmethod
    def key_value_fn2(**kwargs):
        return "VALUE2"

    @staticmethod
    def key_value_fn3(**kwargs):
        return "VALUE3"


class TestBaseMapper(TestCase):
    """Tests for BaseMapper class."""

    def setUp(self):
        self.input_annotations = [
            {"key1a": "value1a", "key1b": "value1b", "key1c": "value1c"},
            {"key2a": "value2a", "key2b": "value2b", "key2c": "value2c"},
            {"key3a": "value3a", "key3b": "value3b", "key3c": "value3c"},
            {"key4a": "value4a", "key4b": "value4b", "key4c": "value4c"},
        ]

    def tearDown(self):
        BaseMapper.label_name = "user_label"

    def test_mapper_builtin_keys_only(self):
        BaseMapper.label_name = "test_label"
        output_annotations = [
            {"KEY1A": "value1a", "key1b": "value1b", "key1c": "value1c", "test_label": "xyz"},
            {"key2a": "value2a", "key2b": "value2b", "key2c": "value2c", "test_label": "xyz"},
            {"key3a": "value3a", "KEY3B": "value3b", "key3c": "value3c", "test_label": "xyz"},
            {"key4a": "value4a", "key4b": "value4b", "KEY4C": "value4c", "test_label": "xyz"},
        ]
        mapper = BaseMapper(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "xyz",
            builtin_key_mappings={"key1a": "KEY1A", "key3b": "KEY3B", "key4c": "KEY4C"},
            builtin_value_mappings={},
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_builtin_values_only(self):
        BaseMapper.label_name = "test_label"
        output_annotations = [
            {"key1a": "VALUE1", "key1b": "value1b", "key1c": "value1c", "test_label": "xyz"},
            {"key2a": "value2a", "key2b": "VALUE2", "key2c": "value2c", "test_label": "xyz"},
            {"key3a": "value3a", "key3b": "value3b", "key3c": "VALUE3", "test_label": "xyz"},
            {"key4a": "value4a", "key4b": "value4b", "key4c": "value4c", "test_label": "xyz"},
        ]
        mapper = BaseMapperExtended(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "xyz",
            builtin_key_mappings={},
            builtin_value_mappings={"key1a": "key_value_fn1", "key2b": "key_value_fn2", "key3c": "key_value_fn3"},
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_builtin_keys_and_values(self):
        output_annotations = [
            {"KEY1A": "VALUE1", "key1b": "value1b", "key1c": "value1c", "user_label": "abc"},
            {"key2a": "value2a", "key2b": "value2b", "key2c": "value2c", "user_label": "abc"},
            {"key3a": "value3a", "KEY3B": "value3b", "key3c": "VALUE3", "user_label": "abc"},
            {"key4a": "value4a", "key4b": "value4b", "key4c": "value4c", "user_label": "abc"},
        ]
        mapper = BaseMapperExtended(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "abc",
            builtin_key_mappings={"key1a": "KEY1A", "key3b": "KEY3B"},
            builtin_value_mappings={"key1a": "key_value_fn1", "key3c": "key_value_fn3"},
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_builtin_keys_only_extra(self):
        BaseMapper.label_name = "test_label"
        output_annotations = [
            {
                "KEY1A": "value1a",
                "key1b": "value1b",
                "key1c": "value1c",
                "extra_key1": "",
                "extra_key2": "",
                "test_label": "xyz",
            },
            {
                "key2a": "value2a",
                "key2b": "value2b",
                "key2c": "value2c",
                "extra_key1": "",
                "extra_key2": "",
                "test_label": "xyz",
            },
            {
                "key3a": "value3a",
                "KEY3B": "value3b",
                "key3c": "value3c",
                "extra_key1": "",
                "extra_key2": "",
                "test_label": "xyz",
            },
            {
                "key4a": "value4a",
                "key4b": "value4b",
                "KEY4C": "value4c",
                "extra_key1": "",
                "extra_key2": "",
                "test_label": "xyz",
            },
        ]
        mapper = BaseMapper(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "xyz",
            builtin_key_mappings={"key1a": "KEY1A", "key3b": "KEY3B", "key4c": "KEY4C"},
            builtin_value_mappings={},
            extra_keys=["extra_key1", "extra_key2"],
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_custom_keys_only(self):
        BaseMapper.label_name = "test_label"
        output_annotations = [
            {"KEY1A": "value1a", "key1b": "value1b", "key1c": "value1c", "test_label": "xyz"},
            {"key2a": "value2a", "key2b": "value2b", "key2c": "value2c", "test_label": "xyz"},
            {"key3a": "value3a", "KEY3B": "value3b", "key3c": "value3c", "test_label": "xyz"},
            {"key4a": "value4a", "key4b": "value4b", "KEY4C": "value4c", "test_label": "xyz"},
        ]
        mapper = BaseMapper(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "xyz",
            builtin_key_mappings={},
            custom_key_mappings={"key1a": "KEY1A", "key3b": "KEY3B", "key4c": "KEY4C"},
            builtin_value_mappings={},
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_custom_values_only(self):
        BaseMapper.label_name = "test_label"
        output_annotations = [
            {"key1a": "cVALUE1", "key1b": "value1b", "key1c": "value1c", "test_label": "xyz"},
            {"key2a": "value2a", "key2b": "cVALUE2", "key2c": "value2c", "test_label": "xyz"},
            {"key3a": "value3a", "key3b": "value3b", "key3c": "cVALUE3", "test_label": "xyz"},
            {"key4a": "value4a", "key4b": "value4b", "key4c": "value4c", "test_label": "xyz"},
        ]
        mapper = BaseMapper(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "xyz",
            builtin_key_mappings={},
            builtin_value_mappings={},
            custom_value_mappings={
                "key1a": lambda **x: "cVALUE1",
                "key2b": lambda **x: "cVALUE2",
                "key3c": lambda **x: "cVALUE3",
            },
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_custom_keys_and_values(self):
        output_annotations = [
            {"key1a": "value1a", "key1b": "value1b", "KEY1C": "cVALUE1", "user_label": "abc"},
            {"key2a": "value2a", "KEY2B": "cVALUE2", "key2c": "value2c", "user_label": "abc"},
            {"key3a": "value3a", "key3b": "cVALUE3", "key3c": "value3c", "user_label": "abc"},
            {"KEY4A": "value4a", "key4b": "value4b", "key4c": "value4c", "user_label": "abc"},
        ]
        mapper = BaseMapper(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "abc",
            builtin_key_mappings={},
            builtin_value_mappings={},
            custom_key_mappings={"key1c": "KEY1C", "key2b": "KEY2B", "key4a": "KEY4A"},
            custom_value_mappings={
                "key1c": lambda **x: "cVALUE1",
                "key2b": lambda **x: "cVALUE2",
                "key3b": lambda **x: "cVALUE3",
            },
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_builtin_and_custom_keys_and_values(self):
        output_annotations = [
            {"KEY1A": "VALUE1", "key1b": "value1b", "KEY1C": "cVALUE1", "user_label": "abc"},
            {"key2a": "value2a", "KEY2B": "cVALUE2", "key2c": "value2c", "user_label": "abc"},
            {"key3a": "value3a", "KEY3B": "cVALUE3", "key3c": "VALUE3", "user_label": "abc"},
            {"KEY4A": "value4a", "key4b": "value4b", "TEST": "value4c", "user_label": "abc"},
        ]
        mapper = BaseMapperExtended(
            annotations=self.input_annotations,
            get_label_fn=lambda **x: "abc",
            builtin_key_mappings={"key1a": "KEY1A", "key3b": "KEY3B", "key4c": "KEY4C"},
            builtin_value_mappings={"key1a": "key_value_fn1", "key2b": "key_value_fn2", "key3c": "key_value_fn3"},
            custom_key_mappings={"key1c": "KEY1C", "key2b": "KEY2B", "key4a": "KEY4A", "key4c": "TEST"},
            custom_value_mappings={
                "key1c": lambda **x: "cVALUE1",
                "key2b": lambda **x: "cVALUE2",
                "key3b": lambda **x: "cVALUE3",
            },
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_label_exists(self):
        BaseMapper.label_name = "key1a"
        with self.assertRaises(ValueError):
            BaseMapper(
                annotations=self.input_annotations,
                get_label_fn=lambda **x: "abc",
                builtin_key_mappings={},
                builtin_value_mappings={},
            )

    def test_mapper_extra_key_exists(self):
        with self.assertRaises(ValueError):
            BaseMapper(
                annotations=self.input_annotations,
                get_label_fn=lambda **x: "abc",
                builtin_key_mappings={},
                builtin_value_mappings={},
                extra_keys=["user_label"],
            )


class TestBaseMapperStaticMethods(TestCase):
    """Tests for static methods in BaseMapper class."""

    def test_mapper__add_keys1(self):
        data_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}
        extra_keys = ["key4", "key5"]
        output_data = {"key1": "value1", "key2": "value2", "key3": "value3", "key4": "", "key5": ""}
        BaseMapper._add_keys(data_dict, extra_keys)
        self.assertEqual(data_dict, output_data)

    def test_mapper__add_keys2(self):
        data_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}
        extra_keys = ["key4", "key2"]
        with self.assertRaises(ValueError):
            BaseMapper._add_keys(data_dict, extra_keys)

    def test_mapper__add_keys3(self):
        data_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}
        extra_keys = ["key4", "key4"]
        with self.assertRaises(ValueError):
            BaseMapper._add_keys(data_dict, extra_keys)
