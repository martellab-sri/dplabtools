# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for Sedeen annotation reader and embedded classes."""

import os
from unittest import TestCase
import xml.etree.ElementTree as ET

from dplabtools.slides.annotations.parsers.sedeen import SedeenParser
from dplabtools.slides.annotations.validators.sedeen import SedeenValidator
from dplabtools.slides.annotations.mappers.sedeen import SedeenMapper
from dplabtools.slides.annotations.readers.sedeen import SedeenReader
from dplabtools.slides.utils import AnnotationPolygon, PolygonData
from dplabtools.slides.annotations.mappers.utils import discretize_ellipse
from testutils import make_test_path


class TestSedeenParser(TestCase):
    """Tests for SedeenParser class."""

    def test_parser(self):
        xml_file = make_test_path("ref_data/slides/annotations/sedeen1.xml")
        parser = SedeenParser(raw_data_file=xml_file)
        output_annotations = [
            {
                "type": "polygon",
                "name": "Region 0",
                "description": "test 0",
                "color": "#00ff00ff",
                "point-list": [(5493, 23178), (4757, 25322), (7189, 26026), (8149, 23914), (6005, 22506)],
            },
            {
                "type": "rectangle",
                "name": "Region 1",
                "description": "test 1",
                "color": "#00ff00ff",
                "point-list": [(10709, 24842), (13461, 24842), (13461, 26954), (10709, 26954)],
            },
            {
                "type": "ellipse",
                "name": "Region 2",
                "description": "test 2",
                "color": "#00ff00ff",
                "point-list": [(7893, 20074), (9493, 20074), (9493, 22058), (7893, 22058)],
            },
            {
                "type": "polygon",
                "name": "Region 3",
                "description": "test 3",
                "color": "#00ff00ff",
                "point-list": [
                    (11765, 21642),
                    (11797, 21642),
                    (11829, 21642),
                    (11861, 21706),
                    (11893, 21706),
                    (11925, 21674),
                    (12085, 21610),
                    (12181, 21610),
                    (12213, 21610),
                    (12245, 21610),
                    (12277, 21578),
                    (12309, 21578),
                    (12341, 21546),
                    (12373, 21546),
                    (12405, 21514),
                    (12437, 21514),
                    (12437, 21482),
                    (12501, 21450),
                    (12533, 21418),
                    (12597, 21386),
                    (12629, 21354),
                    (12693, 21290),
                    (12725, 21258),
                    (12725, 21226),
                    (12757, 21194),
                    (12757, 21162),
                    (12757, 21130),
                    (12789, 21066),
                    (12821, 21066),
                    (12821, 21034),
                    (12821, 21002),
                    (12821, 20970),
                    (12821, 20938),
                    (12821, 20874),
                    (12821, 20842),
                    (12821, 20778),
                    (12821, 20746),
                    (12789, 20714),
                    (12757, 20650),
                    (12725, 20586),
                    (12693, 20586),
                    (12629, 20522),
                    (12597, 20490),
                    (12565, 20458),
                    (12501, 20458),
                    (12501, 20426),
                    (12469, 20426),
                    (12437, 20426),
                    (12405, 20394),
                    (12373, 20394),
                    (12309, 20394),
                    (12245, 20394),
                    (12181, 20394),
                    (12149, 20394),
                    (12085, 20394),
                    (11989, 20426),
                    (11957, 20426),
                    (11765, 20458),
                    (11733, 20458),
                    (11669, 20458),
                    (11605, 20458),
                    (11541, 20490),
                    (11509, 20522),
                    (11477, 20554),
                    (11445, 20554),
                    (11413, 20618),
                    (11413, 20650),
                    (11381, 20682),
                    (11349, 20746),
                    (11317, 20810),
                    (11317, 20842),
                    (11317, 20874),
                    (11317, 20938),
                    (11285, 20970),
                    (11285, 21002),
                    (11285, 21034),
                    (11285, 21098),
                    (11285, 21194),
                    (11285, 21226),
                    (11317, 21290),
                    (11317, 21322),
                    (11349, 21354),
                    (11381, 21354),
                    (11413, 21354),
                ],
            },
        ]
        result_annotations = parser.annotations
        self.assertEqual(len(result_annotations), 4)
        self.assertEqual(result_annotations, output_annotations)

    def test__get_points(self):
        xml_data = """<?xml version="1.0"?>
        <point-list>
            <point>5491.100000,23171.000000</point>
            <point>7183.000000,26023.000000</point>
            <point>8144.900000,23914.000000</point>
            <point>6005.000000,22505.900000</point>
        </point-list>
        """
        root = ET.fromstring(xml_data)
        point_data = root.findall("point")
        output_points = [(5491, 23171), (7183, 26023), (8145, 23914), (6005, 22506)]
        result_points = SedeenParser._get_points(point_data)
        self.assertEqual(result_points, output_points)


class TestSedeenValidator(TestCase):
    """Tests for SedeenValidator class."""

    def test_validator_rectangle1(self):
        annotations = [
            {
                "type": "rectangle",
                "name": "Region 1",
                "point-list": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        with self.assertRaises(ValueError):
            SedeenValidator(annotations=annotations)

    def test_validator_rectangle2(self):
        annotations = [
            {
                "type": "rectangle",
                "name": "Region 1",
                "point-list": [(10709, 24842), (13461, 24842), (13461, 26954), (1234, 5678)],
            },
        ]
        validator = SedeenValidator(annotations=annotations)
        self.assertTrue(validator)

    def test_validator_polygon1(self):
        annotations = [
            {
                "type": "polygon",
                "name": "Region 2",
                "point-list": [(5493, 23178), (4757, 25322)],
            },
        ]
        with self.assertRaises(ValueError):
            SedeenValidator(annotations=annotations)

    def test_validator_polygon2(self):
        annotations = [
            {
                "type": "polygon",
                "name": "Region 2",
                "point-list": [(5493, 23178), (4757, 25322), (1234, 5678)],
            },
        ]
        validator = SedeenValidator(annotations=annotations)
        self.assertTrue(validator)

    def test_validator_ellipse1(self):
        annotations = [
            {
                "type": "ellipse",
                "name": "Region 3",
                "point-list": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        with self.assertRaises(ValueError):
            SedeenValidator(annotations=annotations)

    def test_validator_ellipse2(self):
        annotations = [
            {
                "type": "ellipse",
                "name": "Region 3",
                "point-list": [(10709, 24842), (13461, 24842), (13461, 26954), (1234, 5678)],
            },
        ]
        validator = SedeenValidator(annotations=annotations)
        self.assertTrue(validator)


class TestSedeenMapper(TestCase):
    """Tests for SedeenMapper class."""

    def test_mapper_builtin_keys_values(self):
        input_annotations = [
            {
                "type": "rectangle",
                "name": "Region 1",
                "color": "#cc0000ff",
                "point-list": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        output_annotations = [
            {
                "type": "rectangle",
                "name": "Region 1",
                "color": "#cc0000",
                "points": [(10709, 24842), (13461, 24842), (13461, 26954)],
                "user_label": "abc",
            },
        ]
        mapper = SedeenMapper(annotations=input_annotations, get_label_fn=lambda **x: "abc")
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_non_builtin(self):
        input_annotations = [
            {
                "type": "rectangle",
                "name": "Region 1",
                "color": "#cc0000ff",
                "point-list": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        output_annotations = [
            {
                "type": "rectangle",
                "new_name": "Region 1",
                "color": "#cc0000",
                "points": [(10709, 24842), (13461, 24842), (13461, 26954)],
                "user_label": "ABC",
                "key1": "KEY1",
                "key2": "",
            },
        ]
        mapper = SedeenMapper(
            annotations=input_annotations,
            get_label_fn=lambda **x: "ABC",
            custom_key_mappings={"name": "new_name"},
            custom_value_mappings={"key1": lambda **x: "KEY1"},
            extra_keys=["key1", "key2"],
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_ellipse(self):
        input_annotations = [
            {
                "type": "ellipse",
                "name": "Region 1",
                "color": "#bb0000ff",
                "point-list": [(7893, 20074), (9493, 20074), (9493, 22058), (7893, 22058)],
            },
        ]
        output_annotations = [
            {
                "type": "ellipse",
                "name": "Region 1",
                "color": "#bb0000",
                "points": [(7893, 20074), (9493, 20074), (9493, 22058), (7893, 22058)],
                "user_label": "abc",
            },
        ]
        mapper = SedeenMapper(annotations=input_annotations, get_label_fn=lambda **x: "abc")
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations[0]["type"], output_annotations[0]["type"])
        self.assertEqual(result_annotations[0]["name"], output_annotations[0]["name"])
        self.assertEqual(result_annotations[0]["color"], output_annotations[0]["color"])
        self.assertEqual(result_annotations[0]["user_label"], output_annotations[0]["user_label"])
        self.assertTrue(len(result_annotations[0]["points"]) > len(output_annotations[0]["points"]))


class TestSedeenReader(TestCase):
    """Tests for SedeenReader class."""

    def test_reader(self):
        xml_file = make_test_path("ref_data/slides/annotations/sedeen1.xml")
        reader = SedeenReader(data_file=xml_file, get_label_fn=lambda **x: x["name"])
        ellipse_points = discretize_ellipse([(7893, 20074), (9493, 20074), (9493, 22058), (7893, 22058)])
        output_polygons = [
            AnnotationPolygon(
                points=[
                    (5493, 23178),
                    (4757, 25322),
                    (7189, 26026),
                    (8149, 23914),
                    (6005, 22506),
                ],
                label="Region 0",
            ),
            AnnotationPolygon(
                points=[
                    (10709, 24842),
                    (13461, 24842),
                    (13461, 26954),
                    (10709, 26954),
                ],
                label="Region 1",
            ),
            AnnotationPolygon(
                points=ellipse_points,
                label="Region 2",
            ),
            AnnotationPolygon(
                points=[
                    (11765, 21642),
                    (11797, 21642),
                    (11829, 21642),
                    (11861, 21706),
                    (11893, 21706),
                    (11925, 21674),
                    (12085, 21610),
                    (12181, 21610),
                    (12213, 21610),
                    (12245, 21610),
                    (12277, 21578),
                    (12309, 21578),
                    (12341, 21546),
                    (12373, 21546),
                    (12405, 21514),
                    (12437, 21514),
                    (12437, 21482),
                    (12501, 21450),
                    (12533, 21418),
                    (12597, 21386),
                    (12629, 21354),
                    (12693, 21290),
                    (12725, 21258),
                    (12725, 21226),
                    (12757, 21194),
                    (12757, 21162),
                    (12757, 21130),
                    (12789, 21066),
                    (12821, 21066),
                    (12821, 21034),
                    (12821, 21002),
                    (12821, 20970),
                    (12821, 20938),
                    (12821, 20874),
                    (12821, 20842),
                    (12821, 20778),
                    (12821, 20746),
                    (12789, 20714),
                    (12757, 20650),
                    (12725, 20586),
                    (12693, 20586),
                    (12629, 20522),
                    (12597, 20490),
                    (12565, 20458),
                    (12501, 20458),
                    (12501, 20426),
                    (12469, 20426),
                    (12437, 20426),
                    (12405, 20394),
                    (12373, 20394),
                    (12309, 20394),
                    (12245, 20394),
                    (12181, 20394),
                    (12149, 20394),
                    (12085, 20394),
                    (11989, 20426),
                    (11957, 20426),
                    (11765, 20458),
                    (11733, 20458),
                    (11669, 20458),
                    (11605, 20458),
                    (11541, 20490),
                    (11509, 20522),
                    (11477, 20554),
                    (11445, 20554),
                    (11413, 20618),
                    (11413, 20650),
                    (11381, 20682),
                    (11349, 20746),
                    (11317, 20810),
                    (11317, 20842),
                    (11317, 20874),
                    (11317, 20938),
                    (11285, 20970),
                    (11285, 21002),
                    (11285, 21034),
                    (11285, 21098),
                    (11285, 21194),
                    (11285, 21226),
                    (11317, 21290),
                    (11317, 21322),
                    (11349, 21354),
                    (11381, 21354),
                    (11413, 21354),
                ],
                label="Region 3",
            ),
        ]
        result_polygons = reader.polygons
        self.assertEqual(result_polygons, output_polygons)
        # test JSON file
        save_dir = make_test_path("saved_data/annotations3")
        file_json = os.path.join(save_dir, "sedeen1.json")
        reader.save_json(file_json)
        polygons = PolygonData(polygon_data=file_json).polygons
        self.assertEqual(len(polygons), 4)
        self.assertEqual(polygons, output_polygons)
