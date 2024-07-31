# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for ASAP annotation reader and embedded classes."""

import os
from unittest import TestCase
import xml.etree.ElementTree as ET

from dplabtools.slides.annotations.parsers.asap import AsapParser
from dplabtools.slides.annotations.validators.asap import AsapValidator
from dplabtools.slides.annotations.mappers.asap import AsapMapper
from dplabtools.slides.annotations.readers.asap import AsapReader
from dplabtools.slides.utils import AnnotationPolygon, PolygonData
from testutils import make_test_path


class TestAsapParser(TestCase):
    """Tests for AsapParser class."""

    def test_parser(self):
        xml_file = make_test_path("ref_data/slides/annotations/asap1.xml")
        parser = AsapParser(raw_data_file=xml_file)
        output_annotations = [
            {
                "Type": "Rectangle",
                "Name": "Annotation 1",
                "PartOfGroup": "None",
                "Color": "#F4FA58",
                "Coordinates": [(5955, 24058), (8337, 24058), (8337, 26361), (5955, 26361)],
            },
            {
                "Type": "Polygon",
                "Name": "Annotation 2",
                "PartOfGroup": "None",
                "Color": "#F4FA58",
                "Coordinates": [(10798, 26837), (9687, 24455), (11751, 23780), (13379, 24574), (12704, 25884)],
            },
            {
                "Type": "Spline",
                "Name": "Annotation 3",
                "PartOfGroup": "None",
                "Color": "#F4FA58",
                "Coordinates": [(6233, 21636), (7980, 20048), (10401, 21160), (10640, 22748)],
            },
        ]
        result_annotations = parser.annotations
        self.assertEqual(len(result_annotations), 3)
        self.assertEqual(result_annotations, output_annotations)

    def test__get_points(self):
        xml_data = """<?xml version="1.0"?>
        <Coordinates>
            <Coordinate Order="0" X="10798.3877" Y="26837.166" />
            <Coordinate Order="1" X="9686.78809" Y="24455.1699" />
            <Coordinate Order="3" X="13378,8838" Y="24574,2695" />
            <Coordinate Order="4" X="12703.9844" Y="25884.8691" />
        </Coordinates>
        """
        root = ET.fromstring(xml_data)
        point_data = root.findall("Coordinate")
        output_points = [(10798, 26837), (9687, 24455), (13379, 24574), (12704, 25885)]
        result_points = AsapParser._get_points(point_data)
        self.assertEqual(result_points, output_points)


class TestAsapValidator(TestCase):
    """Tests for AsapValidator class."""

    def test_validator_rectangle1(self):
        annotations = [
            {
                "Type": "Rectangle",
                "Name": "Region 1",
                "Coordinates": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        with self.assertRaises(ValueError):
            AsapValidator(annotations=annotations)

    def test_validator_rectangle2(self):
        annotations = [
            {
                "Type": "Rectangle",
                "Name": "Region 1",
                "Coordinates": [(10709, 24842), (13461, 24842), (13461, 26954), (1234, 5678)],
            },
        ]
        validator = AsapValidator(annotations=annotations)
        self.assertTrue(validator)

    def test_validator_polygon1(self):
        annotations = [
            {
                "Type": "Polygon",
                "Name": "Region 1",
                "Coordinates": [(10709, 24842), (13461, 24842)],
            },
        ]
        with self.assertRaises(ValueError):
            AsapValidator(annotations=annotations)

    def test_validator_polygon2(self):
        annotations = [
            {
                "Type": "Polygon",
                "Name": "Region 1",
                "Coordinates": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        validator = AsapValidator(annotations=annotations)
        self.assertTrue(validator)

    def test_validator_spline1(self):
        annotations = [
            {
                "Type": "Spline",
                "Name": "Region 1",
                "Coordinates": [(10709, 24842), (13461, 24842)],
            },
        ]
        with self.assertRaises(ValueError):
            AsapValidator(annotations=annotations)

    def test_validator_spline2(self):
        annotations = [
            {
                "Type": "Spline",
                "Name": "Region 1",
                "Coordinates": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        validator = AsapValidator(annotations=annotations)
        self.assertTrue(validator)


class TestAsapMapper(TestCase):
    """Tests for AsapMapper class."""

    def test_mapper_builtin_keys_values(self):
        input_annotations = [
            {
                "Type": "Spline",
                "Name": "Region 1",
                "Color": "#F4FA58",
                "Coordinates": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        output_annotations = [
            {
                "Type": "Spline",
                "Name": "Region 1",
                "Color": "#f4fa58",
                "points": [(10709, 24842), (13461, 24842), (13461, 26954)],
                "user_label": "abc",
            },
        ]
        mapper = AsapMapper(annotations=input_annotations, get_label_fn=lambda **x: "abc")
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)

    def test_mapper_non_builtin(self):
        input_annotations = [
            {
                "Type": "Polygon",
                "Name": "Region 1",
                "Color": "#CC0000",
                "Coordinates": [(10709, 24842), (13461, 24842), (13461, 26954)],
            },
        ]
        output_annotations = [
            {
                "Type": "Polygon",
                "New name": "Region 1",
                "Color": "#cc0000",
                "points": [(10709, 24842), (13461, 24842), (13461, 26954)],
                "user_label": "ABC",
                "key1": "",
                "key2": "KEY2",
            },
        ]
        mapper = AsapMapper(
            annotations=input_annotations,
            get_label_fn=lambda **x: "ABC",
            custom_key_mappings={"Name": "New name"},
            custom_value_mappings={"key2": lambda **x: "KEY2"},
            extra_keys=["key1", "key2"],
        )
        result_annotations = mapper.annotations
        self.assertEqual(result_annotations, output_annotations)


class TestAsapReader(TestCase):
    """Tests for AsapReader class."""

    def test_reader(self):
        xml_file = make_test_path("ref_data/slides/annotations/asap1.xml")
        reader = AsapReader(data_file=xml_file, get_label_fn=lambda **x: x["Name"])
        output_polygons = [
            AnnotationPolygon(
                points=[(5955, 24058), (8337, 24058), (8337, 26361), (5955, 26361)],
                label="Annotation 1",
            ),
            AnnotationPolygon(
                points=[(10798, 26837), (9687, 24455), (11751, 23780), (13379, 24574), (12704, 25884)],
                label="Annotation 2",
            ),
            AnnotationPolygon(
                points=[(6233, 21636), (7980, 20048), (10401, 21160), (10640, 22748)],
                label="Annotation 3",
            ),
        ]
        result_polygons = reader.polygons
        self.assertEqual(result_polygons, output_polygons)
        # test JSON file
        save_dir = make_test_path("saved_data/annotations2")
        file_json = os.path.join(save_dir, "asap1.json")
        reader.save_json(file_json)
        polygons = PolygonData(polygon_data=file_json).polygons
        self.assertEqual(len(polygons), 3)
        self.assertEqual(polygons, output_polygons)
