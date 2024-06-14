# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions used to fix polygons."""

from unittest import TestCase

from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.collection import GeometryCollection

from dplabtools.slides.patches.locations.polyfix import fix_polygon_type


class TestPolyfix(TestCase):
    """Tests for different cases inside fix_polygon_type."""

    def test_case1(self):
        """Input data is GeometryCollection."""
        # polygon and lines
        poly = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        line = LineString([[0, 0], [1, 0], [1, 1]])
        multiline = MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
        collection = GeometryCollection([poly, line, multiline])
        result_polygon = fix_polygon_type(collection)
        output_polygon = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        self.assertEqual(result_polygon, output_polygon)
        self.assertEqual(result_polygon.area, output_polygon.area)
        #
        # multipolygon and lines
        poly1 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        poly2 = Polygon([(1 + 10, 1 + 10), (1 + 10, 2 + 10), (2 + 10, 2 + 10), (2 + 10, 1 + 10)])
        multipolygon = MultiPolygon([poly1, poly2])
        line = LineString([[0, 0], [1, 0], [1, 1]])
        multiline = MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
        collection = GeometryCollection([multipolygon, line, multiline])
        result_polygon = fix_polygon_type(collection)
        output_polygon = MultiPolygon([poly1, poly2])
        self.assertEqual(result_polygon, output_polygon)
        self.assertEqual(result_polygon.area, output_polygon.area)
        #
        # two polygons and lines (also tests area check condition)
        poly1 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
        poly2 = Polygon([(1 + 10, 1 + 10), (1 + 10, 2 + 10), (2 + 10, 2 + 10), (2 + 10, 1 + 10)])
        line = LineString([[0, 0], [1, 0], [1, 1]])
        multiline = MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
        collection = GeometryCollection([poly1, poly2, line, multiline])
        with self.assertRaises(TypeError):
            fix_polygon_type(collection)

    def test_case2(self):
        """Input data is either lines or points."""
        # single point
        input_data = Point(1, 1)
        with self.assertRaises(TypeError):
            fix_polygon_type(input_data)
        # multi point
        input_data = MultiPoint([[0, 0], [1, 1]])
        with self.assertRaises(TypeError):
            fix_polygon_type(input_data)
        # single line
        input_data = LineString([[0, 0], [1, 0], [1, 1]])
        with self.assertRaises(TypeError):
            fix_polygon_type(input_data)
        # multi line
        input_data = MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]])
        with self.assertRaises(TypeError):
            fix_polygon_type(input_data)
