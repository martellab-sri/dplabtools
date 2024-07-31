# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for QuPath annotation reader."""

import os
from unittest import TestCase
from unittest.mock import patch, Mock

import shapely

from dplabtools.slides.annotations import QuPathProjectReader
from dplabtools.slides.utils import AnnotationPolygon, PolygonData
from testutils import make_test_path

# must be set before creating QuPathProjectReader object
os.environ["PAQUO_MOCK_BACKEND"] = "true"


class QuPathProjectMock:
    """Mock QuPathProject class and prepare artificial annotation data."""

    def __init__(self, qupath_project_dir, mode):
        pass

    def __enter__(self):
        # create mock objects with annotations content for two images: image1.tif, image2.tif
        roi1a = shapely.geometry.Polygon([(10, 10), (20, 20), (30, 10)])
        roi1b = shapely.geometry.Polygon([(40, 40), (50, 50), (60, 40)])
        roi1c = shapely.geometry.Polygon([(70, 70), (80, 80), (90, 70)])
        roi2a = shapely.geometry.Polygon([(30, 30), (40, 40), (50, 30)])
        roi2b = shapely.geometry.LineString([(300, 300), (400, 400)])
        labels1 = ["label1a", "label1b", "label1c"]
        labels2 = ["label2a", "label2b"]
        # SO #62552148
        m_path_class1a = Mock()
        m_path_class1a.name = labels1[0]
        m_path_class1b = Mock()
        m_path_class1b.name = labels1[1]
        m_path_class1c = Mock()
        m_path_class1c.name = labels1[2]
        m_path_class2a = Mock()
        m_path_class2a.name = labels2[0]
        m_path_class2b = Mock()
        m_path_class2b.name = labels2[1]
        m_annotations1 = [
            Mock(roi=roi1a, path_class=m_path_class1a),
            Mock(roi=roi1b, path_class=m_path_class1b),
            Mock(roi=roi1c, path_class=m_path_class1c),
        ]
        m_annotations2 = [
            Mock(roi=roi2a, path_class=m_path_class2a),
            Mock(roi=roi2b, path_class=m_path_class2b),
        ]
        m_hierarchy1 = Mock(annotations=m_annotations1)
        m_hierarchy2 = Mock(annotations=m_annotations2)
        m_images = [
            Mock(image_name="image1.tif", hierarchy=m_hierarchy1),
            Mock(image_name="image2.tif", hierarchy=m_hierarchy2),
        ]
        images = Mock(images=m_images)
        return images

    def __exit__(self, *args):
        pass


class TestQuPathProjectReader(TestCase):
    """Tests for QuPathProjectReader class."""

    poly1 = AnnotationPolygon(points=[(10.0, 10.0), (20.0, 20.0), (30.0, 10.0), (10.0, 10.0)], label="label1a")
    poly2 = AnnotationPolygon(points=[(40.0, 40.0), (50.0, 50.0), (60.0, 40.0), (40.0, 40.0)], label="label1b")
    poly3 = AnnotationPolygon(points=[(70.0, 70.0), (80.0, 80.0), (90.0, 70.0), (70.0, 70.0)], label="label1c")
    poly4 = AnnotationPolygon(points=[(30.0, 30.0), (40.0, 40.0), (50.0, 30.0), (30.0, 30.0)], label="label2a")

    @patch("paquo.projects.QuPathProject", QuPathProjectMock)
    def test_project_data(self):
        qppr = QuPathProjectReader(qupath_install_dir="not_relevant", qupath_project_dir="not_relevant")
        mocked_output_data = [
            ("image1.tif", [self.poly1, self.poly2, self.poly3]),
            ("image2.tif", [self.poly4]),
        ]
        # compare file names
        self.assertEqual(qppr.project_data[0][0], mocked_output_data[0][0])
        self.assertEqual(qppr.project_data[1][0], mocked_output_data[1][0])
        # compare annotation polygons
        self.assertEqual(qppr.project_data[0][1], mocked_output_data[0][1])
        self.assertEqual(qppr.project_data[1][1], mocked_output_data[1][1])

    @patch("paquo.projects.QuPathProject", QuPathProjectMock)
    def test_save_dir(self):
        save_dir = make_test_path("saved_data/annotations1")
        qppr = QuPathProjectReader(qupath_install_dir="not_relevant", qupath_project_dir="not_relevant")
        qppr.save_json(save_dir)
        # test image1
        file_json1 = os.path.join(save_dir, "image1.tif.json")
        polygons1 = PolygonData(polygon_data=file_json1).polygons
        self.assertEqual(len(polygons1), 3)
        self.assertEqual(polygons1[0], self.poly1)
        self.assertEqual(polygons1[1], self.poly2)
        self.assertEqual(polygons1[2], self.poly3)
        # test image2
        file_json2 = os.path.join(save_dir, "image2.tif.json")
        polygons2 = PolygonData(polygon_data=file_json2).polygons
        self.assertEqual(len(polygons2), 1)
        self.assertEqual(polygons2[0], self.poly4)
