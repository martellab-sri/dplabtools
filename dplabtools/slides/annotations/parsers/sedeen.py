# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for parsing Sedeen annotation files."""

import xml.etree.ElementTree as ET

from dplabtools.slides.annotations.parsers.base import BaseParser


class SedeenParser(BaseParser):
    """Class for parsing Sedeen annotation files."""

    _annotation_objects_tree = "./image/overlays/graphic"
    _polygon_like_types = ("polygon", "rectangle", "ellipse")

    def _parse(self):
        tree = ET.parse(self._raw_data_file)
        annotation_objects = tree.findall(self._annotation_objects_tree)
        for annotation_object in annotation_objects:
            dict_data = self._process_one_annnotation_object(annotation_object)
            if dict_data:
                self._annotations.append(dict_data)

    def _process_one_annnotation_object(self, annotation_object):
        dict_data = {}
        annotation_type = annotation_object.attrib["type"]
        if annotation_type in self._polygon_like_types:
            name = annotation_object.attrib["name"]
            description = annotation_object.attrib["description"]
            color = annotation_object.find("pen").attrib["color"]
            points_data = annotation_object.find("point-list").findall("point")
            point_list = self._get_points(points_data)
            dict_data["type"] = annotation_type
            dict_data["name"] = name
            dict_data["description"] = description
            dict_data["color"] = color
            dict_data["point-list"] = point_list
        return dict_data

    @staticmethod
    def _get_points(point_data):
        points = []
        for point in point_data:
            point_xy = point.text.split(",")
            point_int = (round(float(point_xy[0])), round(float(point_xy[1])))
            points.append(point_int)
        return points
