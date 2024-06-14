# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for parsing ASAP annotation files."""

import xml.etree.ElementTree as ET

from dplabtools.slides.annotations.parsers.base import BaseParser


class AsapParser(BaseParser):
    """Class for parsing ASAP annotation files."""

    _annotation_objects_tree = "./Annotations/Annotation"
    _polygon_like_types = ("Polygon", "Rectangle", "Spline")

    def _parse(self):
        tree = ET.parse(self._raw_data_file)
        annotation_objects = tree.findall(self._annotation_objects_tree)
        for annotation_object in annotation_objects:
            dict_data = self._process_one_annnotation_object(annotation_object)
            if dict_data:
                self._annotations.append(dict_data)

    def _process_one_annnotation_object(self, annotation_object):
        dict_data = {}
        annotation_type = annotation_object.attrib["Type"]
        if annotation_type in self._polygon_like_types:
            name = annotation_object.attrib["Name"]
            group = annotation_object.attrib["PartOfGroup"]
            color = annotation_object.attrib["Color"]
            points_data = annotation_object.findall("Coordinates/Coordinate")
            point_list = self._get_points(points_data)
            dict_data["Type"] = annotation_type
            dict_data["Name"] = name
            dict_data["PartOfGroup"] = group
            dict_data["Color"] = color
            dict_data["Coordinates"] = point_list
        return dict_data

    @staticmethod
    def _get_points(points_data):
        points = []
        for coordinate in points_data:
            # some ASAP XML files use comma as decimal separator
            x = coordinate.attrib["X"].replace(",", ".")
            y = coordinate.attrib["Y"].replace(",", ".")
            point_int = (round(float(x)), round(float(y)))
            points.append(point_int)
        return points
