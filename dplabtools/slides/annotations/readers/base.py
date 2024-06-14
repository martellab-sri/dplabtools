# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for annotation readers.

Annotation reader classes are final end-user classes for reading annotations.
"""

import json

from dplabtools.slides.utils import AnnotationPolygon


class BaseReader:
    """Base class for annotation readers."""

    _parser_class = None
    _validator_class = None
    _mapper_class = None

    def __init__(self, *, data_file, get_label_fn):
        """Init method for annotation readers derived from the base class.

        Parameters
        ----------
        data_file : str
            File name or path for annotation raw data file.

        get_label_fn : function
            Function returning a value to be used as the label.
        """
        self._parser = self._parser_class(raw_data_file=data_file)
        self._validator = self._validator_class(annotations=self._parser.annotations)
        self._mapper = self._mapper_class(annotations=self._parser.annotations, get_label_fn=get_label_fn)

    def save_json(self, json_file):
        """Save annotations as JSON file with serialized AnnotationPolygon objects.

        Parameters
        ----------
        json_file : str
            JSON file name or path.
        """
        polygons = [poly.data_dict for poly in self.polygons]
        with open(json_file, "w") as jfile:
            json.dump(polygons, jfile)

    @property
    def polygons(self):
        """Return annotations as a list of AnnotationPolygon objects."""
        polygons = []
        points_field = self._mapper.points_name
        label_field = self._mapper.label_name
        for annotation in self._mapper.annotations:
            points = annotation[points_field]
            label = annotation[label_field]
            poly = AnnotationPolygon(points=points, label=label)
            polygons.append(poly)
        return polygons
