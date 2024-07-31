# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for validating annotations returned by AsapParser."""

from dplabtools.slides.annotations.validators.base import BaseValidator


class AsapValidator(BaseValidator):
    """Class for validating ASAP annotations."""

    def _validate(self):
        for annotation in self._annotations:
            annotation_type = annotation["Type"]
            annotation_name = annotation["Name"]
            points_len = len(annotation["Coordinates"])
            if annotation_type == "Rectangle" and points_len != 4:
                raise ValueError("Wrong number of points in annotation: %s" % annotation_name)
            if annotation_type == "Polygon" and points_len < 3:
                raise ValueError("Wrong number of points in annotation: %s" % annotation_name)
            if annotation_type == "Spline" and points_len < 3:
                raise ValueError("Wrong number of points in annotation: %s" % annotation_name)
