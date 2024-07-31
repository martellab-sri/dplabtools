# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for reading QuPath projects and retrieving annotation data."""

import os
import json

from shapely.geometry import Polygon

from dplabtools.slides.utils import AnnotationPolygon


class QuPathProjectReader:
    """Class for reading QuPath projects and retrieving annotation data."""

    def __init__(self, *, qupath_install_dir, qupath_project_dir):
        """Create an object for reading QuPath annotations.

        Parameters
        ----------
        qupath_install_dir : str
            Directory where QuPath is installed.

        qupath_project_dir : str
            Directory with a saved QuPath project.
        """
        self._qupath_install_dir = qupath_install_dir
        self._qupath_project_dir = qupath_project_dir
        self._json_file_template = "%s.json"
        self._set_paquo()
        self._project_data = self._get_project_data()

    def _set_paquo(self):
        mock_env = os.getenv("PAQUO_MOCK_BACKEND", "false")
        if mock_env == "false":
            os.environ["PAQUO_QUPATH_DIR"] = self._qupath_install_dir

    def _get_project_data(self):
        project_data = []
        # paquo must be imported locally to make tests and user programs work correctly,
        # this is because of the check whether QuPath is installed or not, which should trigger
        # only when the user decides to use QuPathProjectReader (not when paquo is imported).
        from paquo.projects import QuPathProject

        with QuPathProject(self._qupath_project_dir, mode="r") as qpp:
            for image in qpp.images:
                image_name = image.image_name
                annotations = image.hierarchy.annotations
                image_data = []
                for annotation in annotations:
                    if not isinstance(annotation.roi, Polygon):
                        continue
                    label = annotation.path_class.name
                    shapely_polygon = annotation.roi
                    points = list(zip(*shapely_polygon.exterior.coords.xy))
                    annotation_polygon = AnnotationPolygon(points=points, label=label)
                    image_data.append(annotation_polygon)
                image_entry = (image_name, image_data)
                project_data.append(image_entry)
        return project_data

    def save_json(self, save_dir):
        """Save the extracted annotations for all images in the project as JSON files.

        Parameters
        ----------
        save_dir : str
            Directory for saving JSON files.
        """
        for file_name, file_data in self._project_data:
            polygons = [poly.data_dict for poly in file_data]
            json_file_path = os.path.join(save_dir, self._json_file_template % file_name)
            with open(json_file_path, "w") as jfile:
                json.dump(polygons, jfile)

    @property
    def project_data(self):
        """Return the project data as list of tuples (`file_name`, `list of AnnotationPolygon objects`)."""
        return self._project_data
