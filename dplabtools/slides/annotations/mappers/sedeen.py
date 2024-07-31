# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Class for mapping Sedeen annotations."""

from dplabtools.slides.annotations.mappers.base import BaseMapper
from dplabtools.slides.annotations.mappers.utils import discretize_ellipse


class SedeenMapper(BaseMapper):
    """Class for mapping Sedeen annotations."""

    _builtin_value_mappings = {"color": "_map_color_fn", "point-list": "_map_points_fn"}
    _builtin_key_mappings = {"point-list": BaseMapper.points_name}

    def __init__(self, *, annotations, get_label_fn, custom_key_mappings={}, custom_value_mappings={}, extra_keys=[]):
        super().__init__(
            annotations=annotations,
            get_label_fn=get_label_fn,
            builtin_key_mappings=self._builtin_key_mappings,
            builtin_value_mappings=self._builtin_value_mappings,
            custom_key_mappings=custom_key_mappings,
            custom_value_mappings=custom_value_mappings,
            extra_keys=extra_keys,
        )

    @staticmethod
    def _map_color_fn(**kwargs):
        return kwargs["color"][:-2]

    @staticmethod
    def _map_points_fn(**kwargs):
        points = kwargs["point-list"]
        if kwargs["type"] == "ellipse":
            points = discretize_ellipse(points)
        return points
