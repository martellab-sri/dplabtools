# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Annotation mapping related utilities."""

import shapely.affinity
from shapely.geometry import Point


def discretize_ellipse(rect_points):
    """Convert ellipse defined by rectangle into polygon points."""
    # top-left, top-right, bottom-right, bottom-left
    [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] = rect_points
    ellipse_center = (round((x1 + x2) / 2), round((y1 + y4) / 2))
    ellipse_width = round((x2 - x1) / 2)
    ellipse_height = round((y4 - y1) / 2)
    circle = Point(ellipse_center).buffer(1)
    ellipse = shapely.affinity.scale(circle, ellipse_width, ellipse_height)
    return [(round(x), round(y)) for x, y in ellipse.exterior.coords]
