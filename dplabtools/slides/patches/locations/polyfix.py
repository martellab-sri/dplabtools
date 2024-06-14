# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Custom functions for fixing specific problems with polygons.

Add a new case when necessary.
"""

from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.geometry.collection import GeometryCollection


def fix_polygon_type(polygon, msg_prefix=""):
    """Run custom methods attempting to fix polygon specific problems.

    - Problems are listed as cases and each case has individual fix
    - Polygon area must be preserved for individual fixes to be accepted
    """
    fixed_polygon = None
    fixed_polygon_area = 0

    # Case #1: polygon is GeometryCollection which consists of a single polygon + some line string objects
    if isinstance(polygon, GeometryCollection):
        polygon_count = linestring_count = other_count = 0
        for geometry in polygon.geoms:
            if isinstance(geometry, (Polygon, MultiPolygon)):
                polygon_count += 1
                fixed_polygon = geometry
            elif isinstance(geometry, (LineString, MultiLineString)):
                linestring_count += 1
            else:
                other_count += 1
        if polygon_count == 1 and linestring_count > 0 and other_count == 0:
            fixed_polygon_area = fixed_polygon.area

    # Case #2: polygon becomes a non polygon object after make_valid
    if isinstance(polygon, (Point, MultiPoint, LineString, MultiLineString)):
        raise TypeError(
            "%s: Received polygon of invalid type: %s. Increasing mask size may help." % (msg_prefix, polygon)
        )

    # Case #3 ...
    # Case #4 ...
    # Case #5 ...

    # check polygon area
    if polygon.area != fixed_polygon_area:
        raise TypeError(
            "%s: Cannot fix polygon and maintain its area: %s [area1=%f, area2=%f]. "
            "Add another case for polygon fixing." % (msg_prefix, type(polygon), polygon.area, fixed_polygon_area)
        )

    return fixed_polygon
