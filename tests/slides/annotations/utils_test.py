# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for functions included in slides.annotations.* namespace"""

from unittest import TestCase

from dplabtools.slides.annotations.mappers import utils as mappersutils


class TestUtilsMappers(TestCase):
    """Tests for functions included in slides.annotations.mappers.utils."""

    def test_discretize_ellipse1(self):
        rect_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
        output_ellipse_points = [
            (4, 0),
            (3, 1),
            (9, 2),
            (3, 10),
            (5, 10),
            (8, 9),
            (10, 6),
            (9, 8),
            (0, 5),
            (10, 3),
            (1, 3),
            (1, 9),
            (7, 1),
            (7, 10),
            (3, 0),
            (5, 0),
            (3, 9),
            (9, 1),
            (9, 7),
            (0, 7),
            (10, 5),
            (1, 2),
            (0, 4),
            (2, 1),
            (7, 0),
            (1, 8),
            (7, 9),
            (6, 10),
            (4, 10),
            (9, 3),
            (8, 1),
            (9, 9),
            (10, 4),
            (1, 1),
            (0, 3),
            (10, 7),
            (0, 6),
            (2, 9),
            (1, 7),
            (6, 0),
        ]
        result_ellipse_points = mappersutils.discretize_ellipse(rect_points)
        self.assertEqual(set(result_ellipse_points), set(output_ellipse_points))

    def test_discretize_ellipse2(self):
        rect_points = [(-50, -20), (50, -20), (50, 20), (-50, 20)]
        output_ellipse_points = [
            (-35, 14),
            (19, -18),
            (32, -15),
            (50, 0),
            (-32, -15),
            (15, -19),
            (-46, 8),
            (42, 11),
            (-39, 13),
            (39, -13),
            (19, 18),
            (-50, -2),
            (-35, -14),
            (49, 4),
            (-49, -4),
            (32, 15),
            (-42, 11),
            (0, 20),
            (-19, -18),
            (-32, 15),
            (35, 14),
            (46, 8),
            (-15, 19),
            (0, -20),
            (42, -11),
            (-46, -8),
            (50, 2),
            (35, -14),
            (46, -8),
            (-42, -11),
            (-19, 18),
            (28, 17),
            (-50, 0),
            (-24, -18),
            (-5, 20),
            (-48, -6),
            (-49, 4),
            (-44, -9),
            (10, 20),
            (28, -17),
            (-48, 6),
            (-5, -20),
            (39, 13),
            (15, 19),
            (10, -20),
            (-39, -13),
            (-44, 9),
            (50, -2),
            (-24, 18),
            (44, -9),
            (24, -18),
            (-10, 20),
            (-15, -19),
            (48, -6),
            (-28, 17),
            (44, 9),
            (5, 20),
            (49, -4),
            (-10, -20),
            (48, 6),
            (-50, 2),
            (24, 18),
            (5, -20),
            (-28, -17),
        ]
        result_ellipse_points = mappersutils.discretize_ellipse(rect_points)
        self.assertEqual(set(result_ellipse_points), set(output_ellipse_points))


"""
import matplotlib.pyplot as plt

width, height = 12, 12
points = [(4, 0), (3, 1), (9, 2), (3, 10), (5, 10), (8, 9), (10, 6), (9, 8), (0, 5), (10, 3), (1, 3), (1, 9), (7, 1), (7, 10), (3, 0), (5, 0), (3, 9), (9, 1), (9, 7), (0, 7), (10, 5), (1, 2), (0, 4), (2, 1), (7, 0), (1, 8), (7, 9), (6, 10), (4, 10), (9, 3), (8, 1), (9, 9), (10, 4), (1, 1), (0, 3), (10, 7), (0, 6), (2, 9), (1, 7), (6, 0)]
points_x = [pt[0] for pt in points]
points_y = [pt[1] for pt in points]
plt.scatter(points_x, points_y, c='b')
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('on')
plt.show()


import matplotlib.pyplot as plt

width, height = 60, 30
points = [(-35, 14), (19, -18), (32, -15), (50, 0), (-32, -15), (15, -19), (-46, 8), (42, 11), (-39, 13), (39, -13), (19, 18), (-50, -2), (-35, -14), (49, 4), (-49, -4), (32, 15), (-42, 11), (0, 20), (-19, -18), (-32, 15), (35, 14), (46, 8), (-15, 19), (0, -20), (42, -11), (-46, -8), (50, 2), (35, -14), (46, -8), (-42, -11), (-19, 18), (28, 17), (-50, 0), (-24, -18), (-5, 20), (-48, -6), (-49, 4), (-44, -9), (10, 20), (28, -17), (-48, 6), (-5, -20), (39, 13), (15, 19), (10, -20), (-39, -13), (-44, 9), (50, -2), (-24, 18), (44, -9), (24, -18), (-10, 20), (-15, -19), (48, -6), (-28, 17), (44, 9), (5, 20), (49, -4), (-10, -20), (48, 6), (-50, 2), (24, 18), (5, -20), (-28, -17)]
points_x = [pt[0] for pt in points]
points_y = [pt[1] for pt in points]
plt.scatter(points_x, points_y, c='b')
plt.xlim(-width, width)
plt.ylim(-height, height)
plt.axis('on')
plt.show()
"""
