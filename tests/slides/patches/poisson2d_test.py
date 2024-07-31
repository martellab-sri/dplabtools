# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Test cases for TestPoissonDisk2DPoints."""

from unittest import TestCase
from collections import Counter

from dplabtools.slides.patches.locations.poisson2d import PoissonDisk2DPoints


class TestPoissonDisk2DPoints(TestCase):
    """Tests for different scenarios of 2D points."""

    def test_points1(self):
        for _ in range(100):
            points = PoissonDisk2DPoints(1, 1, 1)
            self.assertTrue(1 <= len(list(points)) <= 2)

    def test_points2(self):
        for _ in range(100):
            points = PoissonDisk2DPoints(3, 3, 3)
            self.assertTrue(1 <= len(list(points)) <= 2)

    def test_points3(self):
        for _ in range(100):
            points = PoissonDisk2DPoints(3, 3, 2)
            self.assertTrue(1 <= len(list(points)) <= 4)

    def test_points4(self):
        all_points = []
        for _ in range(100):
            points = PoissonDisk2DPoints(1, 1, 2)
            all_points.append(len(list(points)))
        self.assertEqual(max(all_points), 1)

    def test_points5(self):
        all_points = []
        for _ in range(100):
            points = PoissonDisk2DPoints(3, 3, 2)
            all_points.append(len(list(points)))
        self.assertEqual(max(all_points), 4)

    def test_points6(self):
        all_points = []
        counter = Counter()
        for _ in range(100):
            points = PoissonDisk2DPoints(3, 3, 2, k=300)
            all_points.append(len(list(points)))
        counter.update(all_points)
        self.assertGreaterEqual(counter[4], 5)

    def test_points7(self):
        for _ in range(100):
            points = PoissonDisk2DPoints(10, 100, 20)
            self.assertLessEqual(len(list(points)), 5)

    def test_points8(self):
        for _ in range(100):
            points = PoissonDisk2DPoints(100, 10, 20)
            self.assertLessEqual(len(list(points)), 5)

    def test_points9(self):
        for _ in range(10):
            points = PoissonDisk2DPoints(100, 100, 10)
            self.assertGreaterEqual(len(list(points)), 50)
