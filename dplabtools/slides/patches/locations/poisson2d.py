# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""PoissonDisk points generator in 2D.

Based on:

"Fast Poisson Disk Sampling in Arbitrary Dimensions" by Robert Bridson

https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

Sample code:

import matplotlib.pyplot as plt
from poisson2d import PoissonDisk2DPoints

width, height = 100, 70
p = PoissonDisk2DPoints(width, height, 5)
points = list(p)
points_x = [pt[0] for pt in points]
points_y = [pt[1] for pt in points]
plt.scatter(points_x, points_y, c='b')
plt.xlim(0, width)
plt.ylim(0, height)
plt.axis('on')
plt.show()

"""

import math
import random


class PoissonDisk2DPoints:
    """PoissonDisk 2D points generator class."""

    def __init__(self, width, height, radius, k=30):
        """Init method for class.

        Params:
        - width: width of 2D area
        - height: height of 2D area
        - radius: minimal distance between points
        - k: number of attempts in point selection
        """
        self._size = (width, height)
        self._radius = radius
        self._k_limit = k
        self._active_list = []
        self._selected_points = []
        self._step = 0
        self._cell_size = radius / math.sqrt(2)
        self._double_radius = radius * 2
        self._double_pi = math.pi * 2
        self._background_grid = self._init_grid()
        self._nearest_grid_locations = [
            (-1, 2),
            (0, 2),
            (1, 2),
            (2, 2),
            (-2, 1),
            (-1, 1),
            (0, 1),
            (1, 1),
            (2, 1),
            (-2, 0),
            (-1, 0),
            (0, 0),
            (1, 0),
            (2, 0),
            (-2, -1),
            (-1, -1),
            (0, -1),
            (1, -1),
            (2, -1),
            (-2, -2),
            (-1, -2),
            (0, -2),
            (1, -2),
        ]

    def _init_grid(self):
        size_x = int(self._size[0] / self._cell_size) + 1
        size_y = int(self._size[1] / self._cell_size) + 1
        grid = [[None for y in range(size_y)] for x in range(size_x)]
        return grid

    def __iter__(self):
        return self

    def __next__(self):
        point = self._get_next_point()
        if point:
            point_x, point_y = point
            self._background_grid[int(point_x / self._cell_size)][int(point_y / self._cell_size)] = self._step
            self._selected_points.append((point_x, point_y))
            self._active_list.append(self._step)
            self._step += 1
        else:
            raise StopIteration
        return point

    def _get_next_point(self):
        next_point = None
        while self._active_list or not self._selected_points:
            if not self._selected_points:
                # first point
                next_point = (random.uniform(0, self._size[0]), random.uniform(0, self._size[1]))
            else:
                # subsequent points
                random_index = random.choice(self._active_list)
                random_point = self._selected_points[random_index]
                for _ in range(self._k_limit):
                    candidate_point = self._get_candidate_point(random_point)
                    # check if candidate point meets all criteria
                    if not (
                        not (0 < candidate_point[0] < self._size[0])
                        or not (0 < candidate_point[1] < self._size[1])
                        or self._is_candidate_point_within_radius(candidate_point)
                    ):
                        next_point = candidate_point
                        break

            if next_point:
                break
            self._active_list.remove(random_index)

        return next_point

    def _get_candidate_point(self, random_point):
        distance = random.uniform(self._radius, self._double_radius)
        angle = random.uniform(0, self._double_pi)
        candidate_point_x = random_point[0] + distance * math.cos(angle)
        candidate_point_y = random_point[1] + distance * math.sin(angle)
        return (candidate_point_x, candidate_point_y)

    def _is_candidate_point_within_radius(self, candidate_point):
        is_within = False
        candidate_point_grid_x, candidate_point_grid_y = (
            int(candidate_point[0] / self._cell_size),
            int(candidate_point[1] / self._cell_size),
        )
        for (location_x, location_y) in self._nearest_grid_locations:
            point_index = self._get_point_index(
                candidate_point_grid_x + location_x, candidate_point_grid_y + location_y, self._background_grid
            )
            if point_index is None:
                continue
            selected_point = self._selected_points[point_index]
            distance = math.sqrt(
                (candidate_point[0] - selected_point[0]) ** 2 + (candidate_point[1] - selected_point[1]) ** 2
            )
            if distance < self._radius:
                is_within = True
                break
        return is_within

    @staticmethod
    def _get_point_index(x, y, grid):
        point_index = None
        if x >= 0 and y >= 0:
            try:
                point_index = grid[x][y]
            except IndexError:
                pass
        return point_index
