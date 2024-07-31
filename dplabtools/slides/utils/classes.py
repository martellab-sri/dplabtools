# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Utility classes used in different slides.* namespaces."""

import PIL
import numpy as np

from dplabtools.slides.utils.data import get_np_array, get_pil_image, get_json_obj


class AnnotationPolygon:
    """Data container for polygon based annotations."""

    def __init__(self, *, points, label, holes=[]):
        """Init AnnotationPolygon class."""
        self._points = points
        self._label = label
        self._holes = holes

    def __eq__(self, cmp_obj):
        """Compare two AnnotationPolygon objects.

        Points and holes can be represented as (x, y) or [x, y], thus only values of points and holes are compared.
        """
        result = False
        if isinstance(cmp_obj, AnnotationPolygon):
            if (
                len(self.points) == len(cmp_obj.points)
                and self.label == cmp_obj.label
                and len(self.holes) == len(cmp_obj.holes)
            ):
                for points_data1, points_data2 in zip(self.points, cmp_obj.points):
                    if points_data1[0] != points_data2[0] or points_data1[1] != points_data2[1]:
                        result = False
                        break
                    result = True
                if result and self.holes:
                    try:
                        for holes_data1, holes_data2 in zip(self.holes, cmp_obj.holes):
                            for holes_points1, holes_points2 in zip(holes_data1, holes_data2):
                                if holes_points1[0] != holes_points2[0] or holes_points1[1] != holes_points2[1]:
                                    raise StopIteration
                    except StopIteration:
                        result = False

        return result

    def __str__(self):
        """Add string representation to class."""
        return self._get_object_text()

    def __repr__(self):
        """Add string representation to class."""
        return self._get_object_text()

    def _get_object_text(self):
        return str([self._points, self._label, self._holes])

    @property
    def data_dict(self):
        """Return dictionary with polygon data.

        Used for serialization of AnnotationPolygon objects to JSON.
        """
        return {"points": self._points, "label": self._label, "holes": self._holes}

    @property
    def points(self):
        """Return list of points for polygon."""
        return self._points

    @property
    def label(self):
        """Return polygon label."""
        return self._label

    @property
    def holes(self):
        """Return list of points for polygon holes."""
        return self._holes


class MaskData:
    """Wrapper class for different types of mask data: 2D images or numpy arrays."""

    npz_data_key = "data"

    def __init__(self, *, mask_data):
        """Init MaskData class."""
        self._mask_data = mask_data
        # try reading image
        try:
            mask_data_image = get_pil_image(self._mask_data)
        except (TypeError, PIL.UnidentifiedImageError):
            pass
        else:
            mask_data_array = np.asarray(mask_data_image, dtype=bool).transpose()
            self._mask_data_array = mask_data_array
            self._check_mask_array()
            return None
        # try reading array
        try:
            mask_data_array = get_np_array(self._mask_data, data_key=self.npz_data_key)
        except (TypeError, ValueError):
            pass
        else:
            self._mask_data_array = mask_data_array
            self._check_mask_array()
            return None
        # wrong data type
        raise TypeError("Mask data must be image or NumPy array, either file or memory object")

    def _check_mask_array(self):
        mask_data_text = self._get_object_text()
        if len(self._mask_data_array.shape) > 2:
            raise ValueError(
                "Incorrect array shape [%s] in mask data '%s'" % ((self._mask_data_array.shape, mask_data_text))
            )

    def __str__(self):
        """Add string representation to class."""
        return self._get_object_text()

    def __repr__(self):
        """Add string representation to class."""
        return self._get_object_text()

    def _get_object_text(self):
        if isinstance(self._mask_data, str):
            text = self._mask_data
        else:
            text = "NumPy array [shape=%s, size=%s, type=%s]" % (
                self._mask_data_array.shape,
                self._mask_data_array.size,
                self._mask_data_array.dtype,
            )
        return text

    @property
    def mask_array(self):
        """Return internal numpy array."""
        return self._mask_data_array


class PolygonData:
    """Wrapper class for different types of polygon data: list of AnnotationPolygon objects or JSON string/file."""

    def __init__(self, *, polygon_data):
        """Init PolygonData class."""
        self._polygon_data = polygon_data
        self._polygons = None
        err_msg = "Polygon data must be a list of AnnotationPolygon objects or a JSON object (document or file)"

        if isinstance(polygon_data, list):
            try:
                first = polygon_data[0]
            except IndexError:
                pass
            else:
                if isinstance(first, AnnotationPolygon):
                    self._polygons = polygon_data
                    return None
        elif isinstance(polygon_data, str):
            json_obj = get_json_obj(polygon_data)
            try:
                polygons = []
                for poly_data in json_obj:
                    points = poly_data["points"]
                    label = poly_data["label"]
                    holes = poly_data["holes"]
                    polygon = AnnotationPolygon(points=points, label=label, holes=holes)
                    polygons.append(polygon)
                self._polygons = polygons
                return None
            except KeyError:
                pass
        else:
            raise TypeError(err_msg)
        raise ValueError(err_msg)

    def __str__(self):
        """Add string representation to class."""
        return self._get_object_text()

    def __repr__(self):
        """Add string representation to class."""
        return self._get_object_text()

    def _get_object_text(self):
        if isinstance(self._polygon_data, str):
            text = self._polygon_data
        else:
            text = str(self._polygons)
        return text

    @property
    def polygons(self):
        """Return list of AnnotationPolygon objects."""
        return self._polygons
