# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Data related utilities for slides."""

import json

import numpy as np
import PIL.Image


def get_np_array(array_data, data_key="data"):
    """Get NumPy array either from file or variable."""
    np_array = None
    if isinstance(array_data, str):
        np_array = np.load(array_data)
        if isinstance(np_array, np.lib.npyio.NpzFile):
            np_array = np_array[data_key]
    elif isinstance(array_data, np.ndarray):
        np_array = array_data
    else:
        raise TypeError("Incorrect input type: %s" % type(array_data))
    return np_array


def get_pil_image(image_data):
    """Get PIL image either from file or variable."""
    pil_image = None
    if isinstance(image_data, str):
        pil_image = PIL.Image.open(image_data)
        pil_image.load()
    elif isinstance(image_data, PIL.Image.Image):
        pil_image = image_data
    else:
        raise TypeError("Incorrect input type: %s" % type(image_data))
    return pil_image


def get_json_obj(json_data):
    """Get python object representing a serialized JSON document or JSON file.

    Notes
    -----
    - json_data must be already validated to be a string (no checks here)
    - returned value must be always a list of dict objects, each dict representing one AnnotationPolygon
    """
    json_obj = None
    if json_data:
        # check if string is JSON doc
        try:
            json_obj = json.loads(json_data)
        except ValueError:
            # check if string is JSON file name/path
            with open(json_data, "r") as json_file:
                json_obj = json.load(json_file)

    if json_obj is None or not isinstance(json_obj, list):
        raise ValueError("Incorrect input value: %s" % json_data)
    return json_obj
