# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Patch locations related utilities."""


def expand_scalar_param(param, param_name, count):
    """Expand provided scalar parameter into list."""
    expanded_param = []
    if isinstance(param, list):
        if len(param) != count:
            raise ValueError(
                "Incorrect list size for expanding parameter %s: received %d, expected %d"
                % (param_name, len(param), count)
            )
        expanded_param = param
    elif isinstance(param, (int, float)):
        expanded_param = [param] * count
    else:
        raise ValueError(
            "Incorrect data type for expanding parameter %s: received %s, expected int, float or list"
            % (param_name, type(param))
        )
    return expanded_param


def expand_list_param(list_param, param_name, count):
    """Expand provided list parameter into nested list."""
    expanded_list_param = []
    try:
        is_list_nested = any(isinstance(param, list) for param in list_param)
    except TypeError:
        is_list_nested = False

    if is_list_nested:
        if len(list_param) != count:
            raise ValueError(
                "Incorrect list size for expanding parameter %s: received %d, expected %d"
                % (param_name, len(list_param), count)
            )
        expanded_list_param = list_param
    elif isinstance(list_param, list):
        expanded_list_param = [list_param] * count
    else:
        raise ValueError(
            "Incorrect data type for expanding parameter %s: received %s, expected list"
            % (param_name, type(list_param))
        )
    return expanded_list_param
