# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Common functions used in other modules."""

import string
import random

from dplabtools.config import print_messages


def print_out(text):
    """Print package internal messages based on configuration settings.

    Behaviour controlled by print_messages.
    """
    if print_messages is True:
        print(text)
    elif print_messages is False:
        pass
    elif isinstance(print_messages, str):
        screen_log = print_messages
        with open(screen_log, "a") as screen_log_file:
            screen_log_file.write(text + "\n")
    else:
        raise ValueError("Invalid configuration value")


def get_random_string(length):
    """Return string with random characters."""
    charset = string.ascii_lowercase + string.digits
    chars = random.choices(charset, k=length)
    random_string = "".join(chars)
    return random_string


def roundfl(number, decimal_places=10):
    """Return rounded float number with fixed decimal places."""
    return round(number, decimal_places)
