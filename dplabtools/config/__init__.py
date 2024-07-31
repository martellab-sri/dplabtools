# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Import point for package configuration functions."""

from dplabtools.config.slides import slide_library
from dplabtools.config.common import print_messages

__all__ = ["slide_library", "print_messages"]
