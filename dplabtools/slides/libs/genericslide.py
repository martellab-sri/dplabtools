# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Dynamically created GenericSlide class to support different WSI readers."""

from dplabtools.config import slide_library
from .baseslide import BaseSlide
from .libcalls import LibCallsMixin

if slide_library == "openslide":
    from .libopenslide import SlideLib
elif slide_library == "tiffslide":
    from .libtiffslide import SlideLib
elif slide_library == "tiffopenslide":
    from .libtiffopenslide import SlideLib
elif slide_library == "pillow":
    from .libpillow import SlideLib
else:
    raise ValueError("Unknown slide reading library: %s" % slide_library)


class GenericSlide(LibCallsMixin, SlideLib, BaseSlide):
    """Class providing abstraction for existing Whole Slide Image libraries."""

    pass
