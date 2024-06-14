# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""All importable classes from slides.processing namespace."""

from dplabtools.slides.processing.mask.polygon import WSIPolygonMask
from dplabtools.slides.processing.mask.tissue import WSITissueMask
from dplabtools.slides.processing.heatmap import WSIHeatmap

# do not blindly import classes which depend on torch
try:
    import torch
except ImportError:
    pass
else:
    from dplabtools.slides.processing.dataset.dataset import WSIDataset, WSIMultiResDataset
    from dplabtools.slides.processing.inference import WSIInference

__all__ = ["WSIPolygonMask", "WSITissueMask", "WSIHeatmap", "WSIDataset", "WSIMultiResDataset", "WSIInference"]
