# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Base class for mask classes."""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import cv2

from dplabtools.slides import GenericSlide
from dplabtools.slides.utils.wsi import get_level_or_level, get_wsi_level_image
from dplabtools.slides.utils.image import get_transparency, get_color_hex, get_color_rgb


class BaseMask(ABC):
    """Base class for mask classes."""

    def __init__(self, *, wsi_file, level_or_minsize):
        """Create a BaseMask object.

        Parameters
        ----------
        wsi_file : str
            WSI file name or path.

        level_or_minsize : int
            WSI level or miminum size for mask dimensions.
        """
        self._wsi_file = wsi_file
        self._mask_array = None
        self._mask_array_transposed = None
        self._wsi_slide = GenericSlide(wsi_file=wsi_file)
        self._level = get_level_or_level(self._wsi_slide, level_or_minsize)
        self._process_wsi_file(wsi_file)

    @abstractmethod
    def _process_wsi_file(self, wsi_file):
        pass

    def save_array(self, array_file):
        """Save mask as a compressed NumPy array (NPZ file).

        Parameters
        ----------
        array_file : str
            NumPy array file name or path.
        """
        np.savez_compressed(array_file, data=self._mask_array)

    def save_png(self, png_file):
        """Save mask as a PNG image.

        Parameters
        ----------
        png_file : str
            PNG file name or path.
        """
        image = Image.fromarray(self._mask_array_transposed)
        image.save(png_file, "PNG", optimize=True)

    def save_overlay_png(self, png_file, mask_color="green", mask_alpha=0.5, outline_color="blue", outline_thickness=2):
        """Save mask as a PNG tissue overlay image.

        Parameters
        ----------
        png_file : str
            PNG file name or path.

        mask_color : str, default="green"
            Color of the mask layer overlay over the WSI.

        mask_alpha : float, default=0.5
            Level of transparency for the mask layer over the WSI.

        outline_color : str, default="blue"
            Color of the mask outline, use `None` to skip drawing outline.

        outline_thickness : int, default=2
            Outline thickness in pixels.
        """
        wsi_scan = get_wsi_level_image(self._wsi_slide, self._level)
        width, height = self._mask_array_transposed.shape
        overlay_shape = (width, height, 4)
        if mask_color:
            transparency = get_transparency(mask_alpha)
            color_hex = get_color_hex(mask_color)
            r, g, b = get_color_rgb(color_hex)
            overlay_array = np.zeros(overlay_shape, np.uint8)
            overlay_array[self._mask_array_transposed == 1] = [r, g, b, transparency]
            overlay_image = Image.fromarray(overlay_array)
            wsi_scan.paste(overlay_image, (0, 0), overlay_image)
        if outline_color:
            drawing_shape = (width, height)
            color_hex = get_color_hex(outline_color)
            r, g, b = get_color_rgb(color_hex)
            contours, hierarchy = cv2.findContours(
                self._mask_array_transposed.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            drawing_array = np.zeros(drawing_shape, np.uint8)
            cv2.drawContours(drawing_array, contours, -1, (255, 255, 255), outline_thickness)
            outline_array = np.zeros(overlay_shape, np.uint8)
            outline_array[drawing_array == 255] = [r, g, b, 255]  # fixed transparency
            outline_image = Image.fromarray(outline_array)
            wsi_scan.paste(outline_image, (0, 0), outline_image)
        wsi_scan.save(png_file, "PNG")

    @property
    def array(self):
        """Return mask data as a NumPy array."""
        return self._mask_array

    @property
    def level(self):
        """Return calculated or provided mask level."""
        return self._level
