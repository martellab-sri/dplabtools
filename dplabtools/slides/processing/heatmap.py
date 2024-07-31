# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""WSIHeatmap class for visualizing inference results."""

from functools import cached_property

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm

from dplabtools.slides.utils.data import get_np_array
from dplabtools.slides.utils.wsi import (
    get_wsi_downsample_factor,
    compute_wsi_resolution_data,
    get_wsi_level_image,
    find_wsi_level,
)
from dplabtools.slides.utils.image import save_tif_image_with_resolution, get_transparency, get_color_hex, get_color_rgb
from dplabtools.slides import GenericSlide


class WSIHeatmap:
    """Class for visualizing inference results."""

    def __init__(self, *, heatmap_data, background_color="white", color_map="jet", alpha=1, vmin=None, vmax=None):
        """Class for visualizing inference results.

        Parameters
        ----------
        heatmap_data : NumPy array
            One class data from ``WSIInference``.

        background_color : str, default="white"
            Color representing heatmap image background. ``None`` will produce a transparent background.

        color_map : str, default="jet"
            Color map used for visualizing data.

        alpha : float, default=1
            Value representing the heatmap image transparency.

        vmin , vmax : float, optional
            Data range covering the heatmap image.
        """
        self._heatmap_data = get_np_array(heatmap_data)
        self._background_color = background_color
        self._cmap = color_map
        self._alpha = alpha
        self._vmin = vmin
        self._vmax = vmax
        self._probs_array = None
        self._probs_array_transposed = None
        self._process_heatmap_data()

    def _process_heatmap_data(self):
        if self._vmin is None:
            self._vmin = np.nanmin(self._heatmap_data)
        if self._vmax is None:
            self._vmax = np.nanmax(self._heatmap_data)
        if self._vmin == self._vmax:
            raise ValueError(
                "Either vmin equals vmax, or minimum and maximum values in the heatmap data are identical."
            )
        self._probs_array = self._heatmap_data
        self._probs_array_transposed = np.transpose(self._heatmap_data)

    @staticmethod
    def _get_pixel_data(data_array, background_color, cmap, alpha, vmin, vmax):
        cmapped_data = WSIHeatmap._get_colormapped_array(data_array, cmap, vmin, vmax)
        transparency = get_transparency(alpha)
        if background_color:
            color_hex = get_color_hex(background_color)
            WSIHeatmap._apply_background(cmapped_data, data_array, color_hex, transparency)
        if alpha < 1:
            WSIHeatmap._apply_transparency(cmapped_data, data_array, transparency)
        return cmapped_data

    @staticmethod
    def _get_colormapped_array(array_data, cmap, vmin, vmax):
        # this function must color map data in the same way as plt.imshow does
        colormap = cm.get_cmap(cmap)
        norm = plt.Normalize(vmin, vmax)
        colormapped_array = colormap(norm(array_data))
        colormapped_array = np.uint8(colormapped_array * 255)
        return colormapped_array

    @staticmethod
    def _apply_background(cmapped_data, array_data, background_color_hex, transparency):
        """Apply background to all regions which are NaN in source data."""
        r, g, b = get_color_rgb(background_color_hex)
        cmapped_data[:, :][np.where(np.isnan(array_data))] = [r, g, b, transparency]

    @staticmethod
    def _apply_transparency(cmapped_data, array_data, transparency):
        """Apply transparency to all regions which are not NaN in source data."""
        cmapped_data[:, :, 3][np.where(~np.isnan(array_data))] = transparency

    @staticmethod
    def _remove_transparency(pixel_data):
        """Convert array to RGB by removing transparency values with index [3] in RGBA."""
        pixel_data = pixel_data[:, :, 0:3]
        return pixel_data

    def save_png(self, png_file, draw_fn=None, draw_args=()):
        """Save the heatmap as a PNG image.

        Parameters
        ----------
        png_file : str
            File name or path for saving the PNG file.

        draw_fn : function, optional
            Custom function to draw on the heatmap image canvas.

        draw_args : tuple, optional
            Arguments for the custom draw function.
        """
        pixel_data = self._pixel_data
        heatmap = Image.fromarray(pixel_data)
        if draw_fn:
            draw_fn(ImageDraw.Draw(heatmap), *draw_args)
        heatmap.save(png_file, "PNG")

    def save_colorbar_png(self, png_file, interpolation="none", dpi=300, draw_fn=None, draw_args=()):
        """Save the heatmap as a PNG image with color bar.

        Parameters
        ----------
        png_file : str
            File name or path for saving the PNG file.

        interpolation: str, default="none"
            An interpolation method used internally by `matplotlib.pyplot.imshow`

        dpi : int, default=300
            Saved image resolution in dots per inch.

        draw_fn : function, optional
            Custom function to draw on the heatmap image canvas.

        draw_args : tuple, optional
            Arguments for the custom draw function.
        """
        plt.imshow(
            self._probs_array_transposed, cmap=self._cmap, interpolation=interpolation, vmin=self._vmin, vmax=self._vmax
        )
        plt.colorbar()
        plt.axis("off")
        if draw_fn:
            draw_fn(plt.gca(), *draw_args)
        plt.savefig(png_file, bbox_inches="tight", dpi=dpi)
        plt.clf()

    def save_tif(self, tif_file, wsi_file, downsample_factor=None, allow_compression=True, draw_fn=None, draw_args=()):
        """Save the heatmap as a TIF image with embedded resolution information.

        Parameters
        ----------
        tif_file : str
            File name or path for saving the TIF file.

        wsi_file: str
            Name or path to a WSI file that the heatmap is generated for.

        downsample_factor : float, optional
            Downsample factor used for resolution information. If not provided, the value will be determined based
            on the heatmap data size.

        allow_compression : bool, default: True
            If `True`, JPEG compression will be applied automatically when transparency is not used and when the
            image background is defined.

        draw_fn : function, optional
            Custom function to draw on the heatmap image canvas.

        draw_args : tuple, optional
            Arguments for the custom draw function.
        """
        use_compression = False
        wsi_slide = GenericSlide(wsi_file=wsi_file)
        pixel_data = self._pixel_data
        if draw_fn:
            draw_fn(pixel_data, *draw_args)
        if self._alpha == 1 and self._background_color:
            use_compression = True
            pixel_data = self._remove_transparency(pixel_data)
        if not downsample_factor:
            downsample_factor = get_wsi_downsample_factor(wsi_slide, self._probs_array.shape)
        resolution_data = compute_wsi_resolution_data(wsi_slide, downsample_factor)
        save_tif_image_with_resolution(
            pixel_data, tif_file, resolution_data, jpeg_compression=use_compression and allow_compression
        )

    def save_overlay_png(self, png_file, wsi_file, cutoff=None, draw_fn=None, draw_args=()):
        """Save the heatmap as a PNG tissue overlay image.

        Background color value will be ignored.

        Parameters
        ----------
        png_file : str
            File name or path for saving the PNG file.

        wsi_file: str
            Name or path to a WSI file that the heatmap is generated for.

        cutoff : float, optional
            Heatmap data values below the cutoff value will be ignored. This allows for the elimination
            of low probability values.

        draw_fn : function, optional
            Custom function to draw on the heatmap image canvas.

        draw_args : tuple, optional
            Arguments for the custom draw function.
        """
        wsi_slide = GenericSlide(wsi_file=wsi_file)
        level = find_wsi_level(wsi_slide, self._heatmap_data.shape)
        wsi_scan = get_wsi_level_image(wsi_slide, level)
        if cutoff:
            overlay_array = np.where(self._probs_array_transposed < cutoff, np.nan, self._probs_array_transposed)
        else:
            overlay_array = self._probs_array_transposed
        cmapped_data = self._get_colormapped_array(overlay_array, self._cmap, self._vmin, self._vmax)
        if self._alpha < 1:
            transparency = get_transparency(self._alpha)
            self._apply_transparency(cmapped_data, overlay_array, transparency)
        overlay = Image.fromarray(cmapped_data)
        if draw_fn:
            draw_fn(ImageDraw.Draw(overlay), *draw_args)
        wsi_scan.paste(overlay, (0, 0), overlay)
        wsi_scan.save(png_file, "PNG")

    @cached_property
    def _pixel_data(self):
        pixel_data = self._get_pixel_data(
            self._probs_array_transposed, self._background_color, self._cmap, self._alpha, self._vmin, self._vmax
        )
        return pixel_data
