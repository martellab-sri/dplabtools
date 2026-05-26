# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024-2026 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""WSIHeatmap class for visualizing inference results."""

from functools import cached_property

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    """Class for visualizing inference results as a heatmap image."""

    def __init__(
        self,
        *,
        heatmap_data,
        wsi_file,
        overlay=False,
        background_color="white",
        color_map="jet",
        alpha=1,
        vmin=None,
        vmax=None,
        cutoff=None
    ):
        """Class for visualizing inference results.

        Parameters
        ----------
        heatmap_data : str or object
            NumPy array (file or object), or ``classes_array`` property from ``WSIInference`` representing just
            one class.

        wsi_file: str or None
            WSI file (name or path) that the heatmap is generated for. Value ``None`` will disable the following
            class features: heatmap dimensions checks, overlay generation, saving as a TIF image (unless custom
            downsample_factor is provided).

        overlay : bool, default=False
            When ``True``, the heatmap will overlay the tissue image.

        background_color : str, default="white"
            Color representing heatmap image background. ``None`` will produce a transparent background.

        color_map : str, default="jet"
            Color map used for visualizing data. Colormap names are defined in:
            https://matplotlib.org/stable/users/explain/colors/colormaps.html

        alpha : float, default=1
            Value representing the heatmap image transparency (for visualized data only, the heatmap background color is
            not affected).

        vmin: float, optional
            Lower bound of the color map range for the heatmap.

        vmax : float, optional
            Upper bound of the color map range for the heatmap.

        cutoff : float, optional
            Heatmap data values below the ``cutoff`` value will be ignored. This allows for the elimination
            of low probability values.
        """
        self._heatmap_data = get_np_array(heatmap_data)
        self._wsi_file = wsi_file
        self._overlay = overlay
        self._background_color = background_color
        self._cmap = color_map
        self._alpha = alpha
        self._vmin = vmin
        self._vmax = vmax
        self._cutoff = cutoff
        self._wsi_slide = None
        self._wsi_level = None
        self._wsi_scan = None
        self._heatmap_data_transposed = None
        self._background_color_plot = background_color

        # evaluate WSI file before data checks
        if self._wsi_file is None:
            self._overlay = False
        else:
            self._wsi_slide = GenericSlide(wsi_file=self._wsi_file)
            wsi_level = find_wsi_level(self._wsi_slide, self._heatmap_data.shape)

        # check and process data
        self._process_heatmap_data()

        # retrieve image scan after data checks
        if self._overlay:
            self._background_color = None
            self._wsi_scan = get_wsi_level_image(self._wsi_slide, wsi_level)

    def _process_heatmap_data(self):
        vmin = np.nanmin(self._heatmap_data)
        vmax = np.nanmax(self._heatmap_data)

        if self._vmin is None:
            self._vmin = vmin

        if self._vmax is None:
            self._vmax = vmax

        if self._vmin == self._vmax:
            raise ValueError(
                "Either vmin equals vmax, or the minimum and maximum values in the heatmap data are identical."
            )

        if self._vmin > vmin or self._vmax < vmax:
            raise ValueError(
                "vmin must be less than or equal to the lowest heatmap data value (%f), and vmax must be greater than "
                "or equal to the highest heatmap data value (%f)." % (vmin, vmax)
            )

        self._heatmap_data_transposed = np.transpose(self._heatmap_data)

        if self._cutoff:
            self._heatmap_data_transposed = np.where(
                self._heatmap_data_transposed < self._cutoff, np.nan, self._heatmap_data_transposed
            )

    @staticmethod
    def _get_pixel_data(heatmap_data, background_color, cmap, alpha, vmin, vmax):
        cmapped_data = WSIHeatmap._get_colormapped_data(heatmap_data, cmap, vmin, vmax)
        transparency_present = False

        if alpha < 1:
            transparency = get_transparency(alpha)
            transparency_present = True
            WSIHeatmap._apply_transparency(cmapped_data, heatmap_data, transparency)

        if background_color:
            color_hex = get_color_hex(background_color)
            WSIHeatmap._apply_background(cmapped_data, heatmap_data, color_hex, transparency_present)

        return cmapped_data

    @staticmethod
    def _get_colormapped_data(heatmap_data, cmap, vmin, vmax):
        # this function must color map data in the same way as plt.imshow does
        colormap = mpl.colormaps.get_cmap(cmap)
        norm = plt.Normalize(vmin, vmax)
        colormapped_array = colormap(norm(heatmap_data))
        colormapped_array = np.uint8(colormapped_array * 255)
        return colormapped_array

    @staticmethod
    def _apply_transparency(cmapped_data, heatmap_data, transparency):
        """Apply transparency to all regions which are not NaN in source data."""
        cmapped_data[:, :, 3][np.where(~np.isnan(heatmap_data))] = transparency

    @staticmethod
    def _apply_background(cmapped_data, heatmap_data, background_color_hex, transparency_present):
        """Apply full background to all regions which are NaN in source data.

        If transparency is present and the background_color is present, then they both must blended.
        However, transparency must be already applied to cmapped_data.
        """
        r, g, b = get_color_rgb(background_color_hex)
        # - Blend background for non NaN values, if transparency is present
        # - Blending formula: final_color = (1 - alpha) * background_color + alpha * foreground_color
        if transparency_present:
            blend_mask = ~np.isnan(heatmap_data)
            cmapped_data_color = cmapped_data[:, :, :3]
            cmaped_data_alpha = cmapped_data[:, :, 3] / 255.0
            # adding np.round below improves accuracy, as reported in issue #141
            cmapped_data[blend_mask, :3] = np.round(
                (1 - cmaped_data_alpha[blend_mask])[:, np.newaxis] * [r, g, b]
                + cmaped_data_alpha[blend_mask][:, np.newaxis] * cmapped_data_color[blend_mask]
            ).astype(int)
            cmapped_data[blend_mask, 3] = 255

        # Apply full background with no transparency for NaN values
        cmapped_data[:, :][np.where(np.isnan(heatmap_data))] = [r, g, b, 255]

    @staticmethod
    def _remove_transparency(pixel_data):
        """Convert array to RGB by removing transparency values with index [3] in RGBA."""
        return pixel_data[:, :, 0:3]

    def save_image(self, image_file, draw_fn=None, draw_args=(), **save_kwargs):
        """Save the heatmap as an image.

        Parameters
        ----------
        image_file : str
            File name or path for saving the image file. File extension will determine the image format.

        draw_fn : function, optional
            Custom function to draw on the heatmap image canvas.

        draw_args : tuple, optional
            Arguments for the custom draw function.

        save_kwargs : -
            Additional keyword parameters passed to the internal save image function.
            Reference: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
        """
        image_data = self._pixel_data

        if self._alpha == 1 and self._background_color:
            image_data = self._remove_transparency(image_data)

        heatmap_image = Image.fromarray(image_data)

        if self._overlay:
            image = self._wsi_scan.copy()
            image.paste(heatmap_image, (0, 0), heatmap_image)
        else:
            image = heatmap_image
        if draw_fn:
            draw_fn(ImageDraw.Draw(image), *draw_args)

        image.save(image_file, **save_kwargs)

    def save_colorbar_image(
        self, image_file, interpolation="none", dpi=300, draw_fn=None, draw_args=(), **savefig_kwargs
    ):
        """Save the heatmap as an image with color bar.

        Parameters
        ----------
        image_file : str
            File name or path for saving the image file. File extension will determine the image format.

        interpolation: str, default="none"
            An interpolation method used internally by `matplotlib.pyplot.imshow`

        dpi : int, default=300
            Saved image resolution in dots per inch.

        draw_fn : function, optional
            Custom function to draw on the heatmap image canvas.

        draw_args : tuple, optional
            Arguments for the custom draw function.

        savefig_kwargs : -
            Additional keyword parameters passed to the internal save image function.
            Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
        """
        if self._background_color_plot is None:
            background_color = "none"
        else:
            background_color = self._background_color_plot

        fig = plt.figure()
        fig.patch.set_facecolor(background_color)

        if self._overlay:
            plt.imshow(self._wsi_scan)

        overlay = plt.imshow(
            self._heatmap_data_transposed,
            cmap=self._cmap,
            interpolation=interpolation,
            vmin=self._vmin,
            vmax=self._vmax,
            alpha=self._alpha,
        )
        plt.colorbar(overlay)
        plt.axis("off")

        if draw_fn:
            draw_fn(plt.gca(), *draw_args)

        plt.savefig(image_file, bbox_inches="tight", dpi=dpi, **savefig_kwargs)
        plt.close()
        plt.clf()

    def save_tif(self, tif_file, downsample_factor=None, allow_compression=True, draw_fn=None, draw_args=()):
        """Save the heatmap as a TIF image with embedded resolution information.

        Parameters
        ----------
        tif_file : str
            File name or path for saving the TIF file.

        downsample_factor : float, optional
            Downsample factor used for resolution information. If not provided, the value will be determined based
            on the heatmap data size.

        allow_compression : bool, default: True
            If ``True``, JPEG compression will be applied automatically when transparency is not used and when the
            image background is defined.

        draw_fn : function, optional
            Custom function to draw on the heatmap image canvas.

        draw_args : tuple, optional
            Arguments for the custom draw function.
        """
        if not self._wsi_file:
            raise ValueError("wsi_file must be specified for saving TIF images.")

        use_compression = False
        image_data = self._pixel_data

        if self._alpha == 1 and self._background_color:
            use_compression = True
            image_data = self._remove_transparency(image_data)

        if not downsample_factor:
            downsample_factor = get_wsi_downsample_factor(self._wsi_slide, self._heatmap_data.shape)

        resolution_data = compute_wsi_resolution_data(self._wsi_slide, downsample_factor)

        if self._overlay:
            heatmap_image = Image.fromarray(image_data)
            scan_image = self._wsi_scan.copy()
            scan_image.paste(heatmap_image, (0, 0), heatmap_image)
            image_data = np.asarray(scan_image)

        if draw_fn:
            # array copying is required here, as array created from image is always read only
            image_data = np.copy(image_data)
            draw_fn(image_data, *draw_args)

        save_tif_image_with_resolution(
            image_data, tif_file, resolution_data, jpeg_compression=use_compression and allow_compression
        )

    @cached_property
    def _pixel_data(self):
        """Retrieve heatmap data converted into RGB or RGBA pixels."""
        pixel_data = self._get_pixel_data(
            self._heatmap_data_transposed, self._background_color, self._cmap, self._alpha, self._vmin, self._vmax
        )
        return pixel_data
