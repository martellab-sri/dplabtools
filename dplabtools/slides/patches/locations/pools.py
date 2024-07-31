# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Pool classes for patch location computing.

Classes included here provide process level parallelization for patch location computing.
"""

import os
from multiprocessing import Pool as ProcessPool
from collections import Counter
from abc import ABC, abstractmethod

from dplabtools.slides.patches import (
    PatchesMock,
    WholeImageRandomPatches,
    WholeImagePoissonDiskPatches,
    WholeImageGridPatches,
    WholeImageCustomPatches,
    WholeImageInvertedRandomPatches,
    WholeImageInvertedPoissonDiskPatches,
    WholeImageInvertedGridPatches,
    WholeImageInvertedCustomPatches,
    PolygonRegionRandomPatches,
    PolygonRegionPoissonDiskPatches,
    PolygonRegionGridPatches,
    PolygonRegionCustomPatches,
)
from dplabtools.slides.utils.wsi import get_wsi_id, get_wsi_name
from dplabtools.slides.patches.locations.utils import expand_list_param


class BasePatchesPool(ABC):
    """Abstract base class for all patches pool classes."""

    _expandable_list_params = ["included_labels", "excluded_labels"]
    _excluded_patches_params = ["wsi_file", "mask_data", "polygon_data", "points"]

    def __init__(
        self,
        *,
        wsi_file_list,
        mask_data_list,
        patches_args,
        proc_num_workers,
        mp_chunksize=1,
        save_preview_image_args={},
    ):
        """Init method for BasePatchesPool and derived classes.

        Parameters
        ----------
        wsi_file_list : list of str
            List of WSIs (files names or paths).

        mask_data_list : list of str or objects
            List of masks for WSIs. Each mask is: file name or path, NumPy array object, or Pillow image object.

        patches_args : dict
            Dictionary object with arguments for the patches class.

        proc_num_workers : int
            Number of processes in the pool. This value corresponds directly to the number of WSIs to be processed
            simultaneously.

        mp_chunksize : int, default=1
            Data chunk size used in parallel processing.

        save_preview_image_args : dict, optional
            Dictionary object with arguments for ``save_preview_image``.
        """
        self._wsi_file_list = wsi_file_list
        self._mask_data_list = mask_data_list
        self._patches_args = patches_args
        self._num_workers = proc_num_workers
        self._mp_chunksize = mp_chunksize
        self._preview_args = save_preview_image_args
        self._wsi_file_count = len(self._wsi_file_list)
        self._patches_pool = []
        self._pids = []
        self._index = 0
        self._check_patches_params(self._patches_args)

    def __iter__(self):
        """Is required by iterator."""
        return self

    def __next__(self):
        """Is required by iterator."""
        if self._index < len(self._patches_pool):
            patches = self._patches_pool[self._index]
            self._index += 1
        else:
            raise StopIteration
        return patches

    @staticmethod
    def _check_patches_params(patches_args):
        for arg in patches_args:
            if arg in BasePatchesPool._excluded_patches_params:
                raise ValueError("Incorrect patches_args key: %s" % arg)

    @staticmethod
    @abstractmethod
    def _update_process_args(args, process_data):
        pass

    @staticmethod
    def _update_expandable_args(args, process_data_index, data_count):
        for arg_key, arg_value in args.items():
            if arg_key in BasePatchesPool._expandable_list_params:
                expanded_arg_value = expand_list_param(arg_value, arg_key, data_count)
                args[arg_key] = expanded_arg_value[process_data_index]
            elif isinstance(arg_value, list):
                if len(arg_value) != data_count:
                    raise ValueError(
                        "Incorrect list size for parameter %s: received: %d, expected: %d"
                        % (arg_key, len(arg_value), data_count)
                    )
                args[arg_key] = arg_value[process_data_index]

    @abstractmethod
    def _get_process_pool_data(self):
        pass

    def _run_pool(self):
        process_pool_data = self._get_process_pool_data()
        with ProcessPool(self._num_workers) as ppool:
            for patches_bits in ppool.imap_unordered(
                self._run_single_procces,
                process_pool_data,
                chunksize=self._mp_chunksize,
            ):
                patch_data, patch_labels, wsi_file, level_or_mpp, pid, process_args = patches_bits
                patches_mock = PatchesMock(
                    patch_data=patch_data,
                    patch_labels=patch_labels,
                    wsi_file=wsi_file,
                    level_or_mpp=level_or_mpp,
                    check_location=False,
                    class_name=self._patches_class.__name__,
                    class_params=process_args,
                )
                self._patches_pool.append(patches_mock)
                self._pids.append(pid)

    def _run_single_procces(self, process_data):
        # this function must return a picklable object (no references)
        pid = os.getpid()
        process_data_index, process_data_content = process_data
        process_args = self._patches_args.copy()
        self._update_expandable_args(process_args, process_data_index, self._wsi_file_count)
        self._update_process_args(process_args, process_data_content)
        process_patches = self._get_patches(process_args)  # creates a patches object
        if self._preview_args:
            wsi_file = process_data_content[0]
            preview_args = self._preview_args.copy()
            self._update_preview_args(preview_args, wsi_file)
            process_patches.save_preview_image(**preview_args)
        return (
            process_patches.patch_data,
            process_patches.patch_labels,
            process_patches.wsi_file,
            process_patches.level_or_mpp,
            pid,
            process_patches.param_info,
        )

    def _get_patches(self, args):
        patches = self._patches_class(**args)
        return patches

    @staticmethod
    def _update_preview_args(args, wsi_file):
        image_file = args["image_file"]
        file_name = get_wsi_name(image_file)
        wsi_id = get_wsi_id(wsi_file)
        dir_name = os.path.dirname(image_file)
        image_file = os.path.join(dir_name, wsi_id + file_name)
        args["image_file"] = image_file

    @property
    def patch_count(self):
        """Return the combined patch count for all processed WSIs."""
        counter = 0
        for patches in self._patches_pool:
            counter += patches.patch_count
        return counter

    @property
    def patch_info(self):
        """Return the combined patch information for all processed WSIs."""
        counter = Counter()
        for patches in self._patches_pool:
            counter.update(patches.patch_info)
        return dict(sorted(counter.items()))

    @property
    def patch_labels(self):
        """Return the distinct polygon labels for all processed WSIs."""
        labels = []
        for patches in self._patches_pool:
            labels.extend(patches.patch_labels)
        return sorted(list(set(labels)))

    @property
    def patch_details(self):
        """Return the combined patch details for all processed WSIs."""
        info_details = []
        for patches in self._patches_pool:
            info_details.append((patches.wsi_file, patches.patch_info))
        return info_details

    @property
    def pids(self):
        """Return the IDs of the executed processes."""
        return self._pids


class WholeImagePatchesPoolBase(BasePatchesPool):
    """Base class for patches pool classes for whole image processing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._run_pool()

    @staticmethod
    def _update_process_args(args, process_data):
        wsi_file, mask_data = process_data
        args.update({"wsi_file": wsi_file, "mask_data": mask_data})

    def _get_process_pool_data(self):
        return list(enumerate(zip(self._wsi_file_list, self._mask_data_list)))


class WholeImageInvertedPatchesPoolBase(BasePatchesPool):
    """Base class for patches pool classes for inverted whole image processing."""

    def __init__(self, *, polygon_data_list, **kwargs):
        """Init method for WholeImageInvertedPatchesPoolBase and derived classes.

        Parameters
        ----------
        polygon_data_list : list of objects
        Polygon data representing excluded regions for different WSIs. Each data element
        is a list of `AnnotationPolygon` objects or a JSON file/string.
        """
        self._polygon_data_list = polygon_data_list
        super().__init__(**kwargs)
        self._run_pool()

    @staticmethod
    def _update_process_args(args, process_data):
        wsi_file, mask_data, polygon_data = process_data
        args.update({"wsi_file": wsi_file, "mask_data": mask_data, "polygon_data": polygon_data})

    def _get_process_pool_data(self):
        return list(enumerate(zip(self._wsi_file_list, self._mask_data_list, self._polygon_data_list)))


class PolygonRegionPatchesPoolBase(BasePatchesPool):
    """Base class for patches pool classes for polygon region processing."""

    def __init__(self, *, polygon_data_list, **kwargs):
        """Init method for PolygonRegionPatchesPoolBase and derived classes.

        Parameters
        ----------
        polygon_data_list : list of objects
        Polygon data representing regions of interest for different WSIs. Each data element
        is a list of `AnnotationPolygon` objects or a JSON file/string.
        """
        self._polygon_data_list = polygon_data_list
        super().__init__(**kwargs)
        self._run_pool()

    @staticmethod
    def _update_process_args(args, process_data):
        wsi_file, mask_data, polygon_data = process_data
        args.update({"wsi_file": wsi_file, "mask_data": mask_data, "polygon_data": polygon_data})

    def _get_process_pool_data(self):
        return list(enumerate(zip(self._wsi_file_list, self._mask_data_list, self._polygon_data_list)))


class CustomPatchesMixin:
    """Mixin that adds an extra parameter for proper custom patches handling."""

    def __init__(self, *, points_list, **kwargs):
        """Init method for CustomPatchesMixin.

        Parameters
        ----------
        points_list : list of lists of tuples with (x, y) `int` values
            Top-left coordinates of custom patches for different WSIs.
        """
        self._points_list = points_list
        super().__init__(**kwargs)

    def _update_process_args(self, args, process_data):
        # this method overrides the parent static method, but also calls it internally
        *non_points_data, points = process_data
        super()._update_process_args(args, non_points_data)
        args.update({"points": points})

    def _get_process_pool_data(self):
        # append original data with custom points
        org_data = super()._get_process_pool_data()
        appended_data = []
        for elem in org_data:
            elem_index = elem[0]
            elem_tuple = elem[1]
            elem_list = list(elem_tuple)
            elem_list.append(self._points_list[elem_index])
            appended_data.append((elem_index, tuple(elem_list)))
        return appended_data


# Final classes


class WholeImageRandomPatchesPool(WholeImagePatchesPoolBase):
    """Patches pool implementation for `WholeImageRandomPatches`."""

    _patches_class = WholeImageRandomPatches


class WholeImagePoissonDiskPatchesPool(WholeImagePatchesPoolBase):
    """Patches pool implementation for `WholeImagePoissonDiskPatches`."""

    _patches_class = WholeImagePoissonDiskPatches


class WholeImageGridPatchesPool(WholeImagePatchesPoolBase):
    """Patches pool implementation for `WholeImageGridPatches`."""

    _patches_class = WholeImageGridPatches


class WholeImageInvertedRandomPatchesPool(WholeImageInvertedPatchesPoolBase):
    """Patches pool implementation for `WholeImageInvertedRandomPatches`."""

    _patches_class = WholeImageInvertedRandomPatches


class WholeImageInvertedPoissonDiskPatchesPool(WholeImageInvertedPatchesPoolBase):
    """Patches pool implementation for `WholeImageInvertedPoissonDiskPatches`."""

    _patches_class = WholeImageInvertedPoissonDiskPatches


class WholeImageInvertedGridPatchesPool(WholeImageInvertedPatchesPoolBase):
    """Patches pool implementation for `WholeImageInvertedGridPatches`."""

    _patches_class = WholeImageInvertedGridPatches


class PolygonRegionRandomPatchesPool(PolygonRegionPatchesPoolBase):
    """Patches pool implementation for `PolygonRegionRandomPatches`."""

    _patches_class = PolygonRegionRandomPatches


class PolygonRegionPoissonDiskPatchesPool(PolygonRegionPatchesPoolBase):
    """Patches pool implementation for `PolygonRegionPoissonDiskPatches`."""

    _patches_class = PolygonRegionPoissonDiskPatches


class PolygonRegionGridPatchesPool(PolygonRegionPatchesPoolBase):
    """Patches pool implementation for `PolygonRegionGridPatches`."""

    _patches_class = PolygonRegionGridPatches


class WholeImageCustomPatchesPool(CustomPatchesMixin, WholeImagePatchesPoolBase):
    """Patches pool implementation for `WholeImageCustomPatches`."""

    _patches_class = WholeImageCustomPatches


class WholeImageInvertedCustomPatchesPool(CustomPatchesMixin, WholeImageInvertedPatchesPoolBase):
    """Patches pool implementation for `WholeImageInvertedCustomPatches`."""

    _patches_class = WholeImageInvertedCustomPatches


class PolygonRegionCustomPatchesPool(CustomPatchesMixin, PolygonRegionPatchesPoolBase):
    """Patches pool implementation for `PolygonRegionCustomPatches`."""

    _patches_class = PolygonRegionCustomPatches
