# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Patches mock class for precomputed patch locations.

PatchesMock is a lightweight class used in process based paralellization when objects passed between processes must be
pickable. PatchesMock replaces non-pickable objects used for patch location computing (classes: WholeImageGridPatches,
PolygonRegionRandomPatches, etc.) and it is used in pool classes for patch location computing and patch extraction.
"""

from collections import Counter


class PatchesMock:
    """Patches mock class."""

    def __init__(
        self,
        *,
        patch_data,
        wsi_file,
        level_or_mpp,
        patch_labels=[],
        check_location=True,
        class_name=None,
        class_params={},
    ):
        """Init method for PatchesMock class.

        Parameters
        ----------
        patch_data : list
            Precomputed patch data coming from one of patch location/sampling classes.

        wsi_file : str
            WSI file name or path.

        level_or_mpp : int or float
            WSI level or MPP value of used for calculated patches.

        patch_labels : list, optional
            List of all distinct labels present in the data (required to populate counters with zero values).

        check_location : bool, default=True
            Whether to run checks for duplicate locations or not.

        class_name : str, optional
            Class name used for calculating patch locations (used for internal logging).

        class_params : dict, optional
            Parameters of the class used for calculating patch locations (used for internal logging).
        """
        self._init_params = locals()
        self._patch_data = patch_data
        self._wsi_file = wsi_file
        self._level_or_mpp = level_or_mpp
        self._patch_labels = patch_labels
        self._class_name = class_name
        self._class_params = class_params
        self._patch_info = self._get_patch_info()
        if self._patch_labels:
            self._check_labels()
        if check_location:
            self._check_location()

    def _get_patch_info(self):
        all_labels = (patch[-1] for patch in self._patch_data)
        info_dict = dict(Counter(all_labels))
        for label in self._patch_labels:
            if label not in info_dict:
                info_dict[label] = 0
        return dict(sorted(info_dict.items()))

    def _check_labels(self):
        distinct_labels = list(self._patch_info.keys())
        for label in distinct_labels:
            if label not in self._patch_labels:
                raise ValueError("Invalid label found: %s" % label)

    def _check_location(self):
        duplicate_location = self._get_duplicate_location()
        if duplicate_location:
            raise ValueError("Duplicate location found: %s" % duplicate_location)

    def _get_duplicate_location(self):
        locations = (patch[0] for patch in self.patch_data)
        location_counts = Counter(locations)
        location_duplicates = {tuple_key: counter for tuple_key, counter in location_counts.items() if counter > 1}
        return location_duplicates

    @property
    def patch_count(self):
        """Return number of patches to be extracted."""
        return len(self._patch_data)

    @property
    def patch_data(self):
        """Return provided patch data."""
        return self._patch_data

    @property
    def patch_info(self):
        """Return patch information (labels and counts)."""
        return self._patch_info

    @property
    def patch_labels(self):
        """Return distinct patch labels."""
        if self._patch_labels:
            labels = sorted(self._patch_labels)
        else:
            labels = sorted(list(self._patch_info.keys()))
        return labels

    @property
    def param_info(self):
        """Return provided class params or init params."""
        if self._class_params:
            param_info = self._class_params
        else:
            param_info = {k: v for k, v in self._init_params.items() if k not in ["self"]}
        return param_info

    @property
    def level_or_mpp(self):
        """Return provided level or MPP value."""
        return self._level_or_mpp

    @property
    def wsi_file(self):
        """Return provided WSI file name."""
        return self._wsi_file

    @property
    def class_name(self):
        """Return provided patches class name or native object class name."""
        class_name = self._class_name if self._class_name else self.__class__.__name__
        return class_name
