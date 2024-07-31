# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Classes for multi-threaded patch extraction.

Final classes provided:
    MemPatchExtractor - extracting single patches in memory
    MultiResMemPatchExtractor - extracting multi resolution patches in memory

Classes to be expanded by mixins:
    BasePatchExtractor - single patches
    MultiResBasePatchExtractor - multi resolution patches
"""

from dplabtools.slides import GenericSlide
from dplabtools.slides.utils.wsi import get_wsi_id, get_wsi_name, get_level_and_mpp


class PatchExtractor:
    """Base class for any patch extraction class."""

    def __init__(
        self, *, patches, num_workers, mp_chunksize=1, resampling_mode=None, included_labels=[], excluded_labels=[]
    ):
        """Init method for PatchExtractor and derived classes.

        Parameters
        ----------
        patches : object
            Object representing one of the patch location classes.

        num_workers : int
            Number of thread workers in parallel processing.

        mp_chunksize : int, default=1
            Data chunk size used in parallel processing.

        resampling_mode : str, optional
            One of two supported down/up-sampling methods: ``wsi`` or ``tile``

        included_labels : list of str, optional
            Polygon labels included in patch extraction, all other labels will be ignored.

        excluded_labels : list of str, optional
            Polygon labels excluded from patch extraction.
        """
        self._base_params = locals()
        self._patches = patches
        self._num_workers = num_workers
        self._mp_chunksize = mp_chunksize
        self._resampling_mode = resampling_mode
        self._included_labels = included_labels
        self._excluded_labels = excluded_labels
        self._wsi_slide = None
        self._wsi_file = self._patches.wsi_file
        self._wsi_id = get_wsi_id(self._patches.wsi_file)
        self._wsi_name = get_wsi_name(self._patches.wsi_file)
        self._check_included_excluded_args()
        self._patch_data = self._patches.patch_data
        self._patch_labels = self._patches.patch_labels
        self._patch_count = self._patches.patch_count
        if included_labels or excluded_labels:
            self._filter_patches()

    def _check_included_excluded_args(self):
        """Check argument lists with included/excluded labels."""
        if self._included_labels and self._excluded_labels:
            raise ValueError("%s: Only one list of labels can be specified at a time" % self._wsi_name)

    def _filter_patches(self):
        """Apply included and excluded labels to patches list."""
        patch_data = []
        labels = []
        patch_count = 0
        for single_patch in self._patches.patch_data:
            label = single_patch[2]
            if self._included_labels:
                if label not in self._included_labels:
                    continue
            elif self._excluded_labels:
                if label in self._excluded_labels:
                    continue
            patch_data.append(single_patch)
            labels.append(label)
            patch_count += 1
        self._patch_data = patch_data
        self._patch_labels = labels
        self._patch_count = patch_count

    @property
    def patch_data(self):
        """Return the patch data used in the patch extraction process."""
        return self._patch_data

    @property
    def patch_labels(self):
        """Return the distinct patch labels used in the patch extraction process."""
        return sorted(list(set(self._patch_labels)))


class BasePatchExtractor(PatchExtractor):
    """Base class for single patch extraction."""

    def __init__(self, **kwargs):
        """Init method for BasePatchExtractor."""
        self._base_child_params = locals()
        super().__init__(**kwargs)
        self._level_or_mpp = self._patches.level_or_mpp

    def _init_slide(self):
        """Create slide object based on GenericSlide."""
        patch_level, patch_mpp = get_level_and_mpp(self._level_or_mpp)
        if self._resampling_mode == "wsi" and patch_mpp:
            extra_mpps = [patch_mpp]
        else:
            extra_mpps = []
        self._wsi_slide = GenericSlide(
            wsi_file=self._wsi_file, resampling_mode=self._resampling_mode, extra_mpps=extra_mpps
        )

    @property
    def patch_count(self):
        """Return the number of extracted patches."""
        return self._patch_count


class MultiResBasePatchExtractor(PatchExtractor):
    """Base class for multi resolution patch extraction."""

    def __init__(self, *, levels_or_mpps, **kwargs):
        """Init method for MultiResBasePatchExtractor.

        Parameters
        ----------
        levels_or_mpps : list of level_or_mpp values
            `Int` or `Float` numbers representing WSI levels or MPP values for multi resolution patches.
        """
        self._base_child_params = locals()
        super().__init__(**kwargs)
        self._levels_or_mpps = levels_or_mpps

    def _init_slide(self):
        """Create slide object based on GenericSlide."""
        if self._resampling_mode == "wsi":
            extra_mpps = self._get_extra_mpps(self._levels_or_mpps)
        else:
            extra_mpps = []
        self._wsi_slide = GenericSlide(
            wsi_file=self._wsi_file, resampling_mode=self._resampling_mode, extra_mpps=extra_mpps
        )

    @staticmethod
    def _get_extra_mpps(levels_or_mpps):
        """Return values which are MPPs."""
        extra_mpp_values = []
        for level_or_mpp in levels_or_mpps:
            level, mpp = get_level_and_mpp(level_or_mpp)
            if mpp:
                extra_mpp_values.append(mpp)
        return extra_mpp_values

    @property
    def patch_count(self):
        """Return the number of extracted patches."""
        return self._patch_count * len(self._levels_or_mpps)
