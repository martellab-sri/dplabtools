# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Pool classes for patch extraction.

Included classes combine process based and thread based multiprocessing.
"""

import os
import functools
from abc import ABC, abstractmethod
from multiprocessing import Manager, Value, Pool as ProcessPool

from dplabtools.slides.patches import (
    PatchesMock,
    DiskPatchExtractor,
    MultiResDiskPatchExtractor,
    MemPatchExtractor,
    MultiResMemPatchExtractor,
)


class AbstractPatchExtractorPool(ABC):
    """Abstract base class for all extractor pool clases."""

    def __init__(
        self,
        *,
        patches_pool,
        proc_num_workers,
        thread_num_workers,
        proc_mp_chunksize=1,
        thread_mp_chunksize=1,
        resampling_mode=None,
        included_labels=[],
        excluded_labels=[],
    ):
        """Init method for AbstractPatchExtractorPool.

        Parameters
        ----------
        patches_pool : object
            Object representing one of the patch location pool classes.

        proc_num_workers : int
            Number of processes in the pool. This value corresponds directly to the number of WSIs to be processed
            simultaneously.

        thread_num_workers : int
            Number of threads per one worker process. This value indicates how many threads will be used to extract
            patches from a single WSI.

        proc_mp_chunksize : int, default=1
            Data chunk size used in process parallelization processing.

        thread_mp_chunksize : int, default=1
            Data chunk size used in thread parallelization processing.

        resampling_mode : str, optional
            One of two supported down/up-sampling methods: ``wsi`` or ``tile``

        included_labels : list of str, optional
            Polygon labels included in patch extraction, all other labels will be ignored.

        excluded_labels : list of str, optional
            Polygon labels excluded from patch extraction.
        """
        self._patches_pool = patches_pool
        self._proc_num_workers = proc_num_workers
        self._thread_num_workers = thread_num_workers
        self._proc_mp_chunksize = proc_mp_chunksize
        self._thread_mp_chunksize = thread_mp_chunksize
        self._resampling_mode = resampling_mode
        self._included_labels = included_labels
        self._excluded_labels = excluded_labels
        self._pids = []
        self._manifest_ids = []
        self._patch_count = 0
        self._run_pool()

    def _run_pool(self):
        """Run all processing steps."""
        process_pool_data = self._get_process_pool_data(self._patches_pool)
        process_args = self._get_single_process_total_args()
        with ProcessPool(self._proc_num_workers) as ppool:
            for extractor_bits in ppool.imap_unordered(
                functools.partial(
                    self._run_single_procces,
                    **process_args,
                ),
                process_pool_data,
                chunksize=self._proc_mp_chunksize,
            ):
                pid, manifest_id, patch_count = extractor_bits
                self._pids.append(pid)
                self._manifest_ids.append(manifest_id)
                self._patch_count += patch_count

    @staticmethod
    def _get_process_pool_data(patches_pool):
        """Convert all necessary data stored in patches objects to much simpler form."""
        process_pool_data = []
        for patches in patches_pool:
            process_pool_data.append(
                (patches.wsi_file, patches.patch_data, patches.level_or_mpp, patches.class_name, patches.param_info)
            )
        return process_pool_data

    @classmethod
    @abstractmethod
    def _run_single_procces(cls):
        """Run single process defined in child classes.

        Objects with references/pointers cannot be passed to other processes, so in child classes instead of objects
        basic data structures are passed in process_pool_data and then PatchesMock class is used to build
        (reconstruct) almost original patches class objects.
        """
        pass

    def _get_single_process_total_args(self):
        """Prepare complete dictionary with process keyword arguments."""
        total_args = self._get_single_process_base_args()
        extra_args = self._get_single_process_extra_args()
        total_args.update(extra_args)
        return total_args

    def _get_single_process_base_args(self):
        """Get process arguments defined in base/common class."""
        base_args = {}
        base_args["num_workers"] = self._thread_num_workers
        base_args["levels_or_mpps"] = self._levels_or_mpps
        base_args["mp_chunksize"] = self._thread_mp_chunksize
        base_args["resampling_mode"] = self._resampling_mode
        base_args["included_labels"] = self._included_labels
        base_args["excluded_labels"] = self._excluded_labels
        return base_args

    def _get_single_process_extra_args(self):
        """Get process arguments defined in child classes (optional)."""
        return {}

    @property
    def pids(self):
        """Return the IDs of the executed processes."""
        return self._pids

    @property
    def patch_count(self):
        """Return the number of extracted patches."""
        return self._patch_count


class AbstractMemPatchExtractorPool(AbstractPatchExtractorPool):
    """Abstract base class for all memory extractor pool clases."""

    def __init__(self, *, array_mode=False, **kwargs):
        """Init method for AbstractMemPatchExtractorPool."""
        self._array_mode = array_mode
        self._shared_list = Manager().list()
        super().__init__(**kwargs)

    @classmethod
    def _run_single_procces(
        cls,
        process_data,
        num_workers,
        levels_or_mpps,
        mp_chunksize,
        resampling_mode,
        included_labels,
        excluded_labels,
        shared_list,
    ):
        """Run single process for memory patch extraction."""
        pid = os.getpid()
        wsi_file, patch_data, level_or_mpp, patches_class_name, patches_class_params = process_data
        patches = PatchesMock(
            patch_data=patch_data,
            wsi_file=wsi_file,
            level_or_mpp=level_or_mpp,
            check_location=False,
            class_name=patches_class_name,
            class_params=patches_class_params,
        )
        extractor = cls._get_extractor_object(
            patches,
            num_workers,
            levels_or_mpps,
            mp_chunksize,
            resampling_mode,
            included_labels,
            excluded_labels,
        )
        shared_list.extend(list(extractor.patch_stream))
        return pid, None, extractor.patch_count

    def _get_single_process_extra_args(self):
        """Get additional parameters for the process."""
        extra_args = {}
        extra_args["shared_list"] = self._shared_list
        return extra_args

    @staticmethod
    @abstractmethod
    def _get_extractor_object(
        patches, num_workers, levels_or_mpps, mp_chunksize, resampling_mode, included_labels, excluded_labels
    ):
        """Get memory extractor object.

        It will be either MemPatchExtractor or MultiResMemPatchExtractor with corresponding arguments.
        """
        pass

    @property
    def patch_list(self):
        """Return the extracted patches stored in memory."""
        return self._shared_list


class MemPatchExtractorPool(AbstractMemPatchExtractorPool):
    """Extractor pool implementation for `MemPatchExtractor`."""

    def __init__(self, **kwargs):
        """Init method for MemPatchExtractorPool."""
        self._levels_or_mpps = []
        super().__init__(**kwargs)

    @staticmethod
    def _get_extractor_object(
        patches,
        num_workers,
        levels_or_mpps,  # never used here
        mp_chunksize,
        resampling_mode,
        included_labels,
        excluded_labels,
    ):
        """Get MemPatchExtractor object.

        levels_or_mpps is not used here (not a multi resolution extractor)
        """
        extractor = MemPatchExtractor(
            patches=patches,
            num_workers=num_workers,
            mp_chunksize=mp_chunksize,
            resampling_mode=resampling_mode,
            included_labels=included_labels,
            excluded_labels=excluded_labels,
        )
        return extractor


class MultiResMemPatchExtractorPool(AbstractMemPatchExtractorPool):
    """Extractor pool implementation for `MultiResMemPatchExtractor`."""

    def __init__(self, *, levels_or_mpps, **kwargs):
        """Init method for MultiResMemPatchExtractorPool.

        Parameters
        ----------
        levels_or_mpps : list of level_or_mpp values
            `Int` or `Float` numbers representing WSI levels or MPP values for multi resolution patches.
        """
        self._levels_or_mpps = levels_or_mpps
        super().__init__(**kwargs)

    @staticmethod
    def _get_extractor_object(
        patches,
        num_workers,
        levels_or_mpps,
        mp_chunksize,
        resampling_mode,
        included_labels,
        excluded_labels,
    ):
        """Get MultiResMemPatchExtractor object."""
        extractor = MultiResMemPatchExtractor(
            patches=patches,
            num_workers=num_workers,
            levels_or_mpps=levels_or_mpps,
            mp_chunksize=mp_chunksize,
            resampling_mode=resampling_mode,
            included_labels=included_labels,
            excluded_labels=excluded_labels,
        )
        return extractor

    @property
    def patchset_count(self):
        """Return the number of patch sets created during the extraction."""
        return len(self._shared_list)


class AbstractDiskPatchExtractorPool(AbstractPatchExtractorPool):
    """Base class for all disk extractor pool clases."""

    def __init__(
        self,
        *,
        output_dir,
        image_type,
        filename_comment="",
        filename_separator="_",
        create_subdirs=False,
        **kwargs,
    ):
        """Init method for AbstractDiskPatchExtractorPool.

        Parameters
        ----------
        output_dir : str
            Directory name or path for saving the extracted patches.

        image_type : str
            Image type of the saved files (PNG, JPG, etc.).

        filename_comment : str, optional
            Comment to be added to the saved file names.

        filename_separator : str, default="_"
            Separator used in the saved file names.

        create_subdirs : bool, default=False
            Whether to create label specific subdirectories inside `output_dir` or not.
        """
        self._output_dir = output_dir
        self._image_type = image_type
        self._filename_comment = filename_comment
        self._filename_separator = filename_separator
        self._create_subdirs = create_subdirs
        super().__init__(**kwargs)

    @classmethod
    def _run_single_procces(
        cls,
        process_data,
        output_dir,
        num_workers,
        levels_or_mpps,
        image_type,
        filename_comment,
        filename_separator,
        create_subdirs,
        mp_chunksize,
        resampling_mode,
        included_labels,
        excluded_labels,
    ):
        """Run single process for disk patch extraction.

        Objects with references/pointers cannot be passed to other processes, so primitive
        patch data is passed and then PatchesMock class is used to build patches class objects.
        """
        pid = os.getpid()
        num_patches = len(process_data[1])
        if cls._use_global_counter:
            current_counter = cls._increase_pool_counter_value(num_patches)
        else:
            current_counter = None

        wsi_file, patch_data, level_or_mpp, patches_class_name, patches_class_params = process_data
        patches = PatchesMock(
            patch_data=patch_data,
            wsi_file=wsi_file,
            level_or_mpp=level_or_mpp,
            check_location=False,
            class_name=patches_class_name,
            class_params=patches_class_params,
        )
        extractor = cls._get_extractor_object(
            patches,
            output_dir,
            num_workers,
            levels_or_mpps,
            image_type,
            filename_comment,
            filename_separator,
            create_subdirs,
            mp_chunksize,
            resampling_mode,
            included_labels,
            excluded_labels,
            current_counter,
        )
        return pid, extractor.manifest_id, extractor.patch_count

    def _get_single_process_extra_args(self):
        extra_args = {}
        extra_args["output_dir"] = self._output_dir
        extra_args["image_type"] = self._image_type
        extra_args["filename_comment"] = self._filename_comment
        extra_args["filename_separator"] = self._filename_separator
        extra_args["create_subdirs"] = self._create_subdirs
        return extra_args

    @staticmethod
    @abstractmethod
    def _get_extractor_object(
        patches,
        output_dir,
        num_workers,
        levels_or_mpps,
        image_type,
        filename_comment,
        filename_separator,
        create_subdirs,
        mp_chunksize,
        resampling_mode,
        included_labels,
        excluded_labels,
        global_counter,
    ):
        """Get disk extractor object.

        It will be either DiskPatchExtractor or MultiResDiskPatchExtractor with corresponding arguments.
        """
        pass

    @property
    def manifest_ids(self):
        """Return the IDs of the created manifests."""
        return self._manifest_ids


class DiskPatchExtractorPool(AbstractDiskPatchExtractorPool):
    """Extractor pool implementation for `DiskPatchExtractor`."""

    # Not used in this class, but called in the base class, so must exist
    _use_global_counter = False

    def __init__(self, **kwargs):
        """Init method for DiskPatchExtractorPool."""
        self._levels_or_mpps = []
        super().__init__(**kwargs)

    @staticmethod
    def _get_extractor_object(
        patches,
        output_dir,
        num_workers,
        levels_or_mpps,  # never used here
        image_type,
        filename_comment,
        filename_separator,
        create_subdirs,
        mp_chunksize,
        resampling_mode,
        included_labels,
        excluded_labels,
        global_counter,  # never used here
    ):
        """Get DiskPatchExtractor object.

        Since this is not a multi resolution extractor:
        - levels_or_mpps is not used here
        - global_counter is not used here
        """
        extractor = DiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=num_workers,
            image_type=image_type,
            filename_comment=filename_comment,
            filename_separator=filename_separator,
            create_subdirs=create_subdirs,
            mp_chunksize=mp_chunksize,
            resampling_mode=resampling_mode,
            included_labels=included_labels,
            excluded_labels=excluded_labels,
            pool_mode=True,
        )
        return extractor


class MultiResDiskPatchExtractorPool(AbstractDiskPatchExtractorPool):
    """Extractor pool implementation for `MultiResDiskPatchExtractor`."""

    # Shared value between different processes
    _pool_global_counter = Value("I", 0)
    # Flag for using global counter, will be updated automatically if global_counter is defined in init
    _use_global_counter = False

    def __init__(self, *, levels_or_mpps, global_counter, **kwargs):
        """Init method for MultiResDiskPatchExtractorPool.

        Parameters
        ----------
        levels_or_mpps : list of level_or_mpp values
            `Int` or `Float` numbers representing WSI levels or MPP values for multi resolution patches.

        global_counter : int, default=1
            Initial counter value for enumerating patch set directories (`set1`, `set2`, `set3`, ...) for an entire
            collection of WSIs. Setting this value to `None` will cause the patch set counter to be reset for
            each WSI (`wsi1_set1`, `wsi1_set2`, ... `wsi2_set1`, `wsi2_set2`, ...).
        """
        self._levels_or_mpps = levels_or_mpps
        if global_counter is not None:
            MultiResDiskPatchExtractorPool._use_global_counter = True
            MultiResDiskPatchExtractorPool._increase_pool_counter_value(global_counter)
        super().__init__(**kwargs)

    @classmethod
    def _increase_pool_counter_value(cls, value):
        """Increment global counter value in safe way.

        This function will return the previous value.
        """
        with cls._pool_global_counter.get_lock():
            previous_value = cls._pool_global_counter.value
            cls._pool_global_counter.value += value
        return previous_value

    @classmethod
    def reset_global_counter(cls):
        """Reset global counter value to zero.

        Only required in testing or when multiple instances of MultiResDiskPatchExtractorPool are created,
        which is not recommended.
        """
        cls._pool_global_counter = Value("I", 0)

    @classmethod
    def remove_global_counter(cls):
        """Remove global counter flag, as if MultiResDiskPatchExtractorPool was never used.

        Only required in testing or when multiple instances of MultiResDiskPatchExtractorPool are created,
        which is not recommended.
        """
        cls._use_global_counter = False

    @staticmethod
    def _get_extractor_object(
        patches,
        output_dir,
        num_workers,
        levels_or_mpps,
        image_type,
        filename_comment,
        filename_separator,
        create_subdirs,
        mp_chunksize,
        resampling_mode,
        included_labels,
        excluded_labels,
        global_counter,
    ):
        """Get MultiResDiskPatchExtractor object."""
        extractor = MultiResDiskPatchExtractor(
            patches=patches,
            output_dir=output_dir,
            num_workers=num_workers,
            levels_or_mpps=levels_or_mpps,
            image_type=image_type,
            filename_comment=filename_comment,
            filename_separator=filename_separator,
            create_subdirs=create_subdirs,
            mp_chunksize=mp_chunksize,
            resampling_mode=resampling_mode,
            included_labels=included_labels,
            excluded_labels=excluded_labels,
            global_counter=global_counter,
            pool_mode=True,
        )
        return extractor

    @property
    def patchset_count(self):
        """Return the number of patch sets created during the extraction."""
        patchset_count = None
        num_multires = len(self._levels_or_mpps)
        remainder = self._patch_count % num_multires
        if remainder != 0:
            raise ValueError("Invalid patch count values: %s, %s" % (self._patch_count, num_multires))
        patchset_count = self._patch_count // num_multires
        return patchset_count
