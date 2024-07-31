# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""Mixin classes for patch extraction.

Included classes add support for disk based patch extraction to base patch extractor classes.
"""

import os
import platform
import datetime
import glob
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from itertools import count

from dplabtools.slides.utils.wsi import get_level_and_mpp, get_basepatch_mpp
from dplabtools.slides.utils.multires import get_scaled_center_based_locations
from dplabtools.common import get_random_string


class AbstractDiskPatchMixin(ABC):
    """Abstract class for extracting patches and saving to disk.

    Should be extended separately for single and multi resolution patches.
    """

    def __init__(
        self, *, output_dir, image_type, filename_comment="", create_subdirs=False, filename_separator="_", **kwargs
    ):
        """Init method for AbstractDiskPatchMixin and derived classes.

        Parameters
        ----------
        output_dir : str
            Directory name of path for saving extracted patches.

        image_type : str
            Image type of saved files (PNG, JPG, etc.).

        filename_comment : str, optional
            Comment to be added to saved file names.

        create_subdirs : bool, default=False
            Whether to create label specific subdirectories inside `output_dir` or not.

        filename_separator : str, default="_"
            Separator used in saved file names.
        """
        super().__init__(**kwargs)
        self._abstract_params = locals()
        self._output_dir = output_dir
        self._image_type = image_type
        self._filename_comment = filename_comment
        self._create_subdirs = create_subdirs
        self._filename_separator = filename_separator
        self._patch_file_list = []
        self._manifest_dir_name = "manifests"
        self._manifest_file_template = "manifest__%s__%s.txt"
        self._manifest_file_name = None
        self._run()

    def _run(self):
        """Run all patch extraction steps."""
        self._check_output_dir(self._output_dir)
        self._init_slide()
        self._manifest_id = get_random_string(8)
        if self._create_subdirs:
            distinct_labels = self.patch_labels
            self._create_output_subdirs(distinct_labels, self._output_dir, self._pool_mode, self._wsi_name)
        self._create_manifest()
        self._add_manifest_header()
        self._save_patches(self._save_patch_files)
        self._add_manifest_data()
        self._count_saved_patches()

    @staticmethod
    def _check_output_dir(output_dir):
        """Check if patch output directory exists."""
        if not os.path.exists(output_dir):
            raise OSError('Output directory "%s" does not exist' % output_dir)

    @staticmethod
    def _create_output_subdirs(dir_names, output_dir, pool_mode, wsi_name):
        """Create label based directories for saved patches."""
        for name in dir_names:
            if not name:
                raise ValueError("%s: Cannot create directory with empty label name." % wsi_name)
            subdir_name = os.path.join(output_dir, name)
            if not os.path.exists(subdir_name):
                os.makedirs(subdir_name, exist_ok=pool_mode)

    def _create_manifest(self):
        """Create extraction manifest - wrapper function."""
        manifest_dir = self._create_manifest_dir(self._output_dir, self._manifest_dir_name, self._pool_mode)
        manifest_file_name = self._manifest_file_template % (self._manifest_id, self._wsi_id)
        self._manifest_file_name = self._create_manifest_file(manifest_dir, manifest_file_name, self._wsi_name)

    @staticmethod
    def _create_manifest_dir(output_dir, manifest_dir_name, pool_mode):
        """Create manifest directory."""
        manifest_dir = os.path.join(output_dir, manifest_dir_name)
        if not os.path.exists(manifest_dir):
            os.makedirs(manifest_dir, exist_ok=pool_mode)
        return manifest_dir

    @staticmethod
    def _create_manifest_file(manifest_dir, manifest_file_name, wsi_name):
        """Create manifest file."""
        manifest_path = os.path.join(manifest_dir, manifest_file_name)
        if os.path.exists(manifest_path):
            raise OSError('%s: Manifest file "%s" already exists' % (manifest_path, wsi_name))
        # create empty file
        with open(manifest_path, "w"):
            pass
        return manifest_path

    def _add_manifest_header(self):
        """Add header information to manifest file."""
        date_time = datetime.datetime.now()
        patch_object_params_str = "".join(
            [self._indent_line("", str(k) + ": " + str(v)) for k, v in sorted(self._patches.param_info.items())]
        )
        extractor_class_params = self._get_extractor_class_params(
            self._base_params, self._base_child_params, self._abstract_params, self._mixin_params
        )
        extractor_class_params_str = "".join(
            [self._indent_line("", str(elem[0]) + ": " + str(elem[1])) for elem in extractor_class_params]
        )
        with open(self._manifest_file_name, "a") as manifest_file:
            manifest_file.write(self._indent_line("Manifest ID:", self._manifest_id))
            manifest_file.write(self._indent_line("Manifest file path:", os.path.realpath(self._manifest_file_name)))
            manifest_file.write(self._indent_line("Date created:", date_time.strftime("%Y-%b-%d %H:%M:%S")))
            manifest_file.write(self._indent_line("Hostname:", platform.node()))
            manifest_file.write("-" * 50 + "\n")
            manifest_file.write(self._indent_line("Patch class:", self._patches.class_name))
            manifest_file.write("Patch class params:\n")
            manifest_file.write(patch_object_params_str)
            manifest_file.write(self._indent_line("Patch count:", str(self._patches.patch_count)))
            manifest_file.write(self._indent_line("Patch labels:", str(self._patches.patch_info)))
            manifest_file.write(self._indent_line("Extractor class:", self.__class__.__name__))
            manifest_file.write("Extractor class params:\n")
            manifest_file.write(extractor_class_params_str)

    @staticmethod
    def _get_extractor_class_params(*combined_params):
        """Get extractor class parameters."""
        class_params = []
        excluded_params = ("self", "patches", "__class__", "kwargs")
        for params in combined_params:
            param_list = [(pname, pvalue) for pname, pvalue in params.items() if pname not in excluded_params]
            for param in param_list:
                class_params.append((param[0], param[1]))
        return sorted(class_params)

    def _save_patches(self, save_function):
        """Call provided save_function in parallel mode."""
        function_args = self._get_expanded_patch_args_data()
        self._patch_file_list = self._get_patch_file_list(function_args)
        with ThreadPool(self._num_workers) as tpool:
            for _ in tpool.imap_unordered(save_function, function_args, chunksize=self._mp_chunksize):
                pass

    @abstractmethod
    def _save_patch_files(self):
        """Retrieve patch or patchset and save."""
        pass

    @abstractmethod
    def _get_expanded_patch_args_data(self):
        """Get expanded function parameters for multi-threaded execution."""
        pass

    @abstractmethod
    def _get_patch_file_list(self, function_args):
        """Get list of file paths corresponding to patches."""
        pass

    @staticmethod
    def _get_patch_file_data(wsi_id, patch, filename_comment, manifest_id):
        """Return information used to create patch file name."""
        patch_file_data = []
        location = patch[0]
        # size = patch[1] - never used
        label = patch[2]
        patch_file_data.append(wsi_id)
        patch_file_data.append(manifest_id)
        if filename_comment:
            patch_file_data.append(filename_comment)
        if label:
            patch_file_data.append(label)
        patch_file_data.append("x" + str(location[0]))
        patch_file_data.append("y" + str(location[1]))
        return patch_file_data

    @staticmethod
    def _get_patch_file_name(patch_file_data, filename_separator, image_type):
        """Create patch file name."""
        return filename_separator.join(patch_file_data) + "." + image_type

    @staticmethod
    def _get_patch_file_path(patch_file_name, label, create_subdirs, output_dir, patchset_dir=""):
        """Create path for saving patch."""
        subdir = label if create_subdirs else ""
        return os.path.join(output_dir, subdir, patchset_dir, patch_file_name)

    def _add_manifest_data(self):
        """Add list of saved patches to manifest file."""
        with open(self._manifest_file_name, "a") as manifest_file:
            manifest_file.write("Saved patches:\n")
            for patch_file in self._patch_file_list:
                manifest_file.write(self._indent_line("", patch_file))

    def _count_saved_patches(self):
        """Count saved patches.

        This function will raise an error if number of saved patches does not match number of expected patches.
        """
        patch_file_count = self._get_saved_patch_count(self._output_dir, self._manifest_id, self._image_type)
        expected_patch_count = self._get_expected_patch_count()
        if patch_file_count != expected_patch_count:
            raise OSError(
                "%s: Number of saved patches (%s) does not match the number of patch data elements (%s)"
                % (self._wsi_name, patch_file_count, expected_patch_count)
            )
        with open(self._manifest_file_name, "a") as manifest_file:
            manifest_file.write(self._indent_line("Disk file count:", str(patch_file_count)))

    @staticmethod
    def _get_saved_patch_count(output_dir, manifest_id, image_type):
        """Return number of patches found matching manifest id."""
        count_path = os.path.join(output_dir, "**", "*" + manifest_id + "*." + image_type)
        patch_file_count = len(glob.glob(count_path, recursive=True))
        return patch_file_count

    @abstractmethod
    def _get_expected_patch_count(self):
        """Get expected number of patches."""
        pass

    @staticmethod
    def _indent_line(left_str, right_str):
        """Indent and join strings with spaces for better readibility."""
        indent_chars = 24
        return left_str + (indent_chars - len(left_str)) * " " + right_str + "\n"

    @property
    def manifest_id(self):
        """Return current manifest id."""
        return self._manifest_id


class DiskPatchExtractorMixin(AbstractDiskPatchMixin):
    """Abstract class extention for disk single resolution patches."""

    def __init__(self, *, pool_mode=False, **kwargs):
        """Init method for DiskPatchExtractorMixin.

        Parameters
        ----------
        pool_mode : bool, dafault=False
            Internal flag for integration with the pool classes, not to be set by the user.
        """
        self._mixin_params = locals()
        self._pool_mode = pool_mode
        super().__init__(**kwargs)

    def _get_patch_file_list(self, function_args):
        return [elem[3] for elem in function_args]

    def _get_expanded_patch_args_data(self):
        """Implement abstract function."""
        expanded_data = []
        for patch in self._patch_data:
            file_path = self._get_patch_save_path(patch)
            data = [patch[0], self._level_or_mpp, patch[1], file_path]
            expanded_data.append(data)
        return expanded_data

    def _get_patch_save_path(self, patch):
        """Get path where patch will be saved."""
        # patch tuple is: (location, size, label)
        patch_file_data = self._get_patch_file_data(self._wsi_id, patch, self._filename_comment, self._manifest_id)
        patch_file_name = self._get_patch_file_name(patch_file_data, self._filename_separator, self._image_type)
        patch_file_path = self._get_patch_file_path(patch_file_name, patch[2], self._create_subdirs, self._output_dir)
        return patch_file_path

    def _save_patch_files(self, data):
        """Implement abstract function."""
        location, level_or_mpp, size, path = data
        patch = self._wsi_slide.get_region(location, level_or_mpp, size)
        patch.save(path)

    def _get_expected_patch_count(self):
        return self._patch_count


class MultiResDiskPatchExtractorMixin(AbstractDiskPatchMixin):
    """Abstract class extention for disk multi resolution patches."""

    # class variable for continuous patch set values between different WSIs
    _global_patchset_counter = None

    def __init__(self, *, global_counter=1, pool_mode=False, **kwargs):
        """Init method for MultiResDiskPatchExtractorMixin.

        Parameters
        ----------
        global_counter : int, default=1
            Initial counter value for enumerating patch set directories, setting value to ``None`` will
            disable the counter.

        pool_mode : bool, default=False
            See DiskPatchExtractorMixin class.
        """
        self._mixin_params = locals()
        self._patchset_name = "set%s"
        self._patchset_dirs = []
        self._global_counter = global_counter
        self._pool_mode = pool_mode
        self._patchset_counter = None
        self._init_patchset_counter()
        super().__init__(**kwargs)

    def _init_patchset_counter(self):
        if self._global_counter is None:
            self._patchset_counter = 0
        elif self._pool_mode or self._global_patchset_counter is None:
            MultiResDiskPatchExtractorMixin._global_patchset_counter = count(start=self._global_counter, step=1)
            self._patchset_counter = self._global_counter

    def _get_patch_file_list(self, function_args):
        patch_file_list = []
        for elem in function_args:
            file_list = list(elem[3])
            patch_file_list.extend(file_list)
        return patch_file_list

    def _get_expanded_patch_args_data(self):
        expanded_data = self._get_expanded_patch_args_data_custom()
        self._create_patchset_dirs()
        return expanded_data

    def _get_expanded_patch_args_data_custom(self):
        expanded_data = []
        basepatch_mpp = get_basepatch_mpp(self._patches.level_or_mpp, self._wsi_slide.level_mpp_values)
        for patch in self._patch_data:
            self._inc_patchset_counter()
            patchset_dir = self._add_patchset_dir(patch[2], self._patchset_counter)
            scaled_locations = get_scaled_center_based_locations(
                basepatch_mpp, patch[0], self._levels_or_mpps, patch[1], self._wsi_slide.level_mpp_values
            )
            patchset_path = self._get_patchset_save_path(patch, scaled_locations, patchset_dir)
            data = [scaled_locations, self._levels_or_mpps, patch[1], patchset_path]
            expanded_data.append(data)
        return expanded_data

    def _inc_patchset_counter(self):
        if self._global_counter is None:
            self._patchset_counter += 1
        else:
            self._patchset_counter = next(MultiResDiskPatchExtractorMixin._global_patchset_counter)

    def _add_patchset_dir(self, label, counter_value):
        patchset_dir_with_label = None
        patchset_dir = self._patchset_name % str(counter_value)
        if self._global_counter is None:
            patchset_dir = self._wsi_id + "__" + patchset_dir
        if self._create_subdirs:
            patchset_dir_with_label = os.path.join(label, patchset_dir)
        if patchset_dir_with_label:
            self._patchset_dirs.append(patchset_dir_with_label)
        else:
            self._patchset_dirs.append(patchset_dir)
        return patchset_dir

    def _create_patchset_dirs(self):
        for _dir in self._patchset_dirs:
            patchset_path = os.path.join(self._output_dir, _dir)
            if not os.path.exists(patchset_path):
                os.mkdir(patchset_path)
            else:
                raise OSError(
                    '%s: Patch set directory "%s" already exists' % (self._wsi_name, os.path.abspath(patchset_path))
                )

    def _get_patchset_save_path(self, patch, scaled_locations, patchset_dir):
        # patch tuple is: (location, size, label)
        patchset_path = []
        for _index, level_or_mpp in enumerate(self._levels_or_mpps):
            level, mpp = get_level_and_mpp(level_or_mpp)
            comment_suffix = "mpp" + str(mpp) if mpp else "level" + str(level)
            combined_comment = (
                self._filename_comment + self._filename_separator + comment_suffix
                if self._filename_comment
                else comment_suffix
            )
            patch_with_scaled_locations = (scaled_locations[_index], patch[1], patch[2])
            patch_file_data = self._get_patch_file_data(
                self._wsi_id, patch_with_scaled_locations, combined_comment, self._manifest_id
            )
            patch_file_name = self._get_patch_file_name(patch_file_data, self._filename_separator, self._image_type)
            patch_file_path = self._get_patch_file_path(
                patch_file_name, patch[2], self._create_subdirs, self._output_dir, patchset_dir
            )
            patchset_path.append(patch_file_path)
        return patchset_path

    def _save_patch_files(self, data):
        locations, levels_or_mpps, size, paths = data
        for location, level_or_mpp, path in zip(locations, levels_or_mpps, paths):
            patch = self._wsi_slide.get_region(location, level_or_mpp, size)
            patch.save(path)

    def _get_expected_patch_count(self):
        return self._patch_count * len(self._levels_or_mpps)

    @property
    def patchset_counter(self):
        """Return last used value for global patch counter."""
        return self._patchset_counter


class AbstractMemPatchMixin(ABC):
    """Abstract class for extracting in-memory patches.

    Should be extended separately for single and multi resolution patches.
    """

    def __init__(self, *, inference_mode=False, **kwargs):
        """Init method for mem patch extraction."""
        super().__init__(**kwargs)
        self._inference_mode = inference_mode
        self._patch_data_iterator = None
        self._init_slide()
        self._run()

    def _run(self):
        """Run all patch extraction steps."""
        function_args = self._get_expanded_patch_args_data()
        try:
            tpool = ThreadPool(self._num_workers)
            self._patch_data_iterator = tpool.imap_unordered(
                self._get_patch_data, function_args, chunksize=self._mp_chunksize
            )
        except Exception as e:
            # when using MemPatchExtractorPool some exceptions need separate close() action
            tpool.close()
            raise e
        tpool.close()

    @abstractmethod
    def _get_expanded_patch_args_data(self):
        """Get expanded function parameters for multi-threaded execution."""
        pass

    @abstractmethod
    def _get_patch_data(self):
        """Retrieve patch or patchset."""
        pass

    @property
    def patch_stream(self):
        """Return stream of memory images (handled by iterator object)."""
        return self._patch_data_iterator


class MemPatchExtractorMixin(AbstractMemPatchMixin):
    """Abstract class extention for in memory single resolution patches."""

    def _get_expanded_patch_args_data(self):
        expanded_data = []
        for patch_index, patch in enumerate(self._patch_data):
            data = [patch[0], self._level_or_mpp, patch[1], patch[2], patch_index]
            expanded_data.append(data)
        return expanded_data

    def _get_patch_data(self, data):
        location, level_or_mpp, size, label, patch_index = data
        patch = self._wsi_slide.get_region(location, level_or_mpp, size)
        patch_data = (patch, label, patch_index) if self._inference_mode else (patch, label)
        return patch_data


class MultiResMemPatchExtractorMixin(AbstractMemPatchMixin):
    """Abstract class extention for in memory multi resolution patches."""

    def _get_expanded_patch_args_data(self):
        expanded_data = []
        basepatch_mpp = get_basepatch_mpp(self._patches.level_or_mpp, self._wsi_slide.level_mpp_values)
        for patch_index, patch in enumerate(self._patch_data):
            scaled_locations = get_scaled_center_based_locations(
                basepatch_mpp, patch[0], self._levels_or_mpps, patch[1], self._wsi_slide.level_mpp_values
            )
            data = [scaled_locations, self._levels_or_mpps, patch[1], patch[2], patch_index]
            expanded_data.append(data)
        return expanded_data

    def _get_patch_data(self, data):
        patchset = []
        locations, levels_or_mpps, size, label, patch_index = data
        for location, level_or_mpp in zip(locations, levels_or_mpps):
            patch = self._wsi_slide.get_region(location, level_or_mpp, size)
            patch_data = (patch, label, patch_index) if self._inference_mode else (patch, label)
            patchset.append(patch_data)
        return patchset
