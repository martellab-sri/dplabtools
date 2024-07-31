================
Patch Extraction
================

``dplaptools`` provides a set of patch extraction classes, which integrate with the :doc:`/slides/locations`
classes. Extracted patches can be saved to disk or stored in memory.

Other class features include:

* Parallel patch extraction from a single WSI using multi-threading.
* Support for multi resolution patches (MRP).
* Patch filtering using arbitrary labels.
* For patches saved to disk: automated manifest files creation and built-in file count checks.


In memory patch extraction
==========================

``MemPatchExtractor`` is a class designed to perform the extraction of patches which will reside in memory.

Basic usage
-----------

Assuming that ``patches`` represents an object of one of the :doc:`/slides/locations` classes, the following
example will create a stream of in-memory patches and will print them on the screen:

.. code-block:: python

    from dplabtools.slides.patches import MemPatchExtractor

    extractor = MemPatchExtractor(
        patches=patches,
        num_workers=4,
    )

    for patch in extractor.patch_stream:
        print(patch)


Class details
-------------

.. autoclass:: dplabtools.slides.patches.MemPatchExtractor(...)
   :class-doc-from: class
   :members:
   :inherited-members:

.. seealso::
    :ref:`extractor-common-all-parameters-label`


In memory patch extraction (MRP)
================================

``MultiResMemPatchExtractor`` is a class designed to perform the extraction of multi resolution patches which will
reside in memory.

.. seealso::
    :ref:`misc-mrp-label`

Basic usage
-----------

Assuming that ``patches`` represents an object of one of the :doc:`/slides/locations` classes, the following example
will create a stream of sets of in-memory patches and will print them on the screen:

.. code-block:: python

    from dplabtools.slides.patches import MultiResMemPatchExtractor

    extractor = MultiResMemPatchExtractor(
        patches=patches,
        levels_or_mpps=[0, 1],
        num_workers=4,
    )

    for multires_patch in extractor.patch_stream:
        for patch in multires_patch:
            print(patch)

Class details
-------------

.. autoclass:: dplabtools.slides.patches.MultiResMemPatchExtractor(...)
   :class-doc-from: class
   :members:
   :inherited-members:

Parameters specific to ``MultiResMemPatchExtractor``:

.. autoparam::
   :params: levels_or_mpps
   :paths: dplabtools.slides.patches.extractors.base.MultiResBasePatchExtractor,

.. seealso::
    :ref:`extractor-common-all-parameters-label`

.. seealso::
    :ref:`misc-level-or-mpp-label`


To disk patch extraction
========================

``DiskPatchExtractor`` is a class designed to perform the extraction of patches which will be saved to disk.

Basic usage
-----------

Assuming that ``patches`` represents an object of one of the :doc:`/slides/locations` classes, the following
example will save extracted patches into the `/tmp` directory:

.. code-block:: python

    from dplabtools.slides.patches import DiskPatchExtractor

    extractor = DiskPatchExtractor(
        patches=patches,
        output_dir="/tmp",
        image_type="png",
        num_workers=4,
    )

Class details
-------------

.. autoclass:: dplabtools.slides.patches.DiskPatchExtractor(...)
   :class-doc-from: class
   :members:
   :inherited-members:

.. seealso::
    :ref:`extractor-common-disk-parameters-label`

.. seealso::
    :ref:`extractor-common-all-parameters-label`


To disk patch extraction (MRP)
==============================

``MultiResDiskPatchExtractor`` is a class designed to perform the extraction of multi resolution patches which
will be saved to disk.

.. note::
    Each set of patches will be saved into a dedicated subdirectory.

.. seealso::
    :ref:`misc-mrp-label`

Basic usage
-----------

Assuming that ``patches`` represents an object of one of the :doc:`/slides/locations` classes, the following
example will save sets of extracted patches into the `/tmp` directory:

.. code-block:: python

    from dplabtools.slides.patches import MultiResDiskPatchExtractor

    extractor = MultiResDiskPatchExtractor(
        patches=patches,
        levels_or_mpps=[0, 1],
        output_dir="/tmp",
        image_type="png",
        num_workers=4,
    )

Class details
-------------

.. autoclass:: dplabtools.slides.patches.MultiResDiskPatchExtractor(...)
   :class-doc-from: class
   :members:
   :inherited-members:

Parameters specific to ``MultiResDiskPatchExtractor``:

.. autoparam::
   :params: levels_or_mpps, global_counter
   :paths: dplabtools.slides.patches.extractors.base.MultiResBasePatchExtractor,
           dplabtools.slides.patches.extractors.mixins.MultiResDiskPatchExtractorMixin,

.. seealso::
    :ref:`extractor-common-disk-parameters-label`

.. seealso::
    :ref:`extractor-common-all-parameters-label`


.. _extractor-common-disk-parameters-label:

Parameters common to disk patch extraction classes
==================================================

.. autoparam::
   :params: output_dir, image_type, filename_comment, filename_separator, create_subdirs, pool_mode
   :paths: dplabtools.slides.patches.extractors.mixins.AbstractDiskPatchMixin,
           dplabtools.slides.patches.extractors.mixins.AbstractDiskPatchMixin,
           dplabtools.slides.patches.extractors.mixins.AbstractDiskPatchMixin,
           dplabtools.slides.patches.extractors.mixins.AbstractDiskPatchMixin,
           dplabtools.slides.patches.extractors.mixins.AbstractDiskPatchMixin,
           dplabtools.slides.patches.extractors.mixins.DiskPatchExtractorMixin,


.. _extractor-common-all-parameters-label:

Parameters common to all patch extraction classes
=================================================

.. autoparam::
   :params: patches, num_workers, mp_chunksize, resampling_mode, included_labels, excluded_labels
   :paths: dplabtools.slides.patches.extractors.base.PatchExtractor,
           dplabtools.slides.patches.extractors.base.PatchExtractor,
           dplabtools.slides.patches.extractors.base.PatchExtractor,
           dplabtools.slides.patches.extractors.base.PatchExtractor,
           dplabtools.slides.patches.extractors.base.PatchExtractor,
           dplabtools.slides.patches.extractors.base.PatchExtractor,
