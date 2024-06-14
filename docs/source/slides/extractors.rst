================
Patch Extraction
================

``dplaptools`` provides a set of patch extraction classes, which integrate with :doc:`/slides/locations`
classes. Extracted patches can be saved to disk or stored in memory.

Other class features:

* Parallel patch extraction from a single WSI using multithreading.
* Support for multi resolution patches (MRP).
* Patch filtering using labels.
* Automated manifest files creation and patch file count checks.


In memory patch extraction
==========================

.. autoclass:: dplabtools.slides.patches.MemPatchExtractor(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches`` represents an object of one of the :doc:`/slides/locations` classes, the following example
will create a stream of in-memory patches and will print them on the screen:

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

``MemPatchExtractor`` has no class specific parameters.

.. seealso::
    :ref:`extractor-common-all-parameters-label`

.. seealso::
    :ref:`extractor-common-all-properties-label`


In memory patch extraction (MRP)
================================

.. autoclass:: dplabtools.slides.patches.MultiResMemPatchExtractor()
   :class-doc-from: class

.. seealso::
    :ref:`special-mrp-label`

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

Parameters specific to ``MultiResMemPatchExtractor``:

.. autoparam::
   :params: levels_or_mpps
   :paths: dplabtools.slides.patches.extractors.base.MultiResBasePatchExtractor,

.. seealso::
    :ref:`extractor-common-all-parameters-label`

.. seealso::
    :ref:`extractor-common-all-properties-label`

.. seealso::
    :ref:`special-level-or-mpp-label`


To disk patch extraction
========================

.. autoclass:: dplabtools.slides.patches.DiskPatchExtractor(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches`` represents an object of one of the :doc:`/slides/locations` classes, the following example
will save extracted patches into `/tmp` directory:

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

``DiskPatchExtractor`` has no class specific parameters.

.. seealso::
    :ref:`extractor-common-disk-parameters-label`

.. seealso::
    :ref:`extractor-common-all-parameters-label`

.. seealso::
    :ref:`extractor-common-all-properties-label`


To disk patch extraction (MRP)
==============================

.. autoclass:: dplabtools.slides.patches.MultiResDiskPatchExtractor(...)
   :class-doc-from: class

.. note::
    Each set of patches will be saved into a dedicated subdirectory.

.. seealso::
    :ref:`special-mrp-label`

Basic usage
-----------

Assuming that ``patches`` represents an object of one of the :doc:`/slides/locations` classes, the following example
will save sets of extracted patches into `/tmp` directory:

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

Parameters specific to ``MultiResDiskPatchExtractor``:

.. autoparam::
   :params: levels_or_mpps, global_counter
   :paths: dplabtools.slides.patches.extractors.base.MultiResBasePatchExtractor,
           dplabtools.slides.patches.extractors.mixins.MultiResDiskPatchExtractorMixin,

.. seealso::
    :ref:`extractor-common-disk-parameters-label`

.. seealso::
    :ref:`extractor-common-all-parameters-label`

.. seealso::
    :ref:`extractor-common-all-properties-label`


.. _extractor-common-disk-parameters-label:

Parameters common to all disk patch extraction classes
======================================================

.. autoparam::
   :params: output_dir, image_type, filename_comment, create_subdirs, filename_separator, pool_mode
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

.. _extractor-common-all-properties-label:

Properties common to all patch extraction classes
=================================================

.. index::Properties common to all patch extraction classes

Common properties are derived from the base class and also apply to MRP specific classes.

.. autoclass:: dplabtools.slides.patches.extractors.base.BasePatchExtractor(...)
   :class-doc-from: class
   :members:
   :inherited-members:
