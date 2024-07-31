=============================
Patch extraction pool classes
=============================

``dplabtools`` includes a set of :doc:`/slides/pools` providing a parallel execution interface for classes
described in :doc:`/slides/extractors`. Using pool classes for extracting patches allows for the processing of
multiple WSIs at the same time and combining the results either in memory or by saving to disk. Patch extraction
pool classes have been designed to accept :doc:`/slides/locations_pools` as their input, however, they could also
extract patches calculated by other sources.

.. warning::

   Currently ``MemPatchExtractorPool`` and ``MultiResMemPatchExtractorPool`` suffer from a performance degradation
   related to the default inter-process communication present in Python. While both classes deliver correct results,
   their use in large scale experiments is not recommended. This part of the package is currently pending a rewrite using a
   shared memory model.


In memory patch extraction pool
===============================

``MemPatchExtractorPool`` is a class for parallel extraction of patches which will reside in memory.

Basic usage
-----------

Assuming that ``patches_pool`` represents an object from one of the :doc:`/slides/locations_pools` classes, the following
example will create a list of in-memory patches from multiple WSIs and will print them on the screen:

.. code-block:: python

    from dplabtools.slides.patches import MemPatchExtractorPool

    extractor_pool = MemPatchExtractorPool(
        patches_pool=patches_pool,
        thread_num_workers=2,
        proc_num_workers=3,
    )

    for patch in extractor_pool.patch_list:
        print(patch)

Class details
-------------

.. autoclass:: dplabtools.slides.patches.MemPatchExtractorPool(...)
   :class-doc-from: class
   :members:
   :inherited-members:

.. seealso::
    :ref:`extractor-pool-common-all-parameters-label`


In memory patch extraction pool (MRP)
=====================================

``MultiResMemPatchExtractorPool`` is a class for parallel extraction of multi resolution patches which will reside in memory.

.. seealso::
    :ref:`misc-mrp-label`

Basic usage
-----------

Assuming that ``patches_pool`` represents an object from one of the :doc:`/slides/locations_pools` classes, the following
example will create a list of in-memory multi resolution patches from multiple WSIs and will print them on the screen:

.. code-block:: python

    from dplabtools.slides.patches import MultiResMemPatchExtractorPool

    extractor_pool = MultiResMemPatchExtractorPool(
        patches_pool=patches_pool,
        levels_or_mpps=[2, 1, 0],
        thread_num_workers=2,
        proc_num_workers=3,
    )

    for multires_patch in extractor_pool.patch_list:
        for patch in multires_patch:
            print(patch)

Class details
-------------

.. autoclass:: dplabtools.slides.patches.MultiResMemPatchExtractorPool(...)
   :class-doc-from: class
   :members:
   :inherited-members:

Parameters specific to ``MultiResMemPatchExtractorPool``:

.. autoparam::
   :params: levels_or_mpps
   :paths: dplabtools.slides.patches.extractors.pools.MultiResMemPatchExtractorPool,

.. seealso::
    :ref:`extractor-pool-common-all-parameters-label`


To disk patch extraction pool
=============================

``DiskPatchExtractorPool`` is a class for parallel extraction of patches which will be saved to disk.

Basic usage
-----------

Assuming that ``patches_pool`` represents an object from one of the :doc:`/slides/locations_pools` classes, the following
example will save extracted patches from multiple WSIs into `/tmp` directory:

.. code-block:: python

    from dplabtools.slides.patches import DiskPatchExtractorPool

    extractor_pool = DiskPatchExtractorPool(
        patches_pool=patches_pool,
        output_dir="/tmp",
        image_type="png",
        thread_num_workers=2,
        proc_num_workers=3,
    )

Class details
-------------

.. autoclass:: dplabtools.slides.patches.DiskPatchExtractorPool(...)
   :class-doc-from: class
   :members:
   :inherited-members:

.. seealso::
    :ref:`extractor-pool-common-disk-parameters-label`

.. seealso::
    :ref:`extractor-pool-common-all-parameters-label`


To disk patch extraction pool (MRP)
===================================

``MultiResDiskPatchExtractorPool`` is a class for parallel extraction of multi resolution patches which will be saved to disk.

.. seealso::
    :ref:`misc-mrp-label`

Basic usage
-----------

Assuming that ``patches_pool`` represents an object from one of the :doc:`/slides/locations_pools` classes, the following
example will save sets of extracted patches from multiple WSIs into the `/tmp` directory:

.. code-block:: python

    from dplabtools.slides.patches import MultiResDiskPatchExtractorPool

    extractor_pool = DiskPatchExtractorPool(
        patches_pool=patches_pool,
        levels_or_mpps=[0, 1],
        output_dir="/tmp",
        image_type="png",
        thread_num_workers=2,
        proc_num_workers=3,
    )

Class details
-------------

.. autoclass:: dplabtools.slides.patches.MultiResDiskPatchExtractorPool(...)
   :class-doc-from: class
   :members:
   :inherited-members:
   :exclude-members: remove_global_counter, reset_global_counter

Parameters specific to ``MultiResMemPatchExtractorPool``:

.. autoparam::
   :params: levels_or_mpps, global_counter
   :paths: dplabtools.slides.patches.extractors.pools.MultiResDiskPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.MultiResDiskPatchExtractorPool,

.. seealso::
    :ref:`extractor-pool-common-disk-parameters-label`

.. seealso::
    :ref:`extractor-pool-common-all-parameters-label`


.. _extractor-pool-common-disk-parameters-label:

Parameters common to disk patch extraction pool classes
=======================================================

.. autoparam::
   :params: output_dir, image_type, filename_comment, filename_separator, create_subdirs
   :paths: dplabtools.slides.patches.extractors.pools.AbstractDiskPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractDiskPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractDiskPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractDiskPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractDiskPatchExtractorPool,


.. _extractor-pool-common-all-parameters-label:

Parameters common to all patch extraction pool classes
======================================================

.. autoparam::
   :params: patches_pool, proc_num_workers, thread_num_workers, proc_mp_chunksize, thread_mp_chunksize, resampling_mode,
            included_labels, excluded_labels
   :paths: dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
           dplabtools.slides.patches.extractors.pools.AbstractPatchExtractorPool,
