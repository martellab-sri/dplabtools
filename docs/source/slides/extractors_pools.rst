=============================
Patch extraction pool classes
=============================

Set of :doc:`/slides/pools` providing parallel execution interface for classes extracting calculated patches
described in :doc:`/slides/extractors`. Using pool classes for extracting patches allows to process multiple WSIs
at the same time and combine the results either in memory or by saving to disk. Patch extraction pool classes
have been designed to accept :doc:`/slides/locations_pools` as their input, however, they could also extract patches
calculated by other sources.


In memory patch extraction pool
===============================

.. autoclass:: dplabtools.slides.patches.MemPatchExtractorPool(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches_pool`` represents an object of one of the :doc:`/slides/locations_pools` classes, the following
example will create a list of in-memory patches from multiple WSIs and will print them on the screen:

.. code-block:: python

    from dplabtools.slides.patches import MemPatchExtractorPool

    extractor_pool = MemPatchExtractorPool(
        patches_pool=patches_pool,
        thread_num_workers=2,
        proc_num_workers=2,
    )

    for patch in extractor_pool.patch_list:
        print(patch)


In memory patch extraction pool (MRP)
=====================================

.. autoclass:: dplabtools.slides.patches.MultiResMemPatchExtractorPool(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches_pool`` represents an object of one of the :doc:`/slides/locations_pools` classes, the following
example will create a list of in-memory multi resolution patches from multiple WSIs and will print them on the screen:

.. code-block:: python

    from dplabtools.slides.patches import MultiResMemPatchExtractorPool

    extractor_pool = MultiResMemPatchExtractorPool(
        patches_pool=patches_pool,
        levels_or_mpps=[2, 1, 0],
        thread_num_workers=2,
        proc_num_workers=2,
    )

    for multires_patch in extractor_pool.patch_list:
        for patch in multires_patch:
            print(patch)


To disk patch extraction pool
=============================

.. autoclass:: dplabtools.slides.patches.DiskPatchExtractorPool(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches_pool`` represents an object of one of the :doc:`/slides/locations_pools` classes, the following
example will save extracted patches from multiple WSIs into `/tmp` directory:

.. code-block:: python

    from dplabtools.slides.patches import DiskPatchExtractorPool

    extractor_pool = DiskPatchExtractorPool(
        patches_pool=patches_pool,
        output_dir="/tmp",
        image_type="png",
        thread_num_workers=2,
        proc_num_workers=2,
    )


To disk patch extraction pool (MRP)
===================================

.. autoclass:: dplabtools.slides.patches.MultiResDiskPatchExtractorPool(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches_pool`` represents an object of one of the :doc:`/slides/locations_pools` classes, the following
example will save sets of extracted patches from multiple WSIs into `/tmp` directory:

.. code-block:: python

    from dplabtools.slides.patches import MultiResDiskPatchExtractorPool

    extractor_pool = DiskPatchExtractorPool(
        patches_pool=patches_pool,
        levels_or_mpps=[0, 1],
        output_dir="/tmp",
        image_type="png",
        thread_num_workers=2,
        proc_num_workers=2,
    )
