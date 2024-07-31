=====================================
Patch locations/sampling pool classes
=====================================

``dplabtools`` offers a set of :doc:`/slides/pools` providing a parallel execution interface for classes described in
:doc:`/slides/locations`. Using pool classes for calculating patch locations allows for the processing of multiple WSIs
at the same time as well as automatically collecting patch and label counts for an entire collection of WSIs.
Additionally, patch location pool classes can be fed directly into :doc:`/slides/extractors_pools` for the actual
patch extraction process.


Pool classes for patches on whole images
========================================

The following pool classes share the same execution interface:

.. autoclass:: dplabtools.slides.patches.WholeImageRandomPatchesPool(...)
   :class-doc-from: class

.. autoclass:: dplabtools.slides.patches.WholeImagePoissonDiskPatchesPool(...)
   :class-doc-from: class

.. autoclass:: dplabtools.slides.patches.WholeImageGridPatchesPool(...)
   :class-doc-from: class

Basic usage
-----------

Conversion of the code that processes a single WSI:

    .. code-block:: python

        from dplabtools.slides.patches import WholeImageRandomPatches

        wsi_file = "/tmp/wsi1.svs"
        mask_file = "/tmp/mask1.png"

        random_patches = WholeImageRandomPatches(
            wsi_file=wsi_file,
            mask_data=mask_file,
            patch_size=256,
            num_patches=20,
        )

into the code where three WSIs are processed simultaneously using the same set of arguments:

    .. code-block:: python

        from dplabtools.slides.patches import WholeImageRandomPatchesPool

        wsi_file_list = ["/tmp/wsi1.svs", "/tmp/wsi2.svs", "/tmp/wsi3.svs"]
        mask_data_list = ["/tmp/mask1.png", "/tmp/mask2.png", "/tmp/mask3.png"]

        patches_args = {
            "patch_size": 256,
            "num_patches": 20,
        }

        random_patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_file_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=3,
        )

.. seealso::
    :ref:`patches-pool-patches-args-label`

Class details
-------------

Pool classes for patches on whole images have no class specific parameters.

.. seealso::
    :ref:`patches-pool-common-parameters-label`

.. seealso::
    :ref:`patches-pool-common-methods-label`


Pool classes for inverted patches on whole images
=================================================

The following pool classes share the same execution interface:

.. autoclass:: dplabtools.slides.patches.WholeImageInvertedRandomPatchesPool(...)
   :class-doc-from: class

.. autoclass:: dplabtools.slides.patches.WholeImageInvertedPoissonDiskPatchesPool(...)
   :class-doc-from: class

.. autoclass:: dplabtools.slides.patches.WholeImageInvertedGridPatchesPool(...)
   :class-doc-from: class

Basic usage
-----------

Conversion of the code that processes a single WSI:

    .. code-block:: python

        from dplabtools.slides.patches import WholeImageInvertedPoissonDiskPatches

        wsi_file = "/tmp/wsi1.svs"
        mask_file = "/tmp/mask1.png"
        polygons = [poly1, poly2] # list elements are not defined here

        poisson_patches = WholeImageInvertedPoissonDiskPatches(
            wsi_file=wsi_file,
            mask_data=mask_file,
            patch_size=256,
            poisson_spacing=50,
            polygon_data=polygons,
        )

into the code where three WSIs are processed simultaneously using the same set of arguments:

    .. code-block:: python

        from dplabtools.slides.patches import WholeImageInvertedPoissonDiskPatchesPool

        wsi_file_list = ["/tmp/wsi1.svs", "/tmp/wsi2.svs", "/tmp/wsi3.svs"]
        mask_data_list = ["/tmp/mask1.png", "/tmp/mask2.png", "/tmp/mask3.png"]
        polygon_data_list = [polygons1, polygons2, polygons3] # list elements are not defined here

        patches_args = {
            "patch_size": 256,
            "poisson_spacing": 50,
        }

        poisson_patches_pool = WholeImageInvertedPoissonDiskPatchesPool(
            wsi_file_list=wsi_file_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygon_data_list,
            patches_args=patches_args,
            proc_num_workers=3,
        )

.. seealso::
    :ref:`patches-pool-patches-args-label`

Class details
-------------

Parameters specific to pool classes for inverted patches on whole images:

.. autoparam::
   :params: polygon_data_list
   :paths: dplabtools.slides.patches.locations.pools.WholeImageInvertedPatchesPoolBase,

.. seealso::
    :ref:`patches-pool-common-parameters-label`

.. seealso::
    :ref:`patches-pool-common-methods-label`


Pool classes for patches on polygon regions
===========================================

The following pool classes share the same execution interface:

.. autoclass:: dplabtools.slides.patches.PolygonRegionRandomPatchesPool(...)
   :class-doc-from: class

.. autoclass:: dplabtools.slides.patches.PolygonRegionPoissonDiskPatchesPool(...)
   :class-doc-from: class

.. autoclass:: dplabtools.slides.patches.PolygonRegionGridPatchesPool(...)
   :class-doc-from: class

Basic usage
-----------

Conversion of the code that processes a single WSI:

    .. code-block:: python

        from dplabtools.slides.patches import PolygonRegionGridPatches

        wsi_file = "/tmp/wsi1.svs"
        mask_file = "/tmp/mask1.png"
        polygons = [poly1, poly2] # list elements are not defined here

        grid_patches = PolygonRegionGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_file,
            patch_size=256,
            patch_stride=1,
            polygon_data=polygons,
        )

into the code where three WSIs are processed simultaneously using the same set of arguments:

    .. code-block:: python

        from dplabtools.slides.patches import PolygonRegionGridPatchesPool

        wsi_file_list = ["/tmp/wsi1.svs", "/tmp/wsi2.svs", "/tmp/wsi3.svs"]
        mask_data_list = ["/tmp/mask1.png", "/tmp/mask2.png", "/tmp/mask3.png"]
        polygon_data_list = [polygons1, polygons2, polygons3] # list elements are not defined here

        patches_args = {
            "patch_size": 256,
            "patch_stride": 1,
        }

        grid_patches_pool = PolygonRegionGridPatchesPool(
            wsi_file_list=wsi_file_list,
            mask_data_list=mask_data_list,
            polygon_data_list=polygon_data_list,
            patches_args=patches_args,
            proc_num_workers=3,
        )

.. seealso::
    :ref:`patches-pool-patches-args-label`

Class details
-------------

Parameters specific to pool classes for patches on polygon regions:

.. autoparam::
   :params: polygon_data_list
   :paths: dplabtools.slides.patches.locations.pools.PolygonRegionGridPatchesPool,

.. seealso::
    :ref:`patches-pool-common-parameters-label`

.. seealso::
    :ref:`patches-pool-common-methods-label`


.. _patches-pool-patches-args-label:


patches_args
============

``patches_args`` is a Python dictionary object which should hold all necessary arguments for the
:doc:`/slides/locations` class of the user interest.


Preview images
==============

Similarly to the :doc:`/slides/locations` classes, patch related pool classes can also generate preview
images for each processed WSI. Arguments passed to the ``save_preview_image`` method should be passed in
a Python dictionary object assigned to ``save_preview_image_args``:

    .. code-block:: python

        from dplabtools.slides.patches import WholeImageRandomPatchesPool

        wsi_file_list = ["/tmp/wsi1.svs", "/tmp/wsi2.svs", "/tmp/wsi3.svs"]
        mask_data_list = ["/tmp/mask1.png", "/tmp/mask2.png", "/tmp/mask3.png"]

        patches_args = {
            "patch_size": 256,
            "num_patches": 20,
        }

        save_preview_image_args = {
            "image_file": "/tmp/_preview.tif",
            "level_or_minsize": 1,
            "thickness": 1,
        }

        random_patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_file_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=3,
            save_preview_image_args=save_preview_image_args,
        )

The ``image_file`` key in the dictionary object above is a suffix name for the new file names. The actual files
created for the preview images will be as follows:

 .. code-block:: bash

    /tmp/wsi1_preview.tif
    /tmp/wsi2_preview.tif
    /tmp/wsi3_preview.tif


Expandable parameters
=====================

Similarly to the :doc:`/slides/locations` classes, patch related pool classes also support expandable parameters
passed via ``patches_args``. In the example below, different ``foreground_ratio`` values will be used for different WSIs:

    .. code-block:: python

        from dplabtools.slides.patches import WholeImageRandomPatchesPool

        wsi_file_list = ["/tmp/wsi1.svs", "/tmp/wsi2.svs", "/tmp/wsi3.svs"]
        mask_data_list = ["/tmp/mask1.png", "/tmp/mask2.png", "/tmp/mask3.png"]

        patches_args = {
            "foreground_ratio": [0.5, 0.6, 0.7],
            "patch_size": 256,
            "num_patches": 20,
        }

        random_patches_pool = WholeImageRandomPatchesPool(
            wsi_file_list=wsi_file_list,
            mask_data_list=mask_data_list,
            patches_args=patches_args,
            proc_num_workers=3,
        )

In a more advanced scenario involving one of the polygon based classes, ``patches_args`` can also accept nested values
to customize selected arguments within each processed WSI:

    .. code-block:: python

        patches_args = {
            "foreground_ratio": [[0.5, 0.52, 0.54], [0.6, 0.65], [0.7]],
            "patch_size": 256,
            "num_patches": 20,
        }


.. _patches-pool-common-parameters-label:

Parameters common in all patches pool classes
=============================================

.. autoparam::
   :params: wsi_file_list, mask_data_list, patches_args, proc_num_workers, mp_chunksize, save_preview_image_args
   :paths: dplabtools.slides.patches.locations.pools.BasePatchesPool,
           dplabtools.slides.patches.locations.pools.BasePatchesPool,
           dplabtools.slides.patches.locations.pools.BasePatchesPool,
           dplabtools.slides.patches.locations.pools.BasePatchesPool,
           dplabtools.slides.patches.locations.pools.BasePatchesPool,
           dplabtools.slides.patches.locations.pools.BasePatchesPool,


.. _patches-pool-common-methods-label:

Methods and properties common to all patches pool classes
=========================================================

Common methods and properties are derived from the base class.

.. autoclass:: dplabtools.slides.patches.locations.pools.BasePatchesPool(...)
   :class-doc-from: class
   :members:
   :inherited-members:
