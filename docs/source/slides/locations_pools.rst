=====================================
Patch locations/sampling pool classes
=====================================

Set of :doc:`/slides/pools` providing parallel execution interface for classes calculating patch locations described in
:doc:`/slides/locations`. Using pool classes for calculating patch locations allows to process multiple WSIs
at the same time and automatically collect patch/label counts for the entire collection of WSIs. Additionally, patch
location pool classes can be fed directly to :doc:`/slides/extractors_pools` for the actual patch extraction process.


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

Conversion from the single WSI processing code:

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

to the code where three WSIs will be processed at the same time using the same set of arguments:

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
            proc_num_workers=2,
        )

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

Conversion from the single WSI processing code:

    .. code-block:: python

        from dplabtools.slides.patches import WholeImageInvertedPoissonDiskPatches

        wsi_file = "/tmp/wsi1.svs"
        mask_file = "/tmp/mask1.png"
        polygons = [poly1, poly2] # list elements not defined here

        poisson_patches = WholeImageInvertedPoissonDiskPatches(
            wsi_file=wsi_file,
            mask_data=mask_file,
            patch_size=256,
            poisson_spacing=50,
            polygons=polygons,
        )

to the code where three WSIs will be processed at the same time using the same set of arguments:

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
            proc_num_workers=2,
        )

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

Conversion from the single WSI processing code:

    .. code-block:: python

        from dplabtools.slides.patches import PolygonRegionGridPatches

        wsi_file = "/tmp/wsi1.svs"
        mask_file = "/tmp/mask1.png"
        polygons = [poly1, poly2] # list elements not defined here

        grid_patches = PolygonRegionGridPatches(
            wsi_file=wsi_file,
            mask_data=mask_file,
            patch_size=256,
            patch_stride=1,
            polygons=polygons,
        )

to the code where three WSIs will be processed at the same time using the same set of arguments:

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
            proc_num_workers=2,
        )

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
