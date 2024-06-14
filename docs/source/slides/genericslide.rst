============
GenericSlide
============

``GenericSlide`` is a class providing abstraction for various WSI reading libraries, currently supported libraries
are `openslide` (default) and `tiffslide`. ``GenericSlide`` also supports treating flat image files as one level WSIs
using the `Pillow` library.

``GenericSlide`` class has been introduced to ensure that the package could maintain its full functionality in case
a prominent WSI reading library was ever to lose its support or even disappear.

.. note::

   In general it is not required to be familiar with the ``GenericSlide`` class to successfully use ``dplabtools``.
   Various classes and modules of ``dplabtools`` use ``GenericSlide`` internally and the integration details are
   transparent to the end user. ``GenericSlide`` can, however, be used independently as a standalone class in any
   Python context.

Other class features:

* Extracting patches based on WSI levels or arbitrary MPP (Microns Per Pixel) values.
* Down-sampling and up-sampling of WSIs at any MPP value using two different modes.
* Built-in checks for MPP values and location coordinates when reading WSI regions.
* Built-in warnings for invalid coordinates, invalid MPP values or reading close to WSI borders.
* Automated padding of extracted regions.
* Enhanced ability to extract WSI MPP values encoded in non-standard ways.
* Additional support for missing and invalid MPP values.
* Reading WSI embedded properties.


Basic usage
===========

.. code-block:: python

    from dplabtools.slides import GenericSlide

    wsi_file = "/tmp/wsi1.svs"
    slide = GenericSlide(wsi_file=wsi_file)

Changing WSI reading library
============================

The current WSI reading library can be changed dynamically using package settings, which must be done before any other
imports from ``dplabtools``::

    # change current slide library
    import dplabtools.config
    dplabtools.config.slide_library = "tiffslide"

    # import any other class from the package
    from dplabtools.slides import GenericSlide
    # or
    from dplabtools.slides.patches import WholeImageRandomPatches
    # or
    from dplabtools.slides.patches import DiskPatchExtractor


This change will affect all instances of ``GenericSlide`` created in the program code.

Automated resampling
====================

``GenericSlide`` can automatically down-sample/up-sample WSIs in one of the two modes: ``tile`` or ``wsi``.


1. ``tile`` mode - resampling individual regions on the fly:

* ``GenericSlide`` should be initiated with the argument ``resampling_mode="tile"``.
* Resampling procedure will be applied automatically when calling the built-in image region reading function
  :meth:`.get_region()` based on the ``level_or_mpp`` value provided::

    from dplabtools.slides import GenericSlide

    wsi_file = "/tmp/wsi1.svs"
    slide = GenericSlide(wsi_file=wsi_file, resampling_mode="tile")
    region = slide.get_region(location=(0, 0), level_or_mpp=0.333, size=(256, 256))


2. ``wsi`` mode - resampling the whole level upfront:

* In this mode extra WSI levels will be created upfront and stored in memory. Resampling happens immediately when
  the object is created and may take significant amount of resources.
* ``GenericSlide`` should be initiated with the argument ``resampling_mode="wsi"`` and with the desired MPP levels
  included in the list parameter called ``extra_mpps``::

    from dplabtools.slides import GenericSlide

    wsi_file = "/tmp/wsi1.svs"
    slide = GenericSlide(wsi_file=wsi_file, resampling_mode="wsi", extra_mpps=[0.55, 0.99])
    # possibly lengthy resampling process will commence now...
    # ...
    region1 = slide.get_region(location=(0, 0), level_or_mpp=0.55, size=(256, 256))
    region2 = slide.get_region(location=(0, 0), level_or_mpp=0.99, size=(256, 256))

* ``GenericSlide`` built-in image region reading function :meth:`.get_region()` can access all native WSI levels as well
  as the extra levels created for the custom MPP values.


Whether to use one mode or the other will depend largely on the number of patches extracted from a single WSI and on the
resampling performance in the particular environment.


Advanced features
=================

``GenericSlide`` provides a set of advanced features available as dedicated class methods:

* :meth:`.set_external_mpp()`
* :meth:`.set_mpp_level_margin()`
* :meth:`.set_mpp_round_decimal_places()`
* :meth:`.set_padding_margin_pixels()`
* :meth:`.set_range_max_magnification()`
* :meth:`.set_range_min_mpp()`
* :meth:`.set_resampling_filter()`

Example: missing MPP data
-------------------------

In case of missing MPP data in a WSI file its value can be imputed using :meth:`.set_external_mpp()`::

    from dplabtools.slides import GenericSlide
    GenericSlide.set_external_mpp(0.5)

    wsi_file = "/tmp/wsi1.svs"
    slide = GenericSlide(wsi_file=wsi_file, resampling_mode="tile")
    region = slide.get_region(location=(0, 0), level_or_mpp=0.25, size=(256, 256))

From now on every time a missing MPP value is encountered it will be automatically substituted with the value of 0.5.
A confirmation message of this operation will be logged.


Class details
=============

.. autoclass:: dplabtools.slides.GenericSlide(...)
    :members:
    :inherited-members:
    :exclude-members: check_mpp_data, check_mpp_range

.. seealso::
    :ref:`special-level-or-mpp-label`
