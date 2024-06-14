=======
Special
=======

.. _special-mrp-label:

Multi resolution patches (MRP)
==============================

Multi resolution patches are represented by a collection of patches calculated/retrieved at different resolutions
based on the center of the original patch. For the following original/base patch (``level_or_mpp=0.5``):

.. image:: images/special_base_patch.jpg
   :alt: Base patch

the following set is considered its multi resolution counterpart (``levels_or_mpps=[0.3, 0.4, 0.6, 0.7]``):

.. image:: images/special_mrp_patch1.jpg
   :alt: MRP patch1

.. image:: images/special_mrp_patch2.jpg
   :alt: MRP patch2

.. image:: images/special_mrp_patch3.jpg
   :alt: MRP patch3

.. image:: images/special_mrp_patch4.jpg
   :alt: MRP patch4

In ``dplabtools`` multi resolution patches are created by providing a group of ``level_or_mpp`` values into a list based
variable called ``levels_or_mpps``.


.. _special-level-or-mpp-label:

level_or_mpp
============

``level_or_mpp`` is a dual purpose concept with the following interpretation:

* if an `int` value is assigned, then ``level_or_mpp`` will be considered a WSI level.
* if a `float` value is assigned, then ``level_or_mpp`` will be considered an MPP (microns per pixel) value.

``level_or_mpp`` is mainly used in patch location calculations and patch extraction, to expand the range of possible
WSI resolution values.


.. _special-level-or-minsize-label:

level_or_minsize
================

``level_or_minsize`` is a dual purpose concept with the following interpretation:

* if a below the threshold `int` value is assigned, then ``level_or_minsize`` will be considered a WSI level.
* if an above the threshold `int` value is assigned, then the closest lowest WSI level matching the desired size will
  be found and assigned.

``level_or_minsize`` is mainly used in WSI batch processing (e.g. mask generation), when the existence of the specific
WSI levels is not always guaranteed. To alleviate this problem, using ``level_or_minsize`` allows to calculate WSI level
values dynamically for each processed WSI. [add threshold information]


.. _special-annotation-polygon-label:

AnnotationPolygon class
=======================

``AnnotationPolygon`` is an utility class that allows to create easy to interpret polygon objects corresponding to
available WSI annotations:

.. code-block:: python

    from dplabtools.slides.utils import AnnotationPolygon

    polygon = AnnotationPolygon(
        points=[(10000, 22000), (10000, 25000), (13000, 25000), (13000, 22000)],
        label="group1",
        holes=[],
    )

``AnnotationPolygon`` allows the user to feed WSI annotation based data into various ``dplabtools`` classes in
a friendly and standardized way.
