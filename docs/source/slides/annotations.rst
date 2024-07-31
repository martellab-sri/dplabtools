===========
Annotations
===========

``dplabtools`` offers a set of classes capable of reading annotations saved by the following applications:
`Sedeen <https://pathcore.com/sedeen/>`_, `ASAP <https://computationalpathologygroup.github.io/ASAP/>`_, and
`QuPath <https://qupath.github.io/>`_. Processed annotations are converted to a polygon-like format and integrate
with other classes included in the package, they can also be saved as standardized JSON files for reuse or further
processing.

Notes:

* All non-polygon annotations (arrows, rulers, points, pure text labels, etc.) will be ignored during the annotation
  reading process.
* Since annotation labels can be expressed by different entities (text, color, object name, etc.) classes for
  annotation reading offer dedicated functions to provide the user with the necessary flexibility with regards to
  what constitutes a label.


Sedeen annotations
==================

.. autoclass:: dplabtools.slides.annotations.SedeenReader(...)
   :class-doc-from: class

Basic usage
-----------

.. code-block:: python

    from dplabtools.slides.annotations import SedeenReader

    xml_file = "/tmp/sedeen1.xml"

    def get_label(**kwargs):
        return kwargs["color"]

    reader = SedeenReader(data_file=xml_file, get_label_fn=get_label)
    reader.save_json("sedeen1.json")

Property ``reader.polygons`` can be used for in-memory annotation processing.

Class details
-------------

Parameters, methods and properties specific to ``SedeenReader`` are inherited from the base class.

.. seealso::
    :ref:`annotations-common-baseclass-label`


ASAP annotations
================

.. autoclass:: dplabtools.slides.annotations.AsapReader(...)
   :class-doc-from: class

Basic usage
-----------

.. code-block:: python

    from dplabtools.slides.annotations import AsapReader

    xml_file = "/tmp/asap1.xml"

    def get_label(**kwargs):
        return kwargs["color"]

    reader = AsapReader(data_file=xml_file, get_label_fn=get_label)
    reader.save_json("asap1.json")

Property ``reader.polygons`` can be used for in-memory annotation processing.

Class details
-------------

Parameters, methods and properties specific to ``AsapReader`` are inherited from the base class.

.. seealso::
    :ref:`annotations-common-baseclass-label`


.. _annotations-common-baseclass-label:

Parameters, methods and properties of base reader class
=======================================================

.. autoclass:: dplabtools.slides.annotations.readers.base.BaseReader(...)
   :class-doc-from: init
   :members:
   :inherited-members:

.. seealso::
    :ref:`special-annotation-polygon-label`


QuPath annotations
==================

Annotations saved by QuPath are not stored in an easily accessible format and it is necessary to read the whole QuPath
project to extract them. Additionally a working installation of QuPath is also necessary for this task to complete.

``QuPathProjectReader`` is a class dedicated to process saved QuPath projects and extract WSI annotations in bulk:

Basic usage
-----------

.. code-block:: python

    from dplabtools.slides.annotations import QuPathProjectReader

    reader = QuPathProjectReader(qupath_install_dir="/opt/QuPath/", qupath_project_dir="/data/project1/")
    reader.save_json("/tmp")

Output: annotations for all WSI present in the QuPath project will be saved as individual JSON files:

    .. code-block:: bash

        file1.svs.json
        file2.svs.json
        file3.svs.json

It is also possible to process them directly in memory by using the ``reader.polygon_data`` property.

Class details
-------------

.. autoclass:: dplabtools.slides.annotations.QuPathProjectReader(...)
   :class-doc-from: init
   :members:
   :inherited-members:

.. seealso::
    :ref:`special-annotation-polygon-label`


Integration with patch locations
================================

Extracted annotations can be passed to the classes calculating patches on polygon regions, either as a saved JSON file:

.. code-block:: python

    from dplabtools.slides.annotations import SedeenReader
    from dplabtools.slides.patches import PolygonRegionGridPatches

    wsi_file = "/tmp/wsi1.svs"
    mask_file = "/tmp/wsi1_mask.png"
    xml_file = "/tmp/sedeen1.xml"
    json_file = "sedeen1.json"

    def get_label(**kwargs):
        return kwargs["color"]

    reader = SedeenReader(data_file=xml_file, get_label_fn=get_label)
    reader.save_json(json_file)

    grid_patches = PolygonRegionGridPatches(
        wsi_file=wsi_file,
        mask_data=mask_file,
        patch_size=500,
        patch_stride=1,
        polygon_data=json_file,
    )

or as an in-memory object via the ``polygons`` property:

.. code-block:: python

    from dplabtools.slides.annotations import SedeenReader
    from dplabtools.slides.patches import PolygonRegionGridPatches

    wsi_file = "/tmp/wsi1.svs"
    mask_file = "/tmp/wsi1_mask.png"
    xml_file = "/tmp/sedeen1.xml"
    json_file = "sedeen1.json"

    def get_label(**kwargs):
        return kwargs["color"]

    reader = SedeenReader(data_file=xml_file, get_label_fn=get_label)

    grid_patches = PolygonRegionGridPatches(
        wsi_file=wsi_file,
        mask_data=mask_file,
        patch_size=500,
        patch_stride=1,
        polygon_data=reader.polygons,
    )

In case of the ``QuPathProjectReader`` class, the property holding all annotations in memory i.e. ``project_data`` would
have to be manipulated first to pass single image annotations to the ``polygon_data`` argument.
