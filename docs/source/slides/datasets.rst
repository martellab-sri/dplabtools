========
Datasets
========

Dataset classes are required by the ``WSIInference`` class to streamline the process of patch extraction based on
user-defined parameters. Two WSI dataset classes, available in ``dplabtools``, integrate with patch location classes
introduced in :doc:`/slides/locations`.

.. note::

   Since WSI dataset classes are descendants of PyTorch `Dataset` class, correctly installed PyTorch is required to
   import them in Python code.


WSI dataset class
=================

.. autoclass:: dplabtools.slides.processing.WSIDataset(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches`` object represents one of the classes introduced in :doc:`/slides/locations`:

.. code-block:: python

    from dplabtools.slides.processing import WSIDataset

    dataset = WSIDataset(patches=patches)

.. seealso::
    :ref:`inference-full-example-label` using WSIDataset

Class details
-------------

``WSIDataset`` has no class specific parameters.

.. seealso::
    :ref:`wsidataset-common-parameters-label`

.. seealso::
    :ref:`wsidataset-common-methods-label`


WSI dataset class (MRP)
=======================

.. autoclass:: dplabtools.slides.processing.WSIMultiResDataset(...)
   :class-doc-from: class

Basic usage
-----------

Assuming that ``patches`` object represents one of the classes introduced in :doc:`/slides/locations`:

.. code-block:: python

    from dplabtools.slides.processing import WSIMultiResDataset

    dataset = WSIMultiResDataset(patches=patches, levels_or_mpps=[0.5, 0.7. 0.9])


Class details
-------------

Parameters specific to ``WSIMultiResDataset``:

.. autoparam::
   :params: levels_or_mpps
   :paths: dplabtools.slides.processing.dataset.dataset.WSIMultiResDataset,

.. seealso::
    :ref:`wsidataset-common-parameters-label`

.. seealso::
    :ref:`wsidataset-common-methods-label`

.. seealso::
    :ref:`misc-level-or-mpp-label`


.. _wsidataset-common-parameters-label:

Parameters common in all WSI dataset classes
============================================

.. autoparam::
   :params: patches, transform_fn, resampling_mode, extra_mpps, zero_workers, save_patches_dir
   :paths: dplabtools.slides.processing.dataset.base.BaseDataset,
           dplabtools.slides.processing.dataset.base.BaseDataset,
           dplabtools.slides.processing.dataset.base.BaseDataset,
           dplabtools.slides.processing.dataset.base.BaseDataset,
           dplabtools.slides.processing.dataset.base.BaseDataset,
           dplabtools.slides.processing.dataset.base.BaseDataset,


.. _wsidataset-common-methods-label:

Methods and properties common to all WSI dataset classes
========================================================

Common methods and properties are derived from the base class.

.. autoclass:: dplabtools.slides.processing.dataset.base.BaseDataset(...)
    :class-doc-from: class
    :members:
    :exclude-members: worker_init
