================
Package overview
================

Package ``dplabtools`` provides a set of Python classes for automating tasks typical in Digital Pathology experiments.
The package has been designed with high quality, expandability, performance and easiness of future maintenance in mind.
Currently the package is covered by over 800 test cases with over 2k assertions.


Main features
=============

* Support for different Whole Slide Image (WSI) reading libraries by introducing the concept
  of :doc:`/slides/genericslide` class.
* WSI foreground/background :doc:`/slides/masks` generation.
* Support for different :doc:`/slides/locations` methods and ability to generate patch location preview images.
* Efficient :doc:`/slides/extractors` methods for calculated patches.
* Support for :ref:`special-mrp-label` across the whole package.
* :doc:`/slides/inference` class capable of running trained models on WSIs.
* Integrated :doc:`/slides/heatmap` class for inference results visualization.
* Support for the most popular WSI :doc:`/slides/annotations` types.
* Specialized :doc:`/slides/pools` for parallel WSI processing.
* Extensive but not exhausting documentation :-)


Installation
============

To install the latest version available in the linked repository:

.. code-block:: bash

    pip install dplabtools

To install any previous/specific version:

.. code-block:: bash

    pip install dplabtools==X.Y.Z

All required dependencies will be installed automatically. Except for PyTorch, which is expected to be installed
manually, if PyTorch depended ``dplabtools`` classes are planned to be used.


Requirements
============

Package ``dplabtools`` is developed on Linux and tested on Linux. While most of the classes are expected to work
on other operating systems, there are also areas which are highly likely not to function properly (e.g. low level
multiprocessing code).

General requirements:

* Any Linux operating system
* Currently supported Python versions: 3.8 - 3.11
* Currently supported PyTorch versions: 1.x


Conventions
===========

The following conventions are used across the package:

* All user classes should be called with keyword-only arguments e.g.:

 .. code-block:: python

    s = GenericSlide(wsi_file="file1.svs")

 not: ``s = GenericSlide("file1.svs")``.

 Typical error when this convention is not followed:

 .. code-block:: bash

    TypeError: __init__() takes 1 positional argument but N were given

* All strings representing colors (e.g. "red", "blue") are defined in the
  `Matplotlib library. <https://matplotlib.org/stable/gallery/color/named_colors.html>`_


License and copyright
=====================

Package ``dplabtools`` is released under the terms of the Apache 2.0 license available
`here. <https://www.apache.org/licenses/LICENSE-2.0>`_

Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.

All trademarks are property of their respective owners.
