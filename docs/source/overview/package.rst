================
Package overview
================

Package ``dplabtools`` provides a set of Python classes for automating tasks typical in Digital Pathology experiments.
The package has been thoughtfully designed, keeping scalability, performance and ease of future maintenance in mind.
Currently the package is covered by over 850 test cases with over 2450 assertions.


Main features
=============

* Support for different Whole Slide Image (WSI) reading libraries by introducing the concept
  of :doc:`/slides/genericslide` class.
* WSI foreground/background :doc:`/slides/masks` generation.
* Support for different :doc:`/slides/locations` methods and the ability to generate patch location preview images.
* Efficient :doc:`/slides/extractors` methods for calculated patches.
* Support for :ref:`misc-mrp-label` across the whole package.
* :doc:`/slides/inference` class capable of running trained models on WSIs.
* Integrated :doc:`/slides/heatmaps` class for visualizing inference results.
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

All required dependencies will be installed automatically, apart from PyTorch, which should be installed manually
if ``dplabtools`` classes that depend on it are to be used. In that case, it is recommended that PyTorch be installed
first, followed by ``dplabtools``.

.. note::

    To function properly, OpenSlide Python package requires OpenSlide binaries to be installed. This installation 
    process varies between operating systems, more details can be found at `openslide.org <https://openslide.org/>`_.


Code repository
===============

Package ``dplabtools`` is hosted on `GitHub <https://github.com/martellab-sri/dplabtools/>`_.


Requirements
============

Package ``dplabtools`` is developed on Linux and tested on Linux. While most of the classes are expected to work
on other operating systems, there are also areas which have a high likelihood of not functioning properly (e.g. low level
multiprocessing code).

General requirements:

* Any Linux operating system
* Currently supported Python versions: 3.10 - 3.13
* Currently supported PyTorch versions: 1.x - 2.x

.. note::

   Python **3.14** is not currently supported.


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


Acknowledgements
================
Thanks to the following Digital Pathology projects and their maintainers/contributors, whose libraries improved this project:

* `OpenSlide <https://openslide.org/>`_
* `TiffSlide <https://pypi.org/project/tiffslide/>`_
* `Paquo <https://pypi.org/project/paquo/>`_


License and copyright
=====================

Package ``dplabtools`` is released under the terms of the Apache 2.0 license available
`here. <https://www.apache.org/licenses/LICENSE-2.0>`_

Copyright 2024-2026 Sunnybrook Research Institute - All Rights Reserved.

All trademarks are property of their respective owners.
