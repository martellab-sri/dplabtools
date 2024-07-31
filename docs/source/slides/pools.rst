============
Pool classes
============

Pool classes provide a process based parallelization interface for classes that are designed to perform within a single CPU
process/core. In other words, pool classes allow the processing of multiple WSIs at the same time, with each WSI being worked
on by a single non-pool class process. The pool concept is based on the `ProcessPool` class present in the Python standard library.

``dplabtools`` offers two sets of pool classes for the most common and time consuming tasks:

* :doc:`/slides/locations_pools`
* :doc:`/slides/extractors_pools`
