============
Pool classes
============

Pool classes provide process based parallelization interface for classes designed to perform within a single CPU
process/core. In other words, pool classes allow to process multiple WSIs at the same time, with each WSI being worked on by
a single non-pool class process. The pool concept is based on ProcessPool present in the Python standard library.


``dplabtools`` offers two sets of pool classes for the most common and time consuming tasks:

* :doc:`/slides/locations_pools`
* :doc:`/slides/extractors_pools`
