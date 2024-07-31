### Testing tips

  * tif format (not png) must be used as source when reading images with GenericSlide, converting images to numpy array,
    and comparing with reference images (png format is not deterministic)
  * for png images tiffslide and openslide give different output numpy arrays (most likely due to differences in
    decompression algorithms)
  * empty `__init__.py` must be present in directories with python test files to make test discovery work
