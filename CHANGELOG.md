# dplabtools changelog

## [0.0.5] 26-May-2026

- added support for newer Python versions
- added support for PyTorch 2.x
- updated dependencies
- updated documentation
- updated package configuration file
- redesigned `WHIHeatmap` class (breaking changes)
- added context manager to `GenericSlide`
- `WSIInference` now calls `eval()` automatically on loaded models
- renamed parameter in `WSIPolygonMask`: `polygons` --> `polygon_data` (breaking change)
- internal changes to `utils.wsi.compute_wsi_resolution_data` and `utils.image.save_tif_image_with_resolution`

## [0.0.4] 29-Jul-2024

- updated and revised documentation
- changed dependency versions (pillow, opencv-python, matplotlib)
- improvements to `WSIInference` class
- new feature: `level_zero_resampling`

## [0.0.3] 12-Jun-2024

- first public release
