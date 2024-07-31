# This file is part of the Digital Pathology Lab Tools (dplabtools) Python package.
#
# Copyright 2024 Sunnybrook Research Institute - All Rights Reserved.
#
# You may use, modify and distribute this code under the terms of the Apache 2.0 license provided
# in the root of this project, also available at: https://www.apache.org/licenses/LICENSE-2.0


"""dplabtools package configuration/installer."""

import os
import setuptools

package_name = "dplabtools"
url = "https://github.com/martellab-sri/dplabtools"
__version__ = ""

base_path = os.path.dirname(__file__)
version_file = os.path.join(base_path, package_name, "_version.py")
readme_file = os.path.join(base_path, "README.md")

with open(version_file) as f:
    exec(f.read())

with open(readme_file) as f:
    long_description = f.read()

setuptools.setup(
    name=package_name,
    version=__version__,
    description="Digital Pathology Lab Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sunnybrook Research Institute",
    url=url,
    download_url=url,
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pillow>=9.1.0",
        "opencv-python<4.10",
        "scikit-image",
        "matplotlib<3.8",
        "shapely<=1.8.5",
        "tifffile",
        "tiffslide",
        "openslide-python",
        "paquo",
    ],
    license="Apache 2.0",
)
