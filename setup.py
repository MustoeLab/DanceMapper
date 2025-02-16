#########################################################
# Code to compile readMutStrings.pyx
# Anthony Mustoe
# 2018
#
# This file is licensed under the terms of the MIT license
#
#########################################################

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
from pathlib import Path

# From RingMapper
import ringmapper


ringmapper_path = str(Path(ringmapper.__file__).parent)
ext = Extension(
    "accessoryFunctions",
    sources=["accessoryFunctions.pyx"],
    include_dirs=[numpy.get_include(), ringmapper_path],
)
setup(
    name="DanceMapper",
    ext_modules=cythonize(ext),
    scripts=["foldClusters.py", "plotClusters.py", "DanceMapper.py"],
    packages=find_packages(),
)
