
#########################################################
# Code to compile readMutStrings.pyx
# Anthony Mustoe
# 2018
#
# This file is licensed under the terms of the MIT license  
#
#########################################################

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


# get path to functions needed for mutstring I/O
import ringmapperpath

import sys
sys.path.append(ringmapperpath.path())

ext = Extension('accessoryFunctions', 
                sources=['accessoryFunctions.pyx'], 
                include_dirs = [numpy.get_include(), ringmapperpath.path()])

setup(
    name = "accessoryFunctions",
    ext_modules = cythonize(ext),
)
