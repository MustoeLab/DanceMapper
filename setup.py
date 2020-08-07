
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
import externalpaths

import sys
sys.path.append(externalpaths.ringmapper())

ext = Extension('accessoryFunctions', 
                sources=['accessoryFunctions.pyx', 'dSFMT/dSFMT.c'],
                include_dirs = [numpy.get_include(), externalpaths.ringmapper(), 'dSFMT/'],
                extra_compile_args=["-DDSFMT_MEXP=19937"]) # arg for dSFMT

setup(
    name = "accessoryFunctions",
    ext_modules = cythonize(ext),
)
