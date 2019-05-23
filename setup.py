
#########################################################
# Code to compile readMutStrings.pyx
# Anthony Mustoe
# 2018
#
# This file is licensed under the terms of the MIT license  
#
#########################################################


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext = Extension('accessoryFunctions', 
                sources=['accessoryFunctions.pyx'], 
                include_dirs = [numpy.get_include()])

setup(
    name = "accessoryFunctions",
    ext_modules = cythonize(ext)
)
