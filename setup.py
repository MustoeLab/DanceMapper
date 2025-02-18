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

ext = Extension(
    "accessoryFunctions",
    sources=["./dancemapper/accessoryFunctions.pyx"],
    include_dirs=[numpy.get_include()],
)
setup(
    name="DanceMapper",
    packages=find_packages(include=["dancemapper", "dancemapper.*"]),
    package_dir={"dancemapper": "./"},
    ext_modules=cythonize(ext),
    scripts=[
        "./dancemapper/foldClusters.py",
        "./dancemapper/plotClusters.py",
        "./dancemapper/DanceMapper.py",
    ],
)
