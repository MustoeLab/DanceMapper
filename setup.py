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
    sources=["./dance/accessoryFunctions.pyx"],
    include_dirs=[numpy.get_include()],
)
setup(
    name="DanceMapper",
    packages=find_packages(include=["dance", "dance.*"]),
    package_dir={"dance": "./"},
    ext_modules=cythonize(ext),
    scripts=[
        "./dance/foldClusters.py",
        "./dance/plotClusters.py",
        "./dance/DanceMapper.py",
    ],
)
