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
    "dance.accessoryFunctions",
    sources=["./dance/accessoryFunctions.pyx"],
    include_dirs=[numpy.get_include()],
)
setup(
    name="DanceMapper",
    packages=find_packages(include=["dance", "dance.*"]),
    package_dir={"": "./"},
    ext_modules=cythonize(ext),
    scripts=[
        "./dance/foldClusters.py",
        "./dance/plotClusters.py",
        "./dance/DanceMapper.py",
    ],
    install_requires=[
        "cython",
        "numpy",
        "matplotlib",
        "scipy",
        "StructureAnalysisTools @ git+ssh://git@github.com/psirving/StructureAnalysisTools.git@python3",
        "RingMapper @ git+ssh://git@github.com/Weeks-UNC/RingMapper.git@python3",
    ],
)
