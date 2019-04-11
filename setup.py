from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "buildComutationMatrix",
    ext_modules = cythonize("optbuildComutationMatrix.pyx"),
    include_dirs = [numpy.get_include()]
)
