
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = [Extension("test_dsfmt", sources=['test_dsfmt.pyx', 'dSFMT/dSFMT.c'], 
                 include_dirs = ['dSFMT/'],
                 extra_compile_args=["-DDSFMT_MEXP=19937"])]

ext = cythonize(ext)

setup(name='test_dsfmt',
      ext_modules=ext) 



