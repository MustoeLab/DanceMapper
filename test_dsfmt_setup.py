
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = [Extension("test", sources=['test_dsfmt.pyx', 'dSFMT/dSFMT.c'], 
                 extra_compile_args=["-DDSFMT_MEXP=19937"])]

ext = cythonize(ext)

setup(name='test',
      ext_modules=ext) 



