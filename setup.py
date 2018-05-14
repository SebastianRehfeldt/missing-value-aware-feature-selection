from distutils.core import setup
from Cython.Build import cythonize

setup(name="custom_distance", ext_modules=cythonize(
    'project/shared/c_distance.pyx'),)
