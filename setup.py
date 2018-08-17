from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="partial_distance",
    ext_modules=cythonize("project/shared/partial_distance.pyx"),
)

setup(
    name="calculate_ks_contrast",
    ext_modules=cythonize("project/rar/ks_test.pyx"),
    include_dirs=[numpy.get_include()],
)
