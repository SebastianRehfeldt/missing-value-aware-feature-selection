from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    name="partial_distance",
    ext_modules=cythonize(
        Extension(
            "partial_distance",
            ["project/shared/partial_distance.pyx"],
            extra_compile_args=['/openmp'],
        )),
)

setup(
    name="calculate_contrast",
    ext_modules=cythonize(
        Extension(
            "calculate_contrast",
            ["project/rar/contrast.pyx"],
            extra_compile_args=['/openmp'],
        )),
    include_dirs=[numpy.get_include()],
)
