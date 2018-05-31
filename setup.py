from distutils.core import setup
from Cython.Build import cythonize
"""
setup(
    name="partial_distance",
    ext_modules=cythonize('project/shared/partial_distance.pyx'),
)
"""

setup(
    name="calculate_contrast",
    ext_modules=cythonize('project/rar/contrast.pyx'),
)
