from distutils.core import setup, Extension
from Cython.Build import cythonize
"""
setup(
    name="partial_distance",
    ext_modules=cythonize('project/shared/partial_distance.pyx'),
)
"""
"""
setup(
    ext_modules=cythonize(
        Extension(
            "contrast",
            sources=["project/rar/contrast.pyx"],
            language="c++",
        )))
      

from Cython.Distutils import build_ext

ext_modules = [
    Extension('contrast', ['project/rar/contrast.pyx'], language='c++')
]

setup(
    name='calculate_contrast',
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext})
"""
setup(
    name="calculate_contrast",
    ext_modules=cythonize('project/rar/contrast.pyx'),
)