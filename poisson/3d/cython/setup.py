from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

extra_compile_args=["-O3", "-ffast-math"]

# Define extensions
poisson = Extension("poisson",
                    sources=["poisson.pyx"],
                    include_dirs=[numpy.get_include()],
                    extra_compile_args=extra_compile_args)

Q1 = Extension("Q1",
               sources=["Q1.pyx"],
               include_dirs=[numpy.get_include()],
               extra_compile_args=extra_compile_args)

setup(
    name = "poisson",
    version = "0.0.1",
    description = "3D Poisson using Q1 finite elements",
    author = "Constantine Khroulev",
    author_email = "ckhroulev@alaska.edu",
    url = "https://github.com/ckhroulev/finite-elements",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [poisson, Q1]
    )

