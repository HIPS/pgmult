from distutils.core import setup
import numpy as np

from Cython.Build import cythonize

setup(
    name='pgmult',
    version='0.1',
    description=
    "Learning and inference for models with multinomial observations and "
    "underlying Gaussian correlation structure. Examples include correlated "
    "topic model, multinomial linear dynamical systems, and multinomial "
    "Gaussian processes. ",
    author='Scott W. Linderman and Matthew James Johnson',
    author_email='slinderman@seas.harvard.edu, mattjj@csail.mit.edu',
    license="MIT",
    url='https://github.com/HIPS/pgmult',
    packages=['pgmult'],
    install_requires=[
        'Cython >= 0.20.1', 'numpy', 'scipy', 'matplotlib',
        'pybasicbayes', 'pypolyagamma', 'gslrandom', 'pylds'],
    ext_modules=cythonize('pgmult/**/*.pyx'),
    include_dirs=[np.get_include(),],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++',
    ],
    keywords=[
        'multinomial', 'polya', 'gamma', 'correlated topic model', 'ctm',
        'lds', 'linear dynamical system', 'gaussian process', 'gp'],
    platforms="ALL"
)



