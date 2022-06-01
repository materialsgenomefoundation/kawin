import os
from setuptools import setup, Extension


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='kawin',
    author='Nicholas Ury',
    author_email='nury12n@gmail.com',
    description='Tool for simulating precipitation using the KWN model coupled with Calphad.',
    packages=['kawin', 'kawin.tests'],
    license='MIT',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://kawin.org/',
    version='0.1.1',
    install_requires=[
        'matplotlib>=3.3',
        'numpy>=1.13',
        'pycalphad>=0.10.1',
        'scipy',
        'setuptools_scm[toml]>=6.0',
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

)
