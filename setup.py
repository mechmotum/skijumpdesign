#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('safeskijump/version.py').read())

setup(
    name='safeskjump',
    version=__version__,
    author='Jason K. Moore',
    author_email='moorepants@gmail.com',
    url="https://gitlab.com/moorepants/safeskijump/",
    description='Ski Jump Design Tool For Equivalent Fall Height',
    long_description=open('README.rst').read(),
    keywords="engineering sports physics",
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy>=1.0',
                      'matplotlib',
                      'sympy',
                      'cython',
                      'fastcache',
                      ],
    extras_require={'app': ['plotly', 'dash', 'dash-renderer',
                            'dash-html-components', 'dash-core-components'],
                    'dev': ['pytest', 'sphinx']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
        ],
)
