#!/usr/bin/env python

import os

from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(this_dir, 'skijumpdesign', 'version.py')).read())

setup(
    name='skijumpdesign',
    version=__version__,
    author='Jason K. Moore, Bryn Cloud, Mont Hubbard',
    author_email='moorepants@gmail.com',
    url="http://www.skijumpdesign.info",
    description=('Ski Jump Design and Analysis Tool For Specified Equivalent '
                 'Fall Height'),
    long_description=open(os.path.join(this_dir, 'README.rst')).read(),
    keywords="engineering sports physics design analysis",
    license='MIT',
    project_urls={
        'Web Application': 'http://www.skijumpdesign.info',
        'Library Documentation': 'http://skijumpdesign.readthedocs.io',
        'Source Code': 'https://gitlab.com/moorepants/skijumpdesign',
        'Issue Tracker': 'https://gitlab.com/moorepants/skijumpdesign/issues',
    },
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,  # includes things in MANIFEST.in
    zip_safe=False,
    entry_points={'console_scripts':
                  ['skijumpdesign = skijumpdesign.app:app.run_server']},
    install_requires=[
        'cython>=0.28.5',
        'dash-core-components',
        'dash-html-components',
        'dash-renderer',
        'dash-table',
        'dash>=0.22.0',  # when assets/ directory introduced
        'flask>=1.0.2',
        'matplotlib>=2.2.3',
        'numpy>=0.13.0',
        'pandas>=0.24.1',
        'plotly>=3.1.1',
        'scipy>=1.0',  # requires solve_ivp
        'setuptools>=8.0',
        'sympy>=1.2',
        'xlrd>=1.2.0',
        ],
    extras_require={'dev': ['pytest',
                            'pytest-cov',
                            'sphinx',
                            'coverage',
                            'pyinstrument']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Physics',
        ],
)
