#!/usr/bin/env python

import os

from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(this_dir, 'skijumpdesign', 'version.py')).read())

setup(
    name='skijumpdesign',
    version=__version__,
    author='Jason K. Moore, Mont Hubbard',
    author_email='moorepants@gmail.com',
    url="http://www.skijumpdesign.info",
    description='Ski Jump Design Tool For Specified Equivalent Fall Height',
    long_description=open(os.path.join(this_dir, 'README.rst')).read(),
    keywords="engineering sports physics design",
    license='MIT',
    project_urls={
        'Web Application': 'http://www.skijumpdesign.info',
        'Library Documentation': 'http://skijumpdesign.readthedocs.io',
        'Source Code': 'https://gitlab.com/moorepants/skijumpdesign',
        'Issue Tracker': 'https://gitlab.com/moorepants/skijumpdesign/issues',
    },
    python_requires='>=3.5',
    py_modules=['skijumpdesignapp'],
    packages=find_packages(),
    include_package_data=True,  # includes things in MANIFEST.in
    data_files=[('', ['static/skijump.css'])],
    zip_safe=False,
    entry_points={'console_scripts':
                  ['skijumpdesign = skijumpdesignapp:app.run_server']},
    install_requires=['setuptools',
                      'numpy>=0.13.0',
                      'scipy>=1.0',  # requires solve_ivp
                      'matplotlib',
                      'sympy',
                      'cython',
                      'fastcache',
                      'flask',
                      'plotly',
                      'dash',
                      'dash-renderer',
                      'dash-html-components',
                      'dash-core-components'
                      ],
    extras_require={'dev': ['pytest',
                            'pytest-cov',
                            'sphinx',
                            'coverage',
                            'pyinstrument']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
        ],
)
