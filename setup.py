#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('skijumpdesign/version.py').read())

setup(
    name='skijumpdesign',
    version=__version__,
    author='Jason K. Moore',
    author_email='moorepants@gmail.com',
    url="https://gitlab.com/moorepants/skijumpdesign/",
    description='Ski Jump Design Tool For Specified Equivalent Fall Height',
    long_description=open('README.rst').read(),
    keywords="engineering sports physics",
    license='MIT',
    project_urls={
        'App': 'http://skijumpdesign.herokuapp.com',
        'Documentation': 'http://skijumpdesign.readthedocs.io',
        'Source': 'https://gitlab.com/moorepants/skijumpdesign',
        'Tracker': 'https://gitlab.com/moorepants/skijumpdesign/issues',
    },
    python_requires='>=3.5',
    py_modules=['skijumpdesignapp'],
    packages=find_packages(),
    include_package_data=True,  # includes things in MANIFEST.in
    data_files=[('static', ['static/skijump.css'])],
    zip_safe=False,
    entry_points={'console_scripts':
                  ['skijumpdesign = skijumpdesignapp:app.run_server']},
    install_requires=['setuptools',
                      'numpy',
                      'scipy>=1.0',  # requires solve_ivp
                      'matplotlib',
                      'sympy',
                      'cython',
                      'fastcache',
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
