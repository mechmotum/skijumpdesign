#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('skijumpdesign/version.py').read())

setup(
    name='skijumpdesign',
    version=__version__,
    author='Jason K. Moore',
    author_email='moorepants@gmail.com',
    url="https://gitlab.com/moorepants/skijumpdesign/",
    description='Ski Jump Design Tool For Equivalent Fall Height',
    long_description=open('README.rst').read(),
    keywords="engineering sports physics",
    license='MIT',
    py_modules=['dash_app'],
    packages=find_packages(),
    include_package_data=True,  # includes things in MANIFEST.in
    data_files=[('static', ['static/skijump.css'])],
    zip_safe=False,
    entry_points={'console_scripts':
                  ['skijumpdesign = dash_app:app.run_server']},
    install_requires=['numpy',
                      'scipy>=1.0',
                      'matplotlib',
                      'sympy',
                      'cython',
                      'fastcache',
                      ],
    extras_require={'app': ['plotly', 'dash', 'dash-renderer',
                            'dash-html-components', 'dash-core-components'],
                    'dev': ['pytest', 'sphinx', 'coverage']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
        ],
)
