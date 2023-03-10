#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'matplotlib']

test_requirements = ['pytest>=3', ]

setup(
    author="Daniele Prada",
    author_email='prada@imati.cnr.it',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description="A Python package for solving partial differential equations by the Virtual Element Method.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='vempy',
    name='vempy',
    packages=find_packages(include=['vempy', 'vempy.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/dprada85/vempy',
    version='0.1.0',
    zip_safe=False,
)
