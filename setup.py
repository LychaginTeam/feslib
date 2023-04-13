#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "matplotlib>=3.3",
    "scipy",
    "numpy",
]

setup(
    name="gaseostools",
    version="0.0.1",
    description="Thermodinamics, phase transitions, "
    + "applications for different gases, equations of state.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LychaginTeam/gaseostools",
    author="Alexey Batov",
    author_email="batov@ipu.ru",
    classifiers=[
        "Development Status :: Dev",
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "equations of state",
        "Legendrian and Lagrangian manifolds",
        "coexistence curves",
        "phase transitions",
        "thermodynamics",
    ],
    packages=find_packages(),
    include_package_data=False,
    install_requires=install_requires,
    python_requires=">=3.7",
)
