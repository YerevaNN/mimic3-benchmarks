#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='mimic3-benchmarks',
      version="1.0",
      description='Benchmarking ML tools for MIMIC 3',
      packages=find_packages(),
      install_requires=[
          'keras',
          'tensorflow',
          'numpy',
          'pandas',
          'scipy',
          'scikit-learn',
          'pyyaml',
      ],
      entry_points={
      },
      include_package_data=True,
      )
