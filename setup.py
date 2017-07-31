#!/usr/bin/env python

import os
from distutils.core import setup


setup(name='lsm',
      version='0.0.1',
      description='Bayesian inference for populations of networks with latent space models. Based on the work of Durante et al., 2016.',
      author='Scott Linderman',
      author_email='scott.linderman@gmail.com',
      url='https://github.com/blei-lab/factorial-network-models',
      install_requires=['numpy', 'scipy', 'matplotlib', 'pypolyagamma'],
      packages=['lsm']
     )
