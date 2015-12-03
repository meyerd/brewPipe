#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'This is BrewPipe, a pipelined data processing framework.',
    'author': 'Martin Kiechle & Dominik Meyer',
    'url': 'https://brewpipe.hosenhasser.de',
    'download_url': 'https://brewpipe.hosenhasser.de/lastest',
    'author_email': '',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['brewPipe'],
    'scripts': [],
    'name': 'brewPipe'
}

setup(**config)
