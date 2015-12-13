#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Martin Kiechle <martin@cobrainer.com>'


class BrewPipeDataFrame(object):
    """
    The brewPipeDataFrame class is used to encapsulate
    the data with relevant information to it (e.g. naming).
    It can also be used to just pass around references
    to files on the disk, which should be loaded (see
    derived classes).
    """

    def __init__(self, name):
        """
        The class is to be used in the following way:
            - Instantiate the dataframe object with a name
            - Either put data directly into the data object
              variable
            - or put a reference to where the data can be loaded
              from (e.g. LMDB filename)
        :param name: Name of the dataframe (used for persisting)
        """
        self.name = name
        self.data = None
        self.location = None
