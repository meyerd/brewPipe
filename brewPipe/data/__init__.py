#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class BrewPipeDataFrame(object):
    """
    The brewPipeDataFrame class is used to encapsulate
    the data with relevant information to it (e.g. naming).
    It can also be used to just pass around references
    to files on the disk, which should be loaded (see
    derived classes).
    """

    def __init__(self, name, lazy_frame=False, hash=None, callback=None):
        """
        The class is to be used in the following way:
            - Instantiate the dataframe object with a name
            - Either put data directly into the data object
              variable
            - or put a reference to where the data can be loaded
              from (e.g. LMDB filename)
        :param name: Name of the dataframe (used for persisting)
        :param lazy_frame: Whether the data property of the
            dataframe should be evaluated lazily. This means
            the data is not actually retrieved when the
            dataframe is created, but instead a callback function
            will be evaluated upon access to the actual data.
        :param hash: The hash of the data when lazy evaluation
            is to be used in case the hash is already known. Supply
            0 otherwise.
        :param callback: The callback function, that will be called
            upon access to the data if it is requested. The callback
            has to accept as the first argument the name of the
            dataframe data, of which the hash then can be looked
            up by the dataclass that is to be encapsulated
        """
        self.name = name
        self._static_data = None
        self.hash = 0
        self.callback = None
        if lazy_frame:
            if hash is None:
                raise RuntimeError("a hash has to be supplied when using lazy_frame")
            self.hash = hash
            if callback is None:
                raise RuntimeError("a callback has to be supplied when using lazy_frame")
            self.callback = callback
        self._lazy_frame = lazy_frame

    @property
    def data(self):
        if not self._lazy_frame:
            return self._static_data
        return self.callback(self.name)

    @data.setter
    def data(self, value):
        if not self._lazy_frame:
            self._static_data = value
            return
        raise RuntimeError("lazy_frame does not support setting of data")

