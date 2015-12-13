#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from ..pipelineState import PipelineStateInterface
from ..data import BrewPipeDataFrame

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class NumpyNullPreprocessor(PipelineStateInterface):
    """
    This is an example class of preprocessor, that
    takes numpy data from the data loader and outputs
    numpy data again. Basically, it does nothing, and is
    just a testcase to get some interface definitions going.
    """

    def __init__(self, intermediate_directory="intermediates"):
        """
        :param intermediate_directory: Directory, where the
            intermediate pandas dataframe should be persisted
            to.
        """
        super(NumpyNullPreprocessor, self).__init__()

        self._intermediate_directory = intermediate_directory

    def _persist_numpy(self, arr, name):
        filename = os.path.join(self._intermediate_directory,
                                'NumpyNullPreprocessor' + name)
        with open(filename, 'w') as f:
            np.save(f, arr)
        return filename

    def _load_numpy(self, name):
        filename = os.path.join(self._intermediate_directory,
                                'NumpyNullPreprocessor' + name)
        with open(filename, 'r') as f:
            arr = np.load(f)
        return arr

    def preprocess(self, dataframe):
        name = self.get(dataframe.name)
        tmp = None
        if not name:
            org = dataframe.data
            # preprocessing would happen here and be put to tmp
            tmp = org
            self._persist_numpy(tmp, dataframe.name)
            self.put(dataframe.name, dataframe.name)
            name = dataframe.name
        else:
            tmp = self._load_numpy(name)
        r = BrewPipeDataFrame(name)
        r.data = tmp
        return r


