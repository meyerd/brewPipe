#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from ..pipelineState import PipelineStateInterface

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class WintonStockDataOutput(PipelineStateInterface):
    """
    Output the fitted data in the winton stock market
    challenge output format.
    """
    def __init__(self, output_file, overwrite=False):
        """
        Initialize the output writer
        :param output_file: The path to the output .csv file to
            be written.
        :param overwrite: Overwrite the output file if it already
            exists.
        """
        super(WintonStockDataOutput, self).__init__()

        self._output_file = output_file
        self._overwrite = overwrite

        if os.path.exists(self._output_file) and not self._overwrite:
            raise RuntimeError("output '%s' already exists" % \
                               self._output_file)
        self._data = None
        self._n_samples = 0
        self._prediction_size = 0

    def set_data(self, data):
        """
        Set the data to be output. The data has to be in the
        following shape:

        `data: BrewPipeDataFrame (n_samples x prediction_size)`

        where the `prediction_size` is usually 62 and `n_samples`
        120000.
        """
        self._data = data.data
        self._n_samples = self._data.shape[0]
        self._prediction_size = self._data.shape[1]

    def write(self):
        with open(self._output_file, 'w') as f:
            print >>f, "Id,Predicted"
            for i in xrange(self._n_samples):
                for j in xrange(self._prediction_size):
                    print >>f, "%i_%i,%f" % (i+1, j+1, self._data[i,j])

        return True
