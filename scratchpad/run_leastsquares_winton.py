#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from brewPipe.data.winton import WintonStockData
from brewPipe.data import BrewPipeDataFrame
from brewPipe.output.winton import WintonStockDataOutput
from brewPipe.preprocess.numpy_null import NumpyNullPreprocessor
from brewPipe.models.tf_leastsquares import TensorflowLeastSquares


__author__ = 'Dominik Meyer <meyerd@mytum.de>'


if __name__ == '__main__':
    winton_training = WintonStockData(data_directory=os.path.join("data", "winton"),
                                      intermediate_directory="intermediates/",
                                      data_source="train")

    x = winton_training.intraday_2_120()
    y = winton_training.intraday_120_180()

    preproc_x = NumpyNullPreprocessor()
    preproc_y = NumpyNullPreprocessor()
    x = preproc_x.preprocess(x)
    y = preproc_y.preprocess(y)

    lsq = TensorflowLeastSquares(x.data.shape[1], y.data.shape[1],
                                 learn_rate=0.1, batch_size=1000,
                                 silent=False)
    lsq.set_data(x, y)
    lsq.run(max_steps=1000000)

    result = lsq.apply_model(x)
    errors = np.sum(np.abs(result.data - y.data), axis=1)

    # # plot
    # plt.figure()
    # plt.plot(np.arange(100), errors[:100])
    # plt.xlabel("sample")
    # plt.ylabel("error")
    # plt.show()

    winton_testing = WintonStockData(data_directory="data/", intermediate_directory="intermediates/",
                                     data_source="test")

    x1 = winton_testing.intraday_2_120()
    x2 = winton_testing.returns_last_days()

    preproc_x1 = NumpyNullPreprocessor()
    preproc_x2 = NumpyNullPreprocessor()
    x1 = preproc_x1.preprocess(x1)
    x2 = preproc_x1.preprocess(x2)

    # apply model to generate 121 - 180 data
    y1 = lsq.apply_model(x1)
    n_samples = y1.data.shape[0]

    # fill the rest of the data with zeros
    y2 = np.zeros((n_samples, 2))

    y = np.hstack([y1.data, y2])
    yframe = BrewPipeDataFrame('winton_output')
    yframe.data = y

    out = WintonStockDataOutput(output_file="submission_winton.csv",
                                overwrite=True)
    out.set_data(yframe)
    out.write()